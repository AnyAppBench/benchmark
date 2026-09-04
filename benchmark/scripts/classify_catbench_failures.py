#!/usr/bin/env python3
"""Classify CATBench validator failures with a VLM failure-mode judge.

The script reads a CATBench matrix manifest, extracts failed episode pickle
records, compacts the available traces/logs into text, and writes per-episode
failure-mode judgments plus aggregate summaries. The judge is diagnostic only:
it never changes the programmatic validator's pass/fail verdict.
"""

from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import re
import shlex
import sys
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
  from PIL import Image  # type: ignore
  _PIL_AVAILABLE = True
except ImportError:
  _PIL_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

DEFAULT_FAILURE_MODES = (
    "planning",
    "grounding",
    "mixed_planning_grounding",
    "execution_tooling",
    "environment_or_evaluator",
    "unknown",
)
LEGACY_PICKLE_SHIM_MODULE_PREFIXES = ("android_env", "google.genai")
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")

TEXT_FIELDS_BY_PRIORITY = (
    "plan",
    "current_subgoal",
    "completed_plan",
    "progress_status",
    "last_summary",
    "last_action_thought",
    "last_action",
    "action_history",
    "action_pool",
    "action_outcomes",
    "error_descriptions",
    "raw_response_list",
    "thought",
    "thinking",
    "response",
    "action_desc",
    "action",
    "action_output",
    "agent_output",
    "error",
    "summary",
)

IMAGE_LIKE_FIELDS = {
    "raw_screenshot",
    "screenshot",
    "before_screenshot",
    "after_screenshot",
    "before_screenshot_with_som",
    "after_screenshot_with_som",
}

SYSTEM_PROMPT = """You are a CATBench failure-mode classifier.

You receive a CATBench episode whose programmatic validator already returned
is_successful = 0. Your job is NOT to second-guess that verdict. Your job is
to classify the dominant reason the agent failed.

You see: the goal, the recorded `is_successful` flag (always 0 for cases
routed to you), a compact step trace, side artifacts, and a STITCHED set of
key screenshots (first, last, error-adjacent, repeated-action, and any
transition steps).

Return one JSON object only, exactly:
{
  "primary_failure_mode": "planning|grounding|mixed_planning_grounding|execution_tooling|environment_or_evaluator|unknown",
  "planning_score": 0-3,
  "grounding_score": 0-3,
  "confidence": "low|medium|high",
  "rationale": "one concise paragraph",
  "evidence": ["short quoted/paraphrased evidence", "screenshot:step_4 shows ..."]
}

Rules:
- "planning" applies when the agent chose a wrong subgoal, missed a required
  step, stopped early, or committed to an infeasible strategy.
- "grounding" applies when the plan was right but the agent could not locate
  or interact with the correct UI element.
- "mixed_planning_grounding" applies when both meaningfully contributed.
- "execution_tooling" applies when the agent's framework (parser, tool call,
  API timeout, malformed action) is the dominant failure source.
- "environment_or_evaluator" applies when an app crash, missing permission,
  network outage, or a likely validator false-negative is the dominant
  source. A true validator false-negative is reported separately by the
  validator audit in Sec. 6.7; flagging one here is a hypothesis, not a
  verdict.
- "unknown" when the evidence is genuinely insufficient.
- Scores: 0 absent / 1 possible / 2 likely / 3 dominant for each axis.
- Cite at least one step number or screenshot index per non-trivial mode.
- Do NOT include markdown, prose, or extra fields.
"""


# Strict structured-output contract used by OpenAI-compatible judge servers
# (including the vLLM Qwen3-VL judge).  Keep this synchronized with the JSON
# object requested in SYSTEM_PROMPT.
JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "primary_failure_mode": {
            "type": "string",
            "enum": list(DEFAULT_FAILURE_MODES),
        },
        "planning_score": {"type": "integer", "minimum": 0, "maximum": 3},
        "grounding_score": {"type": "integer", "minimum": 0, "maximum": 3},
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "rationale": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "primary_failure_mode",
        "planning_score",
        "grounding_score",
        "confidence",
        "rationale",
        "evidence",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT_TRIAGE = """You are a strict CATBench fast-screening judge.

Decide ONLY whether the agent's recorded behaviour gives IRREFUTABLE evidence of
success/failure on the goal. Escalate to the next stage whenever the evidence
is incomplete.

You will see: the goal, the recorded `is_successful` flag (which can be wrong),
a compact step trace, side artifacts, and (if provided) a few key screenshots.

Return one JSON object only, exactly:
{
  "verdict": "success|failure|uncertain",
  "agree_with_recorded_label": true|false,
  "primary_failure_mode": "planning|grounding|mixed_planning_grounding|execution_tooling|environment_or_evaluator|unknown",
  "confidence": "low|medium|high",
  "needs_escalation": true|false,
  "rationale": "one short paragraph"
}

Rules:
- "success" requires the screenshots/trace to plainly demonstrate the goal was
  met. If you have to infer or guess, return "uncertain".
- Set needs_escalation=true whenever verdict is "uncertain" OR confidence is
  "low" OR primary_failure_mode is "mixed_planning_grounding"/"unknown".
- agree_with_recorded_label=false flags likely evaluator/environment noise.
- Do NOT include markdown, prose, or extra fields.
"""


SYSTEM_PROMPT_DEEP = """You are a CATBench full-review judge.

You receive a CATBench episode that the first stage could not resolve. You
will see the goal, the compact trace, the side artifacts, and a STITCHED set
of key screenshots (first, last, error-adjacent, repeated-action, and any
transition steps).

Verify the agent's recorded behaviour against the screenshots. Do not infer
information that is not visible in either the trace or the images.

Return one JSON object only, exactly:
{
  "verdict": "success|failure",
  "agree_with_recorded_label": true|false,
  "primary_failure_mode": "planning|grounding|mixed_planning_grounding|execution_tooling|environment_or_evaluator|unknown",
  "planning_score": 0-3,
  "grounding_score": 0-3,
  "confidence": "low|medium|high",
  "rationale": "one concise paragraph",
  "evidence": ["short quoted/paraphrased evidence", "screenshot:step_4 shows ..."],
  "recommended_next_debug": "short concrete next debugging step"
}

Rules:
- If a required UI element or value is NOT visible in any provided screenshot
  AND is NOT clearly attested in the textual trace, treat it as absent (fail).
- For "grounding", you must point to a specific screenshot or step where the
  click/type clearly missed its target.
- For "planning", point to a thought/plan line that committed to a wrong
  strategy or stopped early.
- Score 0 absent / 1 possible / 2 likely / 3 dominant for each axis.
"""


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
  episode_id: str
  model_name: str
  category: str
  app_id: str
  app_name: str
  output_path: Path
  pkl_path: Path
  episode_index: int
  task_template: str
  goal: str
  is_successful: float
  exception_info: str
  finish_dtime: str
  episode: dict[str, Any]


def _load_env_file(path: Path, override: bool = False) -> int:
  if not path.exists():
    return 0
  loaded = 0
  with path.open("r", encoding="utf-8") as handle:
    for raw_line in handle:
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      try:
        tokens = shlex.split(line, comments=True, posix=True)
      except ValueError:
        continue
      if tokens and tokens[0] == "export":
        tokens = tokens[1:]
      for token in tokens:
        if "=" not in token:
          continue
        name, value = token.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
          continue
        if not override and name in os.environ:
          continue
        os.environ[name] = _expand_env_value(value)
        loaded += 1
  return loaded


def _expand_env_value(value: str) -> str:
  def replace(match: re.Match[str]) -> str:
    name = match.group(1)
    default = match.group(3)
    env_value = os.environ.get(name)
    return env_value if env_value else default or ""

  return ENV_PATTERN.sub(replace, value)


def _mask_secret(value: str) -> str:
  if not value:
    return "<empty>"
  if value.startswith("${") and value.endswith("}"):
    return value
  if len(value) <= 10:
    return "*" * len(value)
  return f"{value[:6]}...{value[-4:]}"


_PICKLE_SHIM_CLASSES: dict[tuple[str, str], type] = {}


class _PickleShim:
  """Placeholder for legacy pickle-only classes that are not inspected."""

  def __new__(cls, *args: Any, **kwargs: Any) -> "_PickleShim":
    del args, kwargs
    return object.__new__(cls)

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    if args:
      self.args = args
    if kwargs:
      self.kwargs = kwargs

  def __setstate__(self, state: Any) -> None:
    if isinstance(state, dict):
      self.__dict__.update(state)
    else:
      self.state = state

  def __repr__(self) -> str:
    return f"<legacy pickle shim {self.__class__.__module__}.{self.__class__.__name__}>"


def _pickle_shim_class(module: str, name: str) -> type:
  key = (module, name)
  if key not in _PICKLE_SHIM_CLASSES:
    _PICKLE_SHIM_CLASSES[key] = type(
        name,
        (_PickleShim,),
        {"__module__": module},
    )
  return _PICKLE_SHIM_CLASSES[key]


class _PickleShimModule(types.ModuleType):
  def __getattr__(self, name: str) -> Any:
    if name == "__path__":
      value: list[str] = []
    else:
      value = _pickle_shim_class(self.__name__, name)
    setattr(self, name, value)
    return value


def _install_android_env_shims() -> None:
  if "android_env" in sys.modules:
    return
  module_names = (
      "android_env",
      "android_env.components",
      "android_env.components.action_type",
      "android_env.components.config_classes",
      "android_env.components.errors",
      "android_env.env_interface",
      "android_env.loader",
      "android_env.proto",
      "android_env.proto.adb_pb2",
      "android_env.proto.a11y",
      "android_env.proto.a11y.android_accessibility_forest_pb2",
      "android_env.wrappers",
      "android_env.wrappers.a11y_grpc_wrapper",
      "android_env.wrappers.base_wrapper",
  )
  for module_name in module_names:
    module = _PickleShimModule(module_name)
    module.__path__ = []  # Mark shimmed packages as package-like.
    sys.modules[module_name] = module
  for module_name in module_names:
    if "." not in module_name:
      continue
    parent_name, child_name = module_name.rsplit(".", 1)
    setattr(sys.modules[parent_name], child_name, sys.modules[module_name])


class _CompatUnpickler(pickle.Unpickler):
  def find_class(self, module: str, name: str) -> Any:
    try:
      return super().find_class(module, name)
    except (AttributeError, ModuleNotFoundError) as exc:
      if any(
          module == prefix or module.startswith(f"{prefix}.")
          for prefix in LEGACY_PICKLE_SHIM_MODULE_PREFIXES
      ):
        return _pickle_shim_class(module, name)
      raise exc


def _read_pkl_gz(path: Path) -> Any:
  _install_android_env_shims()
  with path.open("rb") as handle:
    raw = handle.read()
  with gzip.open(io.BytesIO(raw), "rb") as gz:
    return _CompatUnpickler(gz).load()


def _safe_float(value: Any) -> float:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return 0.0
  if math.isnan(out) or math.isinf(out):
    return 0.0
  return out


def _is_skipped(ep: dict[str, Any]) -> bool:
  info = ep.get("exception_info") or ep.get("EXCEPTION_INFO") or ""
  return isinstance(info, str) and info.startswith("[skipped_uninstalled]")


def _prune_episode_for_analysis(
    ep: dict[str, Any], keep_images: bool = False
) -> dict[str, Any]:
  """Drop bulky visual fields unless keep_images=True."""
  pruned = dict(ep)
  if not keep_images:
    for field in IMAGE_LIKE_FIELDS:
      pruned.pop(field, None)

  step_data = pruned.get("episode_data")
  if isinstance(step_data, dict):
    compact_step_data: dict[str, Any] = {}
    for field, value in step_data.items():
      if field in IMAGE_LIKE_FIELDS and not keep_images:
        continue
      if (
          field in TEXT_FIELDS_BY_PRIORITY
          or field in IMAGE_LIKE_FIELDS
          or not isinstance(value, list)
      ):
        compact_step_data[field] = value
    pruned["episode_data"] = compact_step_data
  return pruned


def _to_jsonable(value: Any) -> Any:
  if value is None or isinstance(value, (str, int, bool)):
    return value
  if isinstance(value, float):
    return None if math.isnan(value) or math.isinf(value) else value
  if isinstance(value, dt.datetime):
    return value.isoformat(sep=" ")
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    return {str(k): _to_jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_to_jsonable(v) for v in value]
  if hasattr(value, "shape") and hasattr(value, "dtype"):
    return f"<array shape={getattr(value, 'shape', '?')} dtype={getattr(value, 'dtype', '?')}>"
  return str(value)


def _stable_json(value: Any) -> str:
  return json.dumps(
      _to_jsonable(value),
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
  )


def _sha1_text(value: str, length: int = 16) -> str:
  return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _compact_text(value: Any, max_chars: int = 600) -> str:
  if value is None:
    return ""
  if hasattr(value, "shape") and hasattr(value, "dtype"):
    text = f"<array shape={getattr(value, 'shape', '?')} dtype={getattr(value, 'dtype', '?')}>"
  elif isinstance(value, str):
    text = value
  else:
    try:
      text = json.dumps(_to_jsonable(value), ensure_ascii=False)
    except TypeError:
      text = str(value)
  text = re.sub(r"\s+", " ", text).strip()
  if len(text) <= max_chars:
    return text
  return text[: max_chars - 15].rstrip() + " ...<truncated>"


def _read_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as handle:
    return json.load(handle)


def _jobs_from_manifest(path: Path) -> list[dict[str, Any]]:
  payload = _read_json(path)
  jobs = payload.get("jobs", [])
  return [job for job in jobs if isinstance(job, dict)]


def _record_id(
    model_name: str,
    category: str,
    app_id: str,
    pkl_path: Path,
    episode_index: int,
) -> str:
  raw = f"{model_name}|{category}|{app_id}|{pkl_path}|{episode_index}"
  return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _matches_filter(value: str, filters: set[str]) -> bool:
  return not filters or value in filters


def _collect_records(
    manifest: Path,
    include_successes: bool,
    model_filter: set[str],
    category_filter: set[str],
    app_filter: set[str],
    task_regex: re.Pattern[str] | None,
    newest_first: bool,
    max_records: int = 0,
    keep_images: bool = False,
) -> list[EpisodeRecord]:
  records: list[EpisodeRecord] = []
  for job in _jobs_from_manifest(manifest):
    model_name = str(job.get("model_name", ""))
    category = str(job.get("category", ""))
    app_id = str(job.get("app_id", ""))
    if not _matches_filter(model_name, model_filter):
      continue
    if not _matches_filter(category, category_filter):
      continue
    if not _matches_filter(app_id, app_filter):
      continue

    output_path = Path(str(job.get("output_path", ""))).expanduser()
    if not output_path.exists():
      continue
    for pkl_path in sorted(output_path.rglob("*.pkl.gz")):
      try:
        payload = _read_pkl_gz(pkl_path)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"warning: failed to read {pkl_path}: {exc}", file=sys.stderr)
        continue
      episodes = payload if isinstance(payload, list) else [payload]
      for episode_index, ep in enumerate(episodes):
        if not isinstance(ep, dict) or _is_skipped(ep):
          continue
        score = _safe_float(ep.get("is_successful"))
        if not include_successes and score >= 0.5:
          continue
        analysis_ep = _prune_episode_for_analysis(ep, keep_images=keep_images)
        task_template = str(
            analysis_ep.get("task_template")
            or analysis_ep.get("name")
            or pkl_path.stem
        )
        if task_regex and not task_regex.search(task_template):
          continue
        finish = analysis_ep.get("finish_dtime")
        finish_text = (
            finish.isoformat(sep=" ")
            if isinstance(finish, dt.datetime)
            else str(finish or "")
        )
        records.append(
            EpisodeRecord(
                episode_id=_record_id(
                    model_name, category, app_id, pkl_path, episode_index
                ),
                model_name=model_name,
                category=category,
                app_id=app_id,
                app_name=str(job.get("app_name", app_id)),
                output_path=output_path,
                pkl_path=pkl_path,
                episode_index=episode_index,
                task_template=task_template,
                goal=str(analysis_ep.get("goal") or ""),
                is_successful=score,
                exception_info=str(analysis_ep.get("exception_info") or ""),
                finish_dtime=finish_text,
                episode=analysis_ep,
            )
        )
        if max_records and len(records) >= max_records:
          return records

  def sort_key(record: EpisodeRecord) -> tuple[str, float]:
    try:
      mtime = record.pkl_path.stat().st_mtime
    except OSError:
      mtime = 0.0
    return record.finish_dtime, mtime

  records.sort(key=sort_key, reverse=newest_first)
  return records


def _step_indices(length: int, max_steps: int) -> list[int]:
  if max_steps <= 0:
    return list(range(length))
  if length <= max_steps:
    return list(range(length))
  head = min(2, max_steps // 3)
  tail = max_steps - head
  return list(range(head)) + list(range(length - tail, length))


def _pick_key_step_indices(
    episode: dict[str, Any], max_steps: int
) -> list[int]:
  """Smart frame selection: first, last, error-adjacent, repeated-action,
  action-type-change, plus uniform infill up to max_steps."""
  step_data = episode.get("episode_data") or {}
  if not isinstance(step_data, dict):
    return []
  length = 0
  for value in step_data.values():
    if isinstance(value, list):
      length = max(length, len(value))
  if length == 0:
    return []
  if max_steps <= 0:
    return list(range(length))
  if length <= max_steps:
    return list(range(length))

  priority: list[int] = []
  seen_priority: set[int] = set()

  def add_priority(idx: int) -> None:
    if idx < 0 or idx >= length or idx in seen_priority:
      return
    seen_priority.add(idx)
    priority.append(idx)

  def step_text(field: str, idx: int) -> str:
    seq = step_data.get(field)
    if not isinstance(seq, list) or idx >= len(seq):
      return ""
    return _compact_text(seq[idx], 600)

  error_tokens = (
      "traceback",
      "error",
      "exception",
      "timeout",
      "timed out",
      "failed",
      "cannot",
      "impossible",
  )
  prev_action_type: str | None = None
  prev_action_text: str | None = None
  repeat_run = 1
  repeat_start = 0
  for idx in range(length):
    action_text = step_text("action", idx) or step_text("action_desc", idx)
    thought_text = step_text("thought", idx) or step_text("response", idx)
    err_text = step_text("error", idx) or step_text("exception_info", idx)
    blob = " ".join([action_text, thought_text, err_text]).lower()
    if err_text or any(token in blob for token in error_tokens):
      add_priority(idx)
      add_priority(idx - 1)

    action_type = ""
    if action_text:
      match = re.match(r"\s*([a-zA-Z_]+)", action_text)
      if match:
        action_type = match.group(1).lower()

    if prev_action_text is not None and action_text == prev_action_text:
      repeat_run += 1
      if repeat_run == 4:
        add_priority(repeat_start)
        add_priority(repeat_start - 3)
    else:
      repeat_run = 1
      repeat_start = idx

    if prev_action_type and action_type and action_type != prev_action_type:
      add_priority(idx)

    prev_action_type = action_type or prev_action_type
    prev_action_text = action_text

  picked: list[int] = []

  def add_picked(idx: int) -> None:
    if idx < 0 or idx >= length or idx in picked or len(picked) >= max_steps:
      return
    picked.append(idx)

  add_picked(0)
  add_picked(length - 1)
  for idx in priority:
    add_picked(idx)
    if len(picked) >= max_steps:
      break

  if len(picked) < max_steps:
    remaining = max_steps - len(picked)
    chosen = set(picked)
    candidates = [i for i in range(length) if i not in chosen]
    if candidates and remaining:
      stride = max(1, len(candidates) // remaining)
      filler = candidates[::stride][:remaining]
      for idx in filler:
        add_picked(idx)
  return sorted(picked)


def _encode_image_jpeg_base64(array: Any, max_dim: int, quality: int) -> str:
  """Resize an HxWx3 uint8 array to fit max_dim and return base64 JPEG."""
  if not _PIL_AVAILABLE:
    raise RuntimeError(
        "Pillow (PIL) is required for --with_screenshots. "
        "Install with: pip install Pillow"
    )
  if not hasattr(array, "shape"):
    raise ValueError(f"Cannot encode non-array screenshot: {type(array).__name__}")
  height, width = array.shape[:2]
  scale = min(1.0, max_dim / float(max(height, width)))
  image = Image.fromarray(array)
  if scale < 1.0:
    image = image.resize(
        (int(width * scale), int(height * scale)),
        resample=Image.Resampling.LANCZOS,
    )
  buffer = io.BytesIO()
  image.save(buffer, format="JPEG", quality=quality, optimize=True)
  return base64.b64encode(buffer.getvalue()).decode("ascii")


def _extract_screenshots_for_judge(
    episode: dict[str, Any],
    indices: list[int],
    max_dim: int,
    quality: int,
    prefer_som: bool = True,
) -> list[dict[str, Any]]:
  """Pick screenshots from the episode for the indices and return
  [{"step": int, "field": str, "jpeg_base64": str}, ...] entries."""
  step_data = episode.get("episode_data") or {}
  if not isinstance(step_data, dict):
    return []
  field_order: tuple[str, ...]
  if prefer_som:
    field_order = (
        "after_screenshot_with_som",
        "before_screenshot_with_som",
        "raw_screenshot",
        "screenshot",
        "after_screenshot",
        "before_screenshot",
    )
  else:
    field_order = (
        "raw_screenshot",
        "screenshot",
        "after_screenshot",
        "before_screenshot",
        "after_screenshot_with_som",
        "before_screenshot_with_som",
    )
  selected: list[dict[str, Any]] = []
  for idx in indices:
    for field in field_order:
      seq = step_data.get(field)
      if not isinstance(seq, list) or idx >= len(seq):
        continue
      candidate = seq[idx]
      if candidate is None:
        continue
      try:
        encoded = _encode_image_jpeg_base64(candidate, max_dim, quality)
      except (RuntimeError, ValueError) as exc:
        print(
            f"warning: skipped screenshot step={idx} field={field}: {exc}",
            file=sys.stderr,
        )
        continue
      selected.append(
          {"step": idx + 1, "field": field, "jpeg_base64": encoded}
      )
      break
  return selected


def _load_episode_for_screenshots(record: EpisodeRecord) -> dict[str, Any]:
  payload = _read_pkl_gz(record.pkl_path)
  episodes = payload if isinstance(payload, list) else [payload]
  if record.episode_index >= len(episodes):
    return record.episode
  episode = episodes[record.episode_index]
  return episode if isinstance(episode, dict) else record.episode


def _extract_step_summaries(
    episode: dict[str, Any], max_steps: int, field_chars: int
) -> list[dict[str, str]]:
  step_data = episode.get("episode_data") or {}
  if not isinstance(step_data, dict):
    return []
  length = 0
  for field, value in step_data.items():
    if field in IMAGE_LIKE_FIELDS:
      continue
    if isinstance(value, list):
      length = max(length, len(value))
  if length == 0:
    return []

  summaries: list[dict[str, str]] = []
  for idx in _step_indices(length, max_steps):
    row: dict[str, str] = {"step": str(idx + 1)}
    for field in TEXT_FIELDS_BY_PRIORITY:
      seq = step_data.get(field)
      if not isinstance(seq, list) or idx >= len(seq):
        continue
      value = _compact_text(seq[idx], field_chars)
      if value and value not in {"[]", "{}", "false", "False", "None"}:
        row[field] = value
    if len(row) > 1:
      summaries.append(row)
  return summaries


def _read_text_file(path: Path, max_chars: int) -> str:
  try:
    text = path.read_text(encoding="utf-8", errors="replace")
  except OSError:
    return ""
  if len(text) <= max_chars:
    return text
  half = max_chars // 2
  return (
      text[:half].rstrip()
      + "\n...<middle truncated>...\n"
      + text[-half:].lstrip()
  )


def _load_jsonl_preview(path: Path, max_items: int, max_chars: int) -> str:
  items: list[Any] = []
  try:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
      for line in handle:
        if len(items) >= max_items:
          break
        line = line.strip()
        if not line:
          continue
        try:
          items.append(json.loads(line))
        except json.JSONDecodeError:
          items.append(line)
  except OSError:
    return ""
  return _compact_text(items, max_chars)


def _episode_data_first_text(
    episode: dict[str, Any], field: str, max_chars: int
) -> str:
  step_data = episode.get("episode_data") or {}
  if not isinstance(step_data, dict):
    return ""
  value = step_data.get(field)
  if isinstance(value, list) and value:
    return _compact_text(value[0], max_chars)
  return _compact_text(value, max_chars)


def _matching_autodev_log_dir(record: EpisodeRecord) -> Path | None:
  root = record.output_path / "autodev_logs"
  if not root.exists():
    return None
  exact: list[Path] = []
  fallback: list[Path] = []
  for metadata_path in root.glob("run_*/metadata.json"):
    try:
      metadata = _read_json(metadata_path)
    except Exception:  # pylint: disable=broad-exception-caught
      continue
    if metadata.get("goal") == record.goal:
      exact.append(metadata_path.parent)
    fallback.append(metadata_path.parent)
  if exact:
    return max(exact, key=lambda path: path.stat().st_mtime)
  return None


def _side_artifacts(record: EpisodeRecord, max_chars: int) -> dict[str, str]:
  artifacts: dict[str, str] = {}

  mobile_trace = (
      record.output_path
      / "mobile_agent_v3_traces"
      / record.task_template
      / "action.jsonl"
  )
  if mobile_trace.exists():
    artifacts["mobile_agent_v3_action_trace"] = _load_jsonl_preview(
        mobile_trace, max_items=80, max_chars=max_chars
    )

  workflow = _episode_data_first_text(
      record.episode, "agentprog_workflow", max_chars
  )
  if workflow:
    artifacts["agentprog_workflow"] = workflow
  log_path_text = _episode_data_first_text(
      record.episode, "agentprog_log_path", 2000
  )
  if log_path_text:
    log_path = Path(log_path_text)
    if log_path.exists():
      artifacts["agentprog_log"] = _read_text_file(log_path, max_chars)

  autodev_dir = _matching_autodev_log_dir(record)
  if autodev_dir:
    for file_name in ("summary.txt", "timeline.md", "logs.txt"):
      file_path = autodev_dir / file_name
      if file_path.exists():
        artifacts[f"autodev_{file_name}"] = _read_text_file(
            file_path, max_chars
        )
    artifacts["autodev_log_dir"] = str(autodev_dir)

  # Some AutoDev runs in the corrected C1 manifests are symlinked directly to a
  # run_* directory whose pickle has only columnar counters and no action text.
  # The useful traceback/timing evidence is in sibling resume_*.log files.
  try:
    resolved_parent = record.pkl_path.resolve().parent
  except OSError:
    resolved_parent = record.pkl_path.parent
  if "AutoDev" in str(record.pkl_path) or "autodev" in str(record.pkl_path).lower():
    log_paths = sorted(
        resolved_parent.glob("*.log"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    for idx, log_path in enumerate(log_paths[:3], start=1):
      text = _read_text_file(log_path, max_chars)
      if text:
        artifacts[f"autodev_sibling_log_{idx}"] = text
    if log_paths:
      artifacts["autodev_sibling_log_dir"] = str(resolved_parent)

  return {k: v for k, v in artifacts.items() if v}


def _build_case_payload(
    record: EpisodeRecord,
    max_steps: int,
    max_field_chars: int,
    max_artifact_chars: int,
    use_smart_steps: bool = False,
) -> dict[str, Any]:
  if use_smart_steps:
    key_indices = _pick_key_step_indices(record.episode, max_steps)
    steps = _extract_step_summaries_for_indices(
        record.episode, key_indices, max_field_chars
    )
  else:
    steps = _extract_step_summaries(
        record.episode, max_steps=max_steps, field_chars=max_field_chars
    )
    key_indices = [int(row["step"]) - 1 for row in steps]
  return {
      "episode_id": record.episode_id,
      "model_name": record.model_name,
      "category": record.category,
      "app_id": record.app_id,
      "app_name": record.app_name,
      "task_template": record.task_template,
      "goal": record.goal,
      "is_successful": record.is_successful,
      "exception_info": record.exception_info,
      "finish_dtime": record.finish_dtime,
      "pkl_path": str(record.pkl_path),
      "output_path": str(record.output_path),
      "steps": steps,
      "key_step_indices": key_indices,
      "side_artifacts": _side_artifacts(record, max_artifact_chars),
  }


def _extract_step_summaries_for_indices(
    episode: dict[str, Any], indices: list[int], field_chars: int
) -> list[dict[str, str]]:
  step_data = episode.get("episode_data") or {}
  if not isinstance(step_data, dict):
    return []
  summaries: list[dict[str, str]] = []
  for idx in indices:
    row: dict[str, str] = {"step": str(idx + 1)}
    for field in TEXT_FIELDS_BY_PRIORITY:
      seq = step_data.get(field)
      if not isinstance(seq, list) or idx >= len(seq):
        continue
      value = _compact_text(seq[idx], field_chars)
      if value and value not in {"[]", "{}", "false", "False", "None"}:
        row[field] = value
    if len(row) > 1:
      summaries.append(row)
  return summaries


def _heuristic_judgment(case_payload: dict[str, Any]) -> dict[str, Any]:
  text = json.dumps(case_payload, ensure_ascii=False).lower()
  exception = str(case_payload.get("exception_info") or "").lower()
  repeated_actions = _detect_repeated_actions(case_payload)

  if exception and "traceback" in exception:
    mode = "execution_tooling"
    planning_score = 0
    grounding_score = 0
    rationale = "The episode records an exception/traceback, so tooling failure dominates."
  elif any(
      token in text
      for token in (
          "traceback",
          "api error",
          "timed out",
          "timeout",
          "runtimeerror",
          "connectionerror",
          "exception occurred",
      )
  ):
    mode = "execution_tooling"
    planning_score = 0
    grounding_score = 0
    rationale = "The trace contains runtime or tool error language."
  elif repeated_actions:
    mode = "grounding"
    planning_score = 1
    grounding_score = 3
    rationale = "The action trace repeats the same low-level action, suggesting localization or UI grounding trouble."
  elif any(token in text for token in ("impossible", "cannot complete", "can't complete", "finish_task")):
    mode = "planning"
    planning_score = 3
    grounding_score = 1
    rationale = "The trace suggests the agent decided to stop or declared the task impossible."
  elif any(token in text for token in ("permission", "app not installed", "validator", "emulator")):
    mode = "environment_or_evaluator"
    planning_score = 1
    grounding_score = 1
    rationale = "Environment, permission, or evaluator language appears in the trace."
  else:
    mode = "unknown"
    planning_score = 1
    grounding_score = 1
    rationale = "The compact trace is not enough for a reliable heuristic classification."

  return {
      "primary_failure_mode": mode,
      "planning_score": planning_score,
      "grounding_score": grounding_score,
      "confidence": "low",
      "rationale": rationale,
      "evidence": ["heuristic prelabel; run with --judge_backend=llm for review"],
      "recommended_next_debug": "Run the LLM judge or inspect screenshots for this case.",
  }


def _judge_error_judgment(exc: BaseException) -> dict[str, Any]:
  error = _compact_text(str(exc), 2000)
  return {
      "primary_failure_mode": "unknown",
      "planning_score": 0,
      "grounding_score": 0,
      "confidence": "low",
      "rationale": "The judge request failed, so this row was not classifiable from the LLM output.",
      "evidence": [f"judge_error: {error}"],
      "recommended_next_debug": "Re-run this episode without --continue_on_judge_error and inspect the raw judge error preview.",
      "judge_error": error,
  }


def _detect_repeated_actions(case_payload: dict[str, Any]) -> bool:
  artifacts = case_payload.get("side_artifacts") or {}
  action_trace = artifacts.get("mobile_agent_v3_action_trace")
  if not isinstance(action_trace, str):
    return False
  try:
    actions = json.loads(action_trace)
  except json.JSONDecodeError:
    return False
  if not isinstance(actions, list) or len(actions) < 4:
    return False
  compact = [_compact_text(action, 200) for action in actions]
  run = 1
  last = None
  for item in compact:
    if item == last:
      run += 1
      if run >= 4:
        return True
    else:
      run = 1
      last = item
  return False


def _chat_url(base_url: str) -> str:
  base = base_url.rstrip("/")
  if base.endswith("/chat/completions"):
    return base
  if base.endswith("/v1"):
    return f"{base}/chat/completions"
  return f"{base}/v1/chat/completions"


def _extract_json_object(text: str) -> dict[str, Any]:
  text = text.strip()
  if text.startswith("```"):
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
  try:
    payload = json.loads(text)
  except json.JSONDecodeError:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
      raise
    payload = json.loads(match.group(0))
  return _coerce_json_object(payload, original_text=text)


def _normalize_failure_mode_label(value: Any) -> str:
  label = str(value or "").strip().lower()
  label = re.sub(r"[\s\-]+", "_", label)
  aliases = {
      "mixed": "mixed_planning_grounding",
      "planning_and_grounding": "mixed_planning_grounding",
      "planning_grounding": "mixed_planning_grounding",
      "tooling": "execution_tooling",
      "execution": "execution_tooling",
      "environment": "environment_or_evaluator",
      "evaluator": "environment_or_evaluator",
  }
  return aliases.get(label, label)


def _minimal_judgment_from_mode(mode: str, note: str) -> dict[str, Any]:
  mode = _normalize_failure_mode_label(mode)
  if mode not in DEFAULT_FAILURE_MODES:
    mode = "unknown"
  return {
      "primary_failure_mode": mode,
      "planning_score": 3 if mode == "planning" else 2 if mode == "mixed_planning_grounding" else 0,
      "grounding_score": 3 if mode == "grounding" else 2 if mode == "mixed_planning_grounding" else 0,
      "confidence": "low",
      "rationale": note,
      "evidence": [],
      "recommended_next_debug": "Re-run this case or inspect the raw trace because the judge returned an abbreviated classification.",
  }


def _find_judgment_object(payload: Any) -> dict[str, Any] | None:
  if isinstance(payload, dict):
    if any(
        key in payload
        for key in (
            "primary_failure_mode",
            "failure_mode",
            "mode",
            "classification",
            "planning_score",
            "grounding_score",
        )
    ):
      return payload
    for value in payload.values():
      found = _find_judgment_object(value)
      if found is not None:
        return found
  elif isinstance(payload, list):
    for item in payload:
      found = _find_judgment_object(item)
      if found is not None:
        return found
  return None


def _coerce_json_object(payload: Any, original_text: str = "") -> dict[str, Any]:
  if isinstance(payload, dict):
    if "judgment" in payload and isinstance(payload["judgment"], dict):
      return payload["judgment"]
    found = _find_judgment_object(payload)
    if found is not None:
      return found
    return payload
  if isinstance(payload, list):
    found = _find_judgment_object(payload)
    if found is not None:
      return found
    mode_items = [
        _normalize_failure_mode_label(item)
        for item in payload
        if isinstance(item, str)
    ]
    for mode in mode_items:
      if mode in DEFAULT_FAILURE_MODES:
        return _minimal_judgment_from_mode(
            mode, "Judge returned only a JSON array label instead of the full schema."
        )
    dict_items = [item for item in payload if isinstance(item, dict)]
    if len(dict_items) == 1:
      return dict_items[0]
  if isinstance(payload, str):
    stripped = payload.strip()
    if stripped and stripped != original_text.strip():
      try:
        return _extract_json_object(stripped)
      except (ValueError, json.JSONDecodeError):
        mode = _normalize_failure_mode_label(stripped)
        if mode in DEFAULT_FAILURE_MODES:
          return _minimal_judgment_from_mode(
              mode, "Judge returned only a JSON string label instead of the full schema."
          )
  preview = _compact_text(payload if payload is not None else original_text, 1200)
  if not isinstance(payload, dict):
    raise ValueError(
        "LLM response JSON is not an object "
        f"(type={type(payload).__name__}, preview={preview!r})"
    )
  return payload


def _llm_judgment(
    case_payload: dict[str, Any],
    model: str,
    base_url: str,
    api_key: str,
    timeout_sec: float,
    max_retries: int,
    response_format: bool,
) -> dict[str, Any]:
  user_prompt = (
      "Classify this CATBench episode failure.\n\n"
      + json.dumps(case_payload, ensure_ascii=False, indent=2)
  )
  request_payload: dict[str, Any] = {
      "model": model,
      "messages": [
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": user_prompt},
      ],
      "temperature": 0,
  }
  if response_format:
    request_payload["response_format"] = {"type": "json_object"}

  headers = {"Content-Type": "application/json"}
  if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

  data = json.dumps(request_payload).encode("utf-8")
  last_error: Exception | None = None
  for attempt in range(max_retries + 1):
    req = urllib.request.Request(
        _chat_url(base_url), data=data, headers=headers, method="POST"
    )
    try:
      with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8", errors="replace")
      payload = json.loads(body)
      choices = payload.get("choices") or []
      if not choices:
        raise ValueError(f"LLM response has no choices: {body[:500]}")
      message = choices[0].get("message") or {}
      content = message.get("content")
      if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
      if not isinstance(content, str) or not content.strip():
        raise ValueError(f"LLM response has empty content: {body[:500]}")
      return _extract_json_object(content)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
      last_error = exc
      if attempt < max_retries:
        time.sleep(min(2 ** (attempt + 1), 10))
  raise RuntimeError(
      f"LLM judge failed after {max_retries + 1} attempts: {last_error}"
  )


def _gemini_url(base_url: str, model: str) -> str:
  base = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
  model_name = model.removeprefix("models/")
  return f"{base}/models/{urllib.parse.quote(model_name, safe='')}:generateContent"


def _normalize_gemini_model(model: str) -> tuple[str, str | None]:
  """Return the exact Gemini model id to call.

  We intentionally do not rewrite preview model names. The model id is part of
  the experimental configuration, appears in judge_config.json, and affects the
  cache key. If a preview id is unavailable to the current API key, the judge
  should fail loudly rather than silently running a different model.
  """
  return model, None


def _http_error_summary(exc: urllib.error.HTTPError) -> str:
  try:
    body = exc.read().decode("utf-8", errors="replace")
  except Exception:  # pylint: disable=broad-exception-caught
    body = ""
  body = body.strip()
  if len(body) > 4000:
    body = body[:4000].rstrip() + " ...<truncated>"
  return f"HTTP {exc.code} {exc.reason}: {body}"


def _gemini_judgment(
    case_payload: dict[str, Any],
    model: str,
    base_url: str,
    api_key: str,
    timeout_sec: float,
    max_retries: int,
    response_format: bool,
) -> dict[str, Any]:
  if not api_key:
    raise ValueError("Gemini judge requires --judge_api_key or GEMINI_API_KEY/GCP_API_KEY.")
  user_prompt = (
      "Classify this CATBench episode failure.\n\n"
      + json.dumps(case_payload, ensure_ascii=False, indent=2)
  )
  request_payload: dict[str, Any] = {
      "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
      "contents": [
          {"role": "user", "parts": [{"text": user_prompt}]},
      ],
      "generationConfig": {"temperature": 0},
  }
  if response_format:
    request_payload["generationConfig"]["responseMimeType"] = "application/json"

  data = json.dumps(request_payload).encode("utf-8")
  headers = {
      "Content-Type": "application/json",
      "x-goog-api-key": api_key,
  }
  last_error: Exception | None = None
  for attempt in range(max_retries + 1):
    req = urllib.request.Request(
        _gemini_url(base_url, model),
        data=data,
        headers=headers,
        method="POST",
    )
    try:
      with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8", errors="replace")
      payload = json.loads(body)
      candidates = payload.get("candidates") or []
      if not candidates:
        raise ValueError(f"Gemini response has no candidates: {body[:500]}")
      content = candidates[0].get("content") or {}
      parts = content.get("parts") or []
      text = "".join(
          part.get("text", "") if isinstance(part, dict) else str(part)
          for part in parts
      )
      if not text.strip():
        raise ValueError(f"Gemini response has empty text: {body[:500]}")
      try:
        return _extract_json_object(text)
      except (ValueError, json.JSONDecodeError) as exc:
        text_preview = _compact_text(text, 1200)
        body_preview = _compact_text(body, 1200)
        raise ValueError(
            f"{exc}; gemini_text_preview={text_preview!r}; "
            f"gemini_body_preview={body_preview!r}"
        ) from exc
    except urllib.error.HTTPError as exc:
      last_error = RuntimeError(_http_error_summary(exc))
      if attempt < max_retries:
        time.sleep(min(2 ** (attempt + 1), 10))
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
      last_error = exc
      if attempt < max_retries:
        time.sleep(min(2 ** (attempt + 1), 10))
  raise RuntimeError(
      f"Gemini judge failed after {max_retries + 1} attempts: {last_error}"
  )


def _llm_judgment_v2(
    case_payload: dict[str, Any],
    images: list[dict[str, Any]],
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout_sec: float,
    max_retries: int,
    response_format: bool,
    strict_json: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """OpenAI-compatible chat with optional image parts.

  Returns (judgment_dict, usage_dict). Images are sent as
  image_url with data: URIs (the format both OpenAI and most OAI-compatible
  vision endpoints accept).
  """
  user_parts: list[dict[str, Any]] = [
      {
          "type": "text",
          "text": (
              "Classify this CATBench episode.\n\n"
              + json.dumps(case_payload, ensure_ascii=False, indent=2)
          ),
      }
  ]
  for image in images:
    user_parts.append(
        {
            "type": "text",
            "text": (
                f"screenshot:step_{image['step']} "
                f"(field={image['field']})"
            ),
        }
    )
    user_parts.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image['jpeg_base64']}",
                "detail": "low",
            },
        }
    )

  request_payload: dict[str, Any] = {
      "model": model,
      "messages": [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_parts if images else user_parts[0]["text"]},
      ],
      "temperature": 0,
      "max_tokens": 800,
  }
  if response_format:
    if system_prompt == SYSTEM_PROMPT:
      request_payload["response_format"] = {
          "type": "json_schema",
          "json_schema": {
              "name": "catbench_failure_classification",
              "strict": True,
              "schema": JUDGE_OUTPUT_SCHEMA,
          },
      }
    else:
      # Progressive triage/deep prompts use different output contracts. Keep
      # their existing generic JSON constraint rather than forcing the basic
      # failure-classification schema onto incompatible fields.
      request_payload["response_format"] = {"type": "json_object"}

  headers = {"Content-Type": "application/json"}
  if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
  data = json.dumps(request_payload).encode("utf-8")
  last_error: Exception | None = None
  for attempt in range(max_retries + 1):
    req = urllib.request.Request(
        _chat_url(base_url), data=data, headers=headers, method="POST"
    )
    try:
      with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8", errors="replace")
      payload = json.loads(body)
      choices = payload.get("choices") or []
      if not choices:
        raise ValueError(f"LLM response has no choices: {body[:500]}")
      message = choices[0].get("message") or {}
      content = message.get("content")
      if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
      if not isinstance(content, str) or not content.strip():
        raise ValueError(f"LLM response has empty content: {body[:500]}")
      usage = payload.get("usage") or {}
      if strict_json:
        judgment = json.loads(content)
        if not isinstance(judgment, dict):
          raise ValueError(
              "Strict structured judge response is not a top-level JSON object"
          )
      else:
        judgment = _extract_json_object(content)
      return judgment, {
          "prompt_tokens": usage.get("prompt_tokens"),
          "completion_tokens": usage.get("completion_tokens"),
          "total_tokens": usage.get("total_tokens"),
          "model": payload.get("model") or model,
          "num_images": len(images),
      }
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
      last_error = exc
      if attempt < max_retries:
        time.sleep(min(2 ** (attempt + 1), 10))
  raise RuntimeError(
      f"LLM judge failed after {max_retries + 1} attempts: {last_error}"
  )


def _gemini_judgment_v2(
    case_payload: dict[str, Any],
    images: list[dict[str, Any]],
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout_sec: float,
    max_retries: int,
    response_format: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
  if not api_key:
    raise ValueError("Gemini judge requires --judge_api_key or GEMINI_API_KEY/GCP_API_KEY.")
  user_parts: list[dict[str, Any]] = [
      {
          "text": (
              "Classify this CATBench episode.\n\n"
              + json.dumps(case_payload, ensure_ascii=False, indent=2)
          )
      }
  ]
  for image in images:
    user_parts.append(
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image["jpeg_base64"],
            }
        }
    )
    user_parts.append(
        {
            "text": (
                f"^ above image is step {image['step']} "
                f"(field={image['field']})."
            )
        }
    )

  request_payload: dict[str, Any] = {
      "systemInstruction": {"parts": [{"text": system_prompt}]},
      "contents": [{"role": "user", "parts": user_parts}],
      "generationConfig": {"temperature": 0},
  }
  if response_format:
    request_payload["generationConfig"]["responseMimeType"] = "application/json"

  data = json.dumps(request_payload).encode("utf-8")
  headers = {
      "Content-Type": "application/json",
      "x-goog-api-key": api_key,
  }
  last_error: Exception | None = None
  for attempt in range(max_retries + 1):
    req = urllib.request.Request(
        _gemini_url(base_url, model), data=data, headers=headers, method="POST"
    )
    try:
      with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8", errors="replace")
      payload = json.loads(body)
      candidates = payload.get("candidates") or []
      if not candidates:
        raise ValueError(f"Gemini response has no candidates: {body[:500]}")
      content = candidates[0].get("content") or {}
      parts = content.get("parts") or []
      text = "".join(
          part.get("text", "") if isinstance(part, dict) else str(part)
          for part in parts
      )
      if not text.strip():
        raise ValueError(f"Gemini response has empty text: {body[:500]}")
      usage_meta = payload.get("usageMetadata") or {}
      try:
        judgment = _extract_json_object(text)
      except (ValueError, json.JSONDecodeError) as exc:
        text_preview = _compact_text(text, 1200)
        body_preview = _compact_text(body, 1200)
        raise ValueError(
            f"{exc}; gemini_text_preview={text_preview!r}; "
            f"gemini_body_preview={body_preview!r}"
        ) from exc
      return judgment, {
          "prompt_tokens": usage_meta.get("promptTokenCount"),
          "completion_tokens": usage_meta.get("candidatesTokenCount"),
          "total_tokens": usage_meta.get("totalTokenCount"),
          "model": model,
          "num_images": len(images),
      }
    except urllib.error.HTTPError as exc:
      last_error = RuntimeError(_http_error_summary(exc))
      if attempt < max_retries:
        time.sleep(min(2 ** (attempt + 1), 10))
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
      last_error = exc
      if attempt < max_retries:
        time.sleep(min(2 ** (attempt + 1), 10))
  raise RuntimeError(
      f"Gemini judge failed after {max_retries + 1} attempts: {last_error}"
  )


def _call_judge(
    backend: str,
    case_payload: dict[str, Any],
    images: list[dict[str, Any]],
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout_sec: float,
    max_retries: int,
    response_format: bool,
    strict_json: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Dispatch to the right backend (llm/gemini) for one judge call."""
  if backend == "gemini":
    return _gemini_judgment_v2(
        case_payload=case_payload,
        images=images,
        system_prompt=system_prompt,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        response_format=response_format,
    )
  return _llm_judgment_v2(
      case_payload=case_payload,
      images=images,
      system_prompt=system_prompt,
      model=model,
      base_url=base_url,
      api_key=api_key,
      timeout_sec=timeout_sec,
      max_retries=max_retries,
      response_format=response_format,
      strict_json=strict_json,
  )


def _progressive_judgment(
    case_payload: dict[str, Any],
    triage_images: list[dict[str, Any]],
    deep_images: list[dict[str, Any]],
    triage_backend: str,
    triage_model: str,
    triage_base_url: str,
    triage_api_key: str,
    deep_backend: str,
    deep_model: str,
    deep_base_url: str,
    deep_api_key: str,
    timeout_sec: float,
    max_retries: int,
    response_format: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  """Stage 1 (triage) → Stage 2 (deep verification) only if escalation is needed.

  Returns (final_judgment, [stage_records]) where each stage_record contains
  stage, backend, model, judgment, usage.
  """
  stages: list[dict[str, Any]] = []
  triage_judgment, triage_usage = _call_judge(
      backend=triage_backend,
      case_payload=case_payload,
      images=triage_images,
      system_prompt=SYSTEM_PROMPT_TRIAGE,
      model=triage_model,
      base_url=triage_base_url,
      api_key=triage_api_key,
      timeout_sec=timeout_sec,
      max_retries=max_retries,
      response_format=response_format,
  )
  stages.append(
      {
          "stage": "triage",
          "backend": triage_backend,
          "model": triage_model,
          "judgment": triage_judgment,
          "usage": triage_usage,
      }
  )

  needs_escalation = bool(triage_judgment.get("needs_escalation"))
  verdict = str(triage_judgment.get("verdict") or "").lower()
  confidence = str(triage_judgment.get("confidence") or "").lower()
  if verdict == "uncertain" or confidence == "low":
    needs_escalation = True

  if not needs_escalation:
    final = dict(triage_judgment)
    final.setdefault(
        "planning_score",
        3 if final.get("primary_failure_mode") == "planning" else 0,
    )
    final.setdefault(
        "grounding_score",
        3 if final.get("primary_failure_mode") == "grounding" else 0,
    )
    final.setdefault("evidence", [])
    final.setdefault("recommended_next_debug", "")
    final["stage_resolved"] = "triage"
    return final, stages

  deep_judgment, deep_usage = _call_judge(
      backend=deep_backend,
      case_payload=case_payload,
      images=deep_images,
      system_prompt=SYSTEM_PROMPT_DEEP,
      model=deep_model,
      base_url=deep_base_url,
      api_key=deep_api_key,
      timeout_sec=timeout_sec,
      max_retries=max_retries,
      response_format=response_format,
  )
  stages.append(
      {
          "stage": "deep",
          "backend": deep_backend,
          "model": deep_model,
          "judgment": deep_judgment,
          "usage": deep_usage,
      }
  )
  final = dict(deep_judgment)
  final["stage_resolved"] = "deep"
  return final, stages


def _normalize_judgment(judgment: dict[str, Any]) -> dict[str, Any]:
  mode = _normalize_failure_mode_label(
      judgment.get("primary_failure_mode")
      or judgment.get("failure_mode")
      or judgment.get("mode")
      or judgment.get("classification")
      or "unknown"
  )
  if mode not in DEFAULT_FAILURE_MODES:
    mode = "unknown"
  confidence = str(judgment.get("confidence") or "low").lower()
  if confidence not in {"low", "medium", "high"}:
    confidence = "low"

  def score(name: str) -> int:
    try:
      value = int(judgment.get(name, 0))
    except (TypeError, ValueError):
      value = 0
    return max(0, min(3, value))

  evidence = judgment.get("evidence") or []
  if isinstance(evidence, str):
    evidence = [evidence]
  if not isinstance(evidence, list):
    evidence = []
  verdict = str(judgment.get("verdict") or "").lower()
  if verdict not in {"success", "failure", "uncertain"}:
    verdict = ""
  out: dict[str, Any] = {
      "primary_failure_mode": mode,
      "planning_score": score("planning_score"),
      "grounding_score": score("grounding_score"),
      "confidence": confidence,
      "rationale": str(judgment.get("rationale") or ""),
      "evidence": [str(item) for item in evidence[:5]],
  }
  recommended_next_debug = str(judgment.get("recommended_next_debug") or "")
  if recommended_next_debug:
    out["recommended_next_debug"] = recommended_next_debug
  if verdict:
    out["verdict"] = verdict
  if "agree_with_recorded_label" in judgment:
    out["agree_with_recorded_label"] = bool(
        judgment.get("agree_with_recorded_label")
    )
  if "stage_resolved" in judgment:
    out["stage_resolved"] = str(judgment.get("stage_resolved"))
  return out


def _judge_config(
    args: argparse.Namespace,
    triage_backend: str,
    triage_model: str,
    triage_base_url: str,
) -> dict[str, Any]:
  """Return the judge configuration that affects outputs or cache validity."""
  return {
      "judge_backend": args.judge_backend,
      "judge_model": args.judge_model,
      "judge_base_url": args.judge_base_url,
      "response_format": not args.no_response_format,
      "response_schema_sha1": (
          _sha1_text(_stable_json(JUDGE_OUTPUT_SCHEMA), 40)
          if args.judge_backend == "llm" and not args.no_response_format
          else ""
      ),
      "judge_max_tokens": 800 if args.judge_backend == "llm" else None,
      "timeout_sec": args.timeout_sec,
      "max_retries": args.max_retries,
      "include_successes": bool(args.include_successes),
      "with_screenshots": bool(args.with_screenshots),
      "screenshot_max_dim": args.screenshot_max_dim,
      "screenshot_max_frames": args.screenshot_max_frames,
      "screenshot_quality": 80,
      "smart_steps": bool(args.smart_steps or args.progressive),
      "max_steps": args.max_steps,
      "max_field_chars": args.max_field_chars,
      "max_artifact_chars": args.max_artifact_chars,
      "progressive": bool(args.progressive),
      "triage_backend": triage_backend if args.progressive else "",
      "triage_model": triage_model if args.progressive else "",
      "triage_base_url": triage_base_url if args.progressive else "",
      "triage_screenshot_max_frames": (
          args.triage_screenshot_max_frames if args.progressive else 0
      ),
      "system_prompt_sha1": _sha1_text(SYSTEM_PROMPT, 40),
      "triage_prompt_sha1": (
          _sha1_text(SYSTEM_PROMPT_TRIAGE, 40) if args.progressive else ""
      ),
      "deep_prompt_sha1": (
          _sha1_text(SYSTEM_PROMPT_DEEP, 40) if args.progressive else ""
      ),
  }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")


def _write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
  counts = collections.Counter(
      row["judgment"]["primary_failure_mode"] for row in rows
  )
  by_model: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
  by_category: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
  for row in rows:
    mode = row["judgment"]["primary_failure_mode"]
    by_model[row["model_name"]][mode] += 1
    by_category[row["category"]][mode] += 1

  summary = {
      "total_cases": len(rows),
      "counts_by_failure_mode": dict(counts),
      "counts_by_model": {key: dict(value) for key, value in by_model.items()},
      "counts_by_category": {
          key: dict(value) for key, value in by_category.items()
      },
  }
  (out_dir / "failure_mode_summary.json").write_text(
      json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )

  lines = ["# CATBench Failure Mode Summary", ""]
  lines.append(f"Total judged cases: {len(rows)}")
  lines.append("")
  lines.append("## By Failure Mode")
  lines.append("")
  lines.append("| Failure mode | Count |")
  lines.append("|---|---:|")
  for mode, count in counts.most_common():
    lines.append(f"| {mode} | {count} |")
  lines.append("")
  lines.append("## By Model")
  lines.append("")
  lines.append("| Model | " + " | ".join(DEFAULT_FAILURE_MODES) + " |")
  lines.append("|---" + "|---:" * len(DEFAULT_FAILURE_MODES) + "|")
  for model in sorted(by_model):
    counts_for_model = by_model[model]
    values = [str(counts_for_model.get(mode, 0)) for mode in DEFAULT_FAILURE_MODES]
    lines.append(f"| {model} | " + " | ".join(values) + " |")
  lines.append("")
  lines.append("## High-Confidence Examples")
  lines.append("")
  examples = [
      row for row in rows
      if row["judgment"].get("confidence") == "high"
  ][:20]
  if not examples:
    lines.append("No high-confidence examples in this batch.")
  else:
    lines.append("| Mode | Model | App | Task | Rationale |")
    lines.append("|---|---|---|---|---|")
    for row in examples:
      rationale = str(row["judgment"].get("rationale", "")).replace("|", "\\|")
      lines.append(
          f"| {row['judgment']['primary_failure_mode']} | "
          f"{row['model_name']} | {row['app_name']} | "
          f"{row['task_template']} | {rationale[:280]} |"
      )
  (out_dir / "failure_mode_summary.md").write_text(
      "\n".join(lines) + "\n", encoding="utf-8"
  )


def _result_row(
    record: EpisodeRecord,
    case_payload: dict[str, Any],
    judgment: dict[str, Any],
    judge_backend: str,
    judge_config_hash: str,
) -> dict[str, Any]:
  # Don't leak base64 images into the saved case_payload (cache bloat).
  trimmed_payload = dict(case_payload)
  if "side_artifacts" in trimmed_payload:
    trimmed_payload["side_artifacts"] = trimmed_payload["side_artifacts"]
  row = {
      "episode_id": record.episode_id,
      "judge_backend": judge_backend,
      "judge_config_hash": judge_config_hash,
      "model_name": record.model_name,
      "category": record.category,
      "app_id": record.app_id,
      "app_name": record.app_name,
      "task_template": record.task_template,
      "goal": record.goal,
      "is_successful": record.is_successful,
      "finish_dtime": record.finish_dtime,
      "pkl_path": str(record.pkl_path),
      "output_path": str(record.output_path),
      "judgment": _normalize_judgment(judgment),
      "case_payload": trimmed_payload,
  }
  if "_usage" in judgment:
    row["usage"] = judgment["_usage"]
  if "_stages" in judgment:
    row["stages"] = [
        {
            "stage": stage["stage"],
            "backend": stage["backend"],
            "model": stage["model"],
            "judgment": stage["judgment"],
            "usage": stage.get("usage"),
        }
        for stage in judgment["_stages"]
    ]
  return row


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, help="CATBench matrix manifest.")
  parser.add_argument(
      "--out_dir",
      default="",
      help="Output directory. Defaults to <manifest-dir>/failure_mode_analysis.",
  )
  parser.add_argument(
      "--env_file",
      default="",
      help="Optional KEY=VALUE file loaded before resolving LLM settings.",
  )
  parser.add_argument(
      "--judge_backend",
      choices=("llm", "gemini", "heuristic", "dry_run"),
      default="dry_run",
      help=(
          "dry_run writes judge inputs only; heuristic is no-network; "
          "llm calls an OpenAI-compatible chat-completions endpoint; "
          "gemini calls the Gemini generateContent REST API."
      ),
  )
  parser.add_argument(
      "--judge_model",
      default="",
  )
  parser.add_argument(
      "--judge_base_url",
      default="",
  )
  parser.add_argument(
      "--judge_api_key",
      default="",
  )
  parser.add_argument("--timeout_sec", type=float, default=120.0)
  parser.add_argument("--max_retries", type=int, default=3)
  parser.add_argument(
      "--no_response_format",
      action="store_true",
      help="Do not send the strict JSON-schema response format.",
  )
  parser.add_argument("--limit", type=int, default=0)
  parser.add_argument("--include_successes", action="store_true")
  parser.add_argument("--newest_first", action="store_true", default=True)
  parser.add_argument("--oldest_first", action="store_true")
  parser.add_argument("--model", action="append", default=[])
  parser.add_argument("--category", action="append", default=[])
  parser.add_argument("--app", action="append", default=[])
  parser.add_argument("--task_regex", default="")
  parser.add_argument(
      "--max_steps",
      type=int,
      default=0,
      help=(
          "Number of ordered trajectory steps included in judge evidence; "
          "0 (default) includes the full trajectory. Use 6 only for the "
          "registered limited-evidence ablation."
      ),
  )
  parser.add_argument("--max_field_chars", type=int, default=700)
  parser.add_argument("--max_artifact_chars", type=int, default=7000)
  parser.add_argument(
      "--resume",
      action="store_true",
      help="Reuse cached per-episode judgments when present.",
  )
  parser.add_argument(
      "--continue_on_judge_error",
      action="store_true",
      help=(
          "Record an unknown low-confidence row instead of stopping when an "
          "LLM/Gemini judge call fails."
      ),
  )
  parser.add_argument(
      "--with_screenshots",
      action="store_true",
      help=(
          "Send key-frame screenshots from each pkl.gz to the multimodal judge. "
          "Requires Pillow. Adds notable token cost — pair with --progressive "
          "and --smart_steps for cost control."
      ),
  )
  parser.add_argument(
      "--screenshot_max_dim",
      type=int,
      default=896,
      help="Max edge length (px) for resized screenshots sent to the judge.",
  )
  parser.add_argument(
      "--screenshot_max_frames",
      type=int,
      default=0,
      help=(
          "Maximum screenshots sent to the deep stage / single call; 0 "
          "(default) sends every available ordered step screenshot."
      ),
  )
  parser.add_argument(
      "--triage_screenshot_max_frames",
      type=int,
      default=3,
      help="Number of screenshots sent to the cheap triage stage (last 3 by default).",
  )
  parser.add_argument(
      "--smart_steps",
      action="store_true",
      help=(
          "Use the error/repeat/transition-aware step selector rather than "
          "naive head+tail sampling."
      ),
  )
  parser.add_argument(
      "--progressive",
      action="store_true",
      help=(
          "Two-stage progressive scrutiny: cheap triage call resolves obvious "
          "cases; only uncertain/low-confidence cases escalate to the deep VLM."
      ),
  )
  parser.add_argument(
      "--triage_backend",
      choices=("llm", "gemini"),
      default="",
      help="Backend for the triage stage (default: same as --judge_backend).",
  )
  parser.add_argument("--triage_model", default="")
  parser.add_argument("--triage_base_url", default="")
  parser.add_argument("--triage_api_key", default="")
  args = parser.parse_args()

  if args.env_file:
    loaded = _load_env_file(Path(args.env_file).expanduser(), override=False)
    if loaded:
      print(f"Loaded {loaded} env vars from {args.env_file}", flush=True)
  if args.judge_backend == "gemini":
    args.judge_model = (
        args.judge_model
        or os.environ.get("FAILURE_JUDGE_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or "gemini-3.1-pro-preview"
    )
    args.judge_base_url = (
        args.judge_base_url
        or os.environ.get("FAILURE_JUDGE_BASE_URL")
        or os.environ.get("GEMINI_BASE_URL")
        or "https://generativelanguage.googleapis.com/v1beta"
    )
    args.judge_api_key = (
        args.judge_api_key
        or os.environ.get("FAILURE_JUDGE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GCP_API_KEY")
        or ""
    )
    args.judge_model, model_note = _normalize_gemini_model(args.judge_model)
    if model_note:
      print(model_note, flush=True)
    print(
        "Gemini judge config: "
        f"model={args.judge_model} "
        f"base_url={args.judge_base_url} "
        f"api_key={_mask_secret(args.judge_api_key)}",
        flush=True,
    )
  else:
    args.judge_model = (
        args.judge_model
        or os.environ.get("FAILURE_JUDGE_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-5.1"
    )
    args.judge_base_url = (
        args.judge_base_url
        or os.environ.get("FAILURE_JUDGE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1/chat/completions"
    )
    args.judge_api_key = (
        args.judge_api_key
        or os.environ.get("FAILURE_JUDGE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )

  manifest = Path(args.manifest).expanduser().resolve()
  out_dir = (
      Path(args.out_dir).expanduser().resolve()
      if args.out_dir
      else manifest.parent / "failure_mode_analysis"
  )
  out_dir.mkdir(parents=True, exist_ok=True)
  cache_dir = out_dir / "cache"
  cache_dir.mkdir(parents=True, exist_ok=True)

  task_regex = re.compile(args.task_regex) if args.task_regex else None
  if args.with_screenshots and not _PIL_AVAILABLE:
    print(
        "error: --with_screenshots requires Pillow. Install with: pip install Pillow",
        file=sys.stderr,
    )
    return 2

  triage_backend = args.triage_backend or args.judge_backend
  triage_model = args.triage_model or args.judge_model
  triage_base_url = args.triage_base_url or args.judge_base_url
  triage_api_key = args.triage_api_key or args.judge_api_key
  if args.progressive and triage_backend not in {"llm", "gemini"}:
    print(
        "error: --progressive requires a chat-completion triage backend "
        "(llm or gemini).",
        file=sys.stderr,
    )
    return 2

  judge_config = _judge_config(
      args,
      triage_backend=triage_backend,
      triage_model=triage_model,
      triage_base_url=triage_base_url,
  )
  judge_config_hash = _sha1_text(_stable_json(judge_config), 16)
  judge_config["config_hash"] = judge_config_hash
  (out_dir / "judge_config.json").write_text(
      json.dumps(judge_config, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )

  records = _collect_records(
      manifest=manifest,
      include_successes=args.include_successes,
      model_filter=set(args.model),
      category_filter=set(args.category),
      app_filter=set(args.app),
      task_regex=task_regex,
      newest_first=not args.oldest_first,
      max_records=args.limit,
      keep_images=False,
  )

  print(f"Cases selected: {len(records)}", flush=True)
  rows: list[dict[str, Any]] = []
  judge_inputs: list[dict[str, Any]] = []
  for index, record in enumerate(records, start=1):
    case_payload = _build_case_payload(
        record,
        max_steps=args.max_steps,
        max_field_chars=args.max_field_chars,
        max_artifact_chars=args.max_artifact_chars,
        use_smart_steps=args.smart_steps or args.progressive,
    )
    judge_inputs.append(case_payload)

    cache_path = cache_dir / f"{record.episode_id}_{judge_config_hash}.json"
    if args.resume and cache_path.exists() and args.judge_backend != "dry_run":
      judgment = _read_json(cache_path)
      rows.append(
          _result_row(
              record, case_payload, judgment, args.judge_backend, judge_config_hash
          )
      )
      continue
    if args.judge_backend == "dry_run":
      continue
    if args.judge_backend == "heuristic":
      judgment = _heuristic_judgment(case_payload)
      cache_path.write_text(
          json.dumps(judgment, indent=2, ensure_ascii=False) + "\n",
          encoding="utf-8",
      )
      rows.append(
          _result_row(
              record, case_payload, judgment, args.judge_backend, judge_config_hash
          )
      )
      continue

    print(
        f"[{index}/{len(records)}] judging {record.model_name} "
        f"{record.category}/{record.app_id} {record.task_template}"
        + (" (progressive)" if args.progressive else ""),
        flush=True,
    )

    deep_images: list[dict[str, Any]] = []
    triage_images: list[dict[str, Any]] = []
    if args.with_screenshots:
      key_indices = case_payload.get("key_step_indices") or []
      screenshot_episode = _load_episode_for_screenshots(record)
      deep_pick = (
          key_indices
          if args.screenshot_max_frames <= 0
          else key_indices[: args.screenshot_max_frames]
      )
      deep_images = _extract_screenshots_for_judge(
          screenshot_episode,
          deep_pick,
          max_dim=args.screenshot_max_dim,
          quality=80,
      )
      # Triage gets the last N frames (like MemGUI Stage 1).
      triage_pick = key_indices[-args.triage_screenshot_max_frames :]
      triage_images = _extract_screenshots_for_judge(
          screenshot_episode,
          triage_pick,
          max_dim=min(args.screenshot_max_dim, 640),
          quality=70,
      )

    try:
      if args.progressive:
        judgment, stage_records = _progressive_judgment(
            case_payload=case_payload,
            triage_images=triage_images,
            deep_images=deep_images,
            triage_backend=triage_backend,
            triage_model=triage_model,
            triage_base_url=triage_base_url,
            triage_api_key=triage_api_key,
            deep_backend=args.judge_backend,
            deep_model=args.judge_model,
            deep_base_url=args.judge_base_url,
            deep_api_key=args.judge_api_key,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            response_format=not args.no_response_format,
        )
        judgment["_stages"] = stage_records
      else:
        if args.with_screenshots:
          judgment, usage = _call_judge(
              backend=args.judge_backend,
              case_payload=case_payload,
              images=deep_images,
              system_prompt=SYSTEM_PROMPT,
              model=args.judge_model,
              base_url=args.judge_base_url,
              api_key=args.judge_api_key,
              timeout_sec=args.timeout_sec,
              max_retries=args.max_retries,
              response_format=not args.no_response_format,
          )
          judgment["_usage"] = usage
        elif args.judge_backend == "llm":
          judgment = _llm_judgment(
              case_payload=case_payload,
              model=args.judge_model,
              base_url=args.judge_base_url,
              api_key=args.judge_api_key,
              timeout_sec=args.timeout_sec,
              max_retries=args.max_retries,
              response_format=not args.no_response_format,
          )
        else:
          judgment = _gemini_judgment(
              case_payload=case_payload,
              model=args.judge_model,
              base_url=args.judge_base_url,
              api_key=args.judge_api_key,
              timeout_sec=args.timeout_sec,
              max_retries=args.max_retries,
              response_format=not args.no_response_format,
          )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if not args.continue_on_judge_error:
        raise
      print(f"Judge failed; recording unknown and continuing: {exc}", flush=True)
      judgment = _judge_error_judgment(exc)

    cache_path.write_text(
        json.dumps(judgment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows.append(
        _result_row(
            record, case_payload, judgment, args.judge_backend, judge_config_hash
        )
    )

  _write_jsonl(out_dir / "failure_judge_inputs.jsonl", judge_inputs)
  if args.judge_backend == "dry_run":
    print(f"Wrote judge inputs to {out_dir / 'failure_judge_inputs.jsonl'}")
    return 0

  _write_jsonl(out_dir / "failure_mode_judgments.jsonl", rows)
  _write_summary(out_dir, rows)
  print(f"Wrote judgments to {out_dir / 'failure_mode_judgments.jsonl'}")
  print(f"Wrote summary to {out_dir / 'failure_mode_summary.md'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
