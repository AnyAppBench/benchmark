#!/usr/bin/env python3
"""Human annotation tool for CATBench failure-mode classification.

Aligned with the post-advisor-review pipeline:

  * The programmatic AW/CATBench validator owns the pass/fail verdict and
    defines the success rate we report. Neither the VLM classifier nor
    the human annotator overrides it.
  * The VLM classifier and the human annotator both label the same
    multiclass field — primary_failure_mode — on the SAME validator-failed
    sample. Their agreement measures the VLM's classification accuracy.

For each validator-failed episode in --sample_manifest this tool:

  1. Loads the pkl.gz, picks salience-aware key frames (the same frames
     the VLM sees), and writes a per-episode browsable HTML report.
  2. Opens the HTML report in the system browser (xdg-open / open).
  3. Prompts the human on stdin for:
       primary_failure_mode, planning_score, grounding_score,
       confidence, one-line rationale, and an OPTIONAL
       suspected-validator-error flag.
  4. Appends a JSON record to <out_dir>/annotations.jsonl.

Annotations are resumable per --annotator. Use --pool primary for the main
sample and --pool double_annotation for the inter-rater-agreement subset.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from classify_catbench_failures import (  # noqa: E402
    DEFAULT_FAILURE_MODES,
    IMAGE_LIKE_FIELDS,
    _PIL_AVAILABLE,
    _extract_screenshots_for_judge,
    _extract_step_summaries_for_indices,
    _pick_key_step_indices,
    _read_pkl_gz,
)


# Failure-mode glossary printed once at the start of every session so the
# annotator does not have to keep the definitions in their head. Wording tracks
# the classifier's system prompt while adding annotation-tool guidance where
# needed.
FAILURE_MODE_GLOSSARY: dict[str, str] = {
    "planning":
        "The agent chose a wrong high-level strategy: missed a required "
        "subgoal, stopped early, committed to an infeasible plan, or "
        "marked the task impossible when it wasn't.",
    "grounding":
        "The plan was right but the agent could not locate or interact "
        "with the correct UI element: clicked/typed in the wrong place, "
        "repeated bad coordinates, missed visible controls, OCR/visual "
        "grounding failed.",
    "mixed_planning_grounding":
        "Both planning and grounding meaningfully contributed.",
    "execution_tooling":
        "The agent's framework (parser, tool call, API timeout, model "
        "crash, malformed action, framework exception) is the dominant "
        "failure source — the agent never got a fair chance to act.",
    "environment_or_evaluator":
        "App crashed, permission was missing, network was down, the "
        "AVD got stuck, OR the validator looks like it might be wrong. "
        "If you suspect the validator, answer yes on the separate "
        "validator-audit prompt.",
    "unknown":
        "The trace and screenshots are genuinely insufficient to tell.",
}


# Blind annotation must cover strings inside the trajectory, not only manifest
# metadata.  In particular, several agent frameworks echo their model name,
# API endpoint, or host-side output path in ``raw_response_list``/``summary``.
# Keep the action and reasoning text, but remove those audit-only identifiers.
_PRIVATE_AUDIT_ASSIGNMENT_RE = re.compile(
    r"(?i)\b("
    r"model(?:_name)?|agent(?:_name)?|judge(?:_model|_backend|_label)?|"
    r"condition|pool|source(?:_path)?|endpoint|base_url|api_key|"
    r"pkl_path|jsonl_path|output_path|run_dir"
    r")\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)
_URL_RE = re.compile(r"(?i)\bhttps?://[^\s<>'\"`]+")
_MODEL_IDENTITY_RE = re.compile(
    r"(?i)(?<![\w])(?:"
    r"gpt[-_. ]?\d[\w.-]*|gemini[-_. ]?\d[\w.-]*|"
    r"qwen(?:\d|[-_. ]?vl)[\w.-]*|claude[-_. ]?[\w.-]*|"
    r"internvl[\w.-]*|gui[-_. ]?owl[\w.-]*|mai[-_. ]?ui[\w.-]*|"
    r"mobile[-_. ]?agent[\w.-]*|ui[-_. ]?(?:venus|voyager)[\w.-]*|"
    r"autodev[\w.-]*|catbench[-_. ]?judge[\w.-]*"
    r")(?![\w])"
)
# Extra mount points can be added through CATBENCH_REDACT_MOUNTS, a comma
# separated list of top-level directory names (for example a site-specific
# shared-storage mount), so no deployment path needs to be hard-coded here.
_EXTRA_REDACT_MOUNTS = tuple(
    part.strip().strip("/")
    for part in os.environ.get("CATBENCH_REDACT_MOUNTS", "").split(",")
    if part.strip().strip("/")
)
_HOST_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w.])/(?:"
    + "|".join(
        ("home", "root", "tmp", "var", "mnt", "opt", "workspace", "workspaces")
        + tuple(re.escape(m) for m in _EXTRA_REDACT_MOUNTS)
    )
    + r")"
    r"(?:/[^\s<>'\"`:,;)}\]]+)*"
)
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[A-Z]:\\(?:[^\s<>'\"`:,;)}\]]+\\?)+"
)
_CREDENTIAL_RE = re.compile(
    r"(?i)\b(authorization|api[_ -]?key|token)\s*[:=]\s*"
    r"(?:bearer\s+)?[A-Za-z0-9._~+/=-]{12,}"
)
_PRIVATE_PICK_KEYS = {
    "model_name",
    "agent_name",
    "judge_model",
    "judge_backend",
    "judge_label",
    "condition",
    "pool",
    "source",
    "source_path",
    "endpoint",
    "base_url",
    "api_key",
    "pkl_path",
    "jsonl_path",
    "gemini_jsonl_path",
    "output_path",
    "run_dir",
    "judge_config_hash",
}


def _private_values_from_pick(pick: dict[str, Any]) -> list[str]:
  """Return non-trivial audit values that may be echoed in a trace."""
  values: set[str] = set()
  for key, value in pick.items():
    if key not in _PRIVATE_PICK_KEYS or not isinstance(value, str):
      continue
    value = value.strip()
    # Avoid replacing ordinary short words (for example pool="all") in the
    # agent's reasoning. Assignment-form occurrences are handled separately.
    if len(value) >= 5:
      values.add(value)
  return sorted(values, key=len, reverse=True)


def _redact_blind_text(value: Any, pick: dict[str, Any]) -> str:
  """Remove private run identifiers while retaining goal/action content."""
  text = str(value if value is not None else "")
  for private_value in _private_values_from_pick(pick):
    text = re.sub(
        re.escape(private_value),
        "[REDACTED_PRIVATE_VALUE]",
        text,
        flags=re.IGNORECASE,
    )
  text = _PRIVATE_AUDIT_ASSIGNMENT_RE.sub(
      lambda match: f"{match.group(1)}=[REDACTED]", text
  )
  text = _CREDENTIAL_RE.sub(
      lambda match: f"{match.group(1)}=[REDACTED]", text
  )
  text = _URL_RE.sub("[REDACTED_ENDPOINT]", text)
  text = _MODEL_IDENTITY_RE.sub("[REDACTED_MODEL]", text)
  text = _HOST_ABSOLUTE_PATH_RE.sub("[REDACTED_HOST_PATH]", text)
  text = _WINDOWS_ABSOLUTE_PATH_RE.sub("[REDACTED_HOST_PATH]", text)
  return text


def _print_glossary() -> None:
  sys.stdout.write("\n" + "=" * 78 + "\n")
  sys.stdout.write(
      "FAILURE-MODE GLOSSARY (your label space)\n"
      "Validator owns pass/fail; you only label WHY the agent failed.\n"
      + "=" * 78 + "\n"
  )
  for mode, defn in FAILURE_MODE_GLOSSARY.items():
    sys.stdout.write(f"  {mode}:\n")
    # 76 char wrap
    words = defn.split()
    line = "    "
    for w in words:
      if len(line) + len(w) + 1 > 76:
        sys.stdout.write(line + "\n")
        line = "    "
      line += (w + " ")
    if line.strip():
      sys.stdout.write(line + "\n")
    sys.stdout.write("\n")
  sys.stdout.write("=" * 78 + "\n\n")
  sys.stdout.flush()


def _decode_jpeg(jpeg_b64: str) -> bytes:
  return base64.b64decode(jpeg_b64)


def _all_text_step_indices(episode: dict[str, Any]) -> list[int]:
  step_data = episode.get("episode_data") or {}
  if not isinstance(step_data, dict):
    return []
  length = 0
  for field, value in step_data.items():
    if field in IMAGE_LIKE_FIELDS:
      continue
    if isinstance(value, list):
      length = max(length, len(value))
  return list(range(length))


def _all_screenshot_step_indices(episode: dict[str, Any]) -> list[int]:
  step_data = episode.get("episode_data") or {}
  if not isinstance(step_data, dict):
    return []
  length = 0
  for field in IMAGE_LIKE_FIELDS:
    value = step_data.get(field)
    if isinstance(value, list):
      length = max(length, len(value))
  return list(range(length))


def _render_html_report(
    pick: dict[str, Any],
    episode: dict[str, Any],
    out_dir: Path,
    max_frames: int,
    max_dim: int,
    blind: bool,
) -> dict[str, Any]:
  """Build a self-contained HTML report for one episode and dump frames.

  Returns {frames: [...], html_path: str, summary_md: str}.
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  frame_indices = (
      _all_screenshot_step_indices(episode)
      if max_frames <= 0
      else _pick_key_step_indices(episode, max_frames)
  )
  step_data = episode.get("episode_data") or {}

  saved_frames: list[dict[str, Any]] = []
  if _PIL_AVAILABLE and isinstance(step_data, dict):
    for image in _extract_screenshots_for_judge(
        episode,
        frame_indices,
        max_dim=max_dim,
        quality=80,
    ):
      frame_path = out_dir / f"step_{image['step']:03d}_{image['field']}.jpg"
      frame_path.write_bytes(_decode_jpeg(image["jpeg_base64"]))
      saved_frames.append(
          {"step": image["step"], "field": image["field"], "path": str(frame_path)}
      )

  all_step_indices = _all_text_step_indices(episode)
  step_rows = _extract_step_summaries_for_indices(episode, all_step_indices, 1000)

  def visible(value: Any) -> str:
    if blind:
      return _redact_blind_text(value, pick)
    return str(value if value is not None else "")

  # Markdown summary (kept for backward compat with earlier runs).
  if blind:
    md_lines = [
        f"# {visible(pick['category'])} / {visible(pick['app_id'])} / "
        f"{visible(pick['task_template'])}",
        "",
        f"**Goal:** {visible(pick['goal'])}",
        "",
        "**Validator verdict (read-only):** "
        f"is_successful = {visible(pick['is_successful'])}",
        "",
    ]
  else:
    md_lines = [
        f"# {pick['model_name']} / {pick['category']} / {pick['app_id']} / "
        f"{pick['task_template']}",
        "",
        f"**Goal:** {pick['goal']}",
        "",
        "**Validator verdict (read-only):** "
        f"is_successful = {pick['is_successful']}",
        "",
        f"**pkl:** `{pick['pkl_path']}`",
        "",
    ]
    md_lines.extend([
        f"**Prior judge label:** {pick.get('judge_label', 'unjudged')} "
        f"(confidence {pick.get('judge_confidence', '-')})",
        f"**Prior judge rationale:** {pick.get('judge_rationale', '')}",
        "",
    ])
  frame_heading = "All step screenshots" if max_frames <= 0 else "Key frames"
  md_lines.extend([f"## {frame_heading}", ""])
  for frame in saved_frames:
    frame_ref = Path(frame["path"]).name if blind else frame["path"]
    md_lines.append(
        f"- step {frame['step']} (`{frame['field']}`): {frame_ref}"
    )
  md_lines.extend(["", "## Full step trace", ""])
  for row in step_rows:
    md_lines.append(f"### Step {row.get('step')}")
    for key, value in row.items():
      if key == "step":
        continue
      md_lines.append(f"- **{key}**: {visible(value)}")
    md_lines.append("")
  summary_path = out_dir / "summary.md"
  summary_path.write_text("\n".join(md_lines), encoding="utf-8")

  # HTML viewer — frames inline, trace formatted, validator/judge banners.
  def esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""))

  html_lines = [
      "<!doctype html><html><head><meta charset='utf-8'>",
      f"<title>Annotate {esc(pick['task_template'])}</title>",
      "<style>",
      "  body{font-family:system-ui,sans-serif;max-width:1100px;margin:1.5rem auto;padding:0 1.5rem;color:#222}",
      "  h1{font-size:1.2rem;margin:0 0 .4rem 0}",
      "  .meta{color:#555;font-size:.9rem;margin-bottom:1rem}",
      "  .banner{padding:.6rem .9rem;border-radius:6px;margin:.4rem 0;font-size:.92rem}",
      "  .verdict{background:#fdecec;color:#7a2222;border-left:4px solid #c0392b}",
      "  .verdict.pass{background:#eaf5ec;color:#1f5d2a;border-left-color:#27ae60}",
      "  .judge{background:#eef3fb;color:#1c3b6e;border-left:4px solid #2c5ea8}",
      "  .frames{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;margin:1rem 0}",
      "  .frame{border:1px solid #ddd;border-radius:6px;padding:.4rem;background:#fafafa}",
      "  .frame img{width:100%;height:auto;border-radius:4px;display:block}",
      "  .frame .lbl{font-size:.8rem;color:#666;margin-top:.3rem}",
      "  .step{border-left:3px solid #ccc;padding:.4rem .8rem;margin:.6rem 0;background:#fcfcfc;font-size:.88rem}",
      "  .step h3{margin:.2rem 0 .4rem 0;font-size:.95rem;color:#444}",
      "  .step .kv{margin:.15rem 0}",
      "  .step .kv b{color:#222}",
      "  code{background:#f0f0f0;padding:0 .25rem;border-radius:3px}",
      "</style></head><body>",
      f"<h1>{esc(visible(pick['task_template']))}</h1>",
      "<div class='meta'>",
  ]
  if blind:
    html_lines.extend([
        f"  category: {esc(visible(pick['category']))} · ",
        f"  app: {esc(visible(pick['app_id']))}",
        "</div>",
    ])
  else:
    html_lines.extend([
        f"  agent: <b>{esc(pick['model_name'])}</b> · ",
        f"  category: {esc(pick['category'])} · ",
        f"  app: {esc(pick['app_id'])} · ",
        f"  pkl: <code>{esc(pick['pkl_path'])}</code>",
        "</div>",
    ])
  html_lines.extend([
      f"<div class='banner verdict{' pass' if float(pick.get('is_successful') or 0) >= 0.5 else ''}'>",
      f"  <b>Validator verdict (read-only):</b> is_successful = {esc(visible(pick['is_successful']))}",
      "</div>",
      f"<div><b>Goal:</b> {esc(visible(pick['goal']))}</div>",
  ])
  if not blind:
    html_lines.extend([
        "<div class='banner judge'>",
        f"  <b>Prior judge label:</b> {esc(pick.get('judge_label', 'unjudged'))} "
        f"(confidence {esc(pick.get('judge_confidence', '-'))})<br>",
        f"  <small>{esc(pick.get('judge_rationale', ''))}</small>",
        "</div>",
    ])
  if saved_frames:
    html_lines.append(f"<h2>{esc(frame_heading)}</h2><div class='frames'>")
    for frame in saved_frames:
      rel = Path(frame["path"]).name
      html_lines.append("  <div class='frame'>")
      html_lines.append(f"    <img src='{esc(rel)}' alt='step {frame['step']}'>")
      html_lines.append(
          f"    <div class='lbl'>step {frame['step']} · {esc(frame['field'])}</div>"
      )
      html_lines.append("  </div>")
    html_lines.append("</div>")
  if step_rows:
    html_lines.append("<h2>Full step trace</h2>")
    for row in step_rows:
      html_lines.append(f"<div class='step'><h3>Step {esc(row.get('step'))}</h3>")
      for key, value in row.items():
        if key == "step":
          continue
        html_lines.append(
            f"  <div class='kv'><b>{esc(key)}:</b> {esc(visible(value))}</div>"
        )
      html_lines.append("</div>")
  html_lines.append("</body></html>")
  html_path = out_dir / "report.html"
  html_path.write_text("\n".join(html_lines), encoding="utf-8")

  return {
      "frames": saved_frames,
      "html_path": str(html_path),
      "summary_md": str(summary_path),
  }


def _save_episode_artifacts(
    pick: dict[str, Any],
    episode: dict[str, Any],
    out_dir: Path,
    max_frames: int,
    max_dim: int,
    blind: bool,
) -> dict[str, Any]:
  """Compatibility wrapper used by the browser annotation UI."""
  artifacts = _render_html_report(
      pick=pick,
      episode=episode,
      out_dir=out_dir,
      max_frames=max_frames,
      max_dim=max_dim,
      blind=blind,
  )
  artifacts["summary"] = artifacts["summary_md"]
  return artifacts


def _open_in_browser(path: str) -> None:
  """Best-effort open of the HTML report in the system browser."""
  opener = None
  for candidate in ("xdg-open", "open", "wslview"):
    if shutil.which(candidate):
      opener = candidate
      break
  if not opener:
    return
  try:
    subprocess.Popen(
        [opener, path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
  except OSError:
    pass


def _prompt_choice(prompt: str, choices: tuple[str, ...]) -> str:
  options = "/".join(choices)
  while True:
    sys.stdout.write(f"  {prompt} [{options}]: ")
    sys.stdout.flush()
    answer = sys.stdin.readline().strip().lower()
    if not answer:
      continue
    matches = [choice for choice in choices if choice.startswith(answer)]
    if len(matches) == 1:
      return matches[0]
    sys.stdout.write(f"    not understood, pick one of {choices}\n")


def _prompt_score(prompt: str) -> int:
  while True:
    sys.stdout.write(f"  {prompt} [0-3]: ")
    sys.stdout.flush()
    answer = sys.stdin.readline().strip()
    if answer in {"0", "1", "2", "3"}:
      return int(answer)


def _prompt_text(prompt: str) -> str:
  sys.stdout.write(f"  {prompt}\n> ")
  sys.stdout.flush()
  return sys.stdin.readline().rstrip("\n")


def _prompt_nonempty_text(prompt: str) -> str:
  while True:
    value = _prompt_text(prompt).strip()
    if value:
      return value
    sys.stdout.write("    a rationale is required\n")


def _prompt_yes_no(prompt: str, default_no: bool = True) -> bool:
  hint = "y/N" if default_no else "Y/n"
  while True:
    sys.stdout.write(f"  {prompt} [{hint}]: ")
    sys.stdout.flush()
    answer = sys.stdin.readline().strip().lower()
    if not answer:
      return not default_no
    if answer in {"y", "yes"}:
      return True
    if answer in {"n", "no"}:
      return False


def _load_done_keys(annotations_path: Path, annotator: str) -> set[str]:
  if not annotations_path.exists():
    return set()
  done: set[str] = set()
  with annotations_path.open("r", encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        continue
      if row.get("annotator") != annotator:
        continue
      key = f"{row.get('pool')}::{row.get('episode_id')}"
      done.add(key)
  return done


def _format_eta(remaining: int, avg_seconds: float) -> str:
  if avg_seconds <= 0:
    return "?"
  total = remaining * avg_seconds
  if total < 60:
    return f"{int(total)}s"
  if total < 3600:
    return f"{int(total // 60)}m"
  return f"{total / 3600:.1f}h"


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--sample_manifest", required=True)
  parser.add_argument("--out_dir", required=True)
  parser.add_argument(
      "--annotator", required=True, help="Annotator id (your name/initials)."
  )
  parser.add_argument(
      "--max_frames",
      type=int,
      default=0,
      help=(
          "Number of ordered screenshots shown to the annotator; 0 "
          "(default) shows the full trajectory. Use 6 only for the "
          "limited-evidence ablation."
      ),
  )
  parser.add_argument("--max_dim", type=int, default=896)
  parser.add_argument(
      "--blind",
      dest="blind",
      action="store_true",
      default=True,
      help="Hide the prior judge label from the human annotator.",
  )
  parser.add_argument(
      "--show_prior_judge",
      dest="blind",
      action="store_false",
      help="Show the prior VLM label/rationale. Do not use for validation.",
  )
  parser.add_argument(
      "--pool",
      choices=("all", "primary", "double_annotation"),
      default="all",
      help=(
          "Which sample pool to annotate. Use primary for the first "
          "annotator and double_annotation for the second annotator."
      ),
  )
  parser.add_argument(
      "--no_open",
      action="store_true",
      help="Do not auto-open the HTML report in the browser.",
  )
  parser.add_argument(
      "--no_glossary",
      action="store_true",
      help="Skip printing the failure-mode glossary at the start.",
  )
  parser.add_argument(
      "--restrict_to_failures",
      action="store_true",
      default=True,
      help=(
          "Skip any sampled episode whose validator verdict is success. "
          "On by default; the new pipeline only annotates failures."
      ),
  )
  parser.add_argument(
      "--include_successes",
      action="store_true",
      help=(
          "Also annotate validator-successes (for the OPTIONAL separate "
          "validator-audit pass — see Section 6.7 of the appendix). Suspect-"
          "validator flag becomes the primary signal in this mode."
      ),
  )
  args = parser.parse_args()

  out_dir = Path(args.out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  frames_root = out_dir / "frames"
  frames_root.mkdir(parents=True, exist_ok=True)
  annotations_path = out_dir / "annotations.jsonl"
  audit_path = out_dir / "audit_flags.jsonl"

  if not _PIL_AVAILABLE:
    print(
        "warning: Pillow not installed — screenshots will not be rendered. "
        "Install with: pip install Pillow",
        file=sys.stderr,
    )

  done_keys = _load_done_keys(annotations_path, args.annotator)

  picks: list[dict[str, Any]] = []
  with Path(args.sample_manifest).open("r", encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      pick = json.loads(line)
      if args.pool != "all" and pick.get("pool", "primary") != args.pool:
        continue
      is_success = float(pick.get("is_successful") or 0.0) >= 0.5
      if (
          args.restrict_to_failures
          and is_success
          and not args.include_successes
      ):
        continue
      picks.append(pick)

  if not picks:
    print("Empty sample manifest after filtering.", file=sys.stderr)
    return 1

  if not args.no_glossary:
    _print_glossary()
  sys.stdout.write(
      f"Annotator: {args.annotator}   "
      f"Episodes in pool: {len(picks)}   "
      f"Already done: {len(done_keys)}\n"
  )
  if not args.blind:
    sys.stdout.write(f"HTML reports + frames will be saved under: {frames_root}\n")
  if args.include_successes:
    sys.stdout.write(
        "Mode: validator-audit (successes included). For each episode you\n"
        "label the failure mode AS IF the validator's verdict were correct,\n"
        "and flag --suspect-validator only when the screenshots clearly\n"
        "contradict the validator.\n"
    )
  else:
    sys.stdout.write(
        "Mode: failure-mode classification on validator-failed episodes.\n"
        "You do NOT override the validator's pass/fail.\n"
    )
  sys.stdout.flush()

  total = len(picks)
  start_session = time.time()
  per_case_seconds: list[float] = []
  annotated = 0

  for index, pick in enumerate(picks, start=1):
    key = f"{pick.get('pool', 'primary')}::{pick['episode_id']}"
    if key in done_keys:
      continue

    pkl_path = Path(pick["pkl_path"])
    try:
      payload = _read_pkl_gz(pkl_path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if args.blind:
        print(f"  skipping case {index}: unable to load episode", file=sys.stderr)
      else:
        print(f"  skipping {pkl_path}: {exc}", file=sys.stderr)
      continue
    episodes = payload if isinstance(payload, list) else [payload]
    episode = episodes[pick.get("episode_index", 0)]
    if not isinstance(episode, dict):
      if args.blind:
        print(f"  skipping case {index}: episode not a dict", file=sys.stderr)
      else:
        print(f"  skipping {pkl_path}: episode not a dict", file=sys.stderr)
      continue

    episode_dir = frames_root / pick["episode_id"]
    artifacts = _render_html_report(
        pick, episode, episode_dir, args.max_frames, args.max_dim, args.blind
    )

    case_start = time.time()
    sys.stdout.write("\n" + "=" * 78 + "\n")
    avg = (
        sum(per_case_seconds) / len(per_case_seconds)
        if per_case_seconds
        else 0.0
    )
    eta = _format_eta(total - index, avg)
    identity = (
        f"{pick['category']}/{pick['app_id']}"
        if args.blind
        else f"{pick['model_name']} | {pick['category']}/{pick['app_id']}"
    )
    sys.stdout.write(
        f"[{index}/{total}] {identity} | {pick['task_template']}    "
        f"(annotated {annotated}, ~ETA {eta})\n"
    )
    goal = _redact_blind_text(pick["goal"], pick) if args.blind else pick["goal"]
    sys.stdout.write(f"  Goal: {goal}\n")
    sys.stdout.write(
        f"  Validator verdict (read-only): is_successful = {pick['is_successful']}\n"
    )
    if not args.blind:
      sys.stdout.write(
          f"  Prior judge label: {pick.get('judge_label', '-')} "
          f"({pick.get('judge_confidence', '-')})\n"
      )
    if not args.blind:
      sys.stdout.write(f"  Report: file://{artifacts['html_path']}\n")
      sys.stdout.write(
          f"  Saved {len(artifacts['frames'])} key frames in {episode_dir}\n"
      )
    sys.stdout.flush()

    if not args.no_open:
      _open_in_browser(artifacts["html_path"])

    sys.stdout.write(
        "  Read the report in your browser. Type 'skip' on the first prompt "
        "to skip this case.\n"
    )

    mode = _prompt_choice(
        "primary_failure_mode", DEFAULT_FAILURE_MODES + ("skip",)
    )
    if mode == "skip":
      continue
    planning = _prompt_score("planning_score (0-3)")
    grounding = _prompt_score("grounding_score (0-3)")
    confidence = _prompt_choice("your_confidence", ("low", "medium", "high"))
    notes = _prompt_nonempty_text("one-line rationale (with step/screenshot ref)")
    suspect_validator = _prompt_yes_no(
        "OPTIONAL: do the screenshots contradict the validator's verdict?",
        default_no=True,
    )

    case_seconds = time.time() - case_start
    per_case_seconds.append(case_seconds)
    annotated += 1

    record = {
        "annotator": args.annotator,
        "annotated_at": dt.datetime.now().isoformat(),
        "seconds_spent": round(case_seconds, 1),
        "pool": pick.get("pool", "primary"),
        "episode_id": pick["episode_id"],
        "model_name": pick["model_name"],
        "category": pick["category"],
        "app_id": pick["app_id"],
        "task_template": pick["task_template"],
        "goal": pick["goal"],
        "is_successful": pick["is_successful"],
        "pkl_path": pick["pkl_path"],
        "frames_dir": str(episode_dir),
        "html_report": artifacts["html_path"],
        "human_label": {
            "primary_failure_mode": mode,
            "planning_score": planning,
            "grounding_score": grounding,
            "confidence": confidence,
            "rationale": notes,
        },
        "judge_prior": {
            "primary_failure_mode": pick.get("judge_label"),
            "confidence": pick.get("judge_confidence"),
            "rationale": pick.get("judge_rationale"),
        },
        "suspect_validator": suspect_validator,
    }
    with annotations_path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if suspect_validator:
      with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    done_keys.add(key)

  duration = time.time() - start_session
  sys.stdout.write(
      f"\nSession finished. Annotated {annotated} new cases in "
      f"{duration / 60:.1f} min "
      f"(avg {duration / max(1, annotated):.1f}s/case).\n"
      f"Annotations: {annotations_path}\n"
  )
  if audit_path.exists():
    sys.stdout.write(f"Validator-audit flags: {audit_path}\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
