#!/usr/bin/env python3
"""Rejudge the frozen five-category C1 failure cohort with Qwen3-VL.

This is a paired judge-robustness run.  It does not rebuild the cohort or the
text evidence: every selected case is joined back to its exact historical
Gemini JSONL row and reuses that row's ``case_payload`` verbatim.  Screenshots
are reconstructed from the raw pickle with the historical six-frame settings,
and their count must match the count recorded by the Gemini call before Qwen
is queried.

The API key is read from an environment variable and is never written to an
artifact.  Successful calls are cached per episode, so an interrupted run can
be resumed safely.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import classify_catbench_failures as classifier  # noqa: E402


DEFAULT_GEMINI_CSV = Path(
    "$HOME/anyappbench_results/"
    "vlm_judge_merged_c1/"
    "merged_failure_mode_case_level_paper_models.csv"
)
DEFAULT_CATEGORIES = ("clock", "contacts", "files", "maps", "sms")
DEFAULT_EXPECTED_N = 2195
RUNNER_VERSION = "qwen-c1-paired-v1"


@dataclasses.dataclass(frozen=True)
class FrozenCase:
  flat: dict[str, str]
  gemini_row: dict[str, Any]

  @property
  def episode_id(self) -> str:
    return str(self.flat["episode_id"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      if not line.strip():
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
      if not isinstance(row, dict):
        raise ValueError(f"Expected a JSON object at {path}:{line_number}")
      rows.append(row)
  return rows


def _stable_json(value: Any) -> str:
  return json.dumps(
      value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
  )


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    while chunk := handle.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
  with tempfile.NamedTemporaryFile(
      mode="w",
      encoding="utf-8",
      dir=path.parent,
      prefix=f".{path.name}.",
      suffix=".tmp",
      delete=False,
  ) as handle:
    tmp = Path(handle.name)
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
  tmp.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      mode="w",
      encoding="utf-8",
      dir=path.parent,
      prefix=f".{path.name}.",
      suffix=".tmp",
      delete=False,
  ) as handle:
    tmp = Path(handle.name)
    for row in rows:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
  tmp.replace(path)


def _load_frozen_cases(
    merged_csv: Path,
    categories: set[str],
    expected_n: int,
) -> list[FrozenCase]:
  with merged_csv.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    required = {
        "episode_id",
        "model_name",
        "category",
        "app_id",
        "task_template",
        "pkl_path",
        "jsonl_path",
    }
    missing = required - set(reader.fieldnames or [])
    if missing:
      raise ValueError(f"Merged CSV is missing columns: {sorted(missing)}")
    flat_rows = [
        dict(row) for row in reader if str(row.get("category")) in categories
    ]

  if expected_n and len(flat_rows) != expected_n:
    raise ValueError(
        f"Expected {expected_n} frozen cases, found {len(flat_rows)} in "
        f"{sorted(categories)}"
    )

  seen_ids: set[str] = set()
  by_jsonl: dict[Path, list[dict[str, str]]] = {}
  for flat in flat_rows:
    episode_id = str(flat.get("episode_id") or "")
    if not episode_id:
      raise ValueError("Merged CSV contains an empty episode_id")
    if episode_id in seen_ids:
      raise ValueError(f"Duplicate frozen episode_id: {episode_id}")
    seen_ids.add(episode_id)
    jsonl_path = Path(str(flat["jsonl_path"])).expanduser().resolve()
    by_jsonl.setdefault(jsonl_path, []).append(flat)

  cases: list[FrozenCase] = []
  identity_fields = (
      "model_name",
      "category",
      "app_id",
      "task_template",
      "pkl_path",
  )
  for jsonl_path, selected in by_jsonl.items():
    if not jsonl_path.exists():
      raise FileNotFoundError(f"Historical Gemini JSONL is missing: {jsonl_path}")
    source_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(jsonl_path):
      source_by_id.setdefault(str(row.get("episode_id") or ""), []).append(row)
    for flat in selected:
      episode_id = str(flat["episode_id"])
      matches = source_by_id.get(episode_id, [])
      if len(matches) != 1:
        raise ValueError(
            f"Expected one source row for {episode_id} in {jsonl_path}; "
            f"found {len(matches)}"
        )
      source = matches[0]
      case_payload = source.get("case_payload")
      if not isinstance(case_payload, dict):
        raise ValueError(f"Historical row {episode_id} has no case_payload")
      for field in identity_fields:
        source_value = source.get(field)
        if source_value is None:
          source_value = case_payload.get(field)
        if str(source_value or "") != str(flat.get(field) or ""):
          raise ValueError(
              f"Frozen/source mismatch for {episode_id}, field={field}: "
              f"{flat.get(field)!r} != {source_value!r}"
          )
      if not Path(str(flat["pkl_path"])).exists():
        raise FileNotFoundError(f"Raw episode pickle is missing: {flat['pkl_path']}")
      cases.append(FrozenCase(flat=flat, gemini_row=source))

  cases.sort(
      key=lambda case: (
          case.flat["model_name"],
          case.flat["category"],
          case.flat["app_id"],
          case.flat["task_template"],
          case.episode_id,
      )
  )
  return cases


def _historical_image_count(row: dict[str, Any], episode_id: str) -> int:
  usage = row.get("usage")
  if not isinstance(usage, dict) or usage.get("num_images") is None:
    raise ValueError(f"Historical row {episode_id} has no usage.num_images")
  try:
    value = int(usage["num_images"])
  except (TypeError, ValueError) as exc:
    raise ValueError(
        f"Invalid historical image count for {episode_id}: "
        f"{usage.get('num_images')!r}"
    ) from exc
  if value < 0 or value > 6:
    raise ValueError(f"Historical image count out of range for {episode_id}: {value}")
  return value


def _resolve_episode(case: FrozenCase) -> tuple[dict[str, Any], int]:
  pkl_path = Path(case.flat["pkl_path"])
  payload = classifier._read_pkl_gz(pkl_path)  # pylint: disable=protected-access
  episodes = payload if isinstance(payload, list) else [payload]
  matches = [
      index
      for index, episode in enumerate(episodes)
      if isinstance(episode, dict)
      and classifier._record_id(  # pylint: disable=protected-access
          case.flat["model_name"],
          case.flat["category"],
          case.flat["app_id"],
          pkl_path,
          index,
      )
      == case.episode_id
  ]
  if len(matches) != 1:
    raise ValueError(
        f"Could not resolve unique raw episode for {case.episode_id}; "
        f"matches={matches}, episodes={len(episodes)}"
    )
  index = matches[0]
  return episodes[index], index


def _validate_judgment(judgment: dict[str, Any]) -> None:
  required = set(classifier.JUDGE_OUTPUT_SCHEMA["required"])
  missing = required - set(judgment)
  if missing:
    raise ValueError(f"Qwen judgment is missing fields: {sorted(missing)}")
  extra = set(judgment) - required
  if extra:
    raise ValueError(f"Qwen judgment has unexpected fields: {sorted(extra)}")
  mode = judgment.get("primary_failure_mode")
  if mode not in classifier.DEFAULT_FAILURE_MODES:
    raise ValueError(f"Invalid Qwen failure mode: {mode!r}")
  for name in ("planning_score", "grounding_score"):
    value = judgment.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 3:
      raise ValueError(f"Invalid Qwen {name}: {value!r}")
  if judgment.get("confidence") not in {"low", "medium", "high"}:
    raise ValueError(f"Invalid Qwen confidence: {judgment.get('confidence')!r}")
  if not isinstance(judgment.get("rationale"), str):
    raise ValueError("Qwen rationale is not a string")
  if not isinstance(judgment.get("evidence"), list) or not all(
      isinstance(item, str) for item in judgment["evidence"]
  ):
    raise ValueError("Qwen evidence is not a list of strings")


def _config_hash(config: dict[str, Any]) -> str:
  # Only evidence/model choices belong in a per-episode cache key.  Operational
  # settings such as tunnel URL, worker count, retry count, or a temporary
  # subset filter may change when resuming without changing the judgment.
  semantic = {
      key: config[key]
      for key in (
          "runner_version",
          "merged_csv_sha256",
          "model",
          "system_prompt_sha1",
          "output_schema_sha1",
          "evidence",
      )
  }
  return hashlib.sha256(_stable_json(semantic).encode("utf-8")).hexdigest()[:16]


def _input_fingerprint(case: FrozenCase) -> dict[str, Any]:
  """Cheaply bind a cache entry to its exact text source and raw pickle."""
  pkl_path = Path(case.flat["pkl_path"])
  stat = pkl_path.stat()
  return {
      "case_payload_sha256": hashlib.sha256(
          _stable_json(case.gemini_row["case_payload"]).encode("utf-8")
      ).hexdigest(),
      "pkl_size_bytes": stat.st_size,
      "pkl_mtime_ns": stat.st_mtime_ns,
  }


def _result_metadata(
    case: FrozenCase, config_hash: str, model: str
) -> dict[str, Any]:
  return {
      "episode_id": case.episode_id,
      "judge_backend": "llm",
      "judge_model": model,
      "judge_config_hash": config_hash,
      "model_name": case.flat["model_name"],
      "category": case.flat["category"],
      "app_id": case.flat["app_id"],
      "app_name": case.flat.get("app_name", ""),
      "task_template": case.flat["task_template"],
      "pkl_path": case.flat["pkl_path"],
      "gemini_jsonl_path": case.flat["jsonl_path"],
  }


def _historical_artifact_hashes(
    cases: Sequence[FrozenCase],
) -> tuple[dict[str, str], dict[str, str]]:
  """Hash and audit the immutable Gemini evidence sources for this cohort."""
  jsonl_paths = sorted({Path(case.flat["jsonl_path"]) for case in cases})
  jsonl_hashes = {str(path): _sha256_file(path) for path in jsonl_paths}

  expected = {
      "judge_backend": "gemini",
      "system_prompt_sha1": hashlib.sha1(
          classifier.SYSTEM_PROMPT.encode("utf-8")
      ).hexdigest(),
      "with_screenshots": True,
      "screenshot_max_frames": 6,
      "max_steps": 6,
      "smart_steps": True,
      "screenshot_max_dim": 896,
  }
  config_hashes: dict[str, str] = {}
  for path in sorted({path.parent / "judge_config.json" for path in jsonl_paths}):
    if not path.exists():
      raise FileNotFoundError(f"Historical Gemini judge config is missing: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
      raise ValueError(f"Historical Gemini judge config is not an object: {path}")
    mismatches = {
        key: (expected_value, config.get(key))
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    }
    if mismatches:
      raise ValueError(
          f"Historical Gemini judge config mismatch at {path}: {mismatches}"
      )
    config_hashes[str(path)] = _sha256_file(path)
  return jsonl_hashes, config_hashes


def _valid_cached_result(
    cached: Any,
    case: FrozenCase,
    config_hash: str,
    model: str,
) -> bool:
  if not isinstance(cached, dict):
    return False
  if (
      cached.get("status") != "ok"
      or cached.get("episode_id") != case.episode_id
      or cached.get("judge_config_hash") != config_hash
      or cached.get("judge_model") != model
  ):
    return False
  parity = cached.get("evidence_parity")
  if not isinstance(parity, dict) or parity.get("image_count_matches") is not True:
    return False
  judgment = cached.get("qwen_judgment")
  if not isinstance(judgment, dict):
    return False
  try:
    _validate_judgment(judgment)
  except ValueError:
    return False
  stored_fingerprint = cached.get("input_fingerprint")
  if stored_fingerprint is not None and stored_fingerprint != _input_fingerprint(case):
    return False
  return True


def _judge_one(
    case: FrozenCase,
    *,
    out_dir: Path,
    config_hash: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout_sec: float,
    max_retries: int,
    resume: bool,
) -> dict[str, Any]:
  cache_path = out_dir / "cache" / f"{case.episode_id}_{config_hash}.json"
  if resume and cache_path.exists():
    try:
      cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      cached = None
    if _valid_cached_result(cached, case, config_hash, model):
      cached["cache_hit"] = True
      # Caches produced by the already-running v1 job predate the fingerprint.
      # Migrate them in place without repeating a paid judge call. The global
      # historical source audit above establishes that these are the unchanged
      # frozen inputs used by that job.
      if "input_fingerprint" not in cached:
        cached["input_fingerprint"] = _input_fingerprint(case)
        cached["cache_format_migrated"] = True
        _atomic_json(cache_path, cached)
      return cached

  metadata = _result_metadata(case, config_hash, model)
  case_payload = case.gemini_row["case_payload"]
  expected_images = _historical_image_count(case.gemini_row, case.episode_id)
  episode, episode_index = _resolve_episode(case)
  key_indices = case_payload.get("key_step_indices") or []
  if not isinstance(key_indices, list) or not all(
      isinstance(index, int) and not isinstance(index, bool) and index >= 0
      for index in key_indices
  ):
    raise ValueError(f"Invalid key_step_indices for {case.episode_id}")
  picked_indices = key_indices[:6]
  images = classifier._extract_screenshots_for_judge(  # pylint: disable=protected-access
      episode,
      picked_indices,
      max_dim=896,
      quality=80,
      prefer_som=True,
  )
  actual_images = len(images)
  if actual_images != expected_images:
    raise ValueError(
        f"Image-parity failure for {case.episode_id}: "
        f"Qwen reconstruction={actual_images}, historical Gemini={expected_images}"
    )

  # The encoded images and verbatim historical case payload are sufficient for
  # the API call. Release the potentially large raw pickle before waiting on
  # network inference; reference counting handles it without a process-wide
  # cyclic-GC pause in each worker thread.
  del episode

  judgment, usage = classifier._call_judge(  # pylint: disable=protected-access
      backend="llm",
      case_payload=case_payload,
      images=images,
      system_prompt=classifier.SYSTEM_PROMPT,
      model=model,
      base_url=base_url,
      api_key=api_key,
      timeout_sec=timeout_sec,
      max_retries=max_retries,
      response_format=True,
      strict_json=True,
  )
  _validate_judgment(judgment)
  result = {
      **metadata,
      "status": "ok",
      "cache_hit": False,
      "episode_index": episode_index,
      "source": case.flat.get("source", ""),
      "goal": case.gemini_row.get("goal") or case_payload.get("goal") or "",
      "is_successful": case.gemini_row.get("is_successful", 0.0),
      "gemini_judgment": case.gemini_row.get("judgment", {}),
      "gemini_usage": case.gemini_row.get("usage", {}),
      "qwen_judgment": judgment,
      "qwen_usage": usage,
      "input_fingerprint": _input_fingerprint(case),
      "evidence_parity": {
          "case_payload_reused_verbatim": True,
          "key_step_indices": picked_indices,
          "historical_gemini_num_images": expected_images,
          "qwen_num_images": actual_images,
          "image_count_matches": True,
          "qwen_image_sha256": [
              hashlib.sha256(image["jpeg_base64"].encode("ascii")).hexdigest()
              for image in images
          ],
          "screenshot_max_dim": 896,
          "jpeg_quality": 80,
          "prefer_som": True,
      },
  }
  _atomic_json(cache_path, result)
  return result


def _error_result(
    case: FrozenCase,
    config_hash: str,
    model: str,
    exc: BaseException,
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
  error_text = f"{type(exc).__name__}: {exc}"
  for secret in secrets:
    if secret:
      error_text = error_text.replace(secret, "<redacted>")
  return {
      **_result_metadata(case, config_hash, model),
      "status": "error",
      "qwen_error": error_text,
      "source": case.flat.get("source", ""),
      "gemini_judgment": case.gemini_row.get("judgment", {}),
      "gemini_usage": case.gemini_row.get("usage", {}),
  }


def _select_cases(
    cases: list[FrozenCase],
    models: set[str],
    episode_ids: set[str],
    limit: int,
) -> list[FrozenCase]:
  selected = [
      case
      for case in cases
      if (not models or case.flat["model_name"] in models)
      and (not episode_ids or case.episode_id in episode_ids)
  ]
  if episode_ids:
    found = {case.episode_id for case in selected}
    missing = episode_ids - found
    if missing:
      raise ValueError(f"Requested episode_ids not in frozen roster: {sorted(missing)}")
  return selected[:limit] if limit > 0 else selected


def _assert_output_compatible(
    out_dir: Path,
    config_hash: str,
    selected: Sequence[FrozenCase],
) -> None:
  """Refuse to mix a smoke/subset run with another run in one directory."""
  config_path = out_dir / "judge_config.json"
  if config_path.exists():
    try:
      previous = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
      raise ValueError(f"Cannot read existing run config: {config_path}") from exc
    if not isinstance(previous, dict) or previous.get("config_hash") != config_hash:
      raise ValueError(
          f"Output directory belongs to a different judge configuration: {out_dir}"
      )

  roster_path = out_dir / "selected_roster.jsonl"
  if roster_path.exists():
    previous_ids = [str(row.get("episode_id") or "") for row in _read_jsonl(roster_path)]
    selected_ids = [case.episode_id for case in selected]
    if previous_ids != selected_ids:
      raise ValueError(
          "Output directory already contains a different selected roster; "
          "use a dedicated directory for smoke tests or subsets"
      )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--gemini_csv", type=Path, default=DEFAULT_GEMINI_CSV)
  parser.add_argument("--out_dir", type=Path, required=True)
  parser.add_argument(
      "--categories", nargs="+", default=list(DEFAULT_CATEGORIES)
  )
  parser.add_argument("--expected_n", type=int, default=DEFAULT_EXPECTED_N)
  parser.add_argument("--model_filter", action="append", default=[])
  parser.add_argument("--episode_id", action="append", default=[])
  parser.add_argument("--limit", type=int, default=0)
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--model", default="catbench-judge")
  parser.add_argument(
      "--base_url",
      default=os.environ.get("FAILURE_JUDGE_BASE_URL", ""),
  )
  parser.add_argument("--api_key_env", default="FAILURE_JUDGE_API_KEY")
  parser.add_argument("--timeout_sec", type=float, default=600.0)
  parser.add_argument("--max_retries", type=int, default=2)
  parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--continue_on_error", action="store_true")
  parser.add_argument(
      "--roster_only",
      action="store_true",
      help="Audit and write the exact selected roster without making API calls.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  if args.workers < 1:
    raise ValueError("--workers must be >= 1")
  merged_csv = args.gemini_csv.expanduser().resolve()
  out_dir = args.out_dir.expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  cases = _load_frozen_cases(
      merged_csv,
      set(args.categories),
      args.expected_n,
  )
  historical_jsonl_hashes, historical_config_hashes = (
      _historical_artifact_hashes(cases)
  )
  selected = _select_cases(
      cases, set(args.model_filter), set(args.episode_id), args.limit
  )
  if not selected:
    raise ValueError("No cases selected")

  config: dict[str, Any] = {
      "runner_version": RUNNER_VERSION,
      "merged_csv": str(merged_csv),
      "merged_csv_sha256": _sha256_file(merged_csv),
      "historical_jsonl_sha256": historical_jsonl_hashes,
      "historical_judge_config_sha256": historical_config_hashes,
      "frozen_roster_n": len(cases),
      "selected_n": len(selected),
      "categories": list(args.categories),
      "model_filter": list(args.model_filter),
      "episode_id_filter": list(args.episode_id),
      "model": args.model,
      "base_url": args.base_url,
      "api_key_env": args.api_key_env,
      "workers": args.workers,
      "timeout_sec": args.timeout_sec,
      "max_retries": args.max_retries,
      "strict_top_level_json": True,
      "backend_native_serialization": {
          "semantic_prompt_and_evidence_match": True,
          "gemini_image_label_position": "after_image",
          "qwen_image_label_position": "before_image",
          "gemini_response_constraint": "application/json",
          "qwen_response_constraint": "strict_json_schema",
      },
      "system_prompt_sha1": hashlib.sha1(
          classifier.SYSTEM_PROMPT.encode("utf-8")
      ).hexdigest(),
      "output_schema_sha1": hashlib.sha1(
          _stable_json(classifier.JUDGE_OUTPUT_SCHEMA).encode("utf-8")
      ).hexdigest(),
      "evidence": {
          "reuse_historical_case_payload": True,
          "max_frames": 6,
          "screenshot_max_dim": 896,
          "jpeg_quality": 80,
          "prefer_som": True,
          "require_historical_image_count_parity": True,
      },
      "out_dir": str(out_dir),
  }
  config_hash = _config_hash(config)
  config["config_hash"] = config_hash
  _assert_output_compatible(out_dir, config_hash, selected)
  _atomic_json(out_dir / "judge_config.json", config)
  _write_jsonl(
      out_dir / "selected_roster.jsonl",
      (
          {
              **_result_metadata(case, config_hash, args.model),
              "source": case.flat.get("source", ""),
              "historical_gemini_num_images": _historical_image_count(
                  case.gemini_row, case.episode_id
              ),
          }
          for case in selected
      ),
  )
  print(
      f"Frozen roster: {len(cases)}; selected: {len(selected)}; "
      f"config={config_hash}",
      flush=True,
  )
  if args.roster_only:
    return 0

  if not args.base_url:
    raise ValueError(
        "Missing --base_url or FAILURE_JUDGE_BASE_URL environment variable"
    )
  api_key = os.environ.get(args.api_key_env, "")
  if not api_key:
    raise ValueError(f"API key environment variable is empty: {args.api_key_env}")

  # Install legacy pickle modules before worker threads enter the unpickler.
  classifier._install_android_env_shims()  # pylint: disable=protected-access
  results: dict[str, dict[str, Any]] = {}
  completed = 0
  errors = 0
  lock = threading.Lock()

  def run(case: FrozenCase) -> dict[str, Any]:
    return _judge_one(
        case,
        out_dir=out_dir,
        config_hash=config_hash,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
        resume=args.resume,
    )

  with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    future_to_case = {pool.submit(run, case): case for case in selected}
    for future in concurrent.futures.as_completed(future_to_case):
      case = future_to_case[future]
      try:
        row = future.result()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        if not args.continue_on_error:
          for pending in future_to_case:
            pending.cancel()
          raise
        row = _error_result(
            case, config_hash, args.model, exc, secrets=(api_key,)
        )
      with lock:
        completed += 1
        errors += row.get("status") != "ok"
        results[case.episode_id] = row
        marker = "cache" if row.get("cache_hit") else row.get("status", "")
        print(
            f"[{completed}/{len(selected)}] {marker} "
            f"{case.flat['model_name']} {case.flat['category']}/"
            f"{case.flat['app_id']} {case.flat['task_template']}",
            flush=True,
        )

  ordered = [results[case.episode_id] for case in selected]
  _write_jsonl(out_dir / "failure_mode_judgments.jsonl", ordered)
  _write_jsonl(
      out_dir / "judge_errors.jsonl",
      (row for row in ordered if row.get("status") != "ok"),
  )
  coverage = {
      "frozen_roster_cases": len(cases),
      "selected_cases": len(selected),
      "successful_qwen_calls": len(selected) - errors,
      "qwen_errors": errors,
      "cache_hits": sum(bool(row.get("cache_hit")) for row in ordered),
      "all_successful_cases_have_image_parity": all(
          row.get("status") != "ok"
          or row.get("evidence_parity", {}).get("image_count_matches") is True
          for row in ordered
      ),
      "config_hash": config_hash,
  }
  _atomic_json(out_dir / "coverage.json", coverage)
  print(
      f"Finished: success={coverage['successful_qwen_calls']} "
      f"errors={errors}; output={out_dir}",
      flush=True,
  )
  return 1 if errors else 0


if __name__ == "__main__":
  raise SystemExit(main())
