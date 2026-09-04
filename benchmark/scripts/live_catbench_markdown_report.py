#!/usr/bin/env python3
"""Write a live Markdown report for a running CATBench matrix run."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import gzip
import io
import json
import math
import os
import pickle
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_ROOT.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

from app_generalization_profiles import get_domain_profiles  # pylint: disable=wrong-import-position


MARKDOWN_REDIRECT_FILE = REPO_ROOT / ".catbench_markdown_redirect"
CATEGORY_ORDER = ("sms", "files", "maps", "contacts", "clock")
CONFIG_PATH = BENCHMARK_ROOT / "configs" / "catbench_5cat_models.json"
SCREENSHOT_FIELDS = (
    "raw_screenshot",
    "before_screenshot_with_som",
    "after_screenshot_with_som",
    "screenshot",
)
SKIP_EXCEPTION_MARKERS = (
    "[skipped_uninstalled]",
    "[skipped_environment]",
    "_EnvironmentNetworkError",
    "network/connectivity error dialog visible",
)
APP_TYPE_ORIGINAL = "AW Orig."
APP_TYPE_NEW = "New"
CATEGORY_LABELS = {
    "sms": "SMS",
    "files": "File Manager",
    "contacts": "Contacts",
    "maps": "Maps",
    "clock": "Clock",
}
PROVIDED_TABLE_TASKS = {
    "sms": (
        "Send",
        "Reply",
        "ReplyMostRecent",
        "Resend",
        "SendToContact",
        "SendReceivedAddress",
        "CreateDraftMessage",
        "EditDraftMessage",
        "DeleteConversation",
        "ForwardMessage",
    ),
    "files": (
        "CreateFolder",
        "RenameFile",
        "DeleteFile",
        "MoveFile",
        "SaveCopyOfFile",
        "SearchFile",
        "CompressFiles",
        "ExtractArchive",
        "ViewFileInfo",
        "ShareFile",
    ),
    "contacts": (
        "AddContact",
        "NewContactDraft",
        "EditContact",
        "SearchContact",
        "ViewContactDetails",
        "AddFavoriteContact",
        "RemoveFavoriteContact",
        "DeleteContact",
        "CallContact",
        "MessageContact",
    ),
    "maps": (
        "SearchPlace",
        "AddFavorite",
        "RemoveFavorite",
        "AddMarker",
        "DeleteMarker",
        "RecordTrack",
        "GetDirections",
        "SearchNearbyPlace",
        "ExportLocation",
        "ShareLocation",
    ),
    "clock": (
        "CreateAlarm",
        "EditAlarm",
        "EnableAlarm",
        "DeleteAlarm",
        "CreateTimer",
        "StartTimer",
        "StopwatchRunning",
        "PauseStopwatch",
        "StopwatchReset",
        "AddWorldClock",
    ),
}
PROVIDED_TABLE_APPS = {
    "sms": (
        ("Simple SMS Messenger",),
        ("Fossify Messages", "QUIK SMS", "Messages"),
    ),
    "files": (
        ("Material Files",),
        (
            "Amaze File Manager",
            "Fossify File Manager",
            "Total Commander",
            "X-plore File Manager",
        ),
    ),
    "contacts": (
        ("Google Contacts",),
        (
            "Fossify Contacts",
            "Connect You",
            "Simple Contacts Pro SE",
            "Right Contact",
        ),
    ),
    "maps": (
        ("OsmAnd~",),
        ("Organic Maps", "CoMaps"),
    ),
    "clock": (
        ("Google Clock",),
        ("Clock", "Simple Clock", "Clock You", "Chrono", "Fossify Clock"),
    ),
}
AW_TASK_ORIGINS = {
    "sms": {
        "SmsSend": "SimpleSmsSend",
        "SmsReply": "SimpleSmsReply",
        "SmsReplyMostRecent": "SimpleSmsReplyMostRecent",
        "SmsResend": "SimpleSmsResend",
        "SmsSendClipboard": "SimpleSmsSendClipboardContent",
        "SmsSendReceivedAddress": "SimpleSmsSendReceivedAddress",
    },
    "files": {
        "FilesDeleteFile": "FilesDeleteFile",
        "FilesMoveFile": "FilesMoveFile",
        "FilesSaveCopyOfFile": "SaveCopyOfReceiptTaskEval",
    },
    "contacts": {
        "ContactsAddContact": "ContactsAddContact",
        "ContactsNewContactDraft": "ContactsNewContactDraft",
    },
    "maps": {
        "MapsAddFavorite": "OsmAndFavorite",
        "MapsAddMarker": "OsmAndMarker",
        "MapsRecordTrack": "OsmAndTrack",
    },
    "clock": {
        "ClockCreateTimer": "ClockTimerEntry",
        "ClockStopwatchRunning": "ClockStopWatchRunning",
        "ClockPauseStopwatch": "ClockStopWatchPausedVerify",
    },
}
UNSCHEDULED_AW_TASKS = {
    "sms": (),
    "files": (),
    "contacts": (),
    "maps": (),
    "clock": (),
}
AW_ORIGINAL_APP_IDS = {
    "todo": {"todo_tasks_org"},
    "notes": {"notes_markor", "notes_joplin"},
    "finance": {"finance_pro_expense"},
    "music": {"music_retro_music"},
    "calendar": {"calendar_simple_calendar_pro"},
    "sms": {"sms_simple_sms_messenger"},
    "files": {"files_material_files"},
    "maps": {"maps_osmand"},
    "contacts": {"contacts_google_contacts"},
    "clock": {"clock_google_clock"},
}
CATEGORY_TASK_PREFIX = {
    "sms": "Sms",
    "files": "Files",
    "contacts": "Contacts",
    "maps": "Maps",
    "clock": "Clock",
}
CONTACTS_SQLITE_APP_IDS = {
    "contacts_fossify_contacts",
    "contacts_connect_you",
    "contacts_simple_contacts_pro_se",
}
CONTACTS_UI_FALLBACK_APP_IDS = {"contacts_right_contact"}
CONTACTS_UI_HEURISTIC_TASKS = {"ContactsNewContactDraft"}
SMS_PROVIDER_TASKS = {
    "SmsSend",
    "SmsReply",
    "SmsReplyMostRecent",
    "SmsResend",
    "SmsSendReceivedAddress",
    "SmsCreateDraftMessage",
    "SmsEditDraftMessage",
    "SmsSendToContact",
    "SmsDeleteConversation",
    "SmsForwardMessage",
}
SMS_PROVIDER_SEED_UI_TASKS: set[str] = set()
FILES_FILESYSTEM_TASKS = {
    "FilesCreateFolder",
    "FilesRenameFile",
    "FilesDeleteFile",
    "FilesMoveFile",
    "FilesSaveCopyOfFile",
    "FilesSearchFile",
    "FilesCompressFiles",
    "FilesExtractArchive",
}
FILES_FILESYSTEM_SEED_UI_TASKS = {"FilesViewFileInfo", "FilesShareFile"}
CONTACTS_PROVIDER_TASKS = {
    "ContactsAddContact",
    "ContactsEditContact",
    "ContactsAddFavoriteContact",
    "ContactsRemoveFavoriteContact",
    "ContactsDeleteContact",
}
CONTACTS_PROVIDER_SEED_UI_TASKS = {
    "ContactsSearchContact",
    "ContactsViewContactDetails",
    "ContactsCallContact",
    "ContactsMessageContact",
}
MAPS_STORAGE_TASKS = {
    "MapsAddFavorite",
    "MapsRemoveFavorite",
    "MapsAddMarker",
    "MapsDeleteMarker",
}
MAPS_TRACK_FILE_TASKS = {
    "MapsRecordTrack",
}
MAPS_EXPORT_FILE_TASKS = {
    "MapsExportLocation",
}
MAPS_STORAGE_APP_IDS = {
    "maps_osmand",
    "maps_organic_maps",
    "maps_comaps",
}
CLOCK_SQLITE_TASKS = {
    "ClockCreateAlarm",
    "ClockEditAlarm",
    "ClockEnableAlarm",
    "ClockDeleteAlarm",
    "ClockCreateTimer",
}
CLOCK_SQLITE_APP_IDS = {"clock_fossify_clock"}


@dataclasses.dataclass(frozen=True)
class EpisodeSummary:
  model: str
  family: str
  category: str
  app_id: str
  app_name: str
  task: str
  goal: str
  success: bool
  skipped: bool
  exception: str
  finish_time: str
  run_time: float | None
  episode_length: int | None
  output_path: Path
  checkpoint_path: Path
  trace_path: Path | None
  prompt_excerpt: str
  reasoning_excerpt: str
  output_excerpt: str


def _load_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as handle:
    return json.load(handle)


def _task_names(job: dict[str, Any]) -> list[str]:
  for arg in job.get("command", []):
    if isinstance(arg, str) and arg.startswith("--tasks="):
      return [name for name in arg.split("=", 1)[1].split(",") if name]
  return []


def _slug(value: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def _model_groups(config_path: Path) -> dict[str, str]:
  if not config_path.exists():
    return {}
  payload = _load_json(config_path)
  return {
      model["name"]: model.get("group", "")
      for model in payload.get("models", [])
      if isinstance(model.get("name"), str)
  }


def _model_order(config_path: Path, jobs: list[dict[str, Any]]) -> list[str]:
  names: list[str] = []
  if config_path.exists():
    payload = _load_json(config_path)
    for model in payload.get("models", []):
      name = model.get("name")
      if isinstance(name, str) and name and name not in names:
        names.append(name)
  for job in jobs:
    name = job.get("model_name")
    if isinstance(name, str) and name and name not in names:
      names.append(name)
  selected = {job.get("model_name") for job in jobs}
  return [name for name in names if name in selected]


def _fmt_rate(success: int, total: int) -> str:
  if total <= 0:
    return "--"
  return f"{100.0 * success / total:.1f}%"


def _fmt_progress(done: int, total: int) -> str:
  if total <= 0:
    return "--"
  return f"{100.0 * done / total:.1f}%"


def _fmt_cell(success: int, completed: int, scheduled: int) -> str:
  if scheduled <= 0:
    return "--"
  completed_rate = _fmt_rate(success, completed) if completed else "--"
  return f"{success}/{scheduled} ({completed}/{scheduled}, {completed_rate})"


def _short(value: Any, limit: int = 220) -> str:
  text = "" if value is None else str(value)
  text = re.sub(r"\s+", " ", text).strip()
  if len(text) <= limit:
    return text
  return text[: max(0, limit - 3)].rstrip() + "..."


def _md(value: Any, limit: int = 220) -> str:
  raw = "" if value is None else str(value)
  text = raw if re.fullmatch(r"\[[^\]]+\]\([^)]+\)", raw) else _short(raw, limit)
  text = text.replace("\\", "\\\\").replace("|", "\\|")
  text = text.replace("\n", "<br>")
  return text or "--"


def _link(path: Path | None, label: str) -> str:
  if path is None:
    return "--"
  return f"[{label}]({path})"


def _category_label(category: str) -> str:
  return CATEGORY_LABELS.get(category, category.title())


def _baseline_app_ids() -> dict[str, set[str]]:
  profiles = get_domain_profiles()
  out = {category: set(app_ids) for category, app_ids in AW_ORIGINAL_APP_IDS.items()}
  for category, profile in profiles.items():
    out.setdefault(category, set())
    for app in profile.apps:
      notes = app.notes.lower()
      if "androidworld-original" in notes or "original baseline" in notes:
        out[category].add(app.app_id)
    if not out[category] and profile.apps:
      out[category].add(profile.apps[0].app_id)
  return out


def _app_type(category: str, app_id: str, baseline_app_ids: dict[str, set[str]]) -> str:
  return APP_TYPE_ORIGINAL if app_id in baseline_app_ids.get(category, set()) else APP_TYPE_NEW


def _task_label(category: str, task_name: str) -> str:
  prefix = CATEGORY_TASK_PREFIX.get(category, "")
  return task_name.removeprefix(prefix) if prefix else task_name


def _task_origin_split(category: str, task_names: Iterable[str]) -> tuple[list[str], list[str]]:
  aw: list[str] = []
  new: list[str] = []
  origin_map = AW_TASK_ORIGINS.get(category, {})
  for task_name in task_names:
    label = _task_label(category, task_name)
    upstream = origin_map.get(task_name)
    if upstream:
      aw.append(f"{label} ({upstream})")
    else:
      new.append(label)
  return aw, new


def _provided_task_diff(category: str, task_names: Iterable[str]) -> str:
  provided = PROVIDED_TABLE_TASKS.get(category, ())
  if not provided:
    return "no provided-table baseline"
  repo = tuple(_task_label(category, task_name) for task_name in task_names)
  table_only = [name for name in provided if name not in repo]
  repo_only = [name for name in repo if name not in provided]
  if not table_only and not repo_only:
    return "match"
  parts = []
  if table_only:
    parts.append("table-only: " + ", ".join(table_only))
  if repo_only:
    parts.append("repo-only: " + ", ".join(repo_only))
  return "; ".join(parts)


def _provided_app_diff(
    category: str,
    original_apps: Iterable[str],
    new_apps: Iterable[str],
) -> str:
  provided = PROVIDED_TABLE_APPS.get(category)
  if not provided:
    return "no provided-table baseline"
  expected_original, expected_new = provided
  actual_original = tuple(original_apps)
  actual_new = tuple(new_apps)
  problems = []
  for label, expected, actual in (
      ("AW apps", expected_original, actual_original),
      ("new apps", expected_new, actual_new),
  ):
    expected_set = set(expected)
    actual_set = set(actual)
    missing = [name for name in expected if name not in actual_set]
    extra = [name for name in actual if name not in expected_set]
    if missing or extra:
      details = []
      if missing:
        details.append("missing " + ", ".join(missing))
      if extra:
        details.append("extra " + ", ".join(extra))
      problems.append(f"{label}: " + "; ".join(details))
  return "match" if not problems else " | ".join(problems)


def _canonical_task_name(category: str, task_name: str) -> str:
  profiles = get_domain_profiles()
  profile = profiles.get(category)
  if not profile:
    return task_name.split("For", 1)[0]
  for canonical in sorted(profile.canonical_tasks, key=len, reverse=True):
    if task_name.startswith(canonical):
      return canonical
  return task_name.split("For", 1)[0]


def _validation_mode(category: str, task_name: str, app_id: str = "") -> str:
  canonical = _canonical_task_name(category, task_name)
  if category == "sms":
    if canonical in SMS_PROVIDER_TASKS:
      return "SmsProvider"
    if canonical in SMS_PROVIDER_SEED_UI_TASKS:
      return "SmsProvider seed + UI"
  if category == "files":
    if canonical in FILES_FILESYSTEM_TASKS:
      return "Filesystem"
    if canonical in FILES_FILESYSTEM_SEED_UI_TASKS:
      return "Filesystem seed + UI"
  if category == "contacts":
    if canonical in CONTACTS_UI_HEURISTIC_TASKS:
      return "UI heuristic"
    if app_id in CONTACTS_SQLITE_APP_IDS:
      if canonical in CONTACTS_PROVIDER_TASKS:
        return "SQLite"
      if canonical in CONTACTS_PROVIDER_SEED_UI_TASKS:
        return "SQLite + UI"
    if app_id in CONTACTS_UI_FALLBACK_APP_IDS:
      return "UI fallback"
    if canonical in CONTACTS_PROVIDER_TASKS:
      return "ContactsProvider"
    if canonical in CONTACTS_PROVIDER_SEED_UI_TASKS:
      return "ContactsProvider seed + UI"
  if category == "maps":
    if canonical in MAPS_TRACK_FILE_TASKS:
      return "Filesystem GPX/KML"
    if canonical in MAPS_EXPORT_FILE_TASKS:
      return "Filesystem GPX/KML/link"
    if canonical in MAPS_STORAGE_TASKS:
      if app_id in MAPS_STORAGE_APP_IDS:
        return "Filesystem/SQLite"
      if app_id == "maps_google_maps":
        return "UI heuristic (opaque Google Maps storage)"
    return "UI heuristic"
  if category == "clock":
    if canonical in CLOCK_SQLITE_TASKS and app_id in CLOCK_SQLITE_APP_IDS:
      return "SQLite"
    return "UI heuristic"
  return "Mixed / unknown"


def _validation_mix(category: str, app_id: str, task_names: list[str]) -> str:
  modes = sorted({_validation_mode(category, task, app_id) for task in task_names})
  return ", ".join(modes) if modes else "--"


def _rate_value(success: int, completed: int) -> float | None:
  if completed <= 0:
    return None
  return 100.0 * success / completed


def _fmt_delta(new_success: int, new_completed: int, orig_success: int, orig_completed: int) -> str:
  new_rate = _rate_value(new_success, new_completed)
  orig_rate = _rate_value(orig_success, orig_completed)
  if new_rate is None or orig_rate is None:
    return "--"
  return f"{new_rate - orig_rate:+.1f} pp"


def _read_pkl_gz(path: Path) -> Any:
  with path.open("rb") as handle:
    raw = handle.read()
  with gzip.open(io.BytesIO(raw), "rb") as gz:
    return pickle.load(gz)


def _is_skipped(ep: dict[str, Any]) -> bool:
  info = str(ep.get("exception_info") or "")
  return any(marker in info for marker in SKIP_EXCEPTION_MARKERS)


def _coerce_float(value: Any) -> float:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return 0.0
  if math.isnan(out) or math.isinf(out):
    return 0.0
  return out


def _coerce_int_or_none(value: Any) -> int | None:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return None
  if math.isnan(out) or math.isinf(out):
    return None
  return int(out)


def _last_list_value(step_data: dict[str, Any], *fields: str) -> Any:
  for field in fields:
    value = step_data.get(field)
    if isinstance(value, list) and value:
      return value[-1]
    if value:
      return value
  return ""


def _trace_path(output_path: Path) -> Path | None:
  traces = output_path / "traces" / "trace.jsonl"
  return traces if traces.exists() else None


def _last_trace_record(output_path: Path) -> dict[str, Any]:
  path = _trace_path(output_path)
  if path is None:
    return {}
  try:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
  except OSError:
    return {}
  for line in reversed(lines):
    try:
      payload = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(payload, dict):
      return payload
  return {}


def _summarize_episode(
    ep: dict[str, Any],
    checkpoint_path: Path,
    job: dict[str, Any],
    family: str,
) -> EpisodeSummary:
  step_data = ep.get("episode_data") or {}
  if not isinstance(step_data, dict):
    step_data = {}
  output_path = Path(job["output_path"])
  trace_record = _last_trace_record(output_path)
  exception = str(ep.get("exception_info") or "")
  prompt_excerpt = _last_list_value(
      step_data, "action_prompt", "prompt_user", "summary_prompt"
  )
  reasoning_excerpt = _last_list_value(
      step_data, "action_reason", "thought", "thinking", "summary"
  )
  output_excerpt = _last_list_value(
      step_data, "action_output", "response", "action_raw_response",
      "summary_raw_response"
  )
  if not prompt_excerpt:
    prompt_excerpt = trace_record.get("prompt_user") or trace_record.get("goal")
  if not reasoning_excerpt:
    reasoning_excerpt = (
        trace_record.get("thought")
        or trace_record.get("reasoning")
        or trace_record.get("summary")
    )
  if not output_excerpt:
    output_excerpt = (
        trace_record.get("response")
        or trace_record.get("action_raw")
        or trace_record.get("action_scaled")
    )
  finish = ep.get("finish_dtime") or ""
  return EpisodeSummary(
      model=str(job.get("model_name", "")),
      family=family,
      category=str(job.get("category", "")),
      app_id=str(job.get("app_id", "")),
      app_name=str(job.get("app_name", job.get("app_id", ""))),
      task=str(ep.get("task_template") or checkpoint_path.name.removesuffix(".pkl.gz")),
      goal=str(ep.get("goal") or trace_record.get("goal") or ""),
      success=_coerce_float(ep.get("is_successful")) >= 0.5,
      skipped=_is_skipped(ep),
      exception=exception,
      finish_time=str(finish),
      run_time=_coerce_float(ep.get("run_time")) if ep.get("run_time") else None,
      episode_length=_coerce_int_or_none(ep.get("episode_length")),
      output_path=output_path,
      checkpoint_path=checkpoint_path,
      trace_path=_trace_path(output_path),
      prompt_excerpt=_short(prompt_excerpt, 500),
      reasoning_excerpt=_short(reasoning_excerpt, 500),
      output_excerpt=_short(output_excerpt, 500),
  )


def _load_summaries_for_job(
    job: dict[str, Any],
    family: str,
    cache: dict[Path, tuple[tuple[int, int], list[EpisodeSummary]]],
) -> list[EpisodeSummary]:
  output_path = Path(job["output_path"])
  summaries: list[EpisodeSummary] = []
  if not output_path.exists():
    return summaries
  for checkpoint_path in _checkpoint_paths_for_output(output_path):
    try:
      stat = checkpoint_path.stat()
    except OSError:
      continue
    fingerprint = (stat.st_mtime_ns, stat.st_size)
    cached = cache.get(checkpoint_path)
    if cached and cached[0] == fingerprint:
      summaries.extend(cached[1])
      continue
    try:
      payload = _read_pkl_gz(checkpoint_path)
    except Exception:  # pylint: disable=broad-exception-caught
      cache[checkpoint_path] = (fingerprint, [])
      continue
    episodes = payload if isinstance(payload, list) else [payload]
    file_summaries = [
        _summarize_episode(ep, checkpoint_path, job, family)
        for ep in episodes
        if isinstance(ep, dict)
    ]
    cache[checkpoint_path] = (fingerprint, file_summaries)
    summaries.extend(file_summaries)
  return summaries


def _checkpoint_paths_for_output(output_path: Path) -> list[Path]:
  run_dirs = [
      path for path in output_path.iterdir()
      if path.is_dir() and path.name.startswith("run_")
  ]
  if not run_dirs:
    return sorted(output_path.rglob("*.pkl.gz"))

  def run_dir_key(path: Path) -> tuple[int, int, str]:
    try:
      checkpoint_count = sum(1 for _ in path.rglob("*.pkl.gz"))
      mtime = path.stat().st_mtime_ns
    except OSError:
      return (0, 0, path.name)
    # Prefer a complete retry over a stale skipped/failed run; break ties by time.
    return (checkpoint_count, mtime, path.name)

  best_run_dir = max(run_dirs, key=run_dir_key)
  return sorted(best_run_dir.rglob("*.pkl.gz"))


def _job_key(model: str, category: str, app_name: str) -> tuple[str, str, str]:
  return (model, category, app_name)


def _parse_matrix_log(path: Path) -> tuple[list[tuple[str, str, str, str]], set[tuple[str, str, str]]]:
  if not path.exists():
    return [], set()
  pattern = re.compile(
      r"^\[(RUN|OK|ERR)\] model=(.*?) category=(.*?) app=(.*?) emulator=(.*?)(?: exit=.*)?$"
  )
  started: list[tuple[str, str, str, str]] = []
  finished: set[tuple[str, str, str]] = set()
  for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    match = pattern.match(line)
    if not match:
      continue
    status, model, category, app_name, emulator = match.groups()
    key = _job_key(model, category, app_name)
    if status == "RUN":
      started.append((model, category, app_name, emulator))
    else:
      finished.add(key)
  running = [
      item for item in started
      if _job_key(item[0], item[1], item[2]) not in finished
  ]
  return running, finished


def _matrix_logs_for_manifest(
    manifest: Path, manifest_payload: dict[str, Any]
) -> list[Path]:
  run_root = manifest.parent.parent.parent
  logs_dir = run_root / "logs"
  jobs = manifest_payload.get("jobs", [])
  model_names = sorted({
      str(job.get("model_name"))
      for job in jobs
      if isinstance(job, dict) and job.get("model_name")
  })
  candidates = [
      logs_dir / f"matrix_{_slug(model_name)}.log"
      for model_name in model_names
  ]
  existing = [path for path in candidates if path.exists()]
  aggregate = logs_dir / "catbench_matrix.log"
  if aggregate.exists() or not existing:
    existing.append(aggregate)

  deduped: list[Path] = []
  seen: set[Path] = set()
  for path in existing:
    if path not in seen:
      seen.add(path)
      deduped.append(path)
  return deduped


def _job_lookup(jobs: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
  return {
      _job_key(str(job["model_name"]), str(job["category"]), str(job["app_name"])): job
      for job in jobs
  }


def _latest_trace_excerpt(job: dict[str, Any]) -> tuple[str, str, Path | None]:
  output_path = Path(job["output_path"])
  record = _last_trace_record(output_path)
  if not record:
    return "", "", _trace_path(output_path)
  prompt = record.get("prompt_user") or record.get("goal") or ""
  output = record.get("response") or record.get("action_raw") or record.get("action_scaled") or ""
  return _short(prompt, 260), _short(output, 260), _trace_path(output_path)


def _save_image(value: Any, dest: Path) -> bool:
  if value is None:
    return False
  try:
    import numpy as np  # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel
  except ImportError:
    return False
  try:
    if isinstance(value, np.ndarray):
      array = value if value.dtype == np.uint8 else value.astype(np.uint8)
      dest.parent.mkdir(parents=True, exist_ok=True)
      Image.fromarray(array).save(dest)
      return True
    if hasattr(value, "save"):
      dest.parent.mkdir(parents=True, exist_ok=True)
      value.save(dest)
      return True
  except Exception:  # pylint: disable=broad-exception-caught
    return False
  return False


def _dump_recent_screenshots(
    recent: list[EpisodeSummary],
    artifact_dir: Path,
) -> dict[Path, Path]:
  screenshot_root = artifact_dir / "recent_screenshots"
  if screenshot_root.exists():
    shutil.rmtree(screenshot_root, ignore_errors=True)
  screenshot_root.mkdir(parents=True, exist_ok=True)
  out: dict[Path, Path] = {}
  for idx, summary in enumerate(recent):
    try:
      payload = _read_pkl_gz(summary.checkpoint_path)
    except Exception:  # pylint: disable=broad-exception-caught
      continue
    episodes = payload if isinstance(payload, list) else [payload]
    ep = episodes[0] if episodes and isinstance(episodes[0], dict) else {}
    step_data = ep.get("episode_data") if isinstance(ep, dict) else {}
    if not isinstance(step_data, dict):
      continue
    task_root = screenshot_root / f"{idx:02d}_{_slug(summary.model)}_{_slug(summary.task)}"
    wrote = False
    for field in SCREENSHOT_FIELDS:
      seq = step_data.get(field)
      if not isinstance(seq, list) or not seq:
        continue
      if _save_image(seq[-1], task_root / f"last_{field}.png"):
        wrote = True
    if wrote:
      out[summary.checkpoint_path] = task_root
  return out


def _manifest_stats(
    jobs: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str, str], int], dict[str, int]]:
  scheduled_by_model_category: dict[tuple[str, str], int] = collections.Counter()
  scheduled_by_job: dict[tuple[str, str, str], int] = collections.Counter()
  scheduled_by_model: dict[str, int] = collections.Counter()
  for job in jobs:
    model = str(job["model_name"])
    category = str(job["category"])
    app_name = str(job["app_name"])
    count = len(_task_names(job))
    scheduled_by_model_category[(model, category)] += count
    scheduled_by_job[_job_key(model, category, app_name)] += count
    scheduled_by_model[model] += count
  return scheduled_by_model_category, scheduled_by_job, scheduled_by_model


def _app_type_stats(
    jobs: list[dict[str, Any]],
    completed: list[EpisodeSummary],
    baseline_app_ids: dict[str, set[str]],
) -> tuple[
    dict[tuple[str, str], int],
    dict[tuple[str, str, str], int],
    dict[tuple[str, str], int],
    dict[tuple[str, str, str], int],
    dict[tuple[str, str], int],
    dict[tuple[str, str, str], int],
    dict[tuple[str, str], list[str]],
]:
  scheduled_by_model_type: dict[tuple[str, str], int] = collections.Counter()
  scheduled_by_model_category_type: dict[tuple[str, str, str], int] = collections.Counter()
  completed_by_model_type: dict[tuple[str, str], int] = collections.Counter()
  completed_by_model_category_type: dict[tuple[str, str, str], int] = collections.Counter()
  success_by_model_type: dict[tuple[str, str], int] = collections.Counter()
  success_by_model_category_type: dict[tuple[str, str, str], int] = collections.Counter()
  app_names_by_category_type: dict[tuple[str, str], list[str]] = collections.defaultdict(list)

  for job in jobs:
    model = str(job["model_name"])
    category = str(job["category"])
    app_id = str(job["app_id"])
    app_name = str(job["app_name"])
    app_type = _app_type(category, app_id, baseline_app_ids)
    count = len(_task_names(job))
    scheduled_by_model_type[(model, app_type)] += count
    scheduled_by_model_category_type[(model, category, app_type)] += count
    app_key = (category, app_type)
    if app_name not in app_names_by_category_type[app_key]:
      app_names_by_category_type[app_key].append(app_name)

  for summary in completed:
    app_type = _app_type(summary.category, summary.app_id, baseline_app_ids)
    completed_by_model_type[(summary.model, app_type)] += 1
    completed_by_model_category_type[(summary.model, summary.category, app_type)] += 1
    if summary.success:
      success_by_model_type[(summary.model, app_type)] += 1
      success_by_model_category_type[(summary.model, summary.category, app_type)] += 1

  return (
      scheduled_by_model_type,
      scheduled_by_model_category_type,
      completed_by_model_type,
      completed_by_model_category_type,
      success_by_model_type,
      success_by_model_category_type,
      app_names_by_category_type,
  )


def _write_markdown(
    manifest: Path,
    model_config: Path,
    out_path: Path,
    artifact_dir: Path,
    recent_limit: int,
    cache: dict[Path, tuple[tuple[int, int], list[EpisodeSummary]]],
) -> None:
  manifest_payload = _load_json(manifest)
  jobs = manifest_payload.get("jobs", [])
  model_groups = _model_groups(model_config)
  model_order = _model_order(model_config, jobs)
  job_by_key = _job_lookup(jobs)
  baseline_app_ids = _baseline_app_ids()
  scheduled_by_model_category, scheduled_by_job, scheduled_by_model = _manifest_stats(jobs)

  all_summaries: list[EpisodeSummary] = []
  for job in jobs:
    family = model_groups.get(str(job.get("model_name", "")), "")
    all_summaries.extend(_load_summaries_for_job(job, family, cache))

  completed = [s for s in all_summaries if not s.skipped]
  successful = [s for s in completed if s.success]
  skipped = [s for s in all_summaries if s.skipped]
  failed = [s for s in completed if not s.success]
  total_scheduled = sum(scheduled_by_model.values())
  (
      scheduled_by_model_type,
      scheduled_by_model_category_type,
      completed_by_model_type,
      completed_by_model_category_type,
      success_by_model_type,
      success_by_model_category_type,
      app_names_by_category_type,
  ) = _app_type_stats(jobs, completed, baseline_app_ids)
  orig_scheduled = sum(
      count for (model, app_type), count in scheduled_by_model_type.items()
      if app_type == APP_TYPE_ORIGINAL
  )
  new_scheduled = sum(
      count for (model, app_type), count in scheduled_by_model_type.items()
      if app_type == APP_TYPE_NEW
  )
  orig_completed = sum(
      count for (model, app_type), count in completed_by_model_type.items()
      if app_type == APP_TYPE_ORIGINAL
  )
  new_completed = sum(
      count for (model, app_type), count in completed_by_model_type.items()
      if app_type == APP_TYPE_NEW
  )
  orig_success = sum(
      count for (model, app_type), count in success_by_model_type.items()
      if app_type == APP_TYPE_ORIGINAL
  )
  new_success = sum(
      count for (model, app_type), count in success_by_model_type.items()
      if app_type == APP_TYPE_NEW
  )
  now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  matrix_logs = _matrix_logs_for_manifest(manifest, manifest_payload)
  running: list[tuple[str, str, str, str]] = []
  finished_jobs: set[tuple[str, str, str]] = set()
  for matrix_log in matrix_logs:
    log_running, log_finished = _parse_matrix_log(matrix_log)
    running.extend(log_running)
    finished_jobs.update(log_finished)

  completed_by_model_category: dict[tuple[str, str], int] = collections.Counter()
  success_by_model_category: dict[tuple[str, str], int] = collections.Counter()
  completed_by_model: dict[str, int] = collections.Counter()
  success_by_model: dict[str, int] = collections.Counter()
  for summary in completed:
    key = (summary.model, summary.category)
    completed_by_model_category[key] += 1
    completed_by_model[summary.model] += 1
    if summary.success:
      success_by_model_category[key] += 1
      success_by_model[summary.model] += 1

  recent = sorted(
      completed,
      key=lambda s: (
          s.checkpoint_path.stat().st_mtime if s.checkpoint_path.exists() else 0.0
      ),
      reverse=True,
  )[:recent_limit]
  screenshot_dirs = _dump_recent_screenshots(recent, artifact_dir)

  lines: list[str] = []
  lines.append("# CATBench Live Run")
  lines.append("")
  lines.append(f"Updated: `{now}`")
  lines.append("")
  lines.append(f"Run root: `{manifest.parent.parent.parent}`")
  lines.append(f"Manifest: `{manifest}`")
  if len(matrix_logs) == 1:
    lines.append(f"Matrix log: `{matrix_logs[0]}`")
  else:
    lines.append(
        "Matrix logs: "
        + ", ".join(f"`{matrix_log}`" for matrix_log in matrix_logs)
    )
  lines.append("")
  lines.append("## Overall")
  lines.append("")
  lines.append("| Metric | Value |")
  lines.append("|---|---:|")
  lines.append(f"| Scheduled tasks | {total_scheduled} |")
  lines.append(f"| Completed tasks | {len(completed)} |")
  lines.append(f"| Successful tasks | {len(successful)} |")
  lines.append(f"| Failed tasks | {len(failed)} |")
  lines.append(f"| Skipped tasks | {len(skipped)} |")
  lines.append(f"| Success / completed | {len(successful)}/{len(completed)} ({_fmt_rate(len(successful), len(completed))}) |")
  lines.append(f"| Success / scheduled | {len(successful)}/{total_scheduled} ({_fmt_rate(len(successful), total_scheduled)}) |")
  lines.append(f"| Progress | {len(completed)}/{total_scheduled} ({_fmt_progress(len(completed), total_scheduled)}) |")
  lines.append(f"| AW original apps | {orig_success}/{orig_scheduled} ({orig_completed}/{orig_scheduled}, {_fmt_rate(orig_success, orig_completed)}) |")
  lines.append(f"| Newly installed apps | {new_success}/{new_scheduled} ({new_completed}/{new_scheduled}, {_fmt_rate(new_success, new_completed)}) |")
  lines.append(f"| App-runs finished | {len(finished_jobs)}/{len(jobs)} |")
  lines.append("")
  lines.append("## Benchmark App Split")
  lines.append("")
  lines.append("AW Orig. corresponds to the AndroidWorld-original baseline app(s); New Apps are the newly installed app variants for cross-app generalization.")
  lines.append("")
  lines.append("| Category | Task Templates | # Tmpl. | AW Orig. App(s) | Newly Installed Apps | # New | # Apps |")
  lines.append("|---|---|---:|---|---|---:|---:|")
  profiles = get_domain_profiles()
  for category in CATEGORY_ORDER:
    profile = profiles[category]
    runnable_apps = [app for app in profile.apps if app.implemented_tasks]
    original_apps = [
        app.display_name for app in runnable_apps
        if app.app_id in baseline_app_ids.get(category, set())
    ]
    new_apps = [
        app.display_name for app in runnable_apps
        if app.app_id not in baseline_app_ids.get(category, set())
    ]
    task_templates = ", ".join(
        _task_label(category, task_name) for task_name in profile.canonical_tasks
    )
    row = [
        _category_label(category),
        task_templates,
        str(len(profile.canonical_tasks)),
        ", ".join(original_apps),
        ", ".join(new_apps),
        str(len(new_apps)),
        str(len(runnable_apps)),
    ]
    lines.append("| " + " | ".join(_md(cell, 420) for cell in row) + " |")
  total_templates = sum(len(profiles[category].canonical_tasks) for category in CATEGORY_ORDER)
  total_apps = 0
  total_new_apps = 0
  total_original_apps = 0
  for category in CATEGORY_ORDER:
    runnable_apps = [app for app in profiles[category].apps if app.implemented_tasks]
    total_apps += len(runnable_apps)
    total_original_apps += sum(
        1 for app in runnable_apps if app.app_id in baseline_app_ids.get(category, set())
    )
    total_new_apps += sum(
        1 for app in runnable_apps if app.app_id not in baseline_app_ids.get(category, set())
    )
  lines.append(
      "| Total | -- | "
      f"{total_templates} | {total_original_apps} AW app entries | "
      f"{total_new_apps} newly installed apps | {total_new_apps} | {total_apps} |"
  )
  lines.append("")
  lines.append("## Task Origin And Table Check")
  lines.append("")
  lines.append(
      "Source of truth here is the current runnable CATBench profile and generated"
      " task class names. AW-adapted means the scheduled template preserves an"
      " AndroidWorld task intent; CATBench-new means the template was added for"
      " the cross-app benchmark. The table below verifies the runnable"
      " five-category benchmark against the provided LaTeX table."
  )
  lines.append("")
  lines.append(
      "| Category | App Names vs Provided Table | Task Names vs Provided Table | "
      "AW-Adapted Scheduled Tasks | CATBench-New Scheduled Tasks | "
      "AW Originals Not Scheduled |"
  )
  lines.append("|---|---|---|---|---|---|")
  for category in CATEGORY_ORDER:
    profile = profiles[category]
    runnable_apps = [app for app in profile.apps if app.implemented_tasks]
    original_apps = [
        app.display_name for app in runnable_apps
        if app.app_id in baseline_app_ids.get(category, set())
    ]
    new_apps = [
        app.display_name for app in runnable_apps
        if app.app_id not in baseline_app_ids.get(category, set())
    ]
    aw_tasks, new_tasks = _task_origin_split(category, profile.canonical_tasks)
    row = [
        _category_label(category),
        _provided_app_diff(category, original_apps, new_apps),
        _provided_task_diff(category, profile.canonical_tasks),
        ", ".join(aw_tasks) or "--",
        ", ".join(new_tasks) or "--",
        ", ".join(UNSCHEDULED_AW_TASKS.get(category, ())) or "--",
    ]
    lines.append("| " + " | ".join(_md(cell, 760) for cell in row) + " |")
  lines.append("")
  validation_counts: dict[str, int] = collections.Counter()
  for job in jobs:
    for task_name in _task_names(job):
      validation_counts[
          _validation_mode(
              str(job["category"]), task_name, str(job["app_id"])
          )
      ] += 1
  lines.append("## Validation Modes")
  lines.append("")
  lines.append(
      "This separates strict durable-state checks from UI heuristics. A"
      " provider/filesystem score means the evaluator checks Android storage"
      " after the agent acts; a seed + UI score means a real seeded object must"
      " still exist, but the final state is only visible in app UI."
  )
  lines.append("")
  lines.append("| Mode | Scheduled Tasks | Meaning |")
  lines.append("|---|---:|---|")
  validation_meanings = {
      "SmsProvider": "System Telephony/SmsProvider rows are checked.",
      "SmsProvider seed + UI": "A real SMS row is seeded/verified, then UI evidence is required.",
      "Filesystem": "Files/folders under `/sdcard/CATBench` are checked by adb.",
      "Filesystem seed + UI": "A real file is seeded/verified, then UI evidence is required.",
      "ContactsProvider": "System ContactsProvider rows/starred state are checked.",
      "ContactsProvider seed + UI": "ContactsProvider is seeded/verified, then UI evidence is required.",
      "Filesystem/SQLite": "Map favorites/markers are checked through app GPX/KML files or SQLite tables.",
      "Filesystem GPX/KML": "Saved map tracks/routes are checked through GPX/KML files and waypoint coordinates.",
      "Filesystem GPX/KML/link": "Exported map locations are checked through GPX/KML files or saved map-link text.",
      "SQLite": "App-private SQLite rows are checked directly after the agent acts.",
      "SQLite + UI": "App-private SQLite state is seeded/verified, then UI evidence is required.",
      "UI fallback": "Target app keeps state app-private in this build; only UI evidence is checked.",
      "UI heuristic": "No shared durable validator is currently available; only UI evidence is checked.",
      "UI heuristic (opaque Google Maps storage)": "Google Maps storage is opaque in this image; the evaluator uses visible UI evidence.",
      "Mixed / unknown": "No validation mode metadata is available.",
  }
  for mode, count in sorted(validation_counts.items()):
    lines.append(
        "| "
        + " | ".join(
            _md(cell, 320)
            for cell in (mode, str(count), validation_meanings.get(mode, ""))
        )
        + " |"
    )
  lines.append("")
  lines.append("## Original vs Newly Installed Apps")
  lines.append("")
  lines.append("Cell format: `success/scheduled (completed/scheduled, success rate over completed)`. Delta is new-app success rate minus AW-original success rate, using completed tasks only.")
  lines.append("")
  lines.append("| Family | Model | AW Orig. | Newly Installed | Delta | Progress |")
  lines.append("|---|---|---:|---:|---:|---:|")
  for model in model_order:
    orig_cell = _fmt_cell(
        success_by_model_type[(model, APP_TYPE_ORIGINAL)],
        completed_by_model_type[(model, APP_TYPE_ORIGINAL)],
        scheduled_by_model_type[(model, APP_TYPE_ORIGINAL)],
    )
    new_cell = _fmt_cell(
        success_by_model_type[(model, APP_TYPE_NEW)],
        completed_by_model_type[(model, APP_TYPE_NEW)],
        scheduled_by_model_type[(model, APP_TYPE_NEW)],
    )
    model_completed = completed_by_model_type[(model, APP_TYPE_ORIGINAL)] + completed_by_model_type[(model, APP_TYPE_NEW)]
    model_scheduled = scheduled_by_model_type[(model, APP_TYPE_ORIGINAL)] + scheduled_by_model_type[(model, APP_TYPE_NEW)]
    row = [
        model_groups.get(model, ""),
        model,
        orig_cell,
        new_cell,
        _fmt_delta(
            success_by_model_type[(model, APP_TYPE_NEW)],
            completed_by_model_type[(model, APP_TYPE_NEW)],
            success_by_model_type[(model, APP_TYPE_ORIGINAL)],
            completed_by_model_type[(model, APP_TYPE_ORIGINAL)],
        ),
        f"{model_completed}/{model_scheduled} ({_fmt_progress(model_completed, model_scheduled)})",
    ]
    lines.append("| " + " | ".join(_md(cell, 180) for cell in row) + " |")
  lines.append("")
  lines.append("## Category Original vs New")
  lines.append("")
  lines.append("| Family | Model | Category | AW Orig. App(s) | AW Orig. | Newly Installed Apps | New Apps | Delta |")
  lines.append("|---|---|---|---|---:|---|---:|---:|")
  for model in model_order:
    for category in CATEGORY_ORDER:
      orig_apps = ", ".join(app_names_by_category_type.get((category, APP_TYPE_ORIGINAL), []))
      new_apps = ", ".join(app_names_by_category_type.get((category, APP_TYPE_NEW), []))
      row = [
          model_groups.get(model, ""),
          model,
          _category_label(category),
          orig_apps,
          _fmt_cell(
              success_by_model_category_type[(model, category, APP_TYPE_ORIGINAL)],
              completed_by_model_category_type[(model, category, APP_TYPE_ORIGINAL)],
              scheduled_by_model_category_type[(model, category, APP_TYPE_ORIGINAL)],
          ),
          new_apps,
          _fmt_cell(
              success_by_model_category_type[(model, category, APP_TYPE_NEW)],
              completed_by_model_category_type[(model, category, APP_TYPE_NEW)],
              scheduled_by_model_category_type[(model, category, APP_TYPE_NEW)],
          ),
          _fmt_delta(
              success_by_model_category_type[(model, category, APP_TYPE_NEW)],
              completed_by_model_category_type[(model, category, APP_TYPE_NEW)],
              success_by_model_category_type[(model, category, APP_TYPE_ORIGINAL)],
              completed_by_model_category_type[(model, category, APP_TYPE_ORIGINAL)],
          ),
      ]
      lines.append("| " + " | ".join(_md(cell, 260) for cell in row) + " |")
  lines.append("")
  lines.append("## Model Summary")
  lines.append("")
  lines.append("Cell format: `success/scheduled (completed/scheduled, success rate over completed)`.")
  lines.append("")
  header = ["Family", "Model", *[_category_label(c) for c in CATEGORY_ORDER], "Overall", "Progress"]
  lines.append("| " + " | ".join(header) + " |")
  lines.append("|" + "|".join(["---"] * len(header)) + "|")
  for model in model_order:
    row = [model_groups.get(model, ""), model]
    for category in CATEGORY_ORDER:
      scheduled = scheduled_by_model_category[(model, category)]
      done = completed_by_model_category[(model, category)]
      succ = success_by_model_category[(model, category)]
      row.append(_fmt_cell(succ, done, scheduled))
    row.append(
        _fmt_cell(
            success_by_model[model],
            completed_by_model[model],
            scheduled_by_model[model],
        )
    )
    row.append(
        f"{completed_by_model[model]}/{scheduled_by_model[model]} "
        f"({_fmt_progress(completed_by_model[model], scheduled_by_model[model])})"
    )
    lines.append("| " + " | ".join(_md(cell, 120) for cell in row) + " |")
  lines.append("")
  lines.append("## Running App-Runs")
  lines.append("")
  lines.append("| Model | Category | App Type | App | Emulator | Task Progress | Last Prompt | Last Output | Output Folder | Trace |")
  lines.append("|---|---|---|---|---|---:|---|---|---|---|")
  if not running:
    lines.append("| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
  for model, category, app_name, emulator in running:
    job = job_by_key.get(_job_key(model, category, app_name))
    scheduled = scheduled_by_job.get(_job_key(model, category, app_name), 0)
    if job:
      done = sum(
          1 for s in completed
          if s.model == model and s.category == category and s.app_name == app_name
      )
      prompt, output, trace = _latest_trace_excerpt(job)
      output_path = Path(job["output_path"])
    else:
      done = 0
      prompt, output, trace = "", "", None
      output_path = None
    row = [
        model,
        _category_label(category),
        _app_type(category, str(job.get("app_id", "")), baseline_app_ids) if job else "--",
        app_name,
        emulator,
        f"{done}/{scheduled}",
        prompt,
        output,
        _link(output_path, "folder") if output_path else "--",
        _link(trace, "trace") if trace else "--",
    ]
    lines.append("| " + " | ".join(_md(cell, 260) for cell in row) + " |")
  lines.append("")
  lines.append("## Recent Completed Tasks")
  lines.append("")
  lines.append("| Finished | Model | Category | App Type | App | Task | Validation | Result | Goal / Input Task | Reasoning | Output | Checkpoint | Trace | Screenshots |")
  lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
  if not recent:
    lines.append("| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
  for summary in recent:
    result = "PASS" if summary.success else "FAIL"
    screenshots = screenshot_dirs.get(summary.checkpoint_path)
    row = [
        summary.finish_time or "--",
        summary.model,
        _category_label(summary.category),
        _app_type(summary.category, summary.app_id, baseline_app_ids),
        summary.app_name,
        summary.task,
        _validation_mode(summary.category, summary.task, summary.app_id),
        result,
        summary.goal,
        summary.reasoning_excerpt or summary.prompt_excerpt,
        summary.output_excerpt or summary.exception,
        _link(summary.checkpoint_path, "pkl"),
        _link(summary.trace_path, "trace") if summary.trace_path else _link(summary.output_path, "folder"),
        _link(screenshots, "screens") if screenshots else "--",
    ]
    lines.append("| " + " | ".join(_md(cell, 300) for cell in row) + " |")
  lines.append("")
  lines.append("## App-Level Progress")
  lines.append("")
  lines.append("| Family | Model | Category | App Type | App | Validation Mix | Success / Scheduled | Completed / Scheduled | Success Rate Completed | Output Folder |")
  lines.append("|---|---|---|---|---|---|---:|---:|---:|---|")
  for job in jobs:
    model = str(job["model_name"])
    category = str(job["category"])
    app_name = str(job["app_name"])
    key = _job_key(model, category, app_name)
    scheduled = scheduled_by_job[key]
    app_completed = [
        s for s in completed
        if s.model == model and s.category == category and s.app_name == app_name
    ]
    app_success = [s for s in app_completed if s.success]
    row = [
        model_groups.get(model, ""),
        model,
        _category_label(category),
        _app_type(category, str(job["app_id"]), baseline_app_ids),
        app_name,
        _validation_mix(category, str(job["app_id"]), _task_names(job)),
        f"{len(app_success)}/{scheduled}",
        f"{len(app_completed)}/{scheduled}",
        _fmt_rate(len(app_success), len(app_completed)),
        _link(Path(job["output_path"]), "folder"),
    ]
    lines.append("| " + " | ".join(_md(cell, 180) for cell in row) + " |")
  lines.append("")
  lines.append("## Notes")
  lines.append("")
  lines.append("- The report is generated from checkpoint `*.pkl.gz` files plus live trace JSONL files.")
  lines.append("- `Success / scheduled` is the strict live score over all planned tasks; `Success / completed` is the score on tasks that have already finished.")
  lines.append("- The screenshot links point to the most recent extracted screenshots for the recent completed rows only; they are refreshed on every update.")
  lines.append("- Very long prompts and model outputs are truncated in this Markdown file. The checkpoint and trace links contain the full data.")
  lines.append("")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  artifact_dir.mkdir(parents=True, exist_ok=True)
  tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
  tmp_path.write_text("\n".join(lines), encoding="utf-8")
  tmp_path.replace(out_path)


def _latest_manifest() -> Path:
  candidates = sorted(
      Path("$HOME/anyappbench_results").glob("*/matrix/*/catbench_5cat_manifest.json"),
      key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
      reverse=True,
  )
  if not candidates:
    raise FileNotFoundError("No CATBench manifest found under $HOME/anyappbench_results.")
  return candidates[0]


def _resolve_markdown_out(raw_out: str, allow_redirect: bool = True) -> Path:
  out_path = Path(raw_out).expanduser().resolve()
  if (
      not allow_redirect
      or out_path.name != "markdown_all_models.md"
      or not MARKDOWN_REDIRECT_FILE.exists()
  ):
    return out_path
  redirected = MARKDOWN_REDIRECT_FILE.read_text(encoding="utf-8").strip()
  if not redirected:
    return out_path
  return Path(redirected).expanduser().resolve()


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", default="", help="CATBench matrix manifest.")
  parser.add_argument("--model_config", default=str(CONFIG_PATH))
  parser.add_argument("--out", default=str(REPO_ROOT / "markdown.md"))
  parser.add_argument("--artifact_dir", default=str(REPO_ROOT / "catbench_live_artifacts"))
  parser.add_argument("--recent", type=int, default=20)
  parser.add_argument("--interval", type=int, default=60)
  parser.add_argument("--watch", action="store_true")
  parser.add_argument(
      "--no_redirect",
      action="store_true",
      help="Write exactly to --out, ignoring .catbench_markdown_redirect.",
  )
  args = parser.parse_args()

  manifest = Path(args.manifest).expanduser().resolve() if args.manifest else _latest_manifest()
  model_config = Path(args.model_config).expanduser().resolve()
  out_path = _resolve_markdown_out(args.out, allow_redirect=not args.no_redirect)
  artifact_dir = Path(args.artifact_dir).expanduser().resolve()
  cache: dict[Path, tuple[tuple[int, int], list[EpisodeSummary]]] = {}

  while True:
    _write_markdown(
        manifest=manifest,
        model_config=model_config,
        out_path=out_path,
        artifact_dir=artifact_dir,
        recent_limit=args.recent,
        cache=cache,
    )
    if not args.watch:
      return 0
    time.sleep(max(5, args.interval))


if __name__ == "__main__":
  raise SystemExit(main())
