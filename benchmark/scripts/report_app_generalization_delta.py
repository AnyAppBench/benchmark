#!/usr/bin/env python3
"""Cross-app generalization delta table.

Emits one row per category, four numeric columns:

    Model/Agent  AndroidWorld-original (orig + new)  new-installed (orig + new)  Delta

  * "AndroidWorld-original" = success rate on the upstream apps that ship
    with AndroidWorld (Markor, Tasks.org, Google Clock, Simple Calendar Pro,
    Google Contacts, Simple SMS Messenger). It includes BOTH the original
    upstream tasks and the harder universal cross-app tasks added under
    ``app_generalization_generated`` that target those upstream packages.

  * "new-installed" = success rate on the third-party apps the user
    installed for cross-app generalization (NotallyX, Joplin, ..., Etar,
    Fossify Calendar, Chrono, ..., etc.). Same pair: original-named tasks
    that target a third-party package + new harder cross-app tasks that
    target the third-party package.

  * "Delta" = new-installed (%) - AndroidWorld-original (%); a negative
    delta means the model degraded on alternative apps.

Per-step prompts and reasoning are also surfaced (when present in the
trace) under ``--with_prompts``, so the reporter doubles as a way to read
back what the agent actually saw and how it reasoned.

Skipped-uninstalled tasks (``EXCEPTION_INFO`` starts with
``[skipped_uninstalled]``) are excluded from both numerator and
denominator. They are reported separately at the end.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import io
import json
import os
import pickle
import re
import sys
from typing import Any, Iterable


# Original AndroidWorld task -> category. Anything not in this table that
# does NOT carry a ``For<App>`` suffix is treated as out-of-scope and ignored.
_ORIGINAL_TASKS: dict[str, str] = {
    # Markor
    **{f: "Notes" for f in (
        "MarkorAddNoteHeader", "MarkorChangeNoteContent",
        "MarkorCreateFolder", "MarkorCreateNote",
        "MarkorCreateNoteFromClipboard", "MarkorDeleteAllNotes",
        "MarkorDeleteNewestNote", "MarkorDeleteNote", "MarkorEditNote",
        "MarkorMergeNotes", "MarkorMoveNote", "MarkorTranscribeReceipt",
        "MarkorTranscribeVideo",
    )},
    # Clock (Google Clock)
    **{f: "Clock" for f in (
        "ClockStopWatchPausedVerify", "ClockStopWatchRunning",
        "ClockTimerEntry",
    )},
    # SimpleCalendarPro
    **{f: "Calendar" for f in (
        "SimpleCalendarAddOneEvent", "SimpleCalendarAddOneEventInTwoWeeks",
        "SimpleCalendarAddOneEventRelativeDay",
        "SimpleCalendarAddOneEventTomorrow",
        "SimpleCalendarAddRepeatingEvent", "SimpleCalendarDeleteEvents",
        "SimpleCalendarDeleteEventsOnRelativeDay",
        "SimpleCalendarDeleteOneEvent",
    )},
    # Contacts (Google Contacts)
    "ContactsAddContact": "Contacts",
    "ContactsNewContactDraft": "Contacts",
    # SMS (Simple SMS Messenger)
    **{f: "SMS" for f in (
        "SimpleSmsReply", "SimpleSmsReplyMostRecent", "SimpleSmsResend",
        "SimpleSmsSend", "SimpleSmsSendClipboardContent",
        "SimpleSmsSendReceivedAddress",
    )},
    # Files (upstream)
    "FilesDeleteFile": "Files",
    "FilesMoveFile": "Files",
    # Recipe / Expense / Music upstream evaluators target Broccoli /
    # Pro Expense / Retro Music respectively. Listed only for context;
    # we exclude them from the table because no cross-app cohort exists
    # for them (they are 1-app categories upstream).
}


# Cross-app prefixes (for tasks of the form <Prefix>For<App>). Tied to the
# bases registered in app_generalization_generated/*. Anything else is
# ignored.
_CROSS_APP_PREFIX_TO_CATEGORY: dict[str, str] = {
    "Markor": "Notes", "Notes": "Notes",
    "Clock": "Clock",
    "SimpleCalendar": "Calendar",
    "Contacts": "Contacts",
    "Sms": "SMS",
    "Files": "Files",
    "Finance": "Finance",
    "Maps": "Maps",
    "Music": "Music",
    "Tasks": "ToDo",
}


# AndroidWorld original packages (the apps that ship with the upstream
# AndroidWorld benchmark). Suffixes that resolve to one of these packages
# count toward the AndroidWorld-original column too -- regardless of whether
# the task name is a `For<App>` shim or an upstream-named task.
_AW_ORIGINAL_SUFFIXES: frozenset[str] = frozenset({
    "Markor", "Clock", "Tasksorg", "TasksOrg",
    "SimpleCalendarPro", "GoogleContacts", "SimpleSMSMessenger",
    "MaterialFiles",
    # Categories where the upstream AW evaluator targets a third-party app
    # (not an AOSP app). Including the upstream-targeted suffix here keeps
    # the AW-original column honest -- it captures "the app AndroidWorld
    # ships with" regardless of whether that app is pre-installed or sideloaded.
    "RetroMusic",     # upstream music tasks target Retro Music
    "OsmAnd",         # upstream maps tasks target OsmAnd
    "ProExpense",     # upstream finance tasks target Pro Expense
    "Broccoli",       # upstream recipe tasks target Broccoli
})

CATEGORY_ORDER = (
    "Notes", "Clock", "Calendar", "Contacts", "SMS",
    "Files", "Finance", "Maps", "Music", "ToDo",
)


def _classify(task_name: str) -> tuple[str | None, str]:
  """Returns (category, source) for ``task_name``.

  ``source`` is either ``"aw_original"`` (upstream AW package) or
  ``"new_installed"`` (third-party package), or ``""`` if unclassifiable.

  Recognised shapes:

      * ``<UpstreamTaskName>``                  -> upstream original
      * ``<TaskBase>For<AppSuffix>``            -> cross-app port; category
        is determined by the longest matching prefix in
        ``_CROSS_APP_PREFIX_TO_CATEGORY``; source = ``aw_original`` iff
        ``<AppSuffix>`` is in ``_AW_ORIGINAL_SUFFIXES``.
  """
  if task_name in _ORIGINAL_TASKS:
    return _ORIGINAL_TASKS[task_name], "aw_original"
  # Find the LAST ``For`` followed by a capitalized identifier; that's the
  # app suffix. The rest is the task base name (whose first capitalized
  # token tells us the category). The greedy ``.*`` anchors the match to
  # the right-most ``For``, which matters for names like
  # ``TasksCompletedTasksForDateForNtodotxt`` where ``For`` appears twice.
  match = re.search(r"^(.*)For([A-Z][A-Za-z0-9]+)$", task_name)
  if not match:
    return None, ""
  task_base = match.group(1)
  app_suffix = match.group(2)
  # Pick the longest registered prefix that ``task_base`` starts with.
  category = None
  for pfx in sorted(_CROSS_APP_PREFIX_TO_CATEGORY, key=len, reverse=True):
    if task_base.startswith(pfx):
      category = _CROSS_APP_PREFIX_TO_CATEGORY[pfx]
      break
  if category is None:
    return None, ""
  source = "aw_original" if app_suffix in _AW_ORIGINAL_SUFFIXES else "new_installed"
  return category, source


def _read_pkl_gz(path: str) -> Any:
  with open(path, "rb") as fh:
    raw = fh.read()
  with gzip.open(io.BytesIO(raw), "rb") as gz:
    return pickle.load(gz)


def _load_episodes(checkpoint_dir: str) -> list[dict]:
  if not os.path.isdir(checkpoint_dir):
    raise FileNotFoundError(checkpoint_dir)
  episodes: list[dict] = []
  for fname in sorted(os.listdir(checkpoint_dir)):
    if not fname.endswith(".pkl.gz"):
      continue
    try:
      episodes.extend(_read_pkl_gz(os.path.join(checkpoint_dir, fname)))
    except Exception as exc:  # pylint: disable=broad-except
      print(f"warning: failed to read {fname}: {exc}", file=sys.stderr)
  return episodes


def _is_skipped(ep: dict) -> bool:
  info = ep.get("exception_info") or ""
  return isinstance(info, str) and info.startswith("[skipped_uninstalled]")


def _bucket(
    episodes: Iterable[dict],
) -> tuple[
    dict[str, list[int]],  # aw-original totals: [success, total]
    dict[str, list[int]],  # new-installed totals
    dict[str, int],  # skipped per category
]:
  aw = collections.defaultdict(lambda: [0, 0])
  new = collections.defaultdict(lambda: [0, 0])
  skipped = collections.defaultdict(int)
  for ep in episodes:
    name = ep.get("task_template") or ep.get("name")
    if not name:
      continue
    category, source = _classify(name)
    if category is None:
      continue
    if _is_skipped(ep):
      skipped[category] += 1
      continue
    success = float(ep.get("is_successful") or 0.0) >= 0.5
    bucket = aw if source == "aw_original" else new
    bucket[category][0] += int(success)
    bucket[category][1] += 1
  return aw, new, skipped


def _fmt_rate(stats: tuple[int, int]) -> str:
  s, t = stats
  return "--" if t == 0 else f"{s}/{t} ({100.0 * s / t:5.1f}%)"


def _delta(orig: tuple[int, int], new: tuple[int, int]) -> str:
  if orig[1] == 0 or new[1] == 0:
    return "--"
  return f"{(100.0 * (new[0] / new[1] - orig[0] / orig[1])):+5.1f}pp"


def _render(
    model_label: str,
    aw: dict[str, list[int]],
    new: dict[str, list[int]],
    skipped: dict[str, int],
) -> str:
  rows = [
      ("Category", "AW-original (orig+new)", "new-installed (orig+new)",
       "Delta", "Skipped"),
      ("--------", "----------------------", "------------------------",
       "------", "-------"),
  ]
  total_aw = [0, 0]
  total_new = [0, 0]
  for cat in CATEGORY_ORDER:
    a = tuple(aw.get(cat, [0, 0]))
    n = tuple(new.get(cat, [0, 0]))
    rows.append((
        cat, _fmt_rate(a), _fmt_rate(n), _delta(a, n), str(skipped.get(cat, 0)),
    ))
    total_aw[0] += a[0]
    total_aw[1] += a[1]
    total_new[0] += n[0]
    total_new[1] += n[1]
  rows.append((
      "TOTAL", _fmt_rate(tuple(total_aw)), _fmt_rate(tuple(total_new)),
      _delta(tuple(total_aw), tuple(total_new)),
      str(sum(skipped.values())),
  ))
  widths = [max(len(r[i]) for r in rows) for i in range(5)]
  header = f"{model_label}"
  body = "\n".join(
      "  ".join(r[i].ljust(widths[i]) for i in range(5))
      for r in rows
  )
  return f"{header}\n{'=' * max(len(header), len(body.splitlines()[0]))}\n{body}"


def _emit_prompts(episodes: Iterable[dict], out_dir: str) -> int:
  """Dump per-task prompt + reasoning to JSON for human review.

  Returns the number of tasks dumped. Each task is one ``.json`` file under
  ``out_dir`` with shape::

      {
        "task_template": "...",
        "is_successful": 0.0 or 1.0,
        "goal": "...",
        "steps": [
          {"step": 1, "prompt_system": "...", "prompt_user": "...",
           "response": "...", "thought": "...", "action_desc": "...",
           "action": "..."},
          ...
        ]
      }
  """
  os.makedirs(out_dir, exist_ok=True)
  count = 0
  for ep in episodes:
    name = ep.get("task_template") or ep.get("name")
    if not name:
      continue
    step_data = ep.get("episode_data") or {}
    if not isinstance(step_data, dict):
      continue
    n_steps = len(step_data.get("step_number") or step_data.get("response") or [])
    steps = []
    for i in range(n_steps):
      def _get(field):
        seq = step_data.get(field) or []
        return seq[i] if i < len(seq) else None
      steps.append({
          "step": i + 1,
          "prompt_system": _get("prompt_system"),
          "prompt_user": _get("prompt_user"),
          "response": _get("response"),
          "thought": _get("thought"),
          "action_desc": _get("action_desc"),
          "action": str(_get("action")) if _get("action") is not None else None,
      })
    if not steps:
      continue
    payload = {
        "task_template": name,
        "is_successful": float(ep.get("is_successful") or 0.0),
        "goal": ep.get("goal"),
        "steps": steps,
    }
    fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".json"
    with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fh:
      json.dump(payload, fh, indent=2)
    count += 1
  return count


def _to_csv_rows(
    model_label: str,
    aw: dict[str, list[int]],
    new: dict[str, list[int]],
    skipped: dict[str, int],
) -> list[list[str]]:
  out = [[
      "model_or_agent", "category",
      "aw_original_success", "aw_original_total", "aw_original_rate",
      "new_installed_success", "new_installed_total", "new_installed_rate",
      "delta_pp", "skipped_uninstalled",
  ]]
  for cat in CATEGORY_ORDER:
    a = aw.get(cat, [0, 0])
    n = new.get(cat, [0, 0])
    a_rate = "" if a[1] == 0 else f"{100.0 * a[0] / a[1]:.2f}"
    n_rate = "" if n[1] == 0 else f"{100.0 * n[0] / n[1]:.2f}"
    delta = ""
    if a[1] and n[1]:
      delta = f"{100.0 * (n[0] / n[1] - a[0] / a[1]):.2f}"
    out.append([
        model_label, cat,
        str(a[0]), str(a[1]), a_rate,
        str(n[0]), str(n[1]), n_rate,
        delta, str(skipped.get(cat, 0)),
    ])
  return out


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--checkpoint_dir", required=True,
      help="Directory of *.pkl.gz episodes from one model run.",
  )
  parser.add_argument(
      "--model", required=True,
      help="Label for the model/agent (used as the row header in the table).",
  )
  parser.add_argument(
      "--with_prompts", default="",
      help="If set, write per-task prompt+reasoning JSON files to this dir.",
  )
  parser.add_argument(
      "--csv", default="",
      help="If set, also write a CSV summary to this path (append-friendly).",
  )
  args = parser.parse_args()

  episodes = _load_episodes(os.path.expanduser(args.checkpoint_dir))
  print(f"Loaded {len(episodes)} episodes from {args.checkpoint_dir}\n")

  aw, new, skipped = _bucket(episodes)
  print(_render(args.model, aw, new, skipped))

  if args.with_prompts:
    out = os.path.expanduser(args.with_prompts)
    n_dumped = _emit_prompts(episodes, out)
    print(f"\nWrote per-task prompt+reasoning for {n_dumped} tasks to {out}")

  if args.csv:
    rows = _to_csv_rows(args.model, aw, new, skipped)
    write_header = not os.path.isfile(args.csv)
    with open(args.csv, "a", encoding="utf-8", newline="") as fh:
      writer = csv.writer(fh)
      if write_header:
        writer.writerow(rows[0])
      writer.writerows(rows[1:])
    print(f"\nAppended {len(rows) - 1} category rows to {args.csv}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
