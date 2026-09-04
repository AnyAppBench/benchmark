#!/usr/bin/env python3
"""Report UI-Voyager results for Clock, Contacts, Calendar, and Files."""

from __future__ import annotations

import argparse
import collections
import gzip
import io
import os
import pickle
import sys
from collections.abc import Callable, Iterable


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCHMARK_ROOT = os.path.join(REPO_ROOT, "benchmark")
if BENCHMARK_ROOT not in sys.path:
  sys.path.insert(0, BENCHMARK_ROOT)


ORIGINAL_TASK_TO_CATEGORY: dict[str, str] = {
    "ClockTimerEntry": "Clock",
    "ClockStopWatchPausedVerify": "Clock",
    "ClockStopWatchRunning": "Clock",
    "ContactsAddContact": "Contacts",
    "ContactsNewContactDraft": "Contacts",
    "SimpleCalendarAddOneEvent": "Calendar",
    "SimpleCalendarAddRepeatingEvent": "Calendar",
    "SimpleCalendarDeleteEvents": "Calendar",
    "FilesDeleteFile": "Files",
    "FilesMoveFile": "Files",
}

CATEGORY_ORDER = ("Clock", "Contacts", "Calendar", "Files")


def cross_app_category_of(task_template: str) -> str | None:
  if task_template.startswith("Clock"):
    return "Clock"
  if task_template.startswith("Contacts"):
    return "Contacts"
  if task_template.startswith("SimpleCalendar"):
    return "Calendar"
  if task_template.startswith("Files"):
    return "Files"
  return None


def _read_pkl_gz(path: str):
  with open(path, "rb") as fh:
    raw = fh.read()
  with gzip.open(io.BytesIO(raw), "rb") as gz:
    return pickle.load(gz)


def load_episodes(checkpoint_dir: str) -> list[dict]:
  if not os.path.isdir(checkpoint_dir):
    raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")
  episodes: list[dict] = []
  for fname in sorted(os.listdir(checkpoint_dir)):
    if not fname.endswith(".pkl.gz"):
      continue
    try:
      episodes.extend(_read_pkl_gz(os.path.join(checkpoint_dir, fname)))
    except Exception as exc:  # pylint: disable=broad-except
      print(f"warning: failed to read {fname}: {exc}", file=sys.stderr)
  return episodes


def _category_stats(
    episodes: Iterable[dict],
    classifier: Callable[[str], str | None],
) -> dict[str, tuple[int, int]]:
  totals: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
  for ep in episodes:
    name = ep.get("task_template") or ep.get("name")
    if not name:
      continue
    cat = classifier(name)
    if cat is None:
      continue
    totals[cat][1] += 1
    if float(ep.get("is_successful") or 0.0) >= 0.5:
      totals[cat][0] += 1
  return {cat: tuple(v) for cat, v in totals.items()}


def _per_app_stats(episodes: Iterable[dict]) -> dict[str, dict[str, tuple[int, int]]]:
  out: dict[str, dict[str, list[int]]] = collections.defaultdict(
      lambda: collections.defaultdict(lambda: [0, 0])
  )
  for ep in episodes:
    name = ep.get("task_template") or ep.get("name")
    if not name or "For" not in name:
      continue
    cat = cross_app_category_of(name)
    if cat is None:
      continue
    suffix = name.rsplit("For", 1)[1]
    out[cat][suffix][1] += 1
    if float(ep.get("is_successful") or 0.0) >= 0.5:
      out[cat][suffix][0] += 1
  return {
      cat: {suffix: tuple(v) for suffix, v in apps.items()}
      for cat, apps in out.items()
  }


def _fmt(stats: tuple[int, int]) -> str:
  success, total = stats
  if total == 0:
    return "--"
  return f"{success}/{total} ({100.0 * success / total:5.1f}%)"


def _delta(orig: tuple[int, int], new: tuple[int, int]) -> str:
  if orig[1] == 0 or new[1] == 0:
    return "--"
  pp = 100.0 * (new[0] / new[1] - orig[0] / orig[1])
  sign = "+" if pp >= 0 else ""
  return f"{sign}{pp:5.1f}pp"


def _aggregate(stats: dict[str, tuple[int, int]]) -> tuple[int, int]:
  return (
      sum(v[0] for v in stats.values()),
      sum(v[1] for v in stats.values()),
  )


def render_text(
    orig: dict[str, tuple[int, int]],
    new: dict[str, tuple[int, int]],
) -> str:
  rows = [
      ("Category", "AndroidWorld (orig)", "CATBench (cross-app)", "Delta"),
      ("--------", "-------------------", "--------------------", "------"),
  ]
  for cat in CATEGORY_ORDER:
    rows.append((
        cat,
        _fmt(orig.get(cat, (0, 0))),
        _fmt(new.get(cat, (0, 0))),
        _delta(orig.get(cat, (0, 0)), new.get(cat, (0, 0))),
    ))
  rows.append((
      "TOTAL",
      _fmt(_aggregate(orig)),
      _fmt(_aggregate(new)),
      _delta(_aggregate(orig), _aggregate(new)),
  ))
  widths = [max(len(row[i]) for row in rows) for i in range(4)]
  return "\n".join(
      "  ".join(row[i].ljust(widths[i]) for i in range(4))
      for row in rows
  )


def render_per_app(per_app: dict[str, dict[str, tuple[int, int]]]) -> str:
  if not per_app:
    return "(no per-app data)"
  out: list[str] = []
  for cat in CATEGORY_ORDER:
    apps = per_app.get(cat) or {}
    if not apps:
      continue
    out.append(f"\n{cat} per-app (cross-app run):")
    width = max((len(app) for app in apps), default=20)
    for app in sorted(apps):
      out.append(f"  {app.ljust(width)}  {_fmt(apps[app])}")
  return "\n".join(out)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--original_dir", required=True)
  parser.add_argument("--cross_app_dir", required=True)
  parser.add_argument("--per_app", action="store_true")
  args = parser.parse_args()

  original_dir = os.path.expanduser(args.original_dir)
  cross_app_dir = os.path.expanduser(args.cross_app_dir)
  original_eps = load_episodes(original_dir)
  cross_app_eps = load_episodes(cross_app_dir)

  print(f"Loaded {len(original_eps)} original episodes from {original_dir}")
  print(f"Loaded {len(cross_app_eps)} cross-app episodes from {cross_app_dir}")
  print()

  original_stats = _category_stats(original_eps, ORIGINAL_TASK_TO_CATEGORY.get)
  cross_app_stats = _category_stats(cross_app_eps, cross_app_category_of)

  print("UI-Voyager 4B -- Clock / Contacts / Calendar / Files")
  print("=" * 84)
  print(render_text(original_stats, cross_app_stats))

  if args.per_app:
    print(render_per_app(_per_app_stats(cross_app_eps)))


if __name__ == "__main__":
  main()
