#!/usr/bin/env python3
"""Build a two-column success-rate table comparing the original AndroidWorld
single-app subset (Clock + Contacts + SMS) against the new CATBench cross-app
subset for the same three categories.

Run:
  python scripts/report_clock_contacts_sms.py \
      --original_dir=~/android_world_runs/uiv_ccs/original/<run-id> \
      --cross_app_dir=~/android_world_runs/uiv_ccs/cross_app/<run-id> \
      [--latex]                # also emit a LaTeX-ready row block
      [--per_app]              # per-app breakdown for the cross-app side

Output (default plain text):

  Category   AndroidWorld (orig)   CATBench (cross-app)   Delta
  Clock           x/N (..%)            y/M (..%)         +/- ..pp
  Contacts        ...                  ...               ...
  SMS             ...                  ...               ...
  TOTAL           ...                  ...               ...

Episodes are loaded from the IncrementalCheckpointer .pkl.gz files written by
run_ui_voyager.py. Each episode has the fields described in
android_world.constants.EpisodeConstants (task_template, instance_id,
is_successful, ...).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import io
import os
import pickle
import sys
from typing import Iterable


# Map each task_template to its category. Keep this list explicit so the
# script never silently miscategorises a renamed task.
ORIGINAL_TASK_TO_CATEGORY: dict[str, str] = {
    # Clock
    "ClockTimerEntry": "Clock",
    "ClockStopWatchPausedVerify": "Clock",
    "ClockStopWatchRunning": "Clock",
    # Contacts
    "ContactsAddContact": "Contacts",
    "ContactsNewContactDraft": "Contacts",
    # SMS
    "SimpleSmsSend": "SMS",
    "SimpleSmsReply": "SMS",
    "SimpleSmsReplyMostRecent": "SMS",
    "SimpleSmsResend": "SMS",
    "SimpleSmsSendClipboardContent": "SMS",
    "SimpleSmsSendReceivedAddress": "SMS",
}


CATEGORY_ORDER = ("Clock", "Contacts", "SMS")


def cross_app_category_of(task_template: str) -> str | None:
  """Cross-app classes are named ``{ShortName}For{AppSuffix}``; the prefix
  encodes the category."""
  if task_template.startswith("Clock"):
    return "Clock"
  if task_template.startswith("Contacts"):
    return "Contacts"
  if task_template.startswith("Sms"):
    return "SMS"
  return None


def _read_pkl_gz(path: str):
  with open(path, "rb") as fh:
    raw = fh.read()
  with gzip.open(io.BytesIO(raw), "rb") as gz:
    return pickle.load(gz)


def load_episodes(checkpoint_dir: str) -> list[dict]:
  """Load every episode from a checkpoint directory."""
  if not os.path.isdir(checkpoint_dir):
    raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")
  episodes: list[dict] = []
  for fname in sorted(os.listdir(checkpoint_dir)):
    if not fname.endswith(".pkl.gz"):
      continue
    try:
      episodes.extend(_read_pkl_gz(os.path.join(checkpoint_dir, fname)))
    except Exception as e:  # pylint: disable=broad-except
      print(f"warning: failed to read {fname}: {e}", file=sys.stderr)
  return episodes


def _category_stats(
    episodes: Iterable[dict],
    classifier,  # callable: task_template -> category | None
) -> dict[str, tuple[int, int]]:
  """Return {category: (num_success, num_total)}."""
  totals: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
  for ep in episodes:
    name = ep.get("task_template") or ep.get("name")
    if not name:
      continue
    cat = classifier(name)
    if cat is None:
      continue
    success = float(ep.get("is_successful") or 0.0)
    totals[cat][1] += 1
    if success >= 0.5:
      totals[cat][0] += 1
  return {cat: tuple(v) for cat, v in totals.items()}


def _per_app_stats(episodes: Iterable[dict]) -> dict[str, dict[str, tuple[int, int]]]:
  """For cross-app tasks, group by (category, app suffix)."""
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
    success = float(ep.get("is_successful") or 0.0)
    out[cat][suffix][1] += 1
    if success >= 0.5:
      out[cat][suffix][0] += 1
  return {
      cat: {suf: tuple(v) for suf, v in apps.items()}
      for cat, apps in out.items()
  }


def _fmt(stats: tuple[int, int]) -> str:
  if stats[1] == 0:
    return "  -- "
  s, n = stats
  return f"{s}/{n} ({100.0 * s / n:5.1f}%)"


def _delta(orig: tuple[int, int], new: tuple[int, int]) -> str:
  if orig[1] == 0 or new[1] == 0:
    return "  -- "
  pp = 100.0 * (new[0] / new[1] - orig[0] / orig[1])
  sign = "+" if pp >= 0 else ""
  return f"{sign}{pp:5.1f}pp"


def _aggregate(stats: dict[str, tuple[int, int]]) -> tuple[int, int]:
  s = sum(v[0] for v in stats.values())
  n = sum(v[1] for v in stats.values())
  return s, n


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
  widths = [max(len(r[i]) for r in rows) for i in range(4)]
  out_lines = []
  for r in rows:
    out_lines.append(
        "  ".join(r[i].ljust(widths[i]) for i in range(4))
    )
  return "\n".join(out_lines)


def render_latex(
    orig: dict[str, tuple[int, int]],
    new: dict[str, tuple[int, int]],
) -> str:
  def cell(stats):
    if stats[1] == 0:
      return "--"
    return f"{100.0 * stats[0] / stats[1]:.1f}\\% ({stats[0]}/{stats[1]})"

  lines = [
      r"% Two-column success-rate table.  Drop into a tabular{lcc} block.",
      r"\toprule",
      r"Category & AndroidWorld (orig) & CATBench (cross-app) \\",
      r"\midrule",
  ]
  for cat in CATEGORY_ORDER:
    lines.append(
        f"{cat} & {cell(orig.get(cat, (0, 0)))}"
        f" & {cell(new.get(cat, (0, 0)))} \\\\"
    )
  lines.append(r"\midrule")
  lines.append(
      f"\\textbf{{Total}} & {cell(_aggregate(orig))}"
      f" & {cell(_aggregate(new))} \\\\"
  )
  lines.append(r"\bottomrule")
  return "\n".join(lines)


def render_per_app(per_app: dict[str, dict[str, tuple[int, int]]]) -> str:
  if not per_app:
    return "(no per-app data)"
  out: list[str] = []
  for cat in CATEGORY_ORDER:
    apps = per_app.get(cat) or {}
    if not apps:
      continue
    out.append(f"\n{cat} per-app (cross-app run):")
    width = max((len(a) for a in apps), default=20)
    for app in sorted(apps):
      out.append(f"  {app.ljust(width)}  {_fmt(apps[app])}")
  return "\n".join(out)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--original_dir", required=True,
                  help="checkpoint dir for the AndroidWorld original run.")
  ap.add_argument("--cross_app_dir", required=True,
                  help="checkpoint dir for the cross-app run.")
  ap.add_argument("--latex", action="store_true",
                  help="also emit a LaTeX-ready row block.")
  ap.add_argument("--per_app", action="store_true",
                  help="also print per-app breakdown for the cross-app side.")
  args = ap.parse_args()

  original_dir = os.path.expanduser(args.original_dir)
  cross_app_dir = os.path.expanduser(args.cross_app_dir)

  orig_eps = load_episodes(original_dir)
  new_eps = load_episodes(cross_app_dir)

  print(f"Loaded {len(orig_eps)} original episodes from {original_dir}")
  print(f"Loaded {len(new_eps)} cross-app episodes from {cross_app_dir}")
  print()

  orig_stats = _category_stats(
      orig_eps, ORIGINAL_TASK_TO_CATEGORY.get
  )
  new_stats = _category_stats(new_eps, cross_app_category_of)

  print("UI-Voyager 4B -- Clock / Contacts / SMS")
  print("=" * 76)
  print(render_text(orig_stats, new_stats))

  if args.per_app:
    print(render_per_app(_per_app_stats(new_eps)))

  if args.latex:
    print()
    print("LaTeX:")
    print(render_latex(orig_stats, new_stats))


if __name__ == "__main__":
  main()
