#!/usr/bin/env python3
"""Extract the AW-reproduction table from existing CATBench matrix outputs.

Purpose: defend the paper claim "for the AW-inherited tasks on the AW-baseline
apps, CATBench's measurement matches the published AW numbers" using the data
you ALREADY HAVE. No re-run.

Input:
  --matrix_manifest   the catbench_5cat_manifest.json from the matrix run
  --baseline          benchmark/configs/aw_published_baseline.json (or your
                      filled-in copy)
  --aw_inherited      a CSV/JSON listing which (template, app_id) pairs are
                      considered AW-inherited (i.e. the AW^{AW} cells in your
                      Table 1). Falls back to the default list below.

Output (in --report_dir):
  aw_reproduction_summary.{json,md} : per-cell PASS/FAIL/SKIP
  aw_reproduction_table.tex         : LaTeX table

A cell here is (agent_model x AW-inherited template x AW-baseline app).
Multiple cells can map to the same published_sr if the published number is
agent-level (typical) rather than per-task-instance.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from classify_catbench_failures import _read_pkl_gz, _is_skipped  # noqa: E402


# Default (template -> AW-baseline app_id) mapping derived from your overview
# table. Override with --aw_inherited to be explicit per category. Each tuple
# is (template_substring, app_id). The dynamic per-app class name format used
# in the cross-app modules is "<template>For<AppDisplayName>" so we match by
# substring; the app_id is the matrix's `app_id` (e.g. "clock_google_clock").
DEFAULT_AW_BASELINES: dict[str, tuple[str, ...]] = {
    # category : (app_id of the AW-baseline app for this category, ...)
    "clock":     ("clock_google_clock",),
    "maps":      ("maps_osmand",),
    "contacts":  ("contacts_google_contacts",),
    "sms":       ("sms_simple_sms_messenger",),
    "files":     ("files_material_files",),
    "calendar":  ("calendar_simple_calendar_pro",),
    "notes":     ("notes_joplin", "notes_markor"),
    "music":     ("music_retro_music",),
    "finance":   ("finance_pro_expense",),
    "tasks":     ("tasks_tasks_org",),
}


# Default list of AW-derived template names per category. These match the
# task-template strings recorded in episode pkl.gz (e.g. "ClockTimerEntry").
# Substring match is used for tolerance (CATBench generates per-app variants
# like "ClockTimerEntryForGoogleClock"; we strip the "For<App>" suffix).
DEFAULT_AW_TEMPLATES_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "clock":    ("ClockCreateTimer", "ClockTimerEntry", "ClockStopwatchRunning",
                 "ClockStopWatchRunning", "ClockPauseStopwatch", "ClockStopWatchPausedVerify"),
    "maps":     ("MapsAddFavorite", "MapsAddMarker", "MapsRecordTrack"),
    "contacts": ("ContactsAddContact", "ContactsNewContactDraft"),
    "sms":      ("SmsSend", "SmsReply", "SmsReplyMostRecent", "SmsResend",
                 "SmsSendClipboard", "SmsSendReceivedAddress"),
    "files":    ("FilesDeleteFile", "FilesMoveFile", "FilesSaveCopyOfFile"),
    "calendar": ("SimpleCalendarAddOneEvent", "SimpleCalendarAddTimedEvent",
                 "SimpleCalendarAddRepeatingEvent", "SimpleCalendarDeleteEvent",
                 "SimpleCalendarEventsOnDate", "SimpleCalendarNextEvent",
                 "SimpleCalendarEventsInRange"),
    "notes":    ("NotesCreateNote", "NotesEditNote", "NotesMergeNotes",
                 "NotesDeleteNote", "NotesSearchNote", "NotesShareOrImportNote",
                 "NotesFolderOrMoveNote", "NotesAttachOrTranscribeContent",
                 "NotesCountTodoItems"),
    "music":    ("MusicCreatePlaylist", "MusicAddToPlaylist", "MusicAddToQueue",
                 "MusicSaveOrExportPlaylist", "MusicPlaylistDuration"),
    "finance":  ("FinanceAddExpense", "FinanceAddMultipleExpenses",
                 "FinanceDeleteTransaction", "FinanceDeleteDuplicateTransactions",
                 "FinanceAttachReceipt"),
    "tasks":    ("TasksCompletedTasksForDate", "TasksDueNextWeek",
                 "TasksDueOnDate", "TasksHighPriorityTasks",
                 "TasksHighPriorityDueOnDate", "TasksIncompleteTasksOnDate"),
}


def _strip_app_suffix(template: str) -> str:
  """Turn 'ClockTimerEntryForGoogleClock' into 'ClockTimerEntry'."""
  idx = template.find("For")
  return template[:idx] if idx > 0 else template


def _load_manifest(path: Path) -> list[dict[str, Any]]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  return [job for job in payload.get("jobs", []) if isinstance(job, dict)]


def _harvest_jobs(
    manifest: Path,
    aw_baselines: dict[str, tuple[str, ...]],
    aw_templates: dict[str, tuple[str, ...]],
) -> dict[tuple[str, str, str], list[float]]:
  """{(model, category, template_root): [is_successful, ...]} for AW cells only."""
  out: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
  baseline_set = {
      (cat, app_id)
      for cat, apps in aw_baselines.items()
      for app_id in apps
  }
  template_root_by_category = {
      cat: tuple(_strip_app_suffix(t) for t in templates)
      for cat, templates in aw_templates.items()
  }
  for job in _load_manifest(manifest):
    model = str(job.get("model_name") or "")
    category = str(job.get("category") or "").lower()
    app_id = str(job.get("app_id") or "").lower()
    if (category, app_id) not in baseline_set:
      continue
    aw_roots_for_cat = template_root_by_category.get(category, ())
    output_path = Path(str(job.get("output_path") or "")).expanduser()
    if not output_path.exists():
      continue
    for pkl in sorted(output_path.rglob("*.pkl.gz")):
      try:
        payload = _read_pkl_gz(pkl)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"warning: skip {pkl}: {exc}", file=sys.stderr)
        continue
      episodes = payload if isinstance(payload, list) else [payload]
      for ep in episodes:
        if not isinstance(ep, dict) or _is_skipped(ep):
          continue
        template = str(ep.get("task_template") or pkl.stem)
        template_root = _strip_app_suffix(template)
        if template_root not in aw_roots_for_cat:
          continue
        sr = float(ep.get("is_successful") or 0.0)
        out[(model, category, template_root)].append(1.0 if sr >= 0.5 else 0.0)
  return out


def _aggregate(
    cells: dict[tuple[str, str, str], list[float]],
    published_by_agent: dict[str, dict[str, Any]],
    tolerance_pp: float,
) -> dict[str, Any]:
  by_agent: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
  summary = {"pass": 0, "fail": 0, "skip": 0}
  for (model, category, template_root), srs in sorted(cells.items()):
    if srs:
      catbench_sr: float | None = sum(srs) / len(srs)
      catbench_n = len(srs)
      catbench_stderr = (
          statistics.stdev(srs) / (len(srs) ** 0.5)
          if len(srs) > 1
          else None
      )
    else:
      catbench_sr = None
      catbench_n = 0
      catbench_stderr = None
    pub = (published_by_agent.get(model, {}).get("results", {}) or {}).get(
        template_root, {}
    )
    published_sr = pub.get("published_sr")
    if catbench_sr is None or published_sr is None:
      status = "SKIP"
      delta_pp = None
    else:
      delta_pp = (catbench_sr - published_sr) * 100.0
      status = "PASS" if abs(delta_pp) <= tolerance_pp else "FAIL"
    summary[status.lower()] += 1
    by_agent[model][f"{category}:{template_root}"] = {
        "category": category,
        "template_root": template_root,
        "catbench_sr": catbench_sr,
        "catbench_n": catbench_n,
        "catbench_stderr": catbench_stderr,
        "published_sr": published_sr,
        "n_published": pub.get("n_published"),
        "delta_pp": delta_pp,
        "status": status,
    }
  total = sum(summary.values())
  compared = summary["pass"] + summary["fail"]
  return {
      "tolerance_pp": tolerance_pp,
      "by_agent": dict(by_agent),
      "summary": {"total": total, "compared": compared, **summary},
  }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
  tolerance = report["tolerance_pp"]
  lines: list[str] = [
      "# AW Reproduction — extracted from existing CATBench matrix outputs",
      "",
      f"Tolerance: ±{tolerance:.1f} pp per cell.",
      "Source: pkl.gz episodes from the matrix run, filtered to ",
      "(AW-inherited template × AW-baseline app) pairs only. No re-run.",
      "",
  ]
  for model in sorted(report["by_agent"]):
    cells = report["by_agent"][model]
    lines.append(f"## Agent: `{model}`")
    lines.append("")
    lines.append(
        "| Category | Template (AW-inherited) | CATBench SR (N) ± SE | Published SR | Δ (pp) | Status |"
    )
    lines.append("|---|---|---:|---:|---:|:---:|")
    for key in sorted(cells):
      cell = cells[key]
      if cell["catbench_sr"] is None:
        cb_str = "no data"
      else:
        se = cell["catbench_stderr"]
        cb_str = (
            f"{cell['catbench_sr']:.3f} (n={cell['catbench_n']})"
            + (f" ± {se:.3f}" if se is not None else "")
        )
      pub_str = (
          f"{cell['published_sr']:.3f}" if cell["published_sr"] is not None else "—"
      )
      delta_str = (
          f"{cell['delta_pp']:+.1f}" if cell["delta_pp"] is not None else "—"
      )
      lines.append(
          f"| {cell['category']} | `{cell['template_root']}` | {cb_str} | "
          f"{pub_str} | {delta_str} | {cell['status']} |"
      )
    lines.append("")
  s = report["summary"]
  rate = (100 * s["pass"] / s["compared"]) if s["compared"] else 0.0
  lines.extend(
      [
          "## Aggregate",
          "",
          f"- Cells compared : {s['compared']} (PASS {s['pass']} / FAIL {s['fail']})",
          f"- Cells skipped  : {s['skip']} (no published baseline)",
          "",
          f"**Reproduction claim:** {s['pass']}/{s['compared']} = "
          f"{rate:.1f}% of comparable (model, template) cells within "
          f"±{tolerance:.1f} pp of the published AW number.",
          "",
      ]
  )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(path: Path, report: dict[str, Any]) -> None:
  tol = report["tolerance_pp"]
  buf = [
      "% Auto-generated by extract_aw_reproduction_from_matrix.py.",
      "\\begin{table}[t]\\centering\\small",
      "\\caption{AW reproduction: extracted from the existing matrix run, "
      "filtered to AW-inherited templates on the AW-baseline app per category. "
      f"Cells PASS when $|\\Delta| \\leq {tol:.1f}\\,$pp.}}",
      "\\label{tab:aw-reproduction}",
      "\\begin{tabular}{@{}lllrrrc@{}}",
      "\\toprule",
      "Agent & Category & Template & CATBench SR & $N$ & Published & $\\Delta$ (pp) \\\\",
      "\\midrule",
  ]
  for model in sorted(report["by_agent"]):
    for key in sorted(report["by_agent"][model]):
      c = report["by_agent"][model][key]
      sr = f"{c['catbench_sr']:.3f}" if c["catbench_sr"] is not None else "--"
      pub = f"{c['published_sr']:.3f}" if c["published_sr"] is not None else "--"
      delta = (
          f"{c['delta_pp']:+.1f}" if c["delta_pp"] is not None else "--"
      )
      buf.append(
          f"{model} & {c['category']} & \\texttt{{{c['template_root']}}} & "
          f"{sr} & {c['catbench_n']} & {pub} & {delta} \\\\"
      )
  buf.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
  path.write_text("\n".join(buf) + "\n", encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--matrix_manifest", required=True)
  parser.add_argument(
      "--baseline",
      default="benchmark/configs/aw_published_baseline.json",
  )
  parser.add_argument("--report_dir", required=True)
  parser.add_argument(
      "--aw_inherited",
      default="",
      help=(
          "Optional JSON override of the AW-inherited templates and baseline "
          "apps. See DEFAULT_AW_TEMPLATES_BY_CATEGORY / DEFAULT_AW_BASELINES "
          "in the script for the expected shape."
      ),
  )
  args = parser.parse_args()

  report_dir = Path(args.report_dir).expanduser().resolve()
  report_dir.mkdir(parents=True, exist_ok=True)

  baseline = json.loads(
      Path(args.baseline).expanduser().read_text(encoding="utf-8")
  )
  tolerance = float(baseline.get("tolerance", {}).get("delta_pp", 15.0))
  published_by_agent = baseline.get("published_by_agent", {}) or {}

  if args.aw_inherited:
    override = json.loads(
        Path(args.aw_inherited).expanduser().read_text(encoding="utf-8")
    )
    aw_baselines = {
        k: tuple(v) for k, v in (override.get("baselines") or {}).items()
    } or DEFAULT_AW_BASELINES
    aw_templates = {
        k: tuple(v) for k, v in (override.get("templates") or {}).items()
    } or DEFAULT_AW_TEMPLATES_BY_CATEGORY
  else:
    aw_baselines = DEFAULT_AW_BASELINES
    aw_templates = DEFAULT_AW_TEMPLATES_BY_CATEGORY

  cells = _harvest_jobs(
      Path(args.matrix_manifest).expanduser(), aw_baselines, aw_templates
  )
  if not cells:
    print(
        "error: no AW-inherited cells found. Check --matrix_manifest and "
        "--aw_inherited.",
        file=sys.stderr,
    )
    return 2

  report = _aggregate(cells, published_by_agent, tolerance)
  (report_dir / "aw_reproduction_summary.json").write_text(
      json.dumps(report, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  _write_markdown(report_dir / "aw_reproduction_summary.md", report)
  _write_latex(report_dir / "aw_reproduction_table.tex", report)
  print(f"Wrote {report_dir / 'aw_reproduction_summary.md'}")
  print(
      f"  cells compared={report['summary']['compared']}  "
      f"pass={report['summary']['pass']}  "
      f"fail={report['summary']['fail']}  "
      f"skip={report['summary']['skip']}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
