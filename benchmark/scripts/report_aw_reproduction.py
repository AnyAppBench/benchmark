#!/usr/bin/env python3
"""Compare CATBench-measured SR vs. AW-published SR on the 6 AW-canonical tasks.

Reads the pkl.gz checkpoints produced by ``run_aw_reproduction.py`` and the
published baseline JSON at ``benchmark/configs/aw_published_baseline.json``.
Emits a paper-grade comparison table:

  - ``aw_reproduction_summary.json``  machine-readable per (agent, task) cell
  - ``aw_reproduction_summary.md``    human-readable markdown table
  - ``aw_reproduction_table.tex``     LaTeX table block ready to \\input{}

Cell status:
  PASS  abs(catbench_sr - published_sr) <= tolerance
  FAIL  abs delta > tolerance
  SKIP  published_sr is null in the baseline

A PASS does not prove the validators are byte-identical to AW's, only that
their measured aggregate matches AW's within the stated tolerance under
your environment + agent + model snapshot.
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


def _harvest_pkl(directory: Path) -> dict[str, list[float]]:
  """Return {task_template: [is_successful, ...]} from all pkl.gz in dir."""
  results: dict[str, list[float]] = collections.defaultdict(list)
  for pkl in sorted(directory.rglob("*.pkl.gz")):
    try:
      payload = _read_pkl_gz(pkl)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"warning: skip {pkl}: {exc}", file=sys.stderr)
      continue
    episodes = payload if isinstance(payload, list) else [payload]
    for ep in episodes:
      if not isinstance(ep, dict) or _is_skipped(ep):
        continue
      template = str(ep.get("task_template") or "").strip()
      if not template:
        continue
      sr = float(ep.get("is_successful") or 0.0)
      results[template].append(1.0 if sr >= 0.5 else 0.0)
  return results


def _agent_results(
    out_root: Path,
) -> dict[str, dict[str, list[float]]]:
  """Walk out_root/<agent>/trial_*/  -> {agent: {task: [sr, sr, ...]}}."""
  out: dict[str, dict[str, list[float]]] = {}
  for agent_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
    if agent_dir.name in {"_", "logs", "configs"}:
      continue
    merged: dict[str, list[float]] = collections.defaultdict(list)
    for trial_dir in sorted(agent_dir.iterdir()):
      if not (trial_dir.is_dir() and trial_dir.name.startswith("trial_")):
        continue
      for task, srs in _harvest_pkl(trial_dir).items():
        merged[task].extend(srs)
    if merged:
      out[agent_dir.name] = dict(merged)
  return out


def _format_cell(
    catbench_sr: float | None,
    catbench_n: int,
    catbench_stderr: float | None,
    published_sr: float | None,
    n_published: int | None,
    tolerance_pp: float,
) -> dict[str, Any]:
  if catbench_sr is None or published_sr is None:
    return {
        "catbench_sr": catbench_sr,
        "catbench_n": catbench_n,
        "catbench_stderr": catbench_stderr,
        "published_sr": published_sr,
        "n_published": n_published,
        "delta_pp": None,
        "status": "SKIP",
    }
  delta_pp = (catbench_sr - published_sr) * 100.0
  status = "PASS" if abs(delta_pp) <= tolerance_pp else "FAIL"
  return {
      "catbench_sr": catbench_sr,
      "catbench_n": catbench_n,
      "catbench_stderr": catbench_stderr,
      "published_sr": published_sr,
      "n_published": n_published,
      "delta_pp": delta_pp,
      "status": status,
  }


def _write_markdown(
    path: Path,
    report: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
  tolerance = baseline.get("tolerance", {}).get("delta_pp", 15.0)
  lines: list[str] = [
      "# AW Reproduction Report",
      "",
      f"Tolerance: ±{tolerance:.1f} pp (per-cell).",
      "",
  ]
  for agent, cells in report["by_agent"].items():
    lines.append(f"## Agent: `{agent}`")
    lines.append("")
    lines.append(
        "| Task | CATBench SR (N) ± SE | AW Published SR (N) | Δ (pp) | Status |"
    )
    lines.append("|---|---:|---:|---:|:---:|")
    for task, cell in cells.items():
      if cell["catbench_sr"] is None:
        cb_str = "no data"
      else:
        se = cell["catbench_stderr"]
        cb_str = (
            f"{cell['catbench_sr']:.3f} (n={cell['catbench_n']})"
            + (f" ± {se:.3f}" if se is not None else "")
        )
      if cell["published_sr"] is None:
        pub_str = "—"
      else:
        n_pub = cell["n_published"]
        pub_str = (
            f"{cell['published_sr']:.3f}"
            + (f" (n={n_pub})" if n_pub else "")
        )
      delta_str = (
          f"{cell['delta_pp']:+.1f}" if cell["delta_pp"] is not None else "—"
      )
      lines.append(
          f"| {task} | {cb_str} | {pub_str} | {delta_str} | {cell['status']} |"
      )
    lines.append("")
  summary = report["summary"]
  lines.extend(
      [
          "## Aggregate",
          "",
          f"- Total cells compared : {summary['total']}",
          f"- PASS                 : {summary['pass']}",
          f"- FAIL                 : {summary['fail']}",
          f"- SKIP (no baseline)   : {summary['skip']}",
          "",
          f"**Reproduction claim:** PASS rate = "
          f"{summary['pass']}/{summary['compared']} = "
          f"{(100 * summary['pass'] / summary['compared']) if summary['compared'] else 0.0:.1f}% "
          f"of comparable cells within ±{tolerance:.1f} pp.",
          "",
      ]
  )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(
    path: Path, report: dict[str, Any], baseline: dict[str, Any]
) -> None:
  tolerance = baseline.get("tolerance", {}).get("delta_pp", 15.0)
  buf: list[str] = [
      "% Auto-generated by report_aw_reproduction.py.",
      "\\begin{table}[t]",
      "\\centering",
      "\\small",
      "\\caption{AW reproduction on the 6 AW-canonical Clock and Maps tasks.",
      f" Cells are PASS when $|\\Delta| \\leq {tolerance:.1f}\\,$pp.}}",
      "\\label{tab:aw-reproduction}",
      "\\begin{tabular}{@{}llccccc@{}}",
      "\\toprule",
      "Agent & Task & CATBench SR & $N$ & Published SR & $\\Delta$ (pp) & Status \\\\",
      "\\midrule",
  ]
  for agent, cells in report["by_agent"].items():
    for task, cell in cells.items():
      cb_sr = f"{cell['catbench_sr']:.3f}" if cell["catbench_sr"] is not None else "--"
      pub_sr = f"{cell['published_sr']:.3f}" if cell["published_sr"] is not None else "--"
      n = str(cell["catbench_n"]) if cell["catbench_n"] else "--"
      delta = (
          f"{cell['delta_pp']:+.1f}" if cell["delta_pp"] is not None else "--"
      )
      buf.append(
          f"{agent} & \\texttt{{{task}}} & {cb_sr} & {n} & {pub_sr} & {delta}"
          f" & \\textbf{{{cell['status']}}} \\\\"
      )
  buf.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
  path.write_text("\n".join(buf) + "\n", encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out_root", required=True, help="Reproduction run root.")
  parser.add_argument(
      "--baseline",
      default="benchmark/configs/aw_published_baseline.json",
      help="Path to aw_published_baseline.json.",
  )
  parser.add_argument(
      "--report_dir",
      default="",
      help="Where to write the summary files. Defaults to <out_root>/_report.",
  )
  args = parser.parse_args()

  out_root = Path(args.out_root).expanduser().resolve()
  report_dir = (
      Path(args.report_dir).expanduser().resolve()
      if args.report_dir
      else out_root / "_report"
  )
  report_dir.mkdir(parents=True, exist_ok=True)
  baseline = json.loads(Path(args.baseline).expanduser().read_text(encoding="utf-8"))
  tolerance = float(baseline.get("tolerance", {}).get("delta_pp", 15.0))
  published_by_agent: dict[str, dict[str, Any]] = baseline.get(
      "published_by_agent", {}
  )

  agent_results = _agent_results(out_root)
  if not agent_results:
    print(f"error: no agent results found under {out_root}", file=sys.stderr)
    return 2

  # The folder name on disk uses the safe-short form (alnum/_/-). Map back
  # by exact match if the baseline uses display names with spaces.
  def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)

  baseline_keys_by_safe = {_safe(k): k for k in published_by_agent}

  by_agent: dict[str, dict[str, dict[str, Any]]] = {}
  summary = {"pass": 0, "fail": 0, "skip": 0}
  for agent_safe, by_task in agent_results.items():
    canonical_name = baseline_keys_by_safe.get(agent_safe, agent_safe)
    agent_published = published_by_agent.get(canonical_name, {}).get(
        "results", {}
    )
    cells: dict[str, dict[str, Any]] = {}
    for task, srs in sorted(by_task.items()):
      if srs:
        catbench_sr: float | None = sum(srs) / len(srs)
        catbench_stderr = (
            statistics.stdev(srs) / (len(srs) ** 0.5)
            if len(srs) > 1
            else None
        )
        catbench_n = len(srs)
      else:
        catbench_sr = None
        catbench_stderr = None
        catbench_n = 0
      published_entry = agent_published.get(task, {})
      published_sr = published_entry.get("published_sr")
      n_published = published_entry.get("n_published")
      cell = _format_cell(
          catbench_sr=catbench_sr,
          catbench_n=catbench_n,
          catbench_stderr=catbench_stderr,
          published_sr=published_sr,
          n_published=n_published,
          tolerance_pp=tolerance,
      )
      cells[task] = cell
      summary[cell["status"].lower()] += 1
    by_agent[canonical_name] = cells

  total = sum(summary.values())
  compared = summary["pass"] + summary["fail"]
  report = {
      "tolerance_pp": tolerance,
      "by_agent": by_agent,
      "summary": {
          "total": total,
          "compared": compared,
          **summary,
      },
  }
  (report_dir / "aw_reproduction_summary.json").write_text(
      json.dumps(report, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  _write_markdown(report_dir / "aw_reproduction_summary.md", report, baseline)
  _write_latex(report_dir / "aw_reproduction_table.tex", report, baseline)
  print(f"Wrote {report_dir / 'aw_reproduction_summary.md'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
