#!/usr/bin/env python3
"""Per-app effect of C2 sub-goal augmentation (paper Table E.1).

Reads C2 episode checkpoints (``*.pkl.gz``) from one or more batch roots,
counts successes per (model, app) per replicate, pools them, and reports the
difference in success rate (percentage points) against a C1 baseline taken
either from the validated K=1 baseline JSON or from a C1 batch root.

Root layouts:
  nested  ``<root>/<replicate>/<lane>/matrix/**/<Task>_<inst>.pkl.gz``
  flat    ``<root>/**/*.pkl.gz`` (whole root treated as replicate ``r1``)

Each checkpoint is mapped to (model, category, app_id) from its path
(``.../<model_slug>/<category>/<app_id>/run_*/<file>``); model slugs are
resolved against the model config and any ``catbench_5cat_manifest.json``.

Fail-closed accounting: episodes that are missing from the expected schedule,
invalid (infrastructure exceptions / unknown status), or run under the wrong
condition are counted and reported separately; they are never silently
treated as failures.  Rates use the number of valid completed episodes as the
denominator; the strict rate with the scheduled denominator is kept in JSON.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import itertools
import json
import re
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, BENCHMARK_ROOT):
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from app_generalization_profiles import get_domain_profiles  # pylint: disable=wrong-import-position
from report_catbench_5cat_results import _is_skipped, _read_pkl_gz, _slug  # pylint: disable=wrong-import-position

DEFAULT_MODEL_CONFIG = BENCHMARK_ROOT / "configs" / "catbench_5cat_models.json"
DEFAULT_COHORT = BENCHMARK_ROOT / "configs" / "catbench_5cat_primary_cohort.json"
C2_CONDITION = "c2_g"
C1_CONDITION = "c1"
CATEGORY_LABELS = {
    "sms": "SMS",
    "files": "Files",
    "maps": "Maps",
    "contacts": "Contacts",
    "clock": "Clock",
}
_RUN_DIR_RE = re.compile(r"^run_")
_FILE_INSTANCE_RE = re.compile(r"^(?P<template>.+?)_(?P<instance>\d+)$")


# --------------------------------------------------------------------------
# Cohort / config
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Cohort:
  """Model order plus category -> (app_ids, semantic template ids)."""

  models: tuple[str, ...]
  categories: tuple[str, ...]
  app_ids: dict[str, tuple[str, ...]]
  semantic_ids: dict[str, tuple[str, ...]]
  aw_app_id: dict[str, str]
  app_display: dict[str, str]

  @property
  def cells(self) -> list[tuple[str, str, str]]:
    return [
        (model, category, app_id)
        for model in self.models
        for category in self.categories
        for app_id in self.app_ids[category]
    ]

  def semantic_id_for(self, category: str, task_template: str) -> str | None:
    """Maps ``SmsSendForFossifyMessages`` -> ``SmsSend`` (longest match)."""
    best = None
    for sid in self.semantic_ids.get(category, ()):
      if task_template == sid or task_template.startswith(sid + "For"):
        if best is None or len(sid) > len(best):
          best = sid
    return best


def load_cohort(
    cohort_path: Path,
    model_config: Path,
    exclude_semantic_tasks: tuple[str, ...] = (),
) -> Cohort:
  """Loads the frozen cohort, optionally dropping unevaluable templates.

  A template excluded here leaves the schedule entirely, so its episodes are
  neither counted nor reported as missing. Use it only for a template that no
  agent can be scored on in this environment (e.g. one needing a helper app
  absent from the pinned image), and state the exclusion alongside the table.
  """
  cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
  config = json.loads(model_config.read_text(encoding="utf-8"))
  config_names = [m["name"] for m in config.get("models", []) if m.get("name")]
  models = tuple(m for m in cohort["models"] if m in config_names) or tuple(
      cohort["models"]
  )
  profiles = get_domain_profiles()
  display: dict[str, str] = {}
  for profile in profiles.values():
    for app in profile.apps:
      display[app.app_id] = app.display_name
  categories = tuple(cohort["categories"])
  return Cohort(
      models=models,
      categories=categories,
      app_ids={c: tuple(cohort["categories"][c]["app_ids"]) for c in categories},
      semantic_ids={
          c: tuple(
              t
              for t in cohort["categories"][c]["semantic_task_ids"]
              if t not in exclude_semantic_tasks
          )
          for c in categories
      },
      aw_app_id={c: cohort["categories"][c]["aw_app_id"] for c in categories},
      app_display=display,
  )


# --------------------------------------------------------------------------
# Checkpoint discovery and parsing
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
  """Slim, picklable view of one checkpoint episode."""

  replicate: str
  lane: str
  model_slug: str
  category: str
  app_id: str
  task_template: str
  semantic_task_id: str | None
  instance_id: int | None
  condition: str
  condition_config_valid: bool
  episode_status: str | None
  is_successful: float | None
  has_exception: bool
  skipped: bool
  mtime: float
  path: str


_REPLICATE_DIR_RE = re.compile(r"r\d+")


def discover_checkpoints(root: Path, layout: str) -> list[tuple[str, str, Path]]:
  """Returns (replicate, lane, path) triples for every checkpoint under root."""
  found: list[tuple[str, str, Path]] = []
  if layout == "flat":
    for path in sorted(root.rglob("*.pkl.gz")):
      try:
        lane = path.relative_to(root).parts[0]
      except (ValueError, IndexError):
        lane = ""
      found.append(("r1", lane, path))
    return found
  for rep_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    if not _REPLICATE_DIR_RE.fullmatch(rep_dir.name):
      # Only r1, r2, ... are replicates. Anything else under the batch root
      # (quarantined data, scratch dirs) must never enter the analysis.
      continue
    for lane_dir in sorted(p for p in rep_dir.iterdir() if p.is_dir()):
      search_root = lane_dir / "matrix" if (lane_dir / "matrix").is_dir() else lane_dir
      for path in sorted(search_root.rglob("*.pkl.gz")):
        found.append((rep_dir.name, lane_dir.name, path))
  return found


def _path_identity(path: Path) -> tuple[str, str, str] | None:
  """Infers (model_slug, category, app_id) from ``.../m/c/a/run_*/x.pkl.gz``."""
  parts = path.parts
  for idx in range(len(parts) - 1, 2, -1):
    if _RUN_DIR_RE.match(parts[idx]) and idx >= 3:
      return parts[idx - 3], parts[idx - 2], parts[idx - 1]
  return None


def _coerce_float(value: Any) -> float | None:
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def _coerce_int(value: Any) -> int | None:
  try:
    return int(value)
  except (TypeError, ValueError):
    return None


def parse_checkpoint(
    item: tuple[str, str, Path],
) -> tuple[list[EpisodeRecord], str | None]:
  """Loads one checkpoint; returns (records, error_message)."""
  replicate, lane, path = item
  try:
    payload = _read_pkl_gz(path)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return [], f"{path}: unreadable checkpoint ({exc})"
  if not isinstance(payload, list):
    return [], f"{path}: checkpoint payload is not a list"
  identity = _path_identity(path)
  stem = path.name.removesuffix(".pkl.gz")
  file_match = _FILE_INSTANCE_RE.match(stem)
  records: list[EpisodeRecord] = []
  mtime = path.stat().st_mtime
  for ep in payload:
    if not isinstance(ep, dict):
      continue
    if identity is None:
      model_slug = _slug(str(ep.get("model_name") or ""))
      category, app_id = "", str(ep.get("app_id") or "")
    else:
      model_slug, category, app_id = identity
    if not app_id:
      return [], f"{path}: cannot map checkpoint to (model, category, app_id)"
    instance = _coerce_int(ep.get("instance_id"))
    if instance is None:
      instance = _coerce_int(ep.get("catbench_instance_id"))
    if instance is None and file_match:
      instance = int(file_match.group("instance"))
    task_template = str(ep.get("task_template") or (file_match.group("template") if file_match else stem))
    exception = bool(ep.get("exception_info") or ep.get("catbench_exception_type"))
    records.append(
        EpisodeRecord(
            replicate=replicate,
            lane=lane,
            model_slug=model_slug,
            category=category,
            app_id=app_id,
            task_template=task_template,
            semantic_task_id=(
                str(ep["semantic_task_id"]) if ep.get("semantic_task_id") else None
            ),
            instance_id=instance,
            condition=str(ep.get("catbench_condition") or ""),
            condition_config_valid=bool(ep.get("catbench_condition_config_valid", False)),
            episode_status=(
                str(ep["catbench_episode_status"])
                if ep.get("catbench_episode_status") is not None else None
            ),
            is_successful=_coerce_float(ep.get("is_successful")),
            has_exception=exception,
            skipped=_is_skipped(ep),
            mtime=mtime,
            path=str(path),
        )
    )
  return records, None


def load_records(
    roots: Sequence[Path], layout: str, workers: int
) -> tuple[list[EpisodeRecord], list[str]]:
  """Loads all checkpoints under the given roots."""
  items: list[tuple[str, str, Path]] = []
  for root in roots:
    items.extend(discover_checkpoints(root, layout))
  records: list[EpisodeRecord] = []
  errors: list[str] = []
  if workers > 1 and len(items) > 1:
    with ProcessPoolExecutor(max_workers=workers) as pool:
      results: Iterable[tuple[list[EpisodeRecord], str | None]] = pool.map(
          parse_checkpoint, items, chunksize=8
      )
      for recs, err in results:
        records.extend(recs)
        if err:
          errors.append(err)
  else:
    for item in items:
      recs, err = parse_checkpoint(item)
      records.extend(recs)
      if err:
        errors.append(err)
  return records, errors


def model_slug_map(cohort: Cohort, roots: Sequence[Path]) -> dict[str, str]:
  """slug -> canonical model name, from the config plus any lane manifests."""
  mapping = {_slug(m): m for m in cohort.models}
  mapping.update({m: m for m in cohort.models})
  for root in roots:
    for manifest in root.rglob("catbench_5cat_manifest.json"):
      try:
        jobs = json.loads(manifest.read_text(encoding="utf-8")).get("jobs", [])
      except (OSError, ValueError):
        continue
      for job in jobs:
        name = job.get("model_name") if isinstance(job, dict) else None
        if isinstance(name, str) and name in cohort.models:
          mapping.setdefault(_slug(name), name)
          out = job.get("output_path")
          if isinstance(out, str):
            parts = Path(out).parts
            if len(parts) >= 3:
              mapping.setdefault(parts[-3], name)
  return mapping


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------


@dataclasses.dataclass
class CellCounts:
  """Per (model, app) accounting for one replicate or pooled."""

  scheduled: int = 0
  success: int = 0
  valid: int = 0
  invalid: int = 0
  exception: int = 0
  excluded_condition: int = 0
  duplicates: int = 0
  unexpected: int = 0
  missing: list[str] = dataclasses.field(default_factory=list)

  def add(self, other: "CellCounts") -> None:
    for name in (
        "scheduled", "success", "valid", "invalid", "exception",
        "excluded_condition", "duplicates", "unexpected",
    ):
      setattr(self, name, getattr(self, name) + getattr(other, name))
    self.missing = self.missing + other.missing

  @property
  def rate(self) -> float | None:
    return 100.0 * self.success / self.valid if self.valid else None

  @property
  def rate_scheduled(self) -> float | None:
    return 100.0 * self.success / self.scheduled if self.scheduled else None

  def as_dict(self) -> dict[str, Any]:
    return {
        "scheduled": self.scheduled,
        "success": self.success,
        "valid": self.valid,
        "invalid": self.invalid,
        "exception": self.exception,
        "excluded_condition": self.excluded_condition,
        "duplicates": self.duplicates,
        "unexpected": self.unexpected,
        "missing": len(self.missing),
        "missing_slots": self.missing,
        "rate_percent": self.rate,
        "rate_scheduled_percent": self.rate_scheduled,
    }


CellKey = tuple[str, str, str]
Counts = dict[CellKey, CellCounts]


def count_replicate(
    records: Sequence[EpisodeRecord],
    cohort: Cohort,
    slug_map: dict[str, str],
    condition: str,
    instance_ids: Sequence[int],
) -> tuple[Counts, list[str]]:
  """Builds CellCounts for every cohort cell from one replicate's records."""
  notes: list[str] = []
  by_slot: dict[CellKey, dict[tuple[str, int], list[EpisodeRecord]]] = collections.defaultdict(
      lambda: collections.defaultdict(list)
  )
  counts: Counts = {cell: CellCounts() for cell in cohort.cells}
  for rec in records:
    model = slug_map.get(rec.model_slug)
    cell = (model or rec.model_slug, rec.category, rec.app_id)
    if model is None or cell not in counts:
      notes.append(f"unmapped episode ignored: {rec.path}")
      continue
    sid = rec.semantic_task_id or cohort.semantic_id_for(rec.category, rec.task_template)
    if sid is None or rec.instance_id is None:
      counts[cell].unexpected += 1
      notes.append(f"unrecognised template/instance ignored: {rec.path}")
      continue
    if rec.instance_id not in instance_ids:
      continue
    if rec.condition != condition or not rec.condition_config_valid:
      counts[cell].excluded_condition += 1
      continue
    by_slot[cell][(sid, rec.instance_id)].append(rec)

  for cell, cc in counts.items():
    _, category, _ = cell
    expected = [(sid, inst) for sid in cohort.semantic_ids[category] for inst in instance_ids]
    cc.scheduled = len(expected)
    slots = by_slot.get(cell, {})
    for key in expected:
      recs = slots.get(key, [])
      if not recs:
        cc.missing.append(f"{key[0]}#{key[1]}")
        continue
      if len(recs) > 1:
        cc.duplicates += len(recs) - 1
      rec = max(recs, key=lambda r: (r.mtime, r.path))
      if rec.has_exception:
        cc.exception += 1
      if rec.skipped or rec.is_successful is None:
        cc.invalid += 1
        continue
      cc.valid += 1
      if rec.is_successful >= 0.5:
        cc.success += 1
    for key in slots:
      if key not in expected:
        cc.unexpected += len(slots[key])
  return counts, notes


def pool_counts(per_replicate: dict[str, Counts], cohort: Cohort) -> Counts:
  pooled: Counts = {cell: CellCounts() for cell in cohort.cells}
  for counts in per_replicate.values():
    for cell, cc in counts.items():
      pooled[cell].add(cc)
  return pooled


def aggregate(counts: Counts, cells: Iterable[CellKey]) -> CellCounts:
  total = CellCounts()
  for cell in cells:
    if cell in counts:
      total.add(counts[cell])
  return total


# --------------------------------------------------------------------------
# C1 baseline
# --------------------------------------------------------------------------


def load_c1_baseline_json(path: Path, cohort: Cohort) -> Counts:
  """Reads k1_validated_baseline.json (AW + New split) into CellCounts."""
  payload = json.loads(path.read_text(encoding="utf-8"))
  counts: Counts = {cell: CellCounts() for cell in cohort.cells}
  for row in payload["per_app_rows"]:
    cell = (row["model"], row["category"], row["app_id"])
    if cell not in counts:
      continue
    cc = counts[cell]
    for split in row["K1"].values():
      cc.success += int(split["successes"])
      cc.valid += int(split["total"])
      cc.scheduled += int(split["total"])
  return counts


# --------------------------------------------------------------------------
# Delta table
# --------------------------------------------------------------------------


def fmt_delta(value: float | None) -> str:
  """``+20``, ``-30``, ``0``; one decimal only when not an integer."""
  if value is None:
    return "--"
  rounded = round(value, 1)
  if abs(rounded) < 0.05:
    return "0"
  if abs(rounded - round(rounded)) < 1e-9:
    return f"{int(round(rounded)):+d}"
  return f"{rounded:+.1f}"


def delta(c2: CellCounts, c1: CellCounts) -> float | None:
  if c2.rate is None or c1.rate is None:
    return None
  return c2.rate - c1.rate


@dataclasses.dataclass
class TableRow:
  kind: str  # "app" | "category" | "overall"
  category: str
  label: str
  cells: list[dict[str, Any]]  # one per model: delta, c2, c1


def build_table(c2: Counts, c1: Counts, cohort: Cohort) -> list[TableRow]:
  rows: list[TableRow] = []

  def cell(c2c: CellCounts, c1c: CellCounts) -> dict[str, Any]:
    return {
        "delta": delta(c2c, c1c),
        "c2_success": c2c.success, "c2_valid": c2c.valid, "c2_scheduled": c2c.scheduled,
        "c2_rate": c2c.rate, "c1_success": c1c.success, "c1_valid": c1c.valid,
        "c1_rate": c1c.rate, "c2_incomplete": c2c.valid < c2c.scheduled,
    }

  for category in cohort.categories:
    for app_id in cohort.app_ids[category]:
      rows.append(TableRow(
          "app", category, cohort.app_display.get(app_id, app_id),
          [cell(c2[(m, category, app_id)], c1[(m, category, app_id)]) for m in cohort.models],
      ))
    rows.append(TableRow(
        "category", category, f"{CATEGORY_LABELS.get(category, category)} (all apps)",
        [
            cell(aggregate(c2, [(m, category, a) for a in cohort.app_ids[category]]),
                 aggregate(c1, [(m, category, a) for a in cohort.app_ids[category]]))
            for m in cohort.models
        ],
    ))
  rows.append(TableRow(
      "overall", "", "Overall",
      [
          cell(aggregate(c2, [c for c in cohort.cells if c[0] == m]),
               aggregate(c1, [c for c in cohort.cells if c[0] == m]))
          for m in cohort.models
      ],
  ))
  return rows


def _cell_text(c: dict[str, Any], star: str) -> str:
  text = fmt_delta(c["delta"])
  return text + star if c["c2_incomplete"] and text != "--" else text


def _n_summary(counts: Counts) -> str:
  """Range of valid-episode counts over cells that have any data."""
  values = sorted({cc.valid for cc in counts.values() if cc.valid})
  if not values:
    return "0"
  if len(values) == 1:
    return str(values[0])
  return f"{values[0]}--{values[-1]}"


def write_delta_markdown(
    rows: list[TableRow], cohort: Cohort, c2: Counts, c1: Counts, path: Path, variant: str
) -> None:
  lines = [
      f"# Per-app effect of C2 sub-goal augmentation ({variant})",
      "",
      "Cell = C2 success rate (%) minus C1 success rate (%), in percentage points.",
      f"C2 n per app cell: {_n_summary(c2)} valid episodes (scheduled "
      f"{sorted({cc.scheduled for cc in c2.values()})}); C1 n per app cell: {_n_summary(c1)}.",
      "`*` marks cells where C2 has fewer valid episodes than scheduled.",
      "",
      "| Category | App | " + " | ".join(cohort.models) + " |",
      "|---|---|" + "---:|" * len(cohort.models),
  ]
  for row in rows:
    label = f"**{row.label}**" if row.kind != "app" else row.label
    cat = CATEGORY_LABELS.get(row.category, row.category) if row.kind == "app" else ""
    lines.append(
        f"| {cat} | {label} | " + " | ".join(_cell_text(c, "*") for c in row.cells) + " |"
    )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tex_escape(text: str) -> str:
  return text.replace("&", r"\&").replace("_", r"\_").replace("~", r"\textasciitilde{}")


def write_delta_latex(
    rows: list[TableRow], cohort: Cohort, c2: Counts, c1: Counts, path: Path, variant: str
) -> None:
  ncols = 2 + len(cohort.models)
  header = " & ".join(rf"\textbf{{{_tex_escape(m)}}}" for m in cohort.models)
  lines = [
      "% Auto-generated by report_c2_per_app_delta.py; do not edit by hand.",
      r"\begin{table}[t]",
      r"\centering",
      r"\small",
      rf"\caption{{Per-app effect of C2 sub-goal augmentation ({_tex_escape(variant)}). "
      "Each cell is the C2 success rate minus the C1 success rate, in percentage points.}",
      r"\label{tab:c2_per_app_delta_appendix}",
      rf"\begin{{tabular}}{{ll{'r' * len(cohort.models)}}}",
      r"\toprule",
      rf"\textbf{{Category}} & \textbf{{App}} & {header} \\",
      r"\midrule",
  ]
  for category in cohort.categories:
    cat_rows = [r for r in rows if r.category == category]
    app_rows = [r for r in cat_rows if r.kind == "app"]
    for idx, row in enumerate(app_rows):
      prefix = (
          rf"\multirow{{{len(app_rows)}}}{{*}}{{{CATEGORY_LABELS.get(category, category)}}}"
          if idx == 0 else ""
      )
      cells = " & ".join(_cell_text(c, r"$^{*}$") for c in row.cells)
      lines.append(rf"{prefix} & {_tex_escape(row.label)} & {cells} \\")
    for row in (r for r in cat_rows if r.kind == "category"):
      cells = " & ".join(_cell_text(c, r"$^{*}$") for c in row.cells)
      lines.append(rf" & \textit{{{_tex_escape(row.label)}}} & {cells} \\")
    lines.append(r"\midrule")
  for row in (r for r in rows if r.kind == "overall"):
    cells = " & ".join(rf"\textbf{{{_cell_text(c, r'$^{*}$')}}}" for c in row.cells)
    lines.append(rf"\multicolumn{{2}}{{l}}{{\textbf{{Overall}}}} & {cells} \\")
  lines += [
      r"\bottomrule",
      rf"\multicolumn{{{ncols}}}{{l}}{{\footnotesize C2: $n={_n_summary(c2)}$ valid episodes per app cell "
      rf"(scheduled {sorted({cc.scheduled for cc in c2.values()})[0]}); "
      rf"C1: $n={_n_summary(c1)}$ per app cell. $^{{*}}$: C2 cell has fewer valid episodes than scheduled.}} \\",
      r"\end{tabular}",
      r"\end{table}",
  ]
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Counts table and summary
# --------------------------------------------------------------------------


COUNT_FIELDS = (
    "scheduled", "success", "valid", "invalid", "exception",
    "excluded_condition", "duplicates", "unexpected", "missing",
)


def write_counts(
    per_replicate: dict[str, Counts], pooled: Counts, cohort: Cohort, csv_path: Path, md_path: Path
) -> None:
  rows: list[dict[str, Any]] = []
  for replicate, counts in itertools.chain(per_replicate.items(), [("pooled", pooled)]):
    for cell in cohort.cells:
      cc = counts[cell]
      row = {
          "replicate": replicate,
          "model": cell[0],
          "category": cell[1],
          "app_id": cell[2],
          "app": cohort.app_display.get(cell[2], cell[2]),
      }
      row.update({f: (len(cc.missing) if f == "missing" else getattr(cc, f)) for f in COUNT_FIELDS})
      row["rate_percent"] = "" if cc.rate is None else f"{cc.rate:.2f}"
      row["missing_slots"] = ";".join(cc.missing)
      rows.append(row)
  fields = list(rows[0].keys()) if rows else []
  with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

  lines = [
      "# C2 per-app episode counts",
      "",
      "Cell = successes / valid episodes (scheduled). `inv` = invalid/unknown-status "
      "or infrastructure-skipped episodes (excluded from the denominator), `exc` = "
      "episodes carrying an exception (informational), `cond` = episodes excluded "
      f"because `catbench_condition != \"{C2_CONDITION}\"` or the condition config "
      "was invalid, `miss` = scheduled episodes with no checkpoint.",
      "",
  ]
  for replicate, counts in itertools.chain(per_replicate.items(), [("pooled", pooled)]):
    lines += [f"## Replicate: {replicate}", "",
              "| Category | App | " + " | ".join(cohort.models) + " |",
              "|---|---|" + "---|" * len(cohort.models)]
    for category in cohort.categories:
      for app_id in cohort.app_ids[category]:
        cells = []
        for model in cohort.models:
          cc = counts[(model, category, app_id)]
          extra = []
          if cc.invalid:
            extra.append(f"inv={cc.invalid}")
          if cc.exception:
            extra.append(f"exc={cc.exception}")
          if cc.excluded_condition:
            extra.append(f"cond={cc.excluded_condition}")
          if cc.missing:
            extra.append(f"miss={len(cc.missing)}")
          cells.append(f"{cc.success}/{cc.valid} ({cc.scheduled})" + (f" {' '.join(extra)}" if extra else ""))
        lines.append(
            f"| {CATEGORY_LABELS.get(category, category)} | "
            f"{cohort.app_display.get(app_id, app_id)} | " + " | ".join(cells) + " |"
        )
    totals = [aggregate(counts, [c for c in cohort.cells if c[0] == m]) for m in cohort.models]
    lines.append("| | **Overall** | " + " | ".join(
        f"**{t.success}/{t.valid} ({t.scheduled})**" for t in totals) + " |")
    lines.append("")
  md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replicate_agreement(per_replicate: dict[str, Counts], cohort: Cohort) -> dict[str, Any]:
  """Per model: success rate per replicate and mean |per-app rate difference|."""
  out: dict[str, Any] = {}
  replicates = list(per_replicate)
  for model in cohort.models:
    rates = {
        rep: aggregate(per_replicate[rep], [c for c in cohort.cells if c[0] == model]).rate
        for rep in replicates
    }
    diffs: list[float] = []
    pair_diffs: dict[str, float | None] = {}
    for a, b in itertools.combinations(replicates, 2):
      pair: list[float] = []
      for cell in (c for c in cohort.cells if c[0] == model):
        ra, rb = per_replicate[a][cell].rate, per_replicate[b][cell].rate
        if ra is not None and rb is not None:
          pair.append(abs(ra - rb))
      pair_diffs[f"{a}~{b}"] = statistics.mean(pair) if pair else None
      diffs.extend(pair)
    out[model] = {
        "rate_per_replicate": rates,
        "mean_abs_per_app_diff": statistics.mean(diffs) if diffs else None,
        "pairwise_mean_abs_per_app_diff": pair_diffs,
    }
  return out


def _counts_json(counts: Counts) -> list[dict[str, Any]]:
  return [
      {"model": m, "category": c, "app_id": a, **counts[(m, c, a)].as_dict()}
      for (m, c, a) in counts
  ]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run_variant(
    *,
    suffix: str,
    variant_label: str,
    instance_ids: Sequence[int],
    c2_records: Sequence[EpisodeRecord],
    c1_records: Sequence[EpisodeRecord] | None,
    c1_baseline: Path | None,
    cohort: Cohort,
    slug_map: dict[str, str],
    out_dir: Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
  """Computes and writes all outputs for one instance-selection variant."""
  notes: list[str] = []
  by_rep: dict[str, list[EpisodeRecord]] = collections.defaultdict(list)
  for rec in c2_records:
    by_rep[rec.replicate].append(rec)
  per_replicate: dict[str, Counts] = {}
  for rep in sorted(by_rep):
    per_replicate[rep], rep_notes = count_replicate(
        by_rep[rep], cohort, slug_map, C2_CONDITION, instance_ids
    )
    notes.extend(f"[{rep}] {n}" for n in rep_notes)
  pooled = pool_counts(per_replicate, cohort)

  if c1_baseline is not None:
    c1 = load_c1_baseline_json(c1_baseline, cohort)
    c1_source = {"kind": "k1_validated_baseline_json", "path": str(c1_baseline)}
  else:
    assert c1_records is not None
    c1_by_rep: dict[str, list[EpisodeRecord]] = collections.defaultdict(list)
    for rec in c1_records:
      c1_by_rep[rec.replicate].append(rec)
    c1_per_rep: dict[str, Counts] = {}
    for rep in sorted(c1_by_rep):
      c1_per_rep[rep], rep_notes = count_replicate(
          c1_by_rep[rep], cohort, slug_map, C1_CONDITION, instance_ids
      )
      notes.extend(f"[c1 {rep}] {n}" for n in rep_notes)
    c1 = pool_counts(c1_per_rep, cohort)
    c1_source = {"kind": "c1_root", "replicates": sorted(c1_per_rep)}

  rows = build_table(pooled, c1, cohort)
  write_counts(per_replicate, pooled, cohort,
               out_dir / f"c2_per_app_counts{suffix}.csv", out_dir / f"c2_per_app_counts{suffix}.md")
  write_delta_markdown(rows, cohort, pooled, c1, out_dir / f"c2_per_app_delta{suffix}.md", variant_label)
  write_delta_latex(rows, cohort, pooled, c1, out_dir / f"c2_per_app_delta{suffix}.tex", variant_label)

  overall = {
      m: aggregate(pooled, [c for c in cohort.cells if c[0] == m]).as_dict() for m in cohort.models
  }
  summary = {
      **meta,
      "variant": variant_label,
      "instance_ids": list(instance_ids),
      "c2_condition": C2_CONDITION,
      "c1_source": c1_source,
      "replicates": sorted(per_replicate),
      "c2_per_replicate": {rep: _counts_json(counts) for rep, counts in per_replicate.items()},
      "c2_pooled": _counts_json(pooled),
      "c2_overall_per_model": overall,
      "c1_counts": _counts_json(c1),
      "delta_table": [dataclasses.asdict(r) for r in rows],
      "replicate_agreement": replicate_agreement(per_replicate, cohort),
      "notes": notes,
  }
  (out_dir / f"c2_summary{suffix}.json").write_text(
      json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8"
  )
  return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--c2_root", action="append", required=True, type=Path)
  parser.add_argument("--c2_root_layout", choices=("nested", "flat"), default="nested")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--c1_baseline", type=Path, help="k1_validated_baseline.json")
  group.add_argument("--c1_root", action="append", type=Path, help="C1 batch root(s)")
  parser.add_argument("--c1_root_layout", choices=("nested", "flat"), default="nested")
  parser.add_argument("--model_config", type=Path, default=DEFAULT_MODEL_CONFIG)
  parser.add_argument("--cohort_manifest", type=Path, default=DEFAULT_COHORT)
  parser.add_argument("--out_dir", type=Path, required=True)
  parser.add_argument("--instance_ids", type=int, nargs="+", default=None,
                      help="Restrict to these instance ids (default: emit _k3 and _inst0).")
  parser.add_argument("--workers", type=int, default=8)
  parser.add_argument(
      "--exclude_semantic_tasks",
      nargs="+",
      default=(),
      metavar="TEMPLATE",
      help=(
          "Semantic templates to drop from the schedule entirely (not counted, "
          "not reported missing). For templates that cannot be scored in this "
          "environment; the exclusion is recorded in the outputs."
      ),
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  cohort = load_cohort(
      args.cohort_manifest,
      args.model_config,
      tuple(args.exclude_semantic_tasks),
  )
  args.out_dir.mkdir(parents=True, exist_ok=True)
  c2_roots = [p.expanduser().resolve() for p in args.c2_root]
  c1_roots = [p.expanduser().resolve() for p in (args.c1_root or [])]
  slug_map = model_slug_map(cohort, c2_roots + c1_roots)

  c2_records, errors = load_records(c2_roots, args.c2_root_layout, args.workers)
  c1_records: list[EpisodeRecord] | None = None
  if c1_roots:
    c1_records, c1_errors = load_records(c1_roots, args.c1_root_layout, args.workers)
    errors.extend(c1_errors)
  for err in errors:
    print(f"warning: {err}", file=sys.stderr)

  meta = {
      "c2_roots": [str(p) for p in c2_roots],
      "c2_root_layout": args.c2_root_layout,
      "c1_roots": [str(p) for p in c1_roots],
      "model_config": str(args.model_config),
      "cohort_manifest": str(args.cohort_manifest),
      "checkpoint_errors": errors,
      "c2_episode_records": len(c2_records),
      "excluded_semantic_tasks": list(args.exclude_semantic_tasks),
  }
  n_inst = 3
  variants: list[tuple[str, str, list[int]]]
  if args.instance_ids is None:
    variants = [
        ("_k3", "pooled K=3", list(range(n_inst))),
        ("_inst0", "instance 0 only", [0]),
    ]
  else:
    ids = sorted(set(args.instance_ids))
    variants = [(f"_inst{''.join(map(str, ids))}", f"instances {ids}", ids)]

  for suffix, label, ids in variants:
    summary = run_variant(
        suffix=suffix, variant_label=label, instance_ids=ids,
        c2_records=c2_records, c1_records=c1_records,
        c1_baseline=args.c1_baseline, cohort=cohort, slug_map=slug_map,
        out_dir=args.out_dir, meta=meta,
    )
    print(f"[{label}] C2 overall per model (success/valid, scheduled, missing):")
    for model, o in summary["c2_overall_per_model"].items():
      print(f"  {model:15s} {o['success']}/{o['valid']} (sched {o['scheduled']}, "
            f"missing {o['missing']}, invalid {o['invalid']}, cond-excluded {o['excluded_condition']})")
  print(f"Wrote outputs to {args.out_dir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
