#!/usr/bin/env python3
"""Report strict K=3 C1 app-level results from one frozen consumer state.

This reporter accepts only the fully validated primary schedule bundle and its
append-only consumer state. It never scans legacy manifests, chooses among
retries, or infers template provenance from class names. The whole-triplet
selection ledger chooses the replacement round, and the frozen cohort's
``semantic_origins`` map supplies the AW-adapted/CATBench-new stratum.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, BENCHMARK_ROOT):
  if str(import_root) not in sys.path:
    sys.path.insert(0, str(import_root))

import build_catbench_frozen_schedule as schedule_builder  # noqa: E402
import consume_catbench_frozen_schedule as consumer  # noqa: E402
from app_generalization_profiles import get_domain_profiles  # noqa: E402


ORIGIN_LABELS = {
    schedule_builder.SEMANTIC_ORIGIN_AW: "AW-adapted",
    schedule_builder.SEMANTIC_ORIGIN_NEW: "CATBench-new",
}


class AppLevelReportError(ValueError):
  """Raised when inputs cannot support a complete primary C1 table."""


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_keys(
    cohort: Mapping[str, Any],
) -> set[tuple[str, str, str, str, int]]:
  if int(cohort.get("n_task_combinations") or 0) != 3:
    raise AppLevelReportError("Primary C1 reporting requires frozen K=3")
  expected: set[tuple[str, str, str, str, int]] = set()
  for model in cohort["models"]:
    for category, spec in cohort["categories"].items():
      for app_id in spec["app_ids"]:
        for semantic_task_id in spec["semantic_task_ids"]:
          for instance_id in range(3):
            key = (
                str(model),
                str(category),
                str(app_id),
                str(semantic_task_id),
                instance_id,
            )
            if key in expected:
              raise AppLevelReportError(f"Duplicate frozen C1 key: {key!r}")
            expected.add(key)
  return expected


def _selected_c1_rows(
    bundle: consumer.FrozenBundle,
    state_dir: Path,
) -> list[dict[str, Any]]:
  if bundle.cohort.get("release_id") != schedule_builder.PRIMARY_RELEASE_ID:
    raise AppLevelReportError("Only the frozen primary release is reportable")
  if bundle.schedule_manifest.get("analysis_eligible") is not True:
    raise AppLevelReportError("Schedule manifest is not analysis-eligible")

  journal = consumer.Journal(state_dir / consumer.JOURNAL_FILE)
  recorded = consumer._read_jsonl(  # pylint: disable=protected-access
      state_dir / consumer.SELECTION_FILE
  )
  expected_selection = consumer.ReplacementState(
      bundle.schedule,
      bundle.ledger_seed,
      journal.finished,
  ).selections()
  if consumer._canonical_json(recorded) != consumer._canonical_json(  # pylint: disable=protected-access
      expected_selection
  ):
    raise AppLevelReportError(
        "selected_triplets.jsonl differs from deterministic journal replay"
    )

  expected_keys = _expected_keys(bundle.cohort)
  observed: set[tuple[str, str, str, str, int]] = set()
  rows: list[dict[str, Any]] = []
  for selection in recorded:
    if selection.get("selection_status") != "selected_complete_triplet":
      raise AppLevelReportError(
          f"Incomplete primary triplet selection: {selection.get('pair_id')!r}"
      )
    paired_key = selection.get("paired_key")
    if not isinstance(paired_key, Mapping):
      raise AppLevelReportError("Selection is missing paired_key")
    key = (
        str(paired_key.get("model") or ""),
        str(paired_key.get("category") or ""),
        str(paired_key.get("app_id") or ""),
        str(paired_key.get("semantic_task_id") or ""),
        int(paired_key.get("instance_id")),
    )
    if key not in expected_keys or key in observed:
      raise AppLevelReportError(f"Unexpected or duplicate selected key: {key!r}")
    observed.add(key)

    selected_ids = selection.get("selected_attempt_ids")
    if not isinstance(selected_ids, Mapping) or set(selected_ids) != set(
        schedule_builder.PRIMARY_CONDITIONS
    ):
      raise AppLevelReportError(f"Malformed selected triplet: {key!r}")
    attempt_id = str(selected_ids["c1"])
    finished = journal.finished.get(attempt_id)
    if not finished:
      raise AppLevelReportError(f"Selected C1 attempt is absent: {attempt_id}")
    status = finished.get("status")
    successful = finished.get("is_successful")
    if status == "valid_success" and successful == 1.0:
      success = 1
    elif status == "valid_failure" and successful == 0.0:
      success = 0
    else:
      raise AppLevelReportError(
          f"Selected C1 terminal contract is inconsistent: {attempt_id}"
      )
    provenance_key = (
        str(finished.get("model") or ""),
        str(finished.get("category") or ""),
        str(finished.get("app_id") or ""),
        str(finished.get("semantic_task_id") or ""),
        int(finished.get("instance_id")),
    )
    if provenance_key != key or finished.get("condition") != "c1":
      raise AppLevelReportError(
          f"Selected C1 journal provenance mismatch: {attempt_id}"
      )
    category, semantic_task_id = key[1], key[3]
    origin = bundle.cohort["categories"][category]["semantic_origins"].get(
        semantic_task_id
    )
    if origin != schedule_builder.semantic_origin(category, semantic_task_id):
      raise AppLevelReportError(
          f"Semantic-origin mismatch: {category}/{semantic_task_id}"
      )
    rows.append({
        "model": key[0],
        "category": category,
        "app_id": key[2],
        "semantic_task_id": semantic_task_id,
        "instance_id": key[4],
        "semantic_origin": origin,
        "success": success,
    })

  missing = expected_keys - observed
  if missing:
    raise AppLevelReportError(
        f"Primary C1 selection is missing {len(missing)} frozen keys"
    )
  return rows


def aggregate_rows(
    cohort: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]
) -> dict[tuple[str, str, str, str], tuple[int, int]]:
  """Aggregate complete C1 rows, validating every K=3 origin denominator."""
  counters: dict[
      tuple[str, str, str, str], collections.Counter[str]
  ] = collections.defaultdict(collections.Counter)
  seen: set[tuple[str, str, str, str, int]] = set()
  for row in rows:
    identity = (
        str(row["model"]),
        str(row["category"]),
        str(row["app_id"]),
        str(row["semantic_task_id"]),
        int(row["instance_id"]),
    )
    if identity in seen:
      raise AppLevelReportError(f"Duplicate C1 result row: {identity!r}")
    seen.add(identity)
    origin = str(row["semantic_origin"])
    success = int(row["success"])
    if origin not in ORIGIN_LABELS or success not in (0, 1):
      raise AppLevelReportError(f"Invalid C1 report row: {dict(row)!r}")
    key = (identity[0], identity[1], identity[2], origin)
    counters[key]["scheduled"] += 1
    counters[key]["success"] += success

  result: dict[tuple[str, str, str, str], tuple[int, int]] = {}
  for model in cohort["models"]:
    for category, spec in cohort["categories"].items():
      expected_by_origin = collections.Counter(spec["semantic_origins"].values())
      for app_id in spec["app_ids"]:
        for origin in ORIGIN_LABELS:
          expected_denominator = 3 * expected_by_origin[origin]
          counter = counters.get((model, category, app_id, origin))
          actual_denominator = int(counter["scheduled"]) if counter else 0
          if actual_denominator != expected_denominator:
            raise AppLevelReportError(
                "Incomplete K=3 app/origin cell for "
                f"{model}/{category}/{app_id}/{origin}: "
                f"{actual_denominator}, expected {expected_denominator}"
            )
          result[(model, category, app_id, origin)] = (
              int(counter["success"]),
              actual_denominator,
          )
  return result


def _cell(value: tuple[int, int]) -> str:
  success, denominator = value
  return f"{success}/{denominator} ({100.0 * success / denominator:.1f}%)"


def render_markdown(
    cohort: Mapping[str, Any],
    aggregated: Mapping[tuple[str, str, str, str], tuple[int, int]],
    *,
    cohort_sha256: str,
    schedule_manifest_sha256: str,
    attempt_journal_sha256: str,
    selected_triplets_sha256: str,
) -> str:
  profiles = get_domain_profiles()
  lines = [
      "# Frozen K=3 C1 app-level results",
      "",
      (
          "This report uses only the frozen whole-triplet selection state. "
          "AW-adapted/CATBench-new denotes semantic-template lineage, not app "
          "provenance. Each denominator is three fixed parameter instances "
          "per semantic template; infrastructure replacements do not enlarge it."
      ),
      "",
      f"- Cohort SHA-256: `{cohort_sha256}`",
      f"- Schedule manifest SHA-256: `{schedule_manifest_sha256}`",
      f"- Attempt journal SHA-256: `{attempt_journal_sha256}`",
      f"- Selected-triplets SHA-256: `{selected_triplets_sha256}`",
      (
          "- C1 selected episodes: "
          f"{sum(value[1] for value in aggregated.values()):,}"
      ),
      (
          "- SMS lineage: 5 AW-adapted and 5 CATBench-new templates; "
          "`SmsSendToContact` is CATBench-new and does not inherit the removed "
          "`SmsSendClipboard` label."
      ),
      "",
  ]
  for category, spec in cohort["categories"].items():
    display_names = {
        app.app_id: app.display_name for app in profiles[category].apps
    }
    origin_counts = collections.Counter(spec["semantic_origins"].values())
    lines.extend([
        f"## {profiles[category].domain}",
        "",
        (
            f"Per app: {3 * origin_counts[schedule_builder.SEMANTIC_ORIGIN_AW]} "
            "AW-adapted episodes and "
            f"{3 * origin_counts[schedule_builder.SEMANTIC_ORIGIN_NEW]} "
            "CATBench-new episodes."
        ),
        "",
        "| Model | App | AW-adapted | CATBench-new |",
        "|:--|:--|--:|--:|",
    ])
    for model in cohort["models"]:
      for app_id in spec["app_ids"]:
        aw = aggregated[(model, category, app_id, schedule_builder.SEMANTIC_ORIGIN_AW)]
        new = aggregated[(model, category, app_id, schedule_builder.SEMANTIC_ORIGIN_NEW)]
        lines.append(
            f"| {model} | {display_names[app_id]} | {_cell(aw)} | {_cell(new)} |"
        )
    lines.append("")
  return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cohort", type=Path, required=True)
  parser.add_argument("--schedule-dir", type=Path, required=True)
  parser.add_argument("--state-dir", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  bundle = consumer.load_and_validate_bundle(
      args.schedule_dir.resolve(), args.cohort.resolve()
  )
  rows = _selected_c1_rows(bundle, args.state_dir.resolve())
  aggregated = aggregate_rows(bundle.cohort, rows)
  output = render_markdown(
      bundle.cohort,
      aggregated,
      cohort_sha256=bundle.cohort_sha256,
      schedule_manifest_sha256=bundle.schedule_manifest_sha256,
      attempt_journal_sha256=_sha256(
          args.state_dir.resolve() / consumer.JOURNAL_FILE
      ),
      selected_triplets_sha256=_sha256(
          args.state_dir.resolve() / consumer.SELECTION_FILE
      ),
  )
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(output + "\n", encoding="utf-8")
  print(f"Wrote {args.out} from {len(rows)} selected C1 episodes")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
