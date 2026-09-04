#!/usr/bin/env python3
"""Compare two failure-mode judges on an exact matched CATBench cohort.

The Gemini side is the frozen case-level CSV produced by
``merge_failure_judge_results_for_bar_chart.py``.  The Qwen side is a JSONL
whose rows contain ``episode_id`` and a judgment either under ``judgment`` or
``qwen_judgment`` (a top-level ``primary_failure_mode`` is also accepted).

This script measures *cross-judge robustness*, not accuracy.  Rows with a
missing/failed/malformed Qwen judgment are reported in coverage outputs and
excluded from agreement statistics; they are never converted to ``unknown``.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence


FAILURE_MODES = (
    "planning",
    "grounding",
    "mixed_planning_grounding",
    "execution_tooling",
    "environment_or_evaluator",
    "unknown",
)
DEFAULT_CATEGORIES = ("clock", "contacts", "files", "maps", "sms")
ERROR_STATUSES = {"error", "failed", "timeout", "exception", "cancelled"}
BOOTSTRAP_SEED = 20260712
BOOTSTRAP_CLUSTER_FIELDS = (
    "model_name",
    "category",
    "app_id",
    "task_template",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
      if not isinstance(row, dict):
        raise ValueError(f"Expected a JSON object at {path}:{line_number}")
      row = dict(row)
      row["_input_line_number"] = line_number
      rows.append(row)
  return rows


def _load_gemini_csv(path: Path, categories: set[str]) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    required = {"episode_id", "model_name", "category", "primary_failure_mode"}
    missing = required - set(reader.fieldnames or [])
    if missing:
      raise ValueError(f"Gemini CSV is missing columns: {sorted(missing)}")
    rows = [dict(row) for row in reader if row.get("category") in categories]

  seen: set[str] = set()
  duplicates: list[str] = []
  for row in rows:
    episode_id = str(row.get("episode_id") or "")
    if not episode_id:
      raise ValueError("Gemini CSV contains an empty episode_id")
    if episode_id in seen:
      duplicates.append(episode_id)
    seen.add(episode_id)
    label = str(row.get("primary_failure_mode") or "")
    if label not in FAILURE_MODES:
      raise ValueError(
          f"Gemini row {episode_id} has invalid failure mode {label!r}"
      )
  if duplicates:
    preview = ", ".join(duplicates[:5])
    raise ValueError(f"Gemini CSV has duplicate episode_id values: {preview}")
  return rows


def _is_error_value(value: Any) -> bool:
  if value is None or value is False or value == 0:
    return False
  if isinstance(value, str):
    return value.strip().lower() not in {
        "",
        "0",
        "false",
        "none",
        "null",
        "ok",
        "success",
    }
  if isinstance(value, (list, tuple, dict, set)):
    return bool(value)
  return bool(value)


def _error_text(value: Any) -> str:
  if isinstance(value, str):
    return value.strip()
  try:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
  except TypeError:
    return str(value)


def _judgment_object(row: dict[str, Any]) -> dict[str, Any]:
  for key in ("qwen_judgment", "judgment"):
    candidate = row.get(key)
    if isinstance(candidate, dict):
      return candidate
  return row


def _nonnegative_int(value: Any) -> int | None:
  """Parse an explicit image count without treating a falsey value as zero."""
  if value is None or value == "" or isinstance(value, bool):
    return None
  if isinstance(value, float) and not value.is_integer():
    return None
  try:
    parsed = int(value)
  except (TypeError, ValueError, OverflowError):
    return None
  return parsed if parsed >= 0 else None


def _historical_evidence(row: dict[str, Any]) -> dict[str, Any]:
  """Recover the evidence Gemini saw from an explicit historical count only.

  The rejudge output normally copies the historical row's ``usage`` into
  ``gemini_usage``.  ``evidence_parity`` is the audited fallback.  Qwen's own
  image count is intentionally not used: it cannot establish what Gemini saw.
  """
  gemini_usage = row.get("gemini_usage")
  candidates: list[tuple[str, Any]] = []
  if isinstance(gemini_usage, dict):
    candidates.append(("gemini_usage.num_images", gemini_usage.get("num_images")))
  parity = row.get("evidence_parity")
  if isinstance(parity, dict):
    candidates.append(
        (
            "evidence_parity.historical_gemini_num_images",
            parity.get("historical_gemini_num_images"),
        )
    )
  parsed_candidates = [
      (source, count)
      for source, raw_count in candidates
      if (count := _nonnegative_int(raw_count)) is not None
  ]
  if parsed_candidates:
    source, count = parsed_candidates[0]
    return {
        "historical_num_images": count,
        "evidence_type": "zero_image" if count == 0 else "visual",
        "evidence_count_source": source,
        "evidence_count_conflict": len({value for _, value in parsed_candidates}) > 1,
    }
  return {
      "historical_num_images": "",
      "evidence_type": "unknown",
      "evidence_count_source": "",
      "evidence_count_conflict": False,
  }


def _extract_qwen(row: dict[str, Any]) -> dict[str, Any]:
  judgment = _judgment_object(row)
  errors: list[str] = []

  status = str(row.get("status") or "").strip().lower()
  if status in ERROR_STATUSES:
    errors.append(f"status={status}")
  for owner, key in (
      (row, "qwen_error"),
      (row, "judge_error"),
      (row, "error"),
      (judgment, "error"),
  ):
    value = owner.get(key)
    if _is_error_value(value):
      text = _error_text(value)
      errors.append(f"{key}={text}" if text else key)

  raw_label = judgment.get("primary_failure_mode")
  if raw_label is None and judgment is not row:
    raw_label = row.get("primary_failure_mode")
  label = str(raw_label or "").strip()
  if not label:
    errors.append("missing_primary_failure_mode")
  elif label not in FAILURE_MODES:
    errors.append(f"invalid_primary_failure_mode={label}")

  usage = row.get("qwen_usage")
  if not isinstance(usage, dict):
    usage = row.get("usage")
  if not isinstance(usage, dict):
    usage = judgment.get("_usage")
  if not isinstance(usage, dict):
    usage = {}

  evidence = judgment.get("evidence")
  if not isinstance(evidence, list):
    evidence = []
  historical_evidence = _historical_evidence(row)
  return {
      "label": label if label in FAILURE_MODES else "",
      "error": "; ".join(dict.fromkeys(errors)),
      "status": status or ("error" if errors else "ok"),
      "confidence": str(judgment.get("confidence") or ""),
      "planning_score": judgment.get("planning_score", ""),
      "grounding_score": judgment.get("grounding_score", ""),
      "rationale": str(judgment.get("rationale") or ""),
      "evidence": evidence,
      "num_images": usage.get("num_images", ""),
      "model": usage.get("model") or row.get("judge_model") or "",
      "input_line_number": row.get("_input_line_number", ""),
      **historical_evidence,
  }


def _index_qwen_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  by_id: dict[str, dict[str, Any]] = {}
  counts: collections.Counter[str] = collections.Counter()
  conflicting_ids: set[str] = set()
  prior_labels: dict[str, tuple[str, str]] = {}
  rows_without_id = 0
  for row in rows:
    payload = row.get("case_payload")
    episode_id = str(
        row.get("episode_id")
        or (payload.get("episode_id") if isinstance(payload, dict) else "")
        or ""
    )
    if not episode_id:
      rows_without_id += 1
      continue
    counts[episode_id] += 1
    extracted = _extract_qwen(row)
    signature = (extracted["label"], extracted["error"])
    if episode_id in prior_labels and prior_labels[episode_id] != signature:
      conflicting_ids.add(episode_id)
    prior_labels[episode_id] = signature
    by_id[episode_id] = row  # Last JSONL row wins; duplicates are reported.
  duplicate_ids = {episode_id for episode_id, count in counts.items() if count > 1}
  audit = {
      "qwen_rows_without_episode_id": rows_without_id,
      "qwen_duplicate_episode_ids": len(duplicate_ids),
      "qwen_duplicate_rows": sum(counts[eid] - 1 for eid in duplicate_ids),
      "qwen_conflicting_duplicate_episode_ids": len(conflicting_ids),
  }
  return by_id, audit


def _safe_int(value: Any) -> int | str:
  if value is None or value == "":
    return ""
  try:
    return int(value)
  except (TypeError, ValueError):
    return str(value)


def _pair_rows(
    gemini_rows: list[dict[str, str]],
    qwen_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
  paired: list[dict[str, Any]] = []
  for gemini in gemini_rows:
    episode_id = gemini["episode_id"]
    qwen_raw = qwen_by_id.get(episode_id)
    if qwen_raw is None:
      qwen = {
          "label": "",
          "error": "missing_qwen_row",
          "status": "missing",
          "confidence": "",
          "planning_score": "",
          "grounding_score": "",
          "rationale": "",
          "evidence": [],
          "num_images": "",
          "model": "",
          "input_line_number": "",
          "historical_num_images": "",
          "evidence_type": "unknown",
          "evidence_count_source": "",
          "evidence_count_conflict": False,
      }
    else:
      qwen = _extract_qwen(qwen_raw)
    valid_pair = not qwen["error"] and qwen["label"] in FAILURE_MODES
    agrees = valid_pair and gemini["primary_failure_mode"] == qwen["label"]
    paired.append(
        {
            "episode_id": episode_id,
            "model_name": gemini.get("model_name", ""),
            "category": gemini.get("category", ""),
            "app_id": gemini.get("app_id", ""),
            "app_name": gemini.get("app_name", ""),
            "task_template": gemini.get("task_template", ""),
            "pkl_path": gemini.get("pkl_path", ""),
            "gemini_jsonl_path": gemini.get("jsonl_path", ""),
            "gemini_label": gemini["primary_failure_mode"],
            "gemini_confidence": gemini.get("confidence", ""),
            "gemini_planning_score": _safe_int(gemini.get("planning_score")),
            "gemini_grounding_score": _safe_int(gemini.get("grounding_score")),
            "gemini_rationale": gemini.get("rationale", ""),
            "qwen_label": qwen["label"],
            "qwen_confidence": qwen["confidence"],
            "qwen_planning_score": _safe_int(qwen["planning_score"]),
            "qwen_grounding_score": _safe_int(qwen["grounding_score"]),
            "qwen_rationale": qwen["rationale"],
            "qwen_evidence": qwen["evidence"],
            "qwen_num_images": _safe_int(qwen["num_images"]),
            "qwen_model": qwen["model"],
            "qwen_status": qwen["status"],
            "qwen_error": qwen["error"],
            "qwen_input_line_number": qwen["input_line_number"],
            "historical_num_images": _safe_int(qwen["historical_num_images"]),
            "evidence_type": qwen["evidence_type"],
            "evidence_count_source": qwen["evidence_count_source"],
            "evidence_count_conflict": qwen["evidence_count_conflict"],
            "valid_pair": valid_pair,
            "agrees": agrees if valid_pair else None,
            "transition": (
                f"{gemini['primary_failure_mode']}->{qwen['label']}"
                if valid_pair
                else ""
            ),
        }
    )
  return paired


def _valid_pairs(rows: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
  return [
      (row["gemini_label"], row["qwen_label"])
      for row in rows
      if row["valid_pair"]
  ]


def _agreement_metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
  confusion = {
      gemini: {qwen: 0 for qwen in FAILURE_MODES} for gemini in FAILURE_MODES
  }
  for gemini, qwen in pairs:
    confusion[gemini][qwen] += 1
  n = len(pairs)
  observed = sum(confusion[label][label] for label in FAILURE_MODES) / n if n else None
  gemini_marginal = collections.Counter(gemini for gemini, _ in pairs)
  qwen_marginal = collections.Counter(qwen for _, qwen in pairs)
  expected = (
      sum(
          (gemini_marginal[label] / n) * (qwen_marginal[label] / n)
          for label in FAILURE_MODES
      )
      if n
      else None
  )
  kappa: float | None
  if observed is None or expected is None or math.isclose(expected, 1.0):
    kappa = None
  else:
    kappa = (observed - expected) / (1.0 - expected)
  row_percent: dict[str, dict[str, float | None]] = {}
  for gemini in FAILURE_MODES:
    total = sum(confusion[gemini].values())
    row_percent[gemini] = {
        qwen: (100.0 * confusion[gemini][qwen] / total if total else None)
        for qwen in FAILURE_MODES
    }
  return {
      "n": n,
      "raw_agreement": observed,
      "expected_chance_agreement": expected,
      "cohens_kappa": kappa,
      "confusion_counts": confusion,
      "confusion_row_percent": row_percent,
  }


def _percentile(values: Sequence[float], probability: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  position = (len(ordered) - 1) * probability
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  fraction = position - lower
  return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _percentile_interval(values: Sequence[float]) -> list[float] | None:
  lower = _percentile(values, 0.025)
  upper = _percentile(values, 0.975)
  if lower is None or upper is None:
    return None
  return [lower, upper]


def _cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    replicates: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
  """Paired cluster bootstrap for the two overall agreement estimands."""
  if replicates < 0:
    raise ValueError("bootstrap_replicates must be nonnegative")
  valid = [row for row in rows if row["valid_pair"]]
  grouped: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
  for row in valid:
    missing_fields = [
        field for field in BOOTSTRAP_CLUSTER_FIELDS if not str(row.get(field) or "")
    ]
    if missing_fields:
      raise ValueError(
          f"Cannot cluster episode {row.get('episode_id', '')}: missing "
          f"{', '.join(missing_fields)}"
      )
    key = tuple(str(row.get(field) or "") for field in BOOTSTRAP_CLUSTER_FIELDS)
    grouped[key].append(row)
  keys = sorted(grouped)
  result: dict[str, Any] = {
      "method": "paired_cluster_percentile",
      "confidence_level": 0.95,
      "seed": seed,
      "cluster_fields": list(BOOTSTRAP_CLUSTER_FIELDS),
      "clusters": len(keys),
      "valid_pairs": len(valid),
      "replicates_requested": replicates,
      "raw_agreement_replicates": 0,
      "cohens_kappa_replicates": 0,
      "raw_agreement_ci95": None,
      "cohens_kappa_ci95": None,
  }
  if not keys or replicates == 0:
    return result

  rng = random.Random(seed)
  agreement_samples: list[float] = []
  kappa_samples: list[float] = []
  for _ in range(replicates):
    sampled_rows: list[dict[str, Any]] = []
    for _ in keys:
      sampled_rows.extend(grouped[rng.choice(keys)])
    metrics = _agreement_metrics(_valid_pairs(sampled_rows))
    agreement = metrics["raw_agreement"]
    kappa = metrics["cohens_kappa"]
    if agreement is not None:
      agreement_samples.append(float(agreement))
    if kappa is not None:
      kappa_samples.append(float(kappa))

  result.update(
      {
          "raw_agreement_replicates": len(agreement_samples),
          "cohens_kappa_replicates": len(kappa_samples),
          "raw_agreement_ci95": _percentile_interval(agreement_samples),
          "cohens_kappa_ci95": _percentile_interval(kappa_samples),
      }
  )
  return result


def _marginal_rows(
    rows: list[dict[str, Any]], scope: str, value: str
) -> list[dict[str, Any]]:
  pairs = _valid_pairs(rows)
  n = len(pairs)
  gemini_counts = collections.Counter(gemini for gemini, _ in pairs)
  qwen_counts = collections.Counter(qwen for _, qwen in pairs)
  out: list[dict[str, Any]] = []
  for label in FAILURE_MODES:
    gemini_pct = 100.0 * gemini_counts[label] / n if n else None
    qwen_pct = 100.0 * qwen_counts[label] / n if n else None
    out.append(
        {
            "scope": scope,
            "scope_value": value,
            "label": label,
            "valid_pair_n": n,
            "gemini_count": gemini_counts[label],
            "gemini_percent": gemini_pct,
            "qwen_count": qwen_counts[label],
            "qwen_percent": qwen_pct,
            "delta_pp_qwen_minus_gemini": (
                qwen_pct - gemini_pct
                if qwen_pct is not None and gemini_pct is not None
                else None
            ),
        }
    )
  return out


def _scope_summary(
    rows: list[dict[str, Any]], field: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  for row in rows:
    groups[str(row.get(field) or "")].append(row)
  wide: list[dict[str, Any]] = []
  long: list[dict[str, Any]] = []
  for value in sorted(groups):
    group = groups[value]
    metrics = _agreement_metrics(_valid_pairs(group))
    summary: dict[str, Any] = {
        field: value,
        "roster_n": len(group),
        "valid_pair_n": metrics["n"],
        "excluded_n": len(group) - metrics["n"],
        "raw_agreement": metrics["raw_agreement"],
        "cohens_kappa": metrics["cohens_kappa"],
    }
    shifts = _marginal_rows(group, field, value)
    for shift in shifts:
      label = shift["label"]
      summary[f"gemini_{label}_count"] = shift["gemini_count"]
      summary[f"gemini_{label}_percent"] = shift["gemini_percent"]
      summary[f"qwen_{label}_count"] = shift["qwen_count"]
      summary[f"qwen_{label}_percent"] = shift["qwen_percent"]
      summary[f"delta_{label}_pp"] = shift["delta_pp_qwen_minus_gemini"]
    wide.append(summary)
    long.extend(shifts)
  return wide, long


def _jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
  return dict(row)


def _csv_value(value: Any) -> Any:
  if value is None:
    return ""
  if isinstance(value, (list, dict)):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
  return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
  if fieldnames is None:
    fieldnames = list(rows[0].keys()) if rows else []
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(_jsonable_row(row), ensure_ascii=False) + "\n")


def _write_confusion_csvs(out_dir: Path, metrics: dict[str, Any]) -> None:
  count_rows: list[dict[str, Any]] = []
  percent_rows: list[dict[str, Any]] = []
  for gemini in FAILURE_MODES:
    counts = metrics["confusion_counts"][gemini]
    percentages = metrics["confusion_row_percent"][gemini]
    count_rows.append(
        {
            "gemini_label": gemini,
            **counts,
            "row_total": sum(counts.values()),
        }
    )
    percent_rows.append(
        {"gemini_label": gemini, **percentages}
    )
  fields = ["gemini_label", *FAILURE_MODES, "row_total"]
  _write_csv(out_dir / "confusion_counts.csv", count_rows, fields)
  _write_csv(
      out_dir / "confusion_row_percent.csv",
      percent_rows,
      ["gemini_label", *FAILURE_MODES],
  )


def _fmt_metric(value: Any) -> str:
  return "undefined" if value is None else f"{float(value):.3f}"


def _fmt_ci(value: Any) -> str:
  if not isinstance(value, list) or len(value) != 2:
    return "not computed"
  return f"[{_fmt_metric(value[0])}, {_fmt_metric(value[1])}]"


def _write_markdown(
    path: Path,
    coverage: dict[str, Any],
    metrics: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
) -> None:
  bootstrap = metrics.get("bootstrap") or {}
  lines = [
      "# Qwen vs. Gemini Failure-Judge Robustness",
      "",
      "This is paired cross-judge agreement on verifier-failed episodes; it is not judge accuracy.",
      "",
      f"- Frozen Gemini roster: {coverage['gemini_roster_cases']}",
      f"- Valid matched pairs: {coverage['valid_pairs']}",
      f"- Excluded Qwen missing/error/malformed rows: {coverage['excluded_pairs']}",
      f"- Raw agreement: {_fmt_metric(metrics['raw_agreement'])}",
      f"- Raw agreement 95% cluster-bootstrap CI: "
      f"{_fmt_ci(bootstrap.get('raw_agreement_ci95'))}",
      f"- Cohen's kappa (unweighted, six nominal classes): {_fmt_metric(metrics['cohens_kappa'])}",
      f"- Cohen's kappa 95% cluster-bootstrap CI: "
      f"{_fmt_ci(bootstrap.get('cohens_kappa_ci95'))}",
      f"- Bootstrap: {bootstrap.get('replicates_requested', 0)} replicates, "
      f"{bootstrap.get('clusters', 0)} clusters, seed "
      f"{bootstrap.get('seed', BOOTSTRAP_SEED)}; clusters are "
      f"`{' / '.join(BOOTSTRAP_CLUSTER_FIELDS)}`.",
      "- The intervals condition on Qwen rows with valid judgments; they do "
      "not quantify missing-response uncertainty or judge accuracy.",
      "",
      "## By historical Gemini evidence",
      "",
      "`zero_image` means Gemini received no screenshot; `visual` means it "
      "received at least one. Missing explicit historical counts remain "
      "`unknown` and are never treated as zero.",
      "",
      "| Evidence | Roster | Valid | Excluded | Agreement | Kappa |",
      "|---|---:|---:|---:|---:|---:|",
  ]
  for row in evidence_rows:
    lines.append(
        f"| {row['evidence_type']} | {row['roster_n']} | "
        f"{row['valid_pair_n']} | {row['excluded_n']} | "
        f"{_fmt_metric(row['raw_agreement'])} | "
        f"{_fmt_metric(row['cohens_kappa'])} |"
    )
  lines.extend(
      [
      "",
      "## By category",
      "",
      "| Category | Roster | Valid | Excluded | Agreement | Kappa |",
      "|---|---:|---:|---:|---:|---:|",
      ]
  )
  for row in category_rows:
    lines.append(
        f"| {row['category']} | {row['roster_n']} | {row['valid_pair_n']} | "
        f"{row['excluded_n']} | {_fmt_metric(row['raw_agreement'])} | "
        f"{_fmt_metric(row['cohens_kappa'])} |"
    )
  lines.extend(
      [
          "",
          "## By evaluated model",
          "",
          "| Model | Roster | Valid | Excluded | Agreement | Kappa |",
          "|---|---:|---:|---:|---:|---:|",
      ]
  )
  for row in model_rows:
    lines.append(
        f"| {row['model_name']} | {row['roster_n']} | {row['valid_pair_n']} | "
        f"{row['excluded_n']} | {_fmt_metric(row['raw_agreement'])} | "
        f"{_fmt_metric(row['cohens_kappa'])} |"
    )
  lines.extend(
      [
          "",
          "See `confusion_counts.csv` and the label-shift CSVs for the direction of disagreements.",
          "",
      ]
  )
  path.write_text("\n".join(lines), encoding="utf-8")


def _build_coverage(
    gemini_rows: list[dict[str, str]],
    qwen_rows: list[dict[str, Any]],
    qwen_by_id: dict[str, dict[str, Any]],
    paired: list[dict[str, Any]],
    qwen_audit: dict[str, Any],
    categories: list[str],
) -> dict[str, Any]:
  gemini_ids = {row["episode_id"] for row in gemini_rows}
  qwen_ids = set(qwen_by_id)
  missing = [row for row in paired if row["qwen_status"] == "missing"]
  malformed = [
      row
      for row in paired
      if row["qwen_status"] != "missing"
      and row["qwen_error"]
      and (
          "missing_primary_failure_mode" in row["qwen_error"]
          or "invalid_primary_failure_mode" in row["qwen_error"]
      )
  ]
  explicit_errors = [
      row
      for row in paired
      if row["qwen_status"] != "missing"
      and row["qwen_error"]
      and row not in malformed
  ]
  valid = [row for row in paired if row["valid_pair"]]
  return {
      "categories": categories,
      "labels": list(FAILURE_MODES),
      "gemini_roster_cases": len(gemini_rows),
      "qwen_jsonl_rows": len(qwen_rows),
      "qwen_unique_episode_ids": len(qwen_ids),
      "matched_qwen_episode_ids": len(gemini_ids & qwen_ids),
      "valid_pairs": len(valid),
      "excluded_pairs": len(paired) - len(valid),
      "missing_qwen_rows": len(missing),
      "qwen_explicit_error_pairs": len(explicit_errors),
      "qwen_invalid_or_missing_label_pairs": len(malformed),
      "qwen_extra_episode_ids": len(qwen_ids - gemini_ids),
      "historical_evidence_unknown_pairs": sum(
          row["evidence_type"] == "unknown" for row in paired
      ),
      "historical_evidence_count_conflicts": sum(
          bool(row["evidence_count_conflict"]) for row in paired
      ),
      **qwen_audit,
  }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--gemini_csv", type=Path, required=True)
  parser.add_argument("--qwen_jsonl", type=Path, required=True)
  parser.add_argument("--out_dir", type=Path, required=True)
  parser.add_argument(
      "--categories",
      nargs="+",
      default=list(DEFAULT_CATEGORIES),
      help="Frozen categories to select from the Gemini merged CSV.",
  )
  parser.add_argument(
      "--expected_n",
      type=int,
      default=0,
      help="If nonzero, fail unless the filtered Gemini roster has this size.",
  )
  parser.add_argument(
      "--bootstrap_replicates",
      type=int,
      default=2000,
      help=(
          "Paired cluster-bootstrap replicates for overall agreement and kappa "
          "(default: 2000; use 0 to disable)."
      ),
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  categories = list(dict.fromkeys(args.categories))
  gemini_rows = _load_gemini_csv(args.gemini_csv, set(categories))
  if args.expected_n and len(gemini_rows) != args.expected_n:
    raise ValueError(
        f"Expected {args.expected_n} Gemini cases, found {len(gemini_rows)}"
    )
  qwen_rows = _load_jsonl(args.qwen_jsonl)
  qwen_by_id, qwen_audit = _index_qwen_rows(qwen_rows)
  paired = _pair_rows(gemini_rows, qwen_by_id)
  metrics = _agreement_metrics(_valid_pairs(paired))
  bootstrap = _cluster_bootstrap(paired, args.bootstrap_replicates)
  metrics["bootstrap"] = bootstrap
  metrics["raw_agreement_ci95"] = bootstrap["raw_agreement_ci95"]
  metrics["cohens_kappa_ci95"] = bootstrap["cohens_kappa_ci95"]
  evidence_rows, _ = _scope_summary(paired, "evidence_type")
  evidence_order = {"zero_image": 0, "visual": 1, "unknown": 2}
  evidence_rows.sort(key=lambda row: evidence_order.get(row["evidence_type"], 99))
  category_rows, category_shifts = _scope_summary(paired, "category")
  model_rows, model_shifts = _scope_summary(paired, "model_name")
  overall_shifts = _marginal_rows(paired, "overall", "all")
  coverage = _build_coverage(
      gemini_rows,
      qwen_rows,
      qwen_by_id,
      paired,
      qwen_audit,
      categories,
  )

  out_dir = args.out_dir.expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "coverage.json").write_text(
      json.dumps(coverage, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  (out_dir / "overall.json").write_text(
      json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  _write_jsonl(out_dir / "paired_cases.jsonl", paired)
  _write_csv(out_dir / "paired_cases.csv", paired)
  _write_jsonl(
      out_dir / "excluded_cases.jsonl",
      (row for row in paired if not row["valid_pair"]),
  )
  extra_rows = [
      {
          "episode_id": episode_id,
          "input_line_number": qwen_by_id[episode_id].get("_input_line_number", ""),
      }
      for episode_id in sorted(set(qwen_by_id) - {row["episode_id"] for row in gemini_rows})
  ]
  _write_jsonl(out_dir / "unmatched_qwen_rows.jsonl", extra_rows)
  _write_confusion_csvs(out_dir, metrics)
  _write_csv(out_dir / "label_shift_overall.csv", overall_shifts)
  _write_csv(out_dir / "by_evidence.csv", evidence_rows)
  _write_csv(out_dir / "by_category.csv", category_rows)
  _write_csv(out_dir / "label_shift_by_category.csv", category_shifts)
  _write_csv(out_dir / "by_model.csv", model_rows)
  _write_csv(out_dir / "label_shift_by_model.csv", model_shifts)
  _write_markdown(
      out_dir / "overall.md",
      coverage,
      metrics,
      evidence_rows,
      category_rows,
      model_rows,
  )
  print(
      f"Compared {coverage['valid_pairs']}/{coverage['gemini_roster_cases']} "
      f"valid matched episodes; outputs: {out_dir}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
