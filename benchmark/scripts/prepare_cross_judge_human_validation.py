#!/usr/bin/env python3
"""Freeze the CATBench C1 human-validation sample used in the paper.

The source is the completed paired Gemini/Qwen C1 judgment JSONL.  Sampling
is intentionally blind to every Qwen field and to Gemini--Qwen agreement:
only episode metadata plus the *pre-specified* Gemini label and confidence
may affect inclusion.  The server-facing manifests contain no judge labels,
scores, evidence, or rationales; those values live only in private outputs.

The default design is binding rather than aspirational:

* 200 unique primary failures;
* exact audited model x category and category x Gemini-label margins;
* 20 uniformly sampled primary duplicates for inter-rater agreement; and
* 20 disjoint calibration cases (four per category) outside the primary set.

This script prepares artifacts only.  It never starts the annotation server.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import dataclasses
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "revision_artifacts"
    / "vlm_judge"
    / "qwen3vl30b_c1_5cat_full"
    / "failure_mode_judgments.jsonl"
)

MAIN_SEED = 20260712
DOUBLE_ANNOTATION_SEED = 20260713
PILOT_SEED = 20260711
EXPECTED_SOURCE_ROWS = 2195
PRIMARY_N = 200
DOUBLE_ANNOTATION_N = 20
PILOT_N = 20

CATEGORIES = ("clock", "contacts", "files", "maps", "sms")
FAILURE_MODES = (
    "planning",
    "grounding",
    "mixed_planning_grounding",
    "execution_tooling",
    "environment_or_evaluator",
    "unknown",
)
CONFIDENCE_WEIGHTS = {"low": 3.0, "medium": 1.5, "high": 1.0}

AW_APP_BY_CATEGORY = {
    "clock": "clock_google_clock",
    "contacts": "contacts_google_contacts",
    "files": "files_material_files",
    "maps": "maps_osmand",
    "sms": "sms_simple_sms_messenger",
}

# This allocation was audited against the frozen 2,195-row source.  It gives
# every category exactly 40 primary cases and each model exactly 15 or 16.
MODEL_CATEGORY_QUOTAS: dict[str, dict[str, int]] = {
    "AutoDev": dict(zip(CATEGORIES, (3, 3, 3, 3, 3))),
    "GPT-5.1": dict(zip(CATEGORIES, (3, 3, 3, 3, 3))),
    "GPT-5.1-dagger": dict(zip(CATEGORIES, (4, 4, 2, 3, 3))),
    "GUI-Owl-7B": dict(zip(CATEGORIES, (3, 3, 3, 3, 3))),
    "Gemini-3-Pro": dict(zip(CATEGORIES, (4, 4, 3, 2, 3))),
    "Gemini-3-Pro-dagger": dict(zip(CATEGORIES, (3, 3, 3, 3, 3))),
    "InternVL3-78B": dict(zip(CATEGORIES, (0, 0, 5, 5, 6))),
    "MAI-UI-8B": dict(zip(CATEGORIES, (3, 3, 3, 3, 3))),
    "Mobile-Agent-v3": dict(zip(CATEGORIES, (3, 3, 3, 3, 3))),
    "Qwen3-VL-8B": dict(zip(CATEGORIES, (3, 3, 3, 3, 3))),
    "UI Voyager-4B": dict(zip(CATEGORIES, (3, 3, 3, 3, 3))),
    "UI-Venus-72B": dict(zip(CATEGORIES, (4, 4, 3, 3, 2))),
    "UI-Venus-7B": dict(zip(CATEGORIES, (4, 4, 3, 3, 2))),
}

# Columns follow FAILURE_MODES.  Every category sums to 40; global support is
# [38, 38, 38, 38, 38, 10].
CATEGORY_LABEL_QUOTAS: dict[str, dict[str, int]] = {
    "clock": dict(zip(FAILURE_MODES, (8, 8, 8, 7, 7, 2))),
    "contacts": dict(zip(FAILURE_MODES, (7, 8, 8, 8, 7, 2))),
    "files": dict(zip(FAILURE_MODES, (7, 7, 8, 8, 8, 2))),
    "maps": dict(zip(FAILURE_MODES, (8, 7, 7, 8, 8, 2))),
    "sms": dict(zip(FAILURE_MODES, (8, 8, 7, 7, 8, 2))),
}

# One disjoint pilot case is drawn for each listed stratum.  This gives four
# per category and covers all six failure modes before guideline calibration.
PILOT_CATEGORY_LABELS: dict[str, tuple[str, ...]] = {
    "clock": (
        "planning",
        "grounding",
        "mixed_planning_grounding",
        "unknown",
    ),
    "contacts": ("planning", "grounding", "execution_tooling", "unknown"),
    "files": (
        "planning",
        "mixed_planning_grounding",
        "environment_or_evaluator",
        "unknown",
    ),
    "maps": (
        "grounding",
        "mixed_planning_grounding",
        "execution_tooling",
        "environment_or_evaluator",
    ),
    "sms": (
        "planning",
        "grounding",
        "execution_tooling",
        "environment_or_evaluator",
    ),
}

PUBLIC_FIELDS = (
    "episode_id",
    "model_name",
    "category",
    "app_id",
    "app_name",
    "task_template",
    "goal",
    "is_successful",
    "pkl_path",
    "episode_index",
)


def _canonical_json(value: Any) -> bytes:
  return json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as handle:
    for line_no, raw in enumerate(handle, 1):
      raw = raw.strip()
      if not raw:
        continue
      try:
        row = json.loads(raw)
      except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
      if not isinstance(row, dict):
        raise ValueError(f"Expected object at {path}:{line_no}")
      rows.append(row)
  return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _safe_failure(value: Any) -> bool:
  try:
    score = float(value)
  except (TypeError, ValueError):
    return False
  return math.isfinite(score) and score < 0.5


def _gemini_judgment(row: Mapping[str, Any]) -> Mapping[str, Any]:
  value = row.get("gemini_judgment")
  if not isinstance(value, dict):
    raise ValueError(f"Missing Gemini judgment for {row.get('episode_id')}")
  return value


def _qwen_judgment(row: Mapping[str, Any]) -> Mapping[str, Any]:
  value = row.get("qwen_judgment")
  if not isinstance(value, dict):
    raise ValueError(f"Missing Qwen judgment for {row.get('episode_id')}")
  return value


def _gemini_label(row: Mapping[str, Any]) -> str:
  return str(_gemini_judgment(row).get("primary_failure_mode") or "")


def _gemini_confidence(row: Mapping[str, Any]) -> str:
  return str(_gemini_judgment(row).get("confidence") or "").lower()


def _app_status(row: Mapping[str, Any]) -> str:
  category = str(row["category"])
  return "aw_canonical" if row["app_id"] == AW_APP_BY_CATEGORY[category] else "new"


def _missing_pkl_paths(
    rows: Sequence[Mapping[str, Any]], *, workers: int = 16
) -> list[tuple[str, Path]]:
  """Return missing raw trajectories, parallelizing slow $HOME stat calls."""
  checks = [
      (str(row.get("episode_id") or ""), Path(str(row.get("pkl_path") or "")))
      for row in rows
  ]
  if not checks:
    return []
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=min(workers, len(checks))
  ) as executor:
    exists = executor.map(lambda item: item[1].is_file(), checks)
    return [item for item, is_file in zip(checks, exists) if not is_file]


def _validate_source(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_exact_source: bool = True,
    check_pkl: bool = True,
) -> None:
  """Fail closed before drawing anything from the source cohort."""
  errors: list[str] = []
  if require_exact_source and len(rows) != EXPECTED_SOURCE_ROWS:
    errors.append(
        f"expected exactly {EXPECTED_SOURCE_ROWS} source rows, found {len(rows)}"
    )

  episode_ids: set[str] = set()
  duplicate_ids: set[str] = set()
  models: set[str] = set()
  categories: set[str] = set()
  for index, row in enumerate(rows):
    episode_id = str(row.get("episode_id") or "")
    if not episode_id:
      errors.append(f"row {index} has no episode_id")
    elif episode_id in episode_ids:
      duplicate_ids.add(episode_id)
    episode_ids.add(episode_id)

    model = str(row.get("model_name") or "")
    category = str(row.get("category") or "")
    models.add(model)
    categories.add(category)
    if category not in CATEGORIES:
      errors.append(f"{episode_id}: unexpected category {category!r}")
    if not _safe_failure(row.get("is_successful")):
      errors.append(f"{episode_id}: source is not a validator failure")
    if row.get("status") != "ok":
      errors.append(f"{episode_id}: Qwen status is not 'ok'")

    try:
      gemini = _gemini_judgment(row)
      qwen = _qwen_judgment(row)
    except ValueError as exc:
      errors.append(str(exc))
      continue
    label = str(gemini.get("primary_failure_mode") or "")
    confidence = str(gemini.get("confidence") or "").lower()
    if label not in FAILURE_MODES:
      errors.append(f"{episode_id}: unsupported Gemini label {label!r}")
    if confidence not in CONFIDENCE_WEIGHTS:
      errors.append(f"{episode_id}: unsupported Gemini confidence {confidence!r}")
    if str(qwen.get("primary_failure_mode") or "") not in FAILURE_MODES:
      errors.append(f"{episode_id}: unsupported Qwen failure label")

    for field in PUBLIC_FIELDS:
      if field not in row:
        errors.append(f"{episode_id}: missing required metadata field {field}")

  # $HOME metadata calls are high latency.  Parallel stat calls keep the
  # fail-closed all-source audit practical without changing its semantics.
  if check_pkl:
    for episode_id, pkl_path in _missing_pkl_paths(rows):
      errors.append(f"{episode_id}: pkl does not exist: {pkl_path}")

  if duplicate_ids:
    errors.append(f"duplicate episode IDs: {sorted(duplicate_ids)[:5]}")
  if require_exact_source:
    expected_models = set(MODEL_CATEGORY_QUOTAS)
    if models != expected_models:
      errors.append(
          f"model roster mismatch: missing={sorted(expected_models - models)}, "
          f"extra={sorted(models - expected_models)}"
      )
    if categories != set(CATEGORIES):
      errors.append(
          f"category roster mismatch: expected={list(CATEGORIES)}, "
          f"found={sorted(categories)}"
      )

  if errors:
    preview = "\n - ".join(errors[:30])
    suffix = f"\n ... and {len(errors) - 30} more" if len(errors) > 30 else ""
    raise ValueError(f"Source validation failed:\n - {preview}{suffix}")


@dataclasses.dataclass
class _Edge:
  to: int
  reverse: int
  capacity: int
  cost: int


def _add_edge(graph: list[list[_Edge]], src: int, dst: int, cap: int, cost: int) -> _Edge:
  forward = _Edge(dst, len(graph[dst]), cap, cost)
  reverse = _Edge(src, len(graph[src]), 0, -cost)
  graph[src].append(forward)
  graph[dst].append(reverse)
  return forward


def _minimum_cost_label_allocation(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[tuple[str, str, str], int]:
  """Meet all binding margins while minimizing repeated labels per cell.

  Each category is an independent min-cost flow from model quotas to label
  quotas.  Unit-capacity model--label arcs have increasing costs, so the first
  distinct label in a model/category cell is preferred to a repeated label.
  """
  inventory = collections.Counter(
      (str(row["model_name"]), str(row["category"]), _gemini_label(row))
      for row in rows
  )
  allocation: collections.Counter[tuple[str, str, str]] = collections.Counter()
  rng = random.Random(seed)

  for category in CATEGORIES:
    models = tuple(MODEL_CATEGORY_QUOTAS)
    labels = FAILURE_MODES
    source = 0
    model_offset = 1
    label_offset = model_offset + len(models)
    sink = label_offset + len(labels)
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]

    for model_index, model in enumerate(models):
      _add_edge(
          graph,
          source,
          model_offset + model_index,
          MODEL_CATEGORY_QUOTAS[model][category],
          0,
      )
    for label_index, label in enumerate(labels):
      _add_edge(
          graph,
          label_offset + label_index,
          sink,
          CATEGORY_LABEL_QUOTAS[category][label],
          0,
      )

    tracked: list[tuple[str, str, _Edge]] = []
    for model_index, model in enumerate(models):
      for label_index, label in enumerate(labels):
        capacity = inventory[(model, category, label)]
        # Increasing unit costs are a convex penalty for repeated labels.
        # The small seeded tie-break affects only equally diverse solutions.
        for repetition in range(capacity):
          edge = _add_edge(
              graph,
              model_offset + model_index,
              label_offset + label_index,
              1,
              repetition * 10_000 + rng.randrange(1000),
          )
          tracked.append((model, label, edge))

    target_flow = sum(CATEGORY_LABEL_QUOTAS[category].values())
    flow = 0
    while flow < target_flow:
      distance = [10**18] * len(graph)
      previous: list[tuple[int, int] | None] = [None] * len(graph)
      in_queue = [False] * len(graph)
      queue: collections.deque[int] = collections.deque([source])
      distance[source] = 0
      in_queue[source] = True
      while queue:
        node = queue.popleft()
        in_queue[node] = False
        for edge_index, edge in enumerate(graph[node]):
          if edge.capacity <= 0:
            continue
          candidate = distance[node] + edge.cost
          if candidate >= distance[edge.to]:
            continue
          distance[edge.to] = candidate
          previous[edge.to] = (node, edge_index)
          if not in_queue[edge.to]:
            queue.append(edge.to)
            in_queue[edge.to] = True
      if previous[sink] is None:
        raise ValueError(
            f"Infeasible audited quotas for category {category!r}: "
            f"allocated {flow}/{target_flow}"
        )
      node = sink
      while node != source:
        parent, edge_index = previous[node]  # type: ignore[misc]
        edge = graph[parent][edge_index]
        edge.capacity -= 1
        graph[node][edge.reverse].capacity += 1
        node = parent
      flow += 1

    for model, label, edge in tracked:
      if edge.capacity == 0:
        allocation[(model, category, label)] += 1

  _assert_allocation(allocation, inventory)
  return dict(allocation)


def _assert_allocation(
    allocation: Mapping[tuple[str, str, str], int],
    inventory: Mapping[tuple[str, str, str], int],
) -> None:
  for key, count in allocation.items():
    if count < 0 or count > inventory.get(key, 0):
      raise AssertionError(f"allocation exceeds source support for {key}: {count}")
  for model, category_quotas in MODEL_CATEGORY_QUOTAS.items():
    for category, target in category_quotas.items():
      actual = sum(
          allocation.get((model, category, label), 0) for label in FAILURE_MODES
      )
      if actual != target:
        raise AssertionError(
            f"model/category margin mismatch for {model}/{category}: "
            f"{actual} != {target}"
        )
  for category, label_quotas in CATEGORY_LABEL_QUOTAS.items():
    for label, target in label_quotas.items():
      actual = sum(
          allocation.get((model, category, label), 0)
          for model in MODEL_CATEGORY_QUOTAS
      )
      if actual != target:
        raise AssertionError(
            f"category/label margin mismatch for {category}/{label}: "
            f"{actual} != {target}"
        )


def _weighted_choice(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rng: random.Random,
    app_counts: Mapping[str, int],
    status_counts: Mapping[str, int],
    model_counts: Mapping[str, int] | None = None,
) -> tuple[Mapping[str, Any], dict[str, float]]:
  ordered = sorted(candidates, key=lambda row: str(row["episode_id"]))
  weights: list[float] = []
  components: list[dict[str, float]] = []
  for row in ordered:
    confidence_weight = CONFIDENCE_WEIGHTS[_gemini_confidence(row)]
    app_count = app_counts.get(str(row["app_id"]), 0)
    status_count = status_counts.get(_app_status(row), 0)
    # Strong bonuses for a previously unseen app/status, followed by inverse
    # frequency.  Gemini confidence remains the within-stratum base weight.
    app_diversity = 3.0 if app_count == 0 else 1.0 / (1.0 + app_count)
    status_diversity = 2.0 if status_count == 0 else 1.0 / (1.0 + status_count)
    model_diversity = 1.0
    if model_counts is not None:
      model_count = model_counts.get(str(row["model_name"]), 0)
      model_diversity = 2.0 if model_count == 0 else 1.0 / (1.0 + model_count)
    diversity_weight = app_diversity * status_diversity * model_diversity
    selection_weight = confidence_weight * diversity_weight
    components.append(
        {
            "confidence_weight": confidence_weight,
            "app_diversity_weight": app_diversity,
            "status_diversity_weight": status_diversity,
            "model_diversity_weight": model_diversity,
            "diversity_weight": diversity_weight,
            "selection_weight": selection_weight,
        }
    )
    weights.append(selection_weight)
  chosen_index = rng.choices(range(len(ordered)), weights=weights, k=1)[0]
  audit = dict(components[chosen_index])
  audit["candidate_weight_sum"] = sum(weights)
  audit["candidate_count"] = float(len(ordered))
  return ordered[chosen_index], audit


def _select_primary(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[list[Mapping[str, Any]], dict[str, dict[str, Any]], dict[tuple[str, str, str], int]]:
  allocation = _minimum_cost_label_allocation(rows, seed=seed)
  grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = collections.defaultdict(list)
  for row in rows:
    grouped[(str(row["model_name"]), str(row["category"]), _gemini_label(row))].append(row)

  rng = random.Random(seed)
  selected: list[Mapping[str, Any]] = []
  audits: dict[str, dict[str, Any]] = {}
  for model in MODEL_CATEGORY_QUOTAS:
    for category in CATEGORIES:
      remaining = {
          label: allocation.get((model, category, label), 0)
          for label in FAILURE_MODES
      }
      available = {
          label: list(grouped.get((model, category, label), []))
          for label in FAILURE_MODES
      }
      app_counts: collections.Counter[str] = collections.Counter()
      status_counts: collections.Counter[str] = collections.Counter()
      draw_index = 0
      # Round-robin labels so diversity is considered across the whole
      # model/category cell rather than drawing one complete label at a time.
      while sum(remaining.values()):
        active = [label for label in FAILURE_MODES if remaining[label] > 0]
        rng.shuffle(active)
        for label in active:
          if remaining[label] <= 0:
            continue
          chosen, audit = _weighted_choice(
              available[label],
              rng=rng,
              app_counts=app_counts,
              status_counts=status_counts,
          )
          available[label].remove(chosen)
          remaining[label] -= 1
          selected.append(chosen)
          episode_id = str(chosen["episode_id"])
          audits[episode_id] = {
              "selection_stage": "primary",
              "selection_stratum": {
                  "model_name": model,
                  "category": category,
                  "gemini_label": label,
              },
              "gemini_confidence": _gemini_confidence(chosen),
              "app_status": _app_status(chosen),
              "draw_index_within_model_category": draw_index,
              **audit,
          }
          app_counts[str(chosen["app_id"])] += 1
          status_counts[_app_status(chosen)] += 1
          draw_index += 1

  if len(selected) != PRIMARY_N or len({row["episode_id"] for row in selected}) != PRIMARY_N:
    raise AssertionError("primary selection is not 200 unique episodes")
  _assert_selected_margins(selected)
  return selected, audits, allocation


def _assert_selected_margins(rows: Sequence[Mapping[str, Any]]) -> None:
  model_category = collections.Counter(
      (str(row["model_name"]), str(row["category"])) for row in rows
  )
  category_label = collections.Counter(
      (str(row["category"]), _gemini_label(row)) for row in rows
  )
  for model, category_quotas in MODEL_CATEGORY_QUOTAS.items():
    for category, target in category_quotas.items():
      if model_category[(model, category)] != target:
        raise AssertionError(f"selected model/category margin failed: {model}/{category}")
  for category, label_quotas in CATEGORY_LABEL_QUOTAS.items():
    for label, target in label_quotas.items():
      if category_label[(category, label)] != target:
        raise AssertionError(f"selected category/label margin failed: {category}/{label}")


def _select_pilot(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_episode_ids: set[str],
    seed: int,
) -> tuple[list[Mapping[str, Any]], dict[str, dict[str, Any]]]:
  rng = random.Random(seed)
  pilot: list[Mapping[str, Any]] = []
  audits: dict[str, dict[str, Any]] = {}
  app_counts: collections.Counter[str] = collections.Counter()
  status_counts: collections.Counter[str] = collections.Counter()
  model_counts: collections.Counter[str] = collections.Counter()
  remaining = [row for row in rows if str(row["episode_id"]) not in excluded_episode_ids]

  slots = [
      (category, label)
      for category in CATEGORIES
      for label in PILOT_CATEGORY_LABELS[category]
  ]
  rng.shuffle(slots)
  for draw_index, (category, label) in enumerate(slots):
    candidates = [
        row
        for row in remaining
        if row["category"] == category and _gemini_label(row) == label
    ]
    if not candidates:
      raise ValueError(f"No remaining pilot candidate for {category}/{label}")
    chosen, audit = _weighted_choice(
        candidates,
        rng=rng,
        app_counts=app_counts,
        status_counts=status_counts,
        model_counts=model_counts,
    )
    remaining.remove(chosen)
    pilot.append(chosen)
    episode_id = str(chosen["episode_id"])
    audits[episode_id] = {
        "selection_stage": "calibration",
        "selection_stratum": {
            "category": category,
            "gemini_label": label,
        },
        "gemini_confidence": _gemini_confidence(chosen),
        "app_status": _app_status(chosen),
        "draw_index": draw_index,
        **audit,
    }
    app_counts[str(chosen["app_id"])] += 1
    status_counts[_app_status(chosen)] += 1
    model_counts[str(chosen["model_name"])] += 1

  if len(pilot) != PILOT_N:
    raise AssertionError(f"pilot size is {len(pilot)}, expected {PILOT_N}")
  if excluded_episode_ids & {str(row["episode_id"]) for row in pilot}:
    raise AssertionError("calibration pilot overlaps the primary sample")
  by_category = collections.Counter(str(row["category"]) for row in pilot)
  if by_category != collections.Counter({category: 4 for category in CATEGORIES}):
    raise AssertionError(f"pilot is not balanced four/category: {by_category}")
  if set(_gemini_label(row) for row in pilot) != set(FAILURE_MODES):
    raise AssertionError("pilot does not cover all six Gemini labels")
  return pilot, audits


def _public_row(
    row: Mapping[str, Any], *, pool: str, sample_id: str
) -> dict[str, Any]:
  public = {field: copy.deepcopy(row[field]) for field in PUBLIC_FIELDS}
  public.update(
      {
          "sample_id": sample_id,
          "pool": pool,
          "app_status": _app_status(row),
      }
  )
  return public


def _private_row(
    public: Mapping[str, Any],
    source: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
  gemini_usage = source.get("gemini_usage")
  if not isinstance(gemini_usage, dict):
    gemini_usage = {}
  historical_num_images = gemini_usage.get("num_images")
  return {
      **copy.deepcopy(dict(public)),
      "gemini_label": _gemini_label(source),
      "gemini_confidence": _gemini_confidence(source),
      "gemini_judgment": copy.deepcopy(dict(_gemini_judgment(source))),
      "qwen_label": str(_qwen_judgment(source).get("primary_failure_mode") or ""),
      "qwen_confidence": str(_qwen_judgment(source).get("confidence") or ""),
      "qwen_judgment": copy.deepcopy(dict(_qwen_judgment(source))),
      "gemini_usage": copy.deepcopy(gemini_usage),
      "qwen_usage": copy.deepcopy(source.get("qwen_usage") or {}),
      "evidence_parity": copy.deepcopy(source.get("evidence_parity") or {}),
      "historical_num_images": historical_num_images,
      "evidence_type": (
          "unknown"
          if not isinstance(historical_num_images, int)
          else "zero_image"
          if historical_num_images == 0
          else "visual"
      ),
      "qwen_used_for_sampling": False,
      "judge_agreement_used_for_sampling": False,
      "selection_audit": copy.deepcopy(dict(audit)),
  }


def _assert_public_blinding(rows: Sequence[Mapping[str, Any]]) -> None:
  forbidden_fragments = (
      "judge",
      "judgment",
      "rationale",
      "confidence",
      "failure_mode",
      "agreement",
      "gemini_label",
      "qwen_label",
      "selection_weight",
  )

  def visit(value: Any, location: str) -> None:
    if isinstance(value, dict):
      for key, child in value.items():
        lowered = str(key).lower()
        if any(fragment in lowered for fragment in forbidden_fragments):
          raise AssertionError(f"private judge field leaked into public manifest: {location}.{key}")
        visit(child, f"{location}.{key}")
    elif isinstance(value, list):
      for index, child in enumerate(value):
        visit(child, f"{location}[{index}]")

  for index, row in enumerate(rows):
    visit(row, f"row[{index}]")


def _counter_table(
    lines: list[str], title: str, counter: Mapping[Any, int]
) -> None:
  lines.extend([f"## {title}", "", "| key | count |", "|---|---:|"])
  for key, count in sorted(counter.items(), key=lambda item: str(item[0])):
    if isinstance(key, tuple):
      rendered = " / ".join(str(part) for part in key)
    else:
      rendered = str(key)
    lines.append(f"| {rendered} | {count} |")
  lines.append("")


def _write_summary(
    path: Path,
    *,
    primary: Sequence[Mapping[str, Any]],
    doubles: Sequence[Mapping[str, Any]],
    pilot: Sequence[Mapping[str, Any]],
    source_n: int,
    config_sha256: str,
) -> None:
  lines = [
      "# CATBench Cross-Judge Human-Validation Sample",
      "",
      "The public manifests are blinded to both judges. Selection used only the "
      "pre-specified Gemini label/confidence plus episode metadata; neither Qwen "
      "labels nor cross-judge agreement affected selection.",
      "",
      "## Binding audit",
      "",
      "| check | result |",
      "|---|---:|",
      f"| exact completed C1 source cohort | PASS ({source_n}) |",
      f"| unique primary episodes | PASS ({len(primary)}) |",
      f"| double-annotation copies | PASS ({len(doubles)}) |",
      f"| disjoint calibration pilot | PASS ({len(pilot)}) |",
      "| all selected episodes are validator failures | PASS |",
      "| all selected pkl paths exist | PASS |",
      "| public manifests contain no judge outputs | PASS |",
      f"| sampling config SHA-256 | `{config_sha256}` |",
      "",
  ]
  _counter_table(
      lines,
      "Primary by model",
      collections.Counter(str(row["model_name"]) for row in primary),
  )
  _counter_table(
      lines,
      "Primary by category",
      collections.Counter(str(row["category"]) for row in primary),
  )
  _counter_table(
      lines,
      "Primary by category and pre-specified Gemini label",
      collections.Counter(
          (str(row["category"]), _gemini_label(row)) for row in primary
      ),
  )
  _counter_table(
      lines,
      "Primary by Gemini confidence",
      collections.Counter(_gemini_confidence(row) for row in primary),
  )
  _counter_table(
      lines,
      "Primary by app status",
      collections.Counter(_app_status(row) for row in primary),
  )
  _counter_table(
      lines,
      "Primary by app",
      collections.Counter(str(row["app_id"]) for row in primary),
  )
  _counter_table(
      lines,
      "Calibration by category",
      collections.Counter(str(row["category"]) for row in pilot),
  )
  _counter_table(
      lines,
      "Calibration by pre-specified Gemini label",
      collections.Counter(_gemini_label(row) for row in pilot),
  )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
    source_path: Path,
    main_seed: int = MAIN_SEED,
    double_seed: int = DOUBLE_ANNOTATION_SEED,
    pilot_seed: int = PILOT_SEED,
    all_source_pkl_audited: bool = True,
) -> dict[str, Any]:
  """Select and write the complete frozen annotation bundle."""
  out_dir.mkdir(parents=True, exist_ok=True)
  primary, primary_audits, allocation = _select_primary(rows, seed=main_seed)
  primary_ids = {str(row["episode_id"]) for row in primary}
  pilot, pilot_audits = _select_pilot(
      rows, excluded_episode_ids=primary_ids, seed=pilot_seed
  )
  selected_for_annotation = [*primary, *pilot]
  missing_selected = _missing_pkl_paths(selected_for_annotation)
  if missing_selected:
    preview = ", ".join(
        f"{episode_id}:{path}" for episode_id, path in missing_selected[:5]
    )
    raise FileNotFoundError(
        "Selected annotation trajectories are missing: " + preview
    )
  selected_once = [*primary, *pilot]
  if not all(_safe_failure(row.get("is_successful")) for row in selected_once):
    raise AssertionError("a selected episode is not a validator failure")
  missing_selected = [
      str(row["pkl_path"])
      for row in selected_once
      if not Path(str(row["pkl_path"])).is_file()
  ]
  if missing_selected:
    raise AssertionError(f"selected pkl paths do not exist: {missing_selected[:5]}")

  primary_order = list(primary)
  random.Random(main_seed ^ 0xC47B).shuffle(primary_order)
  primary_public = [
      _public_row(row, pool="primary", sample_id=f"primary-{index:03d}")
      for index, row in enumerate(primary_order, 1)
  ]
  source_by_id = {str(row["episode_id"]): row for row in rows}

  double_rng = random.Random(double_seed)
  duplicate_sources = double_rng.sample(primary_public, DOUBLE_ANNOTATION_N)
  double_public: list[dict[str, Any]] = []
  double_audits: dict[str, dict[str, Any]] = {}
  for index, original in enumerate(duplicate_sources, 1):
    duplicate = copy.deepcopy(original)
    duplicate["pool"] = "double_annotation"
    duplicate["sample_id"] = f"double-{index:03d}"
    duplicate["duplicated_from_sample_id"] = original["sample_id"]
    double_public.append(duplicate)
    double_audits[str(duplicate["episode_id"])] = {
        "selection_stage": "double_annotation",
        "duplicated_from_sample_id": original["sample_id"],
        "uniform_draw_seed": double_seed,
        "uniform_inclusion_probability": DOUBLE_ANNOTATION_N / PRIMARY_N,
        "selection_weight": 1.0,
    }

  pilot_order = list(pilot)
  random.Random(pilot_seed ^ 0xC47B).shuffle(pilot_order)
  pilot_public = [
      _public_row(row, pool="calibration", sample_id=f"calibration-{index:03d}")
      for index, row in enumerate(pilot_order, 1)
  ]
  sample_manifest = primary_public + double_public
  _assert_public_blinding(sample_manifest)
  _assert_public_blinding(pilot_public)

  private_rows: list[dict[str, Any]] = []
  for public in primary_public:
    episode_id = str(public["episode_id"])
    private_rows.append(
        _private_row(public, source_by_id[episode_id], primary_audits[episode_id])
    )
  for public in double_public:
    episode_id = str(public["episode_id"])
    private_rows.append(
        _private_row(public, source_by_id[episode_id], double_audits[episode_id])
    )
  for public in pilot_public:
    episode_id = str(public["episode_id"])
    private_rows.append(
        _private_row(public, source_by_id[episode_id], pilot_audits[episode_id])
    )

  gemini_sample = [
      {
          "episode_id": str(public["episode_id"]),
          "judgment": copy.deepcopy(
              dict(_gemini_judgment(source_by_id[str(public["episode_id"])]))
          ),
      }
      for public in primary_public
  ]
  qwen_sample = [
      {
          "episode_id": str(public["episode_id"]),
          "judgment": copy.deepcopy(
              dict(_qwen_judgment(source_by_id[str(public["episode_id"])]))
          ),
      }
      for public in primary_public
  ]

  model_totals = {
      model: sum(category_quotas.values())
      for model, category_quotas in MODEL_CATEGORY_QUOTAS.items()
  }
  config_without_hash: dict[str, Any] = {
      "schema_version": "catbench_cross_judge_human_validation_v1",
      "source_jsonl": str(source_path.resolve()),
      "source_sha256": _sha256_file(source_path),
      "source_rows": len(rows),
      "all_source_pkl_audited": all_source_pkl_audited,
      "selected_pkl_audited": True,
      "primary_n": PRIMARY_N,
      "double_annotation_n": DOUBLE_ANNOTATION_N,
      "calibration_n": PILOT_N,
      "main_seed": main_seed,
      "double_annotation_seed": double_seed,
      "pilot_seed": pilot_seed,
      "model_totals": model_totals,
      "model_category_quotas": MODEL_CATEGORY_QUOTAS,
      "category_label_quotas": CATEGORY_LABEL_QUOTAS,
      "pilot_category_labels": PILOT_CATEGORY_LABELS,
      "confidence_weights": CONFIDENCE_WEIGHTS,
      "aw_app_by_category": AW_APP_BY_CATEGORY,
      "selection_fields_used": [
          "episode_id",
          "model_name",
          "category",
          "app_id",
          "gemini_judgment.primary_failure_mode",
          "gemini_judgment.confidence",
      ],
      "selection_fields_forbidden": [
          "qwen_judgment",
          "qwen_judgment.primary_failure_mode",
          "qwen_judgment.confidence",
          "Gemini-Qwen agreement",
      ],
      "realized_model_category_label_allocation": {
          "|".join(key): value for key, value in sorted(allocation.items())
      },
  }
  config_sha256 = _sha256_bytes(_canonical_json(config_without_hash))
  config = {**config_without_hash, "config_sha256": config_sha256}

  paths = {
      "sample_manifest.jsonl": sample_manifest,
      "calibration_manifest.jsonl": pilot_public,
      "private_crosswalk.jsonl": private_rows,
      "gemini_judge_sample.jsonl": gemini_sample,
      "qwen_judge_sample.jsonl": qwen_sample,
  }
  for name, output_rows in paths.items():
    _write_jsonl(out_dir / name, output_rows)
  (out_dir / "sampling_config.json").write_text(
      json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  _write_summary(
      out_dir / "sample_summary.md",
      primary=primary,
      doubles=duplicate_sources,
      pilot=pilot,
      source_n=len(rows),
      config_sha256=config_sha256,
  )

  artifact_names = [*paths, "sampling_config.json", "sample_summary.md"]
  hashes = {
      "schema_version": "catbench_annotation_artifact_hashes_v1",
      "config_sha256": config_sha256,
      "artifacts": {
          name: {
              "sha256": _sha256_file(out_dir / name),
              "bytes": (out_dir / name).stat().st_size,
          }
          for name in artifact_names
      },
  }
  (out_dir / "artifact_hashes.json").write_text(
      json.dumps(hashes, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  return {
      "primary": primary,
      "pilot": pilot,
      "sample_manifest": sample_manifest,
      "private_crosswalk": private_rows,
      "config": config,
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source_jsonl", type=Path, default=DEFAULT_SOURCE)
  parser.add_argument("--out_dir", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=MAIN_SEED)
  parser.add_argument("--double_seed", type=int, default=DOUBLE_ANNOTATION_SEED)
  parser.add_argument("--pilot_seed", type=int, default=PILOT_SEED)
  parser.add_argument(
      "--skip_all_source_pkl_audit",
      action="store_true",
      help=(
          "Skip stat calls for unselected source rows. The 220 selected/pilot "
          "trajectories are always checked. Use only when the exact source "
          "cohort has already passed a separate all-row artifact audit."
      ),
  )
  args = parser.parse_args(argv)

  source_path = args.source_jsonl.expanduser().resolve()
  out_dir = args.out_dir.expanduser().resolve()
  rows = _read_jsonl(source_path)
  all_source_pkl_audited = not args.skip_all_source_pkl_audit
  _validate_source(
      rows, require_exact_source=True, check_pkl=all_source_pkl_audited
  )
  result = prepare_artifacts(
      rows,
      out_dir=out_dir,
      source_path=source_path,
      main_seed=args.seed,
      double_seed=args.double_seed,
      pilot_seed=args.pilot_seed,
      all_source_pkl_audited=all_source_pkl_audited,
  )
  print(f"Frozen primary episodes: {len(result['primary'])}")
  print(f"Server queue rows (primary + duplicates): {len(result['sample_manifest'])}")
  print(f"Disjoint calibration episodes: {len(result['pilot'])}")
  print(f"Artifacts: {out_dir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
