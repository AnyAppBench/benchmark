from __future__ import annotations

import json
from unittest import mock

import preflight_maps_track_validator as validator


def test_primary_selection_is_exact_frozen_maps_roster() -> None:
  assert validator.PRIMARY_PACKAGES == (
      ("osmand", "net.osmand.plus"),
      ("organic_maps", "app.organicmaps"),
      ("comaps", "app.comaps.fdroid"),
  )


def test_all_selection_keeps_diagnostic_packages_explicit() -> None:
  selected = validator._selected_packages("all")
  assert [label for label, _ in selected] == [
      "osmand",
      "organic_maps",
      "google_maps",
      "comaps",
      "maps_me",
  ]


def test_missing_package_fails_closed() -> None:
  with mock.patch.object(validator, "_package_installed", return_value=False):
    result = validator._preflight_package(
        serial="emulator-5578",
        label="osmand",
        package="net.osmand.plus",
        waypoints=validator.DEFAULT_WAYPOINTS,
        shared_root=validator.DEFAULT_SHARED_ROOT,
    )
  assert result == {
      "label": "osmand",
      "package_name": "net.osmand.plus",
      "installed": False,
      "passed": False,
      "reason": "package_not_installed",
  }


def test_primary_cohort_binding_rejects_roster_substitution(tmp_path) -> None:
  path = tmp_path / "cohort.json"
  path.write_text(json.dumps({
      "release_id": "test",
      "categories": {"maps": {"app_ids": ["maps_osmand"]}},
  }))
  try:
    validator._validate_primary_cohort(path)
  except ValueError as exc:
    assert "roster mismatch" in str(exc)
  else:
    raise AssertionError("substituted Maps roster was accepted")


def test_checked_primary_cohort_matches_exact_roster() -> None:
  cohort = validator._validate_primary_cohort(validator.DEFAULT_COHORT)
  assert cohort["release_id"] == "catbench_acl_revision_5cat_v1"
