#!/usr/bin/env python3
"""Preflight CATBench Maps GPX/KML/link validators on a device.

This is a fast evaluator smoke test, not a benchmark run.  It writes synthetic
GPX/KML tracks and a map-link text file with known Liechtenstein waypoints,
checks that the same helpers used by ``MapsRecordTrack`` and
``MapsExportLocation`` can find the files and validate coordinates/order, then
deletes the files.

Example:

  PYTHONPATH=benchmark python benchmark/scripts/preflight_maps_track_validator.py \
    --serial emulator-5558
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

from android_world.task_evals.single.app_generalization_generated import (  # noqa: E402
    maps_cross_app_tasks as maps_tasks,
)


DEFAULT_WAYPOINTS = (
    "Balzers, Liechtenstein",
    "Triesen, Liechtenstein",
)
DEFAULT_SHARED_ROOT = "/data/media/0/Download"
PACKAGES = {
    "osmand": maps_tasks._OSMAND_PACKAGE_NAME,
    "organic": maps_tasks._ORGANIC_MAPS_PACKAGE_NAME,
    "organic_maps": maps_tasks._ORGANIC_MAPS_PACKAGE_NAME,
    "google": maps_tasks._GOOGLE_MAPS_PACKAGE_NAME,
    "google_maps": maps_tasks._GOOGLE_MAPS_PACKAGE_NAME,
    "comaps": maps_tasks._COMAPS_PACKAGE_NAME,
    "mapsme": maps_tasks._MAPS_ME_PACKAGE_NAME,
    "maps_me": maps_tasks._MAPS_ME_PACKAGE_NAME,
}
PRIMARY_PACKAGES = (
    ("osmand", maps_tasks._OSMAND_PACKAGE_NAME),
    ("organic_maps", maps_tasks._ORGANIC_MAPS_PACKAGE_NAME),
    ("comaps", maps_tasks._COMAPS_PACKAGE_NAME),
)
PRIMARY_APP_IDS = (
    "maps_osmand",
    "maps_organic_maps",
    "maps_comaps",
)
DEFAULT_COHORT = BENCHMARK_ROOT / "configs" / "catbench_5cat_primary_cohort.json"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, object]:
  def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f"Duplicate JSON key {key!r} in {path}")
      result[key] = value
    return result

  def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} in {path}")

  payload = json.loads(
      path.read_text(encoding="utf-8"),
      object_pairs_hook=reject_duplicates,
      parse_constant=reject_constant,
  )
  if not isinstance(payload, dict):
    raise ValueError(f"Expected JSON object in {path}")
  return payload


def _validate_primary_cohort(path: Path) -> dict[str, object]:
  cohort = _strict_json(path)
  categories = cohort.get("categories")
  if not isinstance(categories, dict):
    raise ValueError("Frozen cohort is missing categories")
  maps = categories.get("maps")
  if not isinstance(maps, dict):
    raise ValueError("Frozen cohort is missing Maps")
  app_ids = maps.get("app_ids")
  if app_ids != list(PRIMARY_APP_IDS):
    raise ValueError(
        f"Frozen Maps app roster mismatch: expected {list(PRIMARY_APP_IDS)}, "
        f"got {app_ids!r}"
    )
  return cohort


def _run(
    args: list[str],
    *,
    timeout: float = 20.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
  proc = subprocess.run(
      args,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      timeout=timeout,
      check=False,
  )
  if check and proc.returncode != 0:
    joined = " ".join(shlex.quote(arg) for arg in args)
    raise RuntimeError(
        f"Command failed ({proc.returncode}): {joined}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
  return proc


def _adb(serial: str, *args: str, check: bool = True) -> str:
  return _run(["adb", "-s", serial, *args], check=check).stdout


def _adb_su(serial: str, cmd: str, *, check: bool = True) -> str:
  quoted = shlex.quote(f"{cmd} || true")
  return _adb(serial, "shell", f"su 0 sh -c {quoted}", check=check)


def _package_installed(serial: str, package: str) -> bool:
  out = _adb(serial, "shell", f"pm path {shlex.quote(package)}", check=False)
  return out.strip().startswith("package:")


def _write_root_file(serial: str, path: str, content: str) -> None:
  encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
  parent = path.rsplit("/", 1)[0]
  _adb_su(
      serial,
      (
          f"mkdir -p {shlex.quote(parent)} && "
          f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
      ),
  )


def _remove_root_file(serial: str, path: str) -> None:
  _adb_su(serial, f"rm -f {shlex.quote(path)}", check=False)


def _track_gpx(waypoints: Iterable[str]) -> str:
  points = []
  for waypoint in waypoints:
    lat, lon = maps_tasks._PLACE_COORDS[waypoint]
    points.append(f'      <trkpt lat="{lat:.7f}" lon="{lon:.7f}" />')
  return (
      '<?xml version="1.0" encoding="UTF-8"?>\n'
      '<gpx version="1.1" creator="CATBenchPreflight"'
      ' xmlns="http://www.topografix.com/GPX/1/1">\n'
      "  <trk><name>CATBench preflight</name><trkseg>\n"
      + "\n".join(points)
      + "\n"
      "  </trkseg></trk>\n"
      "</gpx>\n"
  )


def _track_kml(waypoints: Iterable[str]) -> str:
  coords = []
  for waypoint in waypoints:
    lat, lon = maps_tasks._PLACE_COORDS[waypoint]
    coords.append(f"{lon:.7f},{lat:.7f},0")
  return (
      '<?xml version="1.0" encoding="UTF-8"?>\n'
      '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
      "  <Document>\n"
      "    <Placemark><name>CATBench preflight</name><LineString>\n"
      "      <coordinates>"
      + " ".join(coords)
      + "</coordinates>\n"
      "    </LineString></Placemark>\n"
      "  </Document>\n"
      "</kml>\n"
  )


@contextlib.contextmanager
def _patched_su_shell(serial: str):
  original = maps_tasks._su_shell
  maps_tasks._su_shell = lambda _env, cmd: _adb_su(serial, cmd, check=False)
  try:
    yield
  finally:
    maps_tasks._su_shell = original


def _selected_packages(selection: str) -> list[tuple[str, str]]:
  if selection == "all":
    ordered = (
        ("osmand", maps_tasks._OSMAND_PACKAGE_NAME),
        ("organic_maps", maps_tasks._ORGANIC_MAPS_PACKAGE_NAME),
        ("google_maps", maps_tasks._GOOGLE_MAPS_PACKAGE_NAME),
        ("comaps", maps_tasks._COMAPS_PACKAGE_NAME),
        ("maps_me", maps_tasks._MAPS_ME_PACKAGE_NAME),
    )
    return list(ordered)
  package = PACKAGES.get(selection.casefold(), selection)
  label = next(
      (name for name, value in PACKAGES.items() if value == package),
      package,
  )
  return [(label, package)]


def _preflight_package(
    *,
    serial: str,
    label: str,
    package: str,
    waypoints: tuple[str, ...],
    shared_root: str,
) -> dict[str, object]:
  if not _package_installed(serial, package):
    print(f"{label:13s} FAIL package not installed: {package}")
    return {
        "label": label,
        "package_name": package,
        "installed": False,
        "passed": False,
        "reason": "package_not_installed",
    }

  stamp = f"{int(time.time())}_{package.replace('.', '_')}"
  gpx_path = f"{shared_root}/catbench_preflight_{stamp}.gpx"
  kml_path = f"{shared_root}/catbench_preflight_{stamp}.kml"
  link_path = f"{shared_root}/catbench_preflight_{stamp}.txt"
  reversed_waypoints = tuple(reversed(waypoints))
  place = waypoints[0]
  lat, lon = maps_tasks._PLACE_COORDS[place]
  link = f"https://www.openstreetmap.org/?mlat={lat:.7f}&mlon={lon:.7f}\n"

  try:
    _remove_root_file(serial, gpx_path)
    _remove_root_file(serial, kml_path)
    _remove_root_file(serial, link_path)

    with _patched_su_shell(serial):
      clean = not maps_tasks._track_exists(None, package, waypoints)
      if not clean:
        print(
            f"{label:13s} FAIL stale matching track exists before injection"
        )
        return {
            "label": label,
            "package_name": package,
            "installed": True,
            "clean_before_injection": False,
            "passed": False,
            "reason": "stale_matching_track_before_injection",
        }

      _write_root_file(serial, gpx_path, _track_gpx(waypoints))
      gpx_ok = maps_tasks._track_exists(None, package, waypoints)
      gpx_reversed_ok = maps_tasks._track_exists(
          None, package, reversed_waypoints
      )
      _remove_root_file(serial, gpx_path)

      _write_root_file(serial, kml_path, _track_kml(waypoints))
      kml_ok = maps_tasks._track_exists(None, package, waypoints)
      kml_reversed_ok = maps_tasks._track_exists(
          None, package, reversed_waypoints
      )
      _remove_root_file(serial, kml_path)

      _write_root_file(serial, link_path, link)
      export_ok = maps_tasks._export_exists(None, package, place)
      export_wrong_ok = maps_tasks._export_exists(
          None, package, waypoints[-1]
      )
      _remove_root_file(serial, link_path)

    ok = (
        gpx_ok
        and kml_ok
        and export_ok
        and not gpx_reversed_ok
        and not kml_reversed_ok
        and not export_wrong_ok
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"{label:13s} {status} "
        f"gpx={gpx_ok} gpx_reversed={gpx_reversed_ok} "
        f"kml={kml_ok} kml_reversed={kml_reversed_ok} "
        f"link={export_ok} link_wrong={export_wrong_ok}"
    )
    return {
        "label": label,
        "package_name": package,
        "installed": True,
        "clean_before_injection": clean,
        "gpx_exact": gpx_ok,
        "gpx_reversed": gpx_reversed_ok,
        "kml_exact": kml_ok,
        "kml_reversed": kml_reversed_ok,
        "link_exact": export_ok,
        "link_wrong_place": export_wrong_ok,
        "passed": ok,
        "reason": "passed" if ok else "validator_expectation_mismatch",
    }
  finally:
    _remove_root_file(serial, gpx_path)
    _remove_root_file(serial, kml_path)
    _remove_root_file(serial, link_path)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--serial",
      required=True,
      help="ADB serial, e.g. emulator-5558.",
  )
  parser.add_argument(
      "--package",
      default="primary",
      help=(
          "Package alias/package to test. Use primary (the frozen three-app"
          " Maps cohort), all, osmand, organic_maps, google_maps, comaps,"
          " maps_me, or a raw package name."
      ),
  )
  parser.add_argument(
      "--waypoint",
      action="append",
      dest="waypoints",
      help=(
          "Waypoint name from maps_cross_app_tasks._PLACES. Pass at least two;"
          " defaults to Balzers and Triesen."
      ),
  )
  parser.add_argument(
      "--shared-root",
      default=DEFAULT_SHARED_ROOT,
      help="Writable shared export root included in the validator search paths.",
  )
  parser.add_argument(
      "--output",
      default="",
      help=(
          "Optional exclusive JSON evidence path. The command refuses to"
          " overwrite an existing path."
      ),
  )
  parser.add_argument(
      "--cohort_manifest",
      default=str(DEFAULT_COHORT),
      help="Frozen cohort manifest to bind into machine evidence.",
  )
  parser.add_argument(
      "--docker_image_digest",
      required=True,
      help="Exact Docker emulator image digest used by this worker.",
  )
  args = parser.parse_args()

  waypoints = tuple(args.waypoints or DEFAULT_WAYPOINTS)
  if len(waypoints) < 2:
    parser.error("Need at least two --waypoint values.")
  unknown = [waypoint for waypoint in waypoints if waypoint not in maps_tasks._PLACES]
  if unknown:
    parser.error(f"Unknown waypoint(s): {', '.join(unknown)}")

  cohort_path = Path(args.cohort_manifest).expanduser().resolve()
  cohort = _validate_primary_cohort(cohort_path)

  selected = (
      list(PRIMARY_PACKAGES)
      if args.package.casefold() == "primary"
      else _selected_packages(args.package)
  )
  cases = [
      _preflight_package(
          serial=args.serial,
          label=label,
          package=package,
          waypoints=waypoints,
          shared_root=args.shared_root,
      )
      for label, package in selected
  ]
  all_ok = bool(cases) and all(bool(case["passed"]) for case in cases)
  payload = {
      "schema_version": 1,
      "audit_type": "catbench_maps_live_storage_validator_smoke",
      "artifact_role": "harness_diagnostic_only_not_task_conformance_or_model_result",
      "analysis_eligible": False,
      "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "device": {
          "serial": args.serial,
          "adb_server_port": (
              os.environ.get("ANDROID_ADB_SERVER_PORT")
              or os.environ.get("ADB_SERVER_PORT")
              or "default"
          ),
          "boot_completed": _adb(
              args.serial, "shell", "getprop", "sys.boot_completed"
          ).strip(),
          "android_release": _adb(
              args.serial, "shell", "getprop", "ro.build.version.release"
          ).strip(),
          "api_level": _adb(
              args.serial, "shell", "getprop", "ro.build.version.sdk"
          ).strip(),
          "fingerprint": _adb(
              args.serial, "shell", "getprop", "ro.build.fingerprint"
          ).strip(),
      },
      "scope": {
          "selection": args.package,
          "primary_frozen_maps_scope": args.package.casefold() == "primary",
          "waypoints": list(waypoints),
          "shared_root": args.shared_root,
          "selected_package_count": len(selected),
      },
      "source": {
          "script_path": str(Path(__file__).resolve()),
          "script_sha256": _sha256(Path(__file__).resolve()),
          "maps_task_module_path": str(Path(maps_tasks.__file__).resolve()),
          "maps_task_module_sha256": _sha256(Path(maps_tasks.__file__).resolve()),
          "cohort_manifest_path": str(cohort_path),
          "cohort_manifest_sha256": _sha256(cohort_path),
          "release_id": cohort.get("release_id"),
      },
      "runtime": {"docker_image_digest": args.docker_image_digest},
      "cases": cases,
      "valid_package_count": sum(bool(case["passed"]) for case in cases),
      "invalid_package_count": sum(not bool(case["passed"]) for case in cases),
      "valid": all_ok,
      "execution_claims": {
          "temporary_files_deleted": True,
          "benchmark_episode_executed": False,
          "model_endpoint_called": False,
          "agent_action_generated": False,
          "full_task_or_ui_conformance_established": False,
      },
  }
  if args.output:
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
      json.dump(payload, handle, indent=2)
      handle.write("\n")
  print(json.dumps({
      "valid": all_ok,
      "selected_package_count": len(selected),
      "valid_package_count": payload["valid_package_count"],
      "invalid_package_count": payload["invalid_package_count"],
      "output": str(Path(args.output).expanduser().resolve()) if args.output else "",
  }, indent=2))
  return 0 if all_ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
