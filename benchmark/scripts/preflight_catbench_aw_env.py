#!/usr/bin/env python3
"""Preflight CATBench emulators for AndroidWorld-like app state.

The most important Maps invariant is the AndroidWorld OsmAnd setup asset:
``Liechtenstein_europe.obf``. AndroidWorld copies this map during
``setup_apps`` and later restores app snapshots instead of clearing the
external map directory. CATBench app-generalization runs preserve app data now,
but older runs may already have wiped the file with ``pm clear``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


MAP_PACKAGES = (
    "net.osmand.plus",
    "app.organicmaps",
    "com.google.android.apps.maps",
    "app.comaps.fdroid",
    "com.mapswithme.maps.pro",
)
MAP_PERMISSIONS = (
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.POST_NOTIFICATIONS",
)
OSMAND_MAP_NAME = "Liechtenstein_europe.obf"
OSMAND_MAP_URL = (
    "https://storage.googleapis.com/gresearch/android_world/"
    f"{OSMAND_MAP_NAME}"
)
OSMAND_MAP_DIR = "/storage/emulated/0/Android/data/net.osmand.plus/files"
OSMAND_MAP_PATH = f"{OSMAND_MAP_DIR}/{OSMAND_MAP_NAME}"
OSMAND_MAP_CHECK_PATH = (
    f"/data/media/0/Android/data/net.osmand.plus/files/{OSMAND_MAP_NAME}"
)
OSMAND_MAP_SIZE_BYTES = 7_241_271
OSMAND_MAP_SHA256 = (
    "57a1955c6f517a114999fa50480ecc1402ed9b524db602540ac1c08f57ff140e"
)
DEFAULT_COHORT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "catbench_5cat_primary_cohort.json"
)
PRIMARY_MAP_APP_IDS = (
    "maps_osmand",
    "maps_organic_maps",
    "maps_comaps",
)

MAP_RESOURCE_GROUPS = (
    {
        "name": "Organic Maps",
        "package": "app.organicmaps",
        "base_url": "https://cdn.organicmaps.app/maps/260404",
        "cache_dir": "organic_maps_260404",
        "remote_dir": (
            "/storage/emulated/0/Android/data/app.organicmaps/files/260404"
        ),
        "internal_dir": "/data/data/app.organicmaps/files/260404",
        "files": {
            "WorldCoasts.mwm": 8_458_999,
            "World.mwm": 63_505_843,
            "Liechtenstein.mwm": 3_813_728,
            "Switzerland_Eastern.mwm": 97_260_790,
            "US_California_Chico.mwm": 59_698_005,
        },
        "sha256": {
            "WorldCoasts.mwm": "a858ddeb112509cf5d4a185b8710f2e42ef6ce82426977942429b1ef0f1b62b2",
            "World.mwm": "ceb056db7d7e445b658b99c0d9c1573f02cce6e7434f255b5c706267f24bcc80",
            "Liechtenstein.mwm": "cf732f24f55e20e98dcf1604a8e9c5a51fcb533a8d576ad6b0eb629d79a66016",
            "Switzerland_Eastern.mwm": "24ed57c6d48100cc634ff8c0fa0482c6285252c76ba358c43316e9aa711d7fa4",
            "US_California_Chico.mwm": "a92bda6c9ee1b6043043573bebd5d0b70c922d1f729007f49d03b5e3f181dc43",
        },
    },
    {
        "name": "CoMaps",
        "package": "app.comaps.fdroid",
        # The pinned 2026.04.23 APK advertises MAP_SERIES 2026.04.05 in the
        # corresponding upstream source. A 260421 path never existed on the
        # official mirror and returns HTTP 404.
        "base_url": "https://mapgen-fi-1.comaps.app/maps/260405",
        "cache_dir": "comaps_260405",
        "remote_dir": (
            "/storage/emulated/0/Android/data/app.comaps.fdroid/files/260405"
        ),
        "internal_dir": "/data/data/app.comaps.fdroid/files/260405",
        "files": {
            "WorldCoasts.mwm": 8_492_865,
            "World.mwm": 53_104_836,
            "Liechtenstein.mwm": 3_993_906,
            "Switzerland_Eastern.mwm": 108_002_034,
            "US_California_Chico.mwm": 60_436_041,
        },
        "sha256": {
            "WorldCoasts.mwm": "5a1b573696057250e148afa21b6b01324281322b443d1f584943289a82a05850",
            "World.mwm": "6ee0f7be132895b2cbb350a569ceddfb1eb9f6cf8045bf06b3f9dc0835de6ac2",
            "Liechtenstein.mwm": "11ab91df1671e96ba63e4f2990be6fa15317acc17677d419e450a9c2347ceaf8",
            "Switzerland_Eastern.mwm": "201af2dc4734f0f3b83d71ef1005290bc16f34922b9da111c4f32a8d83d9f0b0",
            "US_California_Chico.mwm": "7f0930feef88ed7ddc7de05e951218bdd0335c7f1396d202b2d395bbea6b254d",
        },
    },
)


def _adb(
    serial: str,
    args: list[str],
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
  command = shlex.split(os.environ.get("ADB_BIN", "adb"))
  port = os.environ.get("ANDROID_ADB_SERVER_PORT") or os.environ.get(
      "ADB_SERVER_PORT"
  )
  if port and "-P" not in command:
    command.extend(["-P", port])
  command.extend(["-s", serial, *args])
  result = subprocess.run(
      command,
      check=False,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  )
  if check and result.returncode != 0:
    rendered = " ".join(command)
    raise RuntimeError(
        f"{rendered} failed with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
  return result


def _adb_root(
    serial: str,
    shell_args: list[str],
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
  return _adb(serial, ["shell", "su", "0", *shell_args], check=check)


def _package_installed(serial: str, package: str) -> bool:
  result = _adb(serial, ["shell", "pm", "path", package])
  return result.returncode == 0 and bool(result.stdout.strip())


def _file_exists(serial: str, path: str, *, root: bool = False) -> bool:
  if root:
    result = _adb_root(serial, ["test", "-f", path])
  else:
    result = _adb(serial, ["shell", "test", "-f", path])
  return result.returncode == 0


def _local_cache_path() -> Path:
  return Path(tempfile.gettempdir()) / "android_world" / "app_data" / OSMAND_MAP_NAME


def _sha256_file(path: Path) -> str:
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
  categories = payload.get("categories")
  maps = categories.get("maps") if isinstance(categories, dict) else None
  app_ids = maps.get("app_ids") if isinstance(maps, dict) else None
  if app_ids != list(PRIMARY_MAP_APP_IDS):
    raise ValueError(
        f"Frozen Maps roster mismatch: expected {list(PRIMARY_MAP_APP_IDS)}, "
        f"got {app_ids!r}"
    )
  return payload


def _download_file(
    url: str,
    cache_path: Path,
    expected_size: int,
    expected_sha256: str,
) -> Path:
  if (
      cache_path.exists()
      and cache_path.stat().st_size == expected_size
      and _sha256_file(cache_path) == expected_sha256
  ):
    return cache_path

  cache_path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
  print(f"[PREFLIGHT] downloading {url} -> {cache_path}", flush=True)
  with urllib.request.urlopen(url, timeout=180) as response:
    tmp_path.write_bytes(response.read())
  actual_size = tmp_path.stat().st_size
  if actual_size != expected_size:
    tmp_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"Downloaded {tmp_path} has unexpected size {actual_size}; "
        f"expected {expected_size}."
    )
  actual_sha256 = _sha256_file(tmp_path)
  if actual_sha256 != expected_sha256:
    tmp_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"Downloaded {tmp_path} has SHA-256 {actual_sha256}; "
        f"expected {expected_sha256}."
    )
  tmp_path.replace(cache_path)
  return cache_path


def _download_osmand_map() -> Path:
  return _download_file(
      OSMAND_MAP_URL,
      _local_cache_path(),
      OSMAND_MAP_SIZE_BYTES,
      OSMAND_MAP_SHA256,
  )


def _repair_osmand_map(serial: str) -> None:
  local_path = _download_osmand_map()
  _adb(serial, ["shell", "mkdir", "-p", OSMAND_MAP_DIR], check=False)
  _adb_root(serial, ["mkdir", "-p", os.path.dirname(OSMAND_MAP_CHECK_PATH)], check=True)
  tmp_path = f"/data/local/tmp/catbench_{OSMAND_MAP_NAME}"
  _adb(serial, ["push", str(local_path), tmp_path], check=True)
  _adb_root(serial, ["cp", tmp_path, OSMAND_MAP_CHECK_PATH], check=True)
  _adb(serial, ["shell", "rm", "-f", tmp_path], check=False)
  _adb_root(serial, ["chmod", "777", OSMAND_MAP_CHECK_PATH], check=False)
  _adb_root(
      serial,
      ["chcon", "u:object_r:media_rw_data_file:s0", OSMAND_MAP_CHECK_PATH],
      check=False,
  )


def _remote_file_has_size(
    serial: str, path: str, expected_size: int, *, root: bool = False
) -> bool:
  if root:
    result = _adb_root(serial, ["wc", "-c", path])
  else:
    result = _adb(serial, ["shell", "wc", "-c", path])
  if result.returncode != 0:
    return False
  try:
    actual_size = int(result.stdout.strip().split()[0])
    return actual_size == expected_size
  except (IndexError, ValueError):
    return False


def _remote_file_size(
    serial: str, path: str, *, root: bool = False
) -> int | None:
  if root:
    result = _adb_root(serial, ["wc", "-c", path])
  else:
    result = _adb(serial, ["shell", "wc", "-c", path])
  if result.returncode != 0:
    return None
  try:
    return int(result.stdout.strip().split()[0])
  except (IndexError, ValueError):
    return None


def _remote_file_sha256(
    serial: str, path: str, *, root: bool = False
) -> str:
  if root:
    result = _adb_root(serial, ["sha256sum", path])
  else:
    result = _adb(serial, ["shell", "sha256sum", path])
  if result.returncode != 0:
    return ""
  fields = result.stdout.strip().split()
  if not fields or len(fields[0]) != 64:
    return ""
  return fields[0].casefold()


def _remote_file_has_identity(
    serial: str,
    path: str,
    expected_size: int,
    expected_sha256: str,
    *,
    root: bool = False,
) -> bool:
  return _remote_file_has_size(
      serial, path, expected_size, root=root
  ) and _remote_file_sha256(serial, path, root=root) == expected_sha256


def _remote_file_identity_record(
    serial: str,
    path: str,
    expected_size: int,
    expected_sha256: str,
    *,
    root: bool = False,
) -> dict[str, object]:
  actual_size = _remote_file_size(serial, path, root=root)
  actual_sha256 = (
      _remote_file_sha256(serial, path, root=root)
      if actual_size is not None
      else ""
  )
  return {
      "path": path,
      "expected_size_bytes": expected_size,
      "actual_size_bytes": actual_size,
      "expected_sha256": expected_sha256,
      "actual_sha256": actual_sha256,
      "valid": (
          actual_size == expected_size
          and actual_sha256 == expected_sha256
      ),
  }


def _media_data_path(path: str) -> str:
  return path.replace("/storage/emulated/0/", "/data/media/0/", 1)


def _app_uid_gid(serial: str, package: str) -> str | None:
  result = _adb(serial, ["shell", "stat", "-c", "%u:%g", f"/data/data/{package}"])
  if result.returncode != 0:
    return None
  uid_gid = result.stdout.strip().splitlines()[0].strip()
  return uid_gid if ":" in uid_gid else None


def _repair_app_internal_file(
    serial: str,
    package: str,
    local_path: Path,
    internal_path: str,
    expected_size: int,
    expected_sha256: str,
    uid_gid: str | None,
) -> None:
  if _remote_file_has_identity(
      serial,
      internal_path,
      expected_size,
      expected_sha256,
      root=True,
  ):
    return
  tmp_path = f"/data/local/tmp/catbench_{package}_{local_path.name}"
  _adb(serial, ["push", str(local_path), tmp_path], check=True)
  _adb_root(serial, ["cp", tmp_path, internal_path], check=True)
  _adb(serial, ["shell", "rm", "-f", tmp_path], check=False)
  if uid_gid:
    _adb_root(serial, ["chown", uid_gid, internal_path], check=False)
  _adb_root(serial, ["chmod", "660", internal_path], check=False)
  _adb_root(serial, ["restorecon", internal_path], check=False)
  if not _remote_file_has_identity(
      serial,
      internal_path,
      expected_size,
      expected_sha256,
      root=True,
  ):
    raise RuntimeError(
        f"{serial}: pushed {package} {internal_path}, but remote identity "
        "does not match the frozen size/SHA-256"
    )


def _repair_map_resource_group(serial: str, group: dict[str, object]) -> None:
  package = str(group["package"])
  name = str(group["name"])
  if not _package_installed(serial, package):
    print(f"[PREFLIGHT] {serial}: {name} is not installed; skipping.", flush=True)
    return

  remote_dir = str(group["remote_dir"])
  check_dir = _media_data_path(remote_dir)
  internal_dir = str(group.get("internal_dir") or "")
  base_url = str(group["base_url"])
  cache_dir = Path(tempfile.gettempdir()) / "android_world" / "app_data" / str(
      group["cache_dir"]
  )
  files = group["files"]
  hashes = group["sha256"]
  assert isinstance(files, dict)
  assert isinstance(hashes, dict)
  if set(files) != set(hashes):
    raise RuntimeError(f"{name}: file/hash roster mismatch")

  print(f"[PREFLIGHT] {serial}: repairing {name} offline map resources.", flush=True)
  _adb(serial, ["shell", "am", "force-stop", package], check=False)
  _adb(serial, ["shell", "mkdir", "-p", remote_dir], check=False)
  _adb_root(serial, ["mkdir", "-p", check_dir], check=True)
  _adb_root(serial, ["chmod", "777", check_dir], check=False)
  uid_gid = _app_uid_gid(serial, package)
  if internal_dir:
    _adb_root(serial, ["mkdir", "-p", internal_dir], check=True)
    if uid_gid:
      _adb_root(
          serial,
          ["chown", "-R", uid_gid, f"/data/data/{package}/files"],
          check=False,
      )
    _adb_root(
        serial,
        ["restorecon", "-R", f"/data/data/{package}/files"],
        check=False,
    )
  partial_paths: list[str] = []
  for filename in files:
    partial_paths.extend((
        f"{check_dir}/{filename}.downloading",
        f"{check_dir}/{filename}.resume",
        f"{remote_dir}/{filename}.downloading",
        f"{remote_dir}/{filename}.resume",
    ))
  _adb(serial, ["shell", "rm", "-f", *partial_paths], check=False)
  _adb_root(serial, ["rm", "-f", *partial_paths], check=False)

  for filename, expected_size in files.items():
    filename = str(filename)
    expected_size = int(expected_size)
    expected_sha256 = str(hashes[filename])
    remote_path = f"{remote_dir}/{filename}"
    check_path = f"{check_dir}/{filename}"
    internal_path = f"{internal_dir}/{filename}" if internal_dir else ""
    external_present = _remote_file_has_identity(
        serial,
        check_path,
        expected_size,
        expected_sha256,
        root=True,
    )
    internal_present = (
        bool(internal_path)
        and _remote_file_has_identity(
            serial,
            internal_path,
            expected_size,
            expected_sha256,
            root=True,
        )
    )
    if external_present and (not internal_path or internal_present):
      print(f"[PREFLIGHT] {serial}: {name} {filename} present.", flush=True)
      continue
    local_path = _download_file(
        f"{base_url}/{filename}",
        cache_dir / filename,
        expected_size,
        expected_sha256,
    )
    if not external_present:
      tmp_path = f"/data/local/tmp/catbench_{package}_{filename}"
      _adb(serial, ["push", str(local_path), tmp_path], check=True)
      _adb_root(serial, ["cp", tmp_path, check_path], check=True)
      _adb(serial, ["shell", "rm", "-f", tmp_path], check=False)
      _adb_root(serial, ["chmod", "666", check_path], check=False)
      _adb_root(
          serial,
          ["chcon", "u:object_r:media_rw_data_file:s0", check_path],
          check=False,
      )
      if not _remote_file_has_identity(
          serial,
          check_path,
          expected_size,
          expected_sha256,
          root=True,
      ):
        raise RuntimeError(
            f"{serial}: pushed {name} {filename}, but remote identity does "
            "not match the frozen size/SHA-256"
        )
    if internal_path and not internal_present:
      _repair_app_internal_file(
          serial,
          package,
          local_path,
          internal_path,
          expected_size,
          expected_sha256,
          uid_gid,
      )


def _map_resource_group_valid(
    serial: str, group: dict[str, object]
) -> bool:
  package = str(group["package"])
  name = str(group["name"])
  if not _package_installed(serial, package):
    print(
        f"[PREFLIGHT] {serial}: required {name} package is not installed.",
        file=sys.stderr,
        flush=True,
    )
    return False
  check_dir = _media_data_path(str(group["remote_dir"]))
  internal_dir = str(group.get("internal_dir") or "")
  files = group["files"]
  hashes = group["sha256"]
  assert isinstance(files, dict)
  assert isinstance(hashes, dict)
  valid = True
  for raw_filename, raw_size in files.items():
    filename = str(raw_filename)
    expected_size = int(raw_size)
    expected_sha256 = str(hashes[filename])
    external_path = f"{check_dir}/{filename}"
    internal_path = f"{internal_dir}/{filename}" if internal_dir else ""
    external_ok = _remote_file_has_identity(
        serial,
        external_path,
        expected_size,
        expected_sha256,
        root=True,
    )
    internal_ok = not internal_path or _remote_file_has_identity(
        serial,
        internal_path,
        expected_size,
        expected_sha256,
        root=True,
    )
    if external_ok and internal_ok:
      print(
          f"[PREFLIGHT] {serial}: {name} {filename} identity valid.",
          flush=True,
      )
    else:
      valid = False
      print(
          f"[PREFLIGHT] {serial}: {name} {filename} identity invalid "
          f"(external={external_ok}, internal={internal_ok}).",
          file=sys.stderr,
          flush=True,
      )
  return valid


def _resource_identity_report(serial: str) -> dict[str, object]:
  osmand = _remote_file_identity_record(
      serial,
      OSMAND_MAP_CHECK_PATH,
      OSMAND_MAP_SIZE_BYTES,
      OSMAND_MAP_SHA256,
      root=True,
  )
  groups: list[dict[str, object]] = []
  for group in MAP_RESOURCE_GROUPS:
    package = str(group["package"])
    files = group["files"]
    hashes = group["sha256"]
    assert isinstance(files, dict)
    assert isinstance(hashes, dict)
    check_dir = _media_data_path(str(group["remote_dir"]))
    internal_dir = str(group.get("internal_dir") or "")
    file_records = []
    for raw_filename, raw_size in files.items():
      filename = str(raw_filename)
      expected_size = int(raw_size)
      expected_sha256 = str(hashes[filename])
      external = _remote_file_identity_record(
          serial,
          f"{check_dir}/{filename}",
          expected_size,
          expected_sha256,
          root=True,
      )
      internal = (
          _remote_file_identity_record(
              serial,
              f"{internal_dir}/{filename}",
              expected_size,
              expected_sha256,
              root=True,
          )
          if internal_dir
          else None
      )
      file_records.append({
          "filename": filename,
          "external": external,
          "internal": internal,
          "valid": bool(external["valid"])
          and (internal is None or bool(internal["valid"])),
      })
    groups.append({
        "name": str(group["name"]),
        "package_name": package,
        "installed": _package_installed(serial, package),
        "map_series": str(group["remote_dir"]).rsplit("/", 1)[-1],
        "files": file_records,
        "valid": _package_installed(serial, package)
        and all(bool(record["valid"]) for record in file_records),
    })
  return {
      "osmand": osmand,
      "groups": groups,
      "valid": bool(osmand["valid"])
      and all(bool(group["valid"]) for group in groups),
  }


def _heal_map_package_state(serial: str) -> None:
  for package in MAP_PACKAGES:
    if not _package_installed(serial, package):
      continue
    _adb(serial, ["shell", "pm", "enable", "--user", "0", package], check=False)
    for permission in MAP_PERMISSIONS:
      _adb(serial, ["shell", "pm", "grant", package, permission], check=False)


def _device_runtime_health(serial: str) -> dict[str, object]:
  """Fail closed on a boot, root-ADB, or system-dialog fault before mutation."""
  boot = _adb(serial, ["shell", "getprop", "sys.boot_completed"])
  uid = _adb(serial, ["shell", "id", "-u"])
  windows = _adb(serial, ["shell", "dumpsys", "window", "windows"])
  window_text = windows.stdout
  fault_markers = (
      "Application Not Responding:",
      "Application Error:",
      "isn't responding",
      "is not responding",
      "has stopped",
      "keeps stopping",
  )
  lowered_windows = window_text.lower()
  faults = [
      marker for marker in fault_markers
      if marker.lower() in lowered_windows
  ]
  valid = (
      boot.returncode == 0
      and boot.stdout.strip() == "1"
      and uid.returncode == 0
      and uid.stdout.strip() == "0"
      and windows.returncode == 0
      and not faults
  )
  return {
      "boot_completed": boot.stdout.strip(),
      "adb_uid": uid.stdout.strip(),
      "window_fault_markers": faults,
      "valid": valid,
  }


def preflight(serial: str, *, repair: bool) -> int:
  health = _device_runtime_health(serial)
  if not health["valid"]:
    print(
        f"[PREFLIGHT] {serial}: emulator runtime health invalid: {health}",
        file=sys.stderr,
        flush=True,
    )
    return 1
  if not _package_installed(serial, "net.osmand.plus"):
    print(
        f"[PREFLIGHT] {serial}: required OsmAnd package is not installed.",
        file=sys.stderr,
        flush=True,
    )
    _heal_map_package_state(serial)
    return 1

  has_map = _remote_file_has_identity(
      serial,
      OSMAND_MAP_CHECK_PATH,
      OSMAND_MAP_SIZE_BYTES,
      OSMAND_MAP_SHA256,
      root=True,
  )
  if not has_map and repair:
    print(f"[PREFLIGHT] {serial}: repairing missing OsmAnd AW map.", flush=True)
    _repair_osmand_map(serial)
    has_map = _remote_file_has_identity(
        serial,
        OSMAND_MAP_CHECK_PATH,
        OSMAND_MAP_SIZE_BYTES,
        OSMAND_MAP_SHA256,
        root=True,
    )

  _heal_map_package_state(serial)
  groups_valid = True
  for group in MAP_RESOURCE_GROUPS:
    if repair:
      _repair_map_resource_group(serial, group)
    groups_valid = _map_resource_group_valid(serial, group) and groups_valid

  if has_map and groups_valid:
    print(
        f"[PREFLIGHT] {serial}: all frozen offline map identities valid.",
        flush=True,
    )
    return 0

  if not has_map:
    print(
        f"[PREFLIGHT] {serial}: missing or invalid "
        f"{OSMAND_MAP_CHECK_PATH}. Run again with --repair on a mutable "
        "provisioning volume before Maps benchmarks.",
        file=sys.stderr,
        flush=True,
    )
  if not groups_valid:
    print(
        f"[PREFLIGHT] {serial}: one or more Organic Maps/CoMaps resources "
        "are missing or byte-invalid.",
        file=sys.stderr,
        flush=True,
    )
  return 1


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--serial",
      default=os.environ.get("ANDROID_SERIAL", ""),
      help="ADB serial, e.g. emulator-5574. Defaults to ANDROID_SERIAL.",
  )
  parser.add_argument(
      "--repair",
      action="store_true",
      help=(
          "Download and push all frozen offline Maps resources to a mutable"
          " provisioning volume, with exact size/SHA-256 verification."
      ),
  )
  parser.add_argument(
      "--output",
      default="",
      help="Optional exclusive machine-readable evidence path.",
  )
  parser.add_argument(
      "--cohort_manifest",
      default=str(DEFAULT_COHORT_MANIFEST),
      help="Frozen cohort manifest bound into the output evidence.",
  )
  parser.add_argument(
      "--docker_image_digest",
      default="",
      help="Exact Docker image digest; required when --output is used.",
  )
  args = parser.parse_args()

  if not args.serial:
    parser.error("--serial is required when ANDROID_SERIAL is not set")
  if args.output and not args.docker_image_digest:
    parser.error("--docker_image_digest is required with --output")
  cohort_path = Path(args.cohort_manifest).expanduser().resolve()
  cohort = _strict_json(cohort_path)
  result = preflight(args.serial, repair=args.repair)
  if args.output:
    resources = _resource_identity_report(args.serial)
    payload = {
        "schema_version": 1,
        "audit_type": "catbench_primary_maps_offline_resource_identity",
        "artifact_role": "provisioning_preflight_only_not_task_or_model_result",
        "analysis_eligible": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repair_performed": args.repair,
        "release_id": cohort.get("release_id"),
        "cohort_manifest": {
            "path": str(cohort_path),
            "sha256": _sha256_file(cohort_path),
        },
        "runtime": {
            "docker_image_digest": args.docker_image_digest,
            "serial": args.serial,
            "adb_server_port": (
                os.environ.get("ANDROID_ADB_SERVER_PORT")
                or os.environ.get("ADB_SERVER_PORT")
                or "default"
            ),
            "boot_completed": _adb(
                args.serial, ["shell", "getprop", "sys.boot_completed"]
            ).stdout.strip(),
            "android_release": _adb(
                args.serial, ["shell", "getprop", "ro.build.version.release"]
            ).stdout.strip(),
            "health": _device_runtime_health(args.serial),
            "api_level": _adb(
                args.serial, ["shell", "getprop", "ro.build.version.sdk"]
            ).stdout.strip(),
            "fingerprint": _adb(
                args.serial, ["shell", "getprop", "ro.build.fingerprint"]
            ).stdout.strip(),
        },
        "source": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "resources": resources,
        "valid": result == 0 and bool(resources["valid"]),
        "execution_claims": {
            "benchmark_episode_executed": False,
            "model_endpoint_called": False,
            "agent_action_generated": False,
            "full_maps_task_or_ui_conformance_established": False,
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
      json.dump(payload, handle, indent=2)
      handle.write("\n")
  return result


if __name__ == "__main__":
  raise SystemExit(main())
