#!/usr/bin/env python3
"""Provision and attest the exact frozen CATBench apps on one Android device.

This utility is deliberately stricter than the general app bootstrapper.  It
accepts only the complete app roster of a frozen cohort, resolves every local
artifact by its pinned base-APK SHA-256, verifies every APK signature with
``apksigner``, checks package/version metadata with ``aapt``/``aapt2``, and
then re-pulls the active APK set from the device for byte-for-byte comparison.

Two explicit modes are supported:

* ``provision`` installs a locally verified APK set when the active device set
  is not already exact, then performs the full post-install attestation.
* ``attest`` is read-only and fails if any active APK, split, version, package,
  or signer differs from the locally verified set.

The JSON output is machine-collected evidence.  It intentionally declares
that it is *not* the independently approved release attestation required by
``consume_catbench_frozen_schedule.py``.  A reviewer must bind the evidence to
an immutable base snapshot and approve the signer identities separately.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
import zipfile


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import audit_pinned_app_signers as signer_audit


DEFAULT_COHORT = BENCHMARK_ROOT / "configs" / "catbench_5cat_primary_cohort.json"
DEFAULT_PINS = BENCHMARK_ROOT / "configs" / "app_versions_pinned.csv"
DEFAULT_APPS = BENCHMARK_ROOT / "app_generalization_apps.csv"
DEFAULT_SIGNER_AUDIT = (
    BENCHMARK_ROOT / "docs" / "audits" / "pinned_app_signer_audit.json"
)
DEFAULT_ARTIFACT_ROOT = Path(
    "$HOME/anyappbench_apks"
)

HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PACKAGE_LINE = re.compile(r"^package:\s+(.*)$", re.MULTILINE)
BADGING_FIELD = re.compile(r"([A-Za-z][A-Za-z0-9_]*)='([^']*)'")
VERSION_CODE = re.compile(r"\bversionCode=(\d+)\b")
VERSION_NAME = re.compile(r"^\s*versionName=(.*?)\s*$", re.MULTILINE)
PM_PATH_LINE = re.compile(r"^package:(/\S.*)$")
BADGING_MISSING_SPLIT_RESOURCE = re.compile(
    r"^AndroidManifest\.xml:\d+:\s+error:\s+ERROR getting "
    r"'android:(?:icon|roundIcon|banner)' attribute:\s+"
    r"attribute value reference does not exist\s*$",
    re.IGNORECASE,
)

ABI_SPLITS = {
    "arm64-v8a": "config.arm64_v8a",
    "armeabi-v7a": "config.armeabi_v7a",
    "x86_64": "config.x86_64",
    "x86": "config.x86",
}
DENSITY_SPLITS = {
    "config.ldpi": 120,
    "config.mdpi": 160,
    "config.tvdpi": 213,
    "config.hdpi": 240,
    "config.xhdpi": 320,
    "config.xxhdpi": 480,
    "config.xxxhdpi": 640,
}


class ProvisionError(RuntimeError):
  """A fail-closed input, local-artifact, ADB, or device validation error."""


@dataclasses.dataclass(frozen=True)
class AppPin:
  app_id: str
  category: str
  package_name: str
  version_name: str
  version_code: str
  apk_sha256: str
  artifact_path: Path
  expected_signers: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ApkFile:
  split_id: str
  path: Path
  sha256: str
  package_name: str
  version_name: str
  version_code: str
  signer_leaf_certificate_sha256: tuple[str, ...]
  verified_schemes: Mapping[str, bool]


@dataclasses.dataclass(frozen=True)
class PreparedApp:
  pin: AppPin
  artifact_sha256: str
  artifact_scope: str
  artifact_member: str
  apks: tuple[ApkFile, ...]


@dataclasses.dataclass(frozen=True)
class DeviceProfile:
  serial: str
  adb_server_port: int | None
  build_fingerprint: str
  api_level: str
  abi_list: tuple[str, ...]
  density: int
  locale: str
  boot_id: str


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _run(
    command: Sequence[str],
    *,
    timeout: int = 180,
    run_command: RunCommand = subprocess.run,
) -> subprocess.CompletedProcess[str]:
  try:
    return run_command(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
  except (OSError, subprocess.TimeoutExpired) as exc:
    raise ProvisionError(f"Command failed to execute: {command[0]}: {exc}") from exc


def _sha256_path(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _sha256_json(value: Any) -> str:
  encoded = json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _regular_input(path: Path, label: str) -> Path:
  expanded = path.expanduser().resolve()
  if path.expanduser().is_symlink() or not expanded.is_file():
    raise ProvisionError(f"{label} must be a regular non-symlink file: {path}")
  return expanded


def _strict_json_loads(raw: str, label: str) -> Any:
  def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
      if key in value:
        raise ProvisionError(f"Duplicate JSON key {key!r} in {label}")
      value[key] = item
    return value

  def reject_constant(value: str) -> None:
    raise ProvisionError(f"Non-finite JSON value {value!r} in {label}")

  return json.loads(
      raw,
      object_pairs_hook=object_from_pairs,
      parse_constant=reject_constant,
  )


def _load_json(path: Path, label: str) -> dict[str, Any]:
  path = _regular_input(path, label)
  try:
    value = _strict_json_loads(path.read_text(encoding="utf-8"), label)
  except (OSError, json.JSONDecodeError) as exc:
    raise ProvisionError(f"Unable to read {label}: {path}: {exc}") from exc
  if not isinstance(value, dict):
    raise ProvisionError(f"{label} must contain one JSON object: {path}")
  return value


def _load_csv(
    path: Path,
    *,
    label: str,
    required_fields: Sequence[str],
) -> tuple[Path, dict[str, dict[str, str]]]:
  path = _regular_input(path, label)
  rows: dict[str, dict[str, str]] = {}
  try:
    with path.open("r", encoding="utf-8", newline="") as handle:
      reader = csv.DictReader(handle)
      if reader.fieldnames is None or any(
          field not in reader.fieldnames for field in required_fields
      ):
        raise ProvisionError(
            f"{label} lacks required columns {list(required_fields)}; "
            f"found {reader.fieldnames}"
        )
      for raw in reader:
        row = {str(key): str(value or "").strip() for key, value in raw.items()}
        app_id = row.get("app_id", "")
        if not app_id or app_id in rows:
          raise ProvisionError(f"{label} has duplicate/empty app_id {app_id!r}")
        rows[app_id] = row
  except OSError as exc:
    raise ProvisionError(f"Unable to read {label}: {path}: {exc}") from exc
  return path, rows


def _frozen_roster(cohort: Mapping[str, Any]) -> list[tuple[str, str]]:
  categories = cohort.get("categories")
  if not isinstance(categories, dict) or not categories:
    raise ProvisionError("Cohort categories must be a non-empty object")
  roster: list[tuple[str, str]] = []
  seen: set[str] = set()
  for category, spec in categories.items():
    if not isinstance(spec, dict) or not isinstance(spec.get("app_ids"), list):
      raise ProvisionError(f"Cohort category {category!r} lacks app_ids")
    for raw_app_id in spec["app_ids"]:
      app_id = str(raw_app_id)
      if not app_id or app_id in seen:
        raise ProvisionError(f"Cohort has duplicate/empty app_id {app_id!r}")
      seen.add(app_id)
      roster.append((str(category), app_id))
  return roster


def _validate_hash(value: Any, label: str) -> str:
  normalized = str(value or "").lower()
  if not HEX_SHA256.fullmatch(normalized):
    raise ProvisionError(f"{label} is not a lowercase SHA-256")
  return normalized


def _signer_expectations(
    audit: Mapping[str, Any],
    *,
    cohort: Mapping[str, Any],
    cohort_sha256: str,
    pins_sha256: str,
    apps_sha256: str,
    roster: Sequence[tuple[str, str]],
    pins: Mapping[str, Mapping[str, str]],
) -> dict[str, tuple[str, ...]]:
  expected_header = {
      "audit_type": "frozen_real_app_signing_certificate_observation",
      "pinning_status": "observational_report_not_an_approved_signer_pinset",
      "auditor_script_sha256": _sha256_path(Path(signer_audit.__file__).resolve()),
      "cohort_release_id": cohort.get("release_id"),
      "cohort_manifest_sha256": cohort_sha256,
      "pins_file_sha256": pins_sha256,
      "app_inventory_file_sha256": apps_sha256,
      "expected_apps": len(roster),
      "artifact_identity_valid_apps": len(roster),
      "signer_identity_known_apps": len(roster),
      "fully_cryptographically_verified_apps": len(roster),
      "release_gate_valid": True,
  }
  for field, expected in expected_header.items():
    if audit.get(field) != expected:
      raise ProvisionError(
          f"Signer audit {field}={audit.get(field)!r}; expected {expected!r}"
      )
  raw_rows = audit.get("apps")
  if not isinstance(raw_rows, list):
    raise ProvisionError("Signer audit apps must be a list")
  expected_ids = [app_id for _, app_id in roster]
  observed_ids = [
      str(row.get("app_id") or "") if isinstance(row, dict) else ""
      for row in raw_rows
  ]
  if observed_ids != expected_ids:
    raise ProvisionError("Signer audit roster/order differs from frozen cohort")
  result: dict[str, tuple[str, ...]] = {}
  for row in raw_rows:
    if not isinstance(row, dict):
      raise ProvisionError("Signer audit contains a malformed app row")
    app_id = str(row["app_id"])
    pin = pins[app_id]
    expected_identity = {
        "category": pin["category"],
        "package_name": pin["package_name"],
        "pinned_version_name": pin["version_name"],
        "pinned_version_code": pin["version_code"],
        "pinned_apk_sha256": pin["apk_sha256"],
        "signer_identity_known": True,
        "fully_cryptographically_verified": True,
    }
    for field, expected in expected_identity.items():
      if row.get(field) != expected:
        raise ProvisionError(f"Signer audit {app_id} {field} mismatch")
    identity = row.get("artifact_identity")
    if not isinstance(identity, dict) or identity.get("valid") is not True:
      raise ProvisionError(f"Signer audit {app_id} artifact identity is invalid")
    extraction = row.get("certificate_extraction")
    verification = row.get("signature_verification")
    if not isinstance(extraction, dict) or not isinstance(verification, dict):
      raise ProvisionError(f"Signer audit {app_id} lacks verification evidence")
    signers = extraction.get("signer_leaf_certificate_sha256")
    if (
        not isinstance(signers, list)
        or not signers
        or signers != sorted(set(map(str, signers)))
    ):
      raise ProvisionError(f"Signer audit {app_id} has invalid signer leaves")
    normalized = tuple(
        _validate_hash(value, f"Signer audit {app_id} certificate")
        for value in signers
    )
    verified = tuple(
        sorted(
            _validate_hash(value, f"Signer audit {app_id} verified certificate")
            for value in verification.get(
                "verified_signer_certificate_sha256", []
            )
        )
    )
    if (
        verification.get("status") != "verified"
        or verification.get("fully_cryptographically_verified") is not True
        or verification.get(
            "certificate_fingerprints_consistent_with_extraction"
        ) is not True
        or verified != normalized
    ):
      raise ProvisionError(f"Signer audit {app_id} is not fully consistent")
    result[app_id] = normalized
  return result


def load_frozen_inputs(
    cohort_path: Path,
    pins_path: Path,
    apps_path: Path,
    signer_audit_path: Path,
    artifact_root: Path,
) -> tuple[
    dict[str, Any],
    str,
    str,
    str,
    str,
    tuple[AppPin, ...],
]:
  cohort_path = _regular_input(cohort_path, "cohort manifest")
  cohort = _load_json(cohort_path, "cohort manifest")
  cohort_sha256 = _sha256_path(cohort_path)
  roster = _frozen_roster(cohort)
  pins_path, pins = _load_csv(
      pins_path,
      label="app pin manifest",
      required_fields=(
          "category",
          "app_id",
          "package_name",
          "version_name",
          "version_code",
          "apk_sha256",
      ),
  )
  apps_path, apps = _load_csv(
      apps_path,
      label="app inventory",
      required_fields=("app_id", "package_name", "apk_filename"),
  )
  pins_sha256 = _sha256_path(pins_path)
  apps_sha256 = _sha256_path(apps_path)
  signer_audit_path = _regular_input(signer_audit_path, "signer audit")
  audit = _load_json(signer_audit_path, "signer audit")
  signers = _signer_expectations(
      audit,
      cohort=cohort,
      cohort_sha256=cohort_sha256,
      pins_sha256=pins_sha256,
      apps_sha256=apps_sha256,
      roster=roster,
      pins=pins,
  )
  root = artifact_root.expanduser().resolve()
  if not root.is_dir():
    raise ProvisionError(f"Artifact root is not a directory: {root}")
  result: list[AppPin] = []
  seen_packages: set[str] = set()
  for category, app_id in roster:
    if app_id not in pins or app_id not in apps:
      raise ProvisionError(f"Frozen app lacks pin or inventory row: {app_id}")
    pin = pins[app_id]
    inventory = apps[app_id]
    for field in ("package_name", "version_name", "version_code"):
      if not pin.get(field):
        raise ProvisionError(f"Pin {app_id} lacks {field}")
    if pin["category"] != category:
      raise ProvisionError(f"Pin {app_id} category differs from cohort")
    if inventory["package_name"] != pin["package_name"]:
      raise ProvisionError(f"Inventory {app_id} package differs from pin")
    if pin["package_name"] in seen_packages:
      raise ProvisionError(
          f"Frozen roster maps multiple app IDs to {pin['package_name']}"
      )
    seen_packages.add(pin["package_name"])
    relative = Path(inventory["apk_filename"])
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
      raise ProvisionError(f"Inventory {app_id} has unsafe apk_filename")
    artifact_path = (root / relative).resolve()
    if root not in artifact_path.parents:
      raise ProvisionError(f"Inventory {app_id} artifact escapes artifact root")
    if (root / relative).is_symlink() or not artifact_path.is_file():
      raise ProvisionError(
          f"Artifact {app_id} must be a regular non-symlink file: {artifact_path}"
      )
    result.append(AppPin(
        app_id=app_id,
        category=category,
        package_name=pin["package_name"],
        version_name=pin["version_name"],
        version_code=pin["version_code"],
        apk_sha256=_validate_hash(pin["apk_sha256"], f"Pin {app_id} APK"),
        artifact_path=artifact_path,
        expected_signers=signers[app_id],
    ))
  return (
      cohort,
      cohort_sha256,
      pins_sha256,
      apps_sha256,
      _sha256_path(signer_audit_path),
      tuple(result),
  )


def _tool_path(explicit: str, candidates: Sequence[str], label: str) -> Path:
  raw = explicit or next(
      (resolved for name in candidates if (resolved := shutil.which(name))), ""
  )
  if not raw:
    raise ProvisionError(f"Required {label} tool is unavailable")
  path = Path(raw).expanduser().resolve()
  if not path.is_file() or not os.access(path, os.X_OK):
    raise ProvisionError(f"Required {label} tool is not executable: {path}")
  return path


def _parse_badging(output: str, source: str) -> tuple[str, str, str, str]:
  match = PACKAGE_LINE.search(output)
  if not match:
    raise ProvisionError(f"aapt did not emit package metadata for {source}")
  fields = dict(BADGING_FIELD.findall(match.group(1)))
  package_name = fields.get("name", "")
  version_name = fields.get("versionName", "")
  version_code = fields.get("versionCode", "")
  split_id = fields.get("split", "base")
  # Android configuration splits commonly encode an empty versionName while
  # retaining the base package/versionCode.  The base must always carry the
  # exact non-empty versionName; split rows may be empty and are checked
  # against the same package, versionCode, signer, and declared split ID.
  if (
      not package_name
      or not version_code
      or (split_id == "base" and not version_name)
  ):
    raise ProvisionError(f"aapt emitted incomplete package metadata for {source}")
  return package_name, version_name, version_code, split_id


def _parse_manifest_xmltree(
    output: str, source: str
) -> tuple[str, str, str, str]:
  """Parses package identity directly from the binary AndroidManifest.

  Unlike ``dump badging``, ``dump xmltree`` does not try to resolve an app
  icon or banner from a configuration split.  It is therefore the
  authoritative fallback for a base APK whose manifest is readable but whose
  presentation resources live in another APK in the same declared bundle.
  Only attributes on the root ``manifest`` element are considered.
  """
  lines = output.splitlines()
  manifest_indexes = [
      index
      for index, line in enumerate(lines)
      if re.match(r"^\s*E:\s+manifest(?:\s|$)", line)
  ]
  if len(manifest_indexes) != 1:
    raise ProvisionError(
        f"aapt xmltree did not emit one manifest root for {source}"
    )
  attributes: dict[str, str] = {}
  for line in lines[manifest_indexes[0] + 1:]:
    stripped = line.strip()
    if stripped.startswith("E:"):
      break
    if not stripped.startswith("A:"):
      continue
    attribute = stripped[2:].strip()
    field = ""
    for candidate in ("versionCode", "versionName", "package", "split"):
      if candidate in {"package", "split"}:
        matched = re.match(
            rf"^{candidate}(?:\(0x[0-9a-fA-F]+\))?=", attribute
        )
      else:
        matched = re.search(
            rf"(?:^|:){candidate}(?:\(0x[0-9a-fA-F]+\))?=", attribute
        )
      if matched:
        field = candidate
        raw_value = attribute[matched.end():].strip()
        break
    if not field:
      continue
    if field in attributes:
      raise ProvisionError(
          f"aapt xmltree emitted duplicate {field} for {source}"
      )
    if field == "versionCode":
      if raw_value.startswith("(type ") and ")" in raw_value:
        raw_value = raw_value.split(")", 1)[1].strip()
      code = re.match(r"^(0x[0-9a-fA-F]+|\d+)(?:\s|$)", raw_value)
      if not code:
        raise ProvisionError(
            f"aapt xmltree emitted invalid versionCode for {source}"
        )
      code_text = code.group(1)
      code_base = 16 if code_text.lower().startswith("0x") else 10
      attributes[field] = str(int(code_text, code_base))
    else:
      string = re.match(r'^"([^"]*)"(?:\s|$)', raw_value)
      if not string:
        raise ProvisionError(
            f"aapt xmltree emitted non-literal {field} for {source}"
        )
      attributes[field] = string.group(1)
  package_name = attributes.get("package", "")
  version_name = attributes.get("versionName", "")
  version_code = attributes.get("versionCode", "")
  split_id = attributes.get("split", "base")
  if (
      not package_name
      or not version_code
      or (split_id == "base" and not version_name)
  ):
    raise ProvisionError(
        f"aapt xmltree emitted incomplete package metadata for {source}"
    )
  return package_name, version_name, version_code, split_id


def _manifest_xmltree_metadata(
    path: Path,
    *,
    aapt_path: Path,
    run_command: RunCommand,
) -> tuple[str, str, str, str]:
  if aapt_path.name.lower().startswith("aapt2"):
    command = [
        str(aapt_path), "dump", "xmltree", "--file", "AndroidManifest.xml",
        str(path),
    ]
  else:
    command = [
        str(aapt_path), "dump", "xmltree", str(path), "AndroidManifest.xml",
    ]
  xmltree = _run(command, timeout=180, run_command=run_command)
  if xmltree.returncode != 0:
    raise ProvisionError(
        f"aapt could not read AndroidManifest.xml for {path}: "
        f"{(xmltree.stdout + xmltree.stderr).strip()}"
    )
  return _parse_manifest_xmltree(xmltree.stdout, str(path))


def _is_only_missing_split_resource_error(output: str) -> bool:
  error_lines = [
      line.strip()
      for line in output.splitlines()
      if re.search(r"\berror:", line, re.IGNORECASE)
  ]
  return (
      len(error_lines) == 1
      and BADGING_MISSING_SPLIT_RESOURCE.fullmatch(error_lines[0]) is not None
  )


def _inspect_apk(
    path: Path,
    *,
    aapt_path: Path,
    apksigner_path: Path,
    expected_signers: tuple[str, ...],
    allow_split_resource_manifest_fallback: bool = False,
    run_command: RunCommand = subprocess.run,
) -> ApkFile:
  badging = _run(
      [str(aapt_path), "dump", "badging", str(path)],
      timeout=180,
      run_command=run_command,
  )
  if badging.returncode == 0:
    package_name, version_name, version_code, split_id = _parse_badging(
        badging.stdout, str(path)
    )
  else:
    diagnostic = badging.stdout + "\n" + badging.stderr
    if (
        not allow_split_resource_manifest_fallback
        or not _is_only_missing_split_resource_error(diagnostic)
    ):
      raise ProvisionError(f"aapt rejected {path}: {diagnostic.strip()}")
    # Do not accept the partial output alone.  Require it to be complete and
    # exactly consistent with a successful direct parse of the binary
    # manifest.  Hash, pinned package/version, split-ID, and signature gates
    # below remain unchanged.
    badging_metadata = _parse_badging(badging.stdout, str(path))
    manifest_metadata = _manifest_xmltree_metadata(
        path,
        aapt_path=aapt_path,
        run_command=run_command,
    )
    if badging_metadata != manifest_metadata:
      raise ProvisionError(
          f"aapt badging/xmltree metadata mismatch for {path}: "
          f"{badging_metadata!r} != {manifest_metadata!r}"
      )
    package_name, version_name, version_code, split_id = manifest_metadata
  extraction = signer_audit.extract_certificates(path)
  verification = signer_audit.verify_signature(
      path,
      extraction,
      apksigner_path=str(apksigner_path),
      jarsigner_path=None,
  )
  extracted = tuple(extraction.get("signer_leaf_certificate_sha256", []))
  verified = tuple(
      verification.get("verified_signer_certificate_sha256", [])
  )
  verified_schemes = dict(verification.get("verified_schemes") or {})
  if (
      verification.get("status") != "verified"
      or verification.get("fully_cryptographically_verified") is not True
      or verification.get(
          "certificate_fingerprints_consistent_with_extraction"
      ) is not True
      or extracted != expected_signers
      or verified != expected_signers
      or not any(verified_schemes.values())
  ):
    raise ProvisionError(f"APK signer verification failed or mismatched: {path}")
  return ApkFile(
      split_id=split_id,
      path=path,
      sha256=_sha256_path(path),
      package_name=package_name,
      version_name=version_name,
      version_code=version_code,
      signer_leaf_certificate_sha256=expected_signers,
      verified_schemes=verified_schemes,
  )


def _safe_extract_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    destination: Path,
) -> Path:
  pure = PurePosixPath(member.filename)
  if (
      member.is_dir()
      or pure.is_absolute()
      or ".." in pure.parts
      or not pure.name.lower().endswith(".apk")
  ):
    raise ProvisionError(f"Unsafe or non-APK XAPK member: {member.filename!r}")
  target = destination / pure.name
  if target.exists():
    raise ProvisionError(f"Duplicate XAPK member basename: {pure.name}")
  with archive.open(member) as source, target.open("xb") as output:
    shutil.copyfileobj(source, output, length=1024 * 1024)
  return target


def _prepare_bundle(
    pin: AppPin,
    destination: Path,
) -> tuple[str, str, list[tuple[str, Path]]]:
  try:
    with zipfile.ZipFile(pin.artifact_path) as archive:
      apk_infos = [
          info
          for info in archive.infolist()
          if not info.is_dir() and info.filename.lower().endswith(".apk")
      ]
      if not apk_infos:
        raise ProvisionError(f"Bundle contains no APK members: {pin.artifact_path}")
      exact = []
      for info in apk_infos:
        with archive.open(info) as handle:
          digest = hashlib.sha256()
          for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() == pin.apk_sha256:
          exact.append(info.filename)
      if len(exact) != 1:
        raise ProvisionError(
            f"Bundle {pin.app_id} must contain exactly one pinned base APK; "
            f"found {exact}"
        )
      manifest_infos = [
          info for info in archive.infolist() if info.filename == "manifest.json"
      ]
      if len(manifest_infos) != 1:
        raise ProvisionError(
            f"Bundle {pin.app_id} must contain exactly one manifest.json"
        )
      try:
        manifest = _strict_json_loads(
            archive.read(manifest_infos[0]).decode("utf-8"),
            f"XAPK manifest for {pin.app_id}",
        )
      except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvisionError(f"Invalid XAPK manifest for {pin.app_id}: {exc}") from exc
      expected_manifest = {
          "package_name": pin.package_name,
          "version_name": pin.version_name,
          "version_code": pin.version_code,
      }
      for field, expected in expected_manifest.items():
        if str(manifest.get(field) or "") != expected:
          raise ProvisionError(f"XAPK manifest {pin.app_id} {field} mismatch")
      split_rows = manifest.get("split_apks")
      if not isinstance(split_rows, list) or not split_rows:
        raise ProvisionError(f"XAPK manifest {pin.app_id} lacks split_apks")
      file_to_id: dict[str, str] = {}
      seen_split_ids: set[str] = set()
      for row in split_rows:
        if not isinstance(row, dict):
          raise ProvisionError(f"Malformed XAPK split row for {pin.app_id}")
        filename = str(row.get("file") or "")
        split_id = str(row.get("id") or "")
        if (
            not filename
            or not split_id
            or filename in file_to_id
            or split_id in seen_split_ids
        ):
          raise ProvisionError(f"Duplicate/empty XAPK split row for {pin.app_id}")
        file_to_id[filename] = split_id
        seen_split_ids.add(split_id)
      member_names = [info.filename for info in apk_infos]
      if set(member_names) != set(file_to_id):
        raise ProvisionError(
            f"XAPK manifest/member APK set differs for {pin.app_id}"
        )
      if file_to_id[exact[0]] != "base":
        raise ProvisionError(f"Pinned XAPK member is not declared base: {pin.app_id}")
      extracted: list[tuple[str, Path]] = []
      for info in apk_infos:
        path = _safe_extract_member(archive, info, destination)
        extracted.append((file_to_id[info.filename], path))
      return "zip_member_exact_hash", exact[0], extracted
  except zipfile.BadZipFile as exc:
    raise ProvisionError(f"Invalid APK bundle {pin.artifact_path}: {exc}") from exc


def prepare_local_apps(
    pins: Sequence[AppPin],
    *,
    workspace: Path,
    aapt_path: Path,
    apksigner_path: Path,
    run_command: RunCommand = subprocess.run,
) -> tuple[PreparedApp, ...]:
  prepared: list[PreparedApp] = []
  for index, pin in enumerate(pins):
    app_root = workspace / f"{index:02d}_{pin.app_id}"
    app_root.mkdir(parents=True, exist_ok=False)
    artifact_sha256 = _sha256_path(pin.artifact_path)
    if pin.artifact_path.suffix.lower() in {".xapk", ".apks"}:
      artifact_scope, artifact_member, sources = _prepare_bundle(pin, app_root)
    else:
      if artifact_sha256 != pin.apk_sha256:
        raise ProvisionError(
            f"Artifact SHA-256 differs from pin for {pin.app_id}"
        )
      artifact_scope, artifact_member = "artifact_file", ""
      sources = [("base", pin.artifact_path)]
    if _sha256_path(pin.artifact_path) != artifact_sha256:
      raise ProvisionError(f"Artifact changed during preflight: {pin.app_id}")
    apk_rows: list[ApkFile] = []
    seen_splits: set[str] = set()
    for declared_split_id, path in sources:
      apk = _inspect_apk(
          path,
          aapt_path=aapt_path,
          apksigner_path=apksigner_path,
          expected_signers=pin.expected_signers,
          allow_split_resource_manifest_fallback=len(sources) > 1,
          run_command=run_command,
      )
      if apk.split_id != declared_split_id:
        raise ProvisionError(
            f"Declared/aapt split ID mismatch for {pin.app_id}: "
            f"{declared_split_id!r} != {apk.split_id!r}"
        )
      if apk.split_id in seen_splits:
        raise ProvisionError(f"Duplicate APK split ID for {pin.app_id}: {apk.split_id}")
      seen_splits.add(apk.split_id)
      if (
          apk.package_name != pin.package_name
          or apk.version_code != pin.version_code
          or (
              apk.split_id == "base"
              and apk.version_name != pin.version_name
          )
          or (
              apk.split_id != "base"
              and apk.version_name not in {"", pin.version_name}
          )
      ):
        raise ProvisionError(f"APK package/version mismatch for {pin.app_id}")
      apk_rows.append(apk)
    bases = [apk for apk in apk_rows if apk.split_id == "base"]
    if len(bases) != 1 or bases[0].sha256 != pin.apk_sha256:
      raise ProvisionError(f"Pinned base APK not uniquely resolved for {pin.app_id}")
    apk_rows.sort(key=lambda apk: (apk.split_id != "base", apk.split_id))
    prepared.append(PreparedApp(
        pin=pin,
        artifact_sha256=artifact_sha256,
        artifact_scope=artifact_scope,
        artifact_member=artifact_member,
        apks=tuple(apk_rows),
    ))
  return tuple(prepared)


def _adb_base(adb_path: Path, serial: str, adb_server_port: int | None) -> list[str]:
  command = [str(adb_path)]
  if adb_server_port is not None:
    command.extend(["-P", str(adb_server_port)])
  command.extend(["-s", serial])
  return command


def _adb(
    adb_path: Path,
    serial: str,
    adb_server_port: int | None,
    args: Sequence[str],
    *,
    timeout: int = 180,
    run_command: RunCommand = subprocess.run,
) -> subprocess.CompletedProcess[str]:
  return _run(
      [*_adb_base(adb_path, serial, adb_server_port), *args],
      timeout=timeout,
      run_command=run_command,
  )


def _required_adb_text(
    adb_path: Path,
    serial: str,
    adb_server_port: int | None,
    args: Sequence[str],
    label: str,
    *,
    run_command: RunCommand = subprocess.run,
) -> str:
  result = _adb(
      adb_path,
      serial,
      adb_server_port,
      args,
      run_command=run_command,
  )
  if result.returncode != 0:
    raise ProvisionError(
        f"ADB {label} failed: {(result.stdout + result.stderr).strip()}"
    )
  return result.stdout.strip()


def collect_device_profile(
    *,
    adb_path: Path,
    serial: str,
    adb_server_port: int | None,
    run_command: RunCommand = subprocess.run,
) -> DeviceProfile:
  state = _required_adb_text(
      adb_path, serial, adb_server_port, ["get-state"], "get-state",
      run_command=run_command,
  )
  if state != "device":
    raise ProvisionError(f"ADB device is not ready: {serial}: {state!r}")
  booted = _required_adb_text(
      adb_path,
      serial,
      adb_server_port,
      ["shell", "getprop", "sys.boot_completed"],
      "boot state",
      run_command=run_command,
  ).replace("\r", "")
  if booted != "1":
    raise ProvisionError(f"Android device is not boot-complete: {serial}")

  def prop(name: str) -> str:
    return _required_adb_text(
        adb_path,
        serial,
        adb_server_port,
        ["shell", "getprop", name],
        name,
        run_command=run_command,
    ).replace("\r", "")

  abi_list = tuple(item for item in prop("ro.product.cpu.abilist").split(",") if item)
  if not abi_list:
    abi_list = (prop("ro.product.cpu.abi"),)
  density_output = _required_adb_text(
      adb_path,
      serial,
      adb_server_port,
      ["shell", "wm", "density"],
      "display density",
      run_command=run_command,
  )
  density_matches = re.findall(r"(?:Physical|Override) density:\s*(\d+)", density_output)
  if not density_matches:
    raise ProvisionError(f"Unable to parse display density: {density_output!r}")
  density = int(density_matches[-1])
  locale = prop("persist.sys.locale") or prop("ro.product.locale")
  if not locale:
    raise ProvisionError("Unable to determine Android locale")
  boot_id = _required_adb_text(
      adb_path,
      serial,
      adb_server_port,
      ["shell", "cat", "/proc/sys/kernel/random/boot_id"],
      "boot ID",
      run_command=run_command,
  ).replace("\r", "")
  return DeviceProfile(
      serial=serial,
      adb_server_port=adb_server_port,
      build_fingerprint=prop("ro.build.fingerprint"),
      api_level=prop("ro.build.version.sdk"),
      abi_list=abi_list,
      density=density,
      locale=locale,
      boot_id=boot_id,
  )


def _language(locale: str) -> str:
  language = re.split(r"[-_]", locale.strip(), maxsplit=1)[0].lower()
  return {"id": "in", "he": "iw", "yi": "ji"}.get(language, language)


def select_device_apks(
    app: PreparedApp, profile: DeviceProfile
) -> tuple[ApkFile, ...]:
  by_split = {apk.split_id: apk for apk in app.apks}
  if "base" not in by_split:
    raise ProvisionError(f"Prepared app lacks base split: {app.pin.app_id}")
  selected = [by_split["base"]]
  remaining = set(by_split) - {"base"}

  abi_candidates = set(ABI_SPLITS.values()) & remaining
  if abi_candidates:
    chosen_abi = next(
        (
            ABI_SPLITS[abi]
            for abi in profile.abi_list
            if abi in ABI_SPLITS and ABI_SPLITS[abi] in abi_candidates
        ),
        "",
    )
    if not chosen_abi:
      raise ProvisionError(
          f"No compatible ABI split for {app.pin.app_id}: {profile.abi_list}"
      )
    selected.append(by_split[chosen_abi])
    remaining -= abi_candidates

  density_candidates = set(DENSITY_SPLITS) & remaining
  if density_candidates:
    chosen_density = min(
        density_candidates,
        key=lambda split: (
            abs(DENSITY_SPLITS[split] - profile.density),
            DENSITY_SPLITS[split],
        ),
    )
    selected.append(by_split[chosen_density])
    remaining -= density_candidates

  language = _language(profile.locale)
  locale_candidates = {
      split
      for split in remaining
      if re.fullmatch(r"config\.[a-z]{2,3}", split)
  }
  language_split = f"config.{language}"
  if language_split in locale_candidates:
    selected.append(by_split[language_split])
  remaining -= locale_candidates

  if remaining:
    raise ProvisionError(
        f"Unsupported split qualifiers for {app.pin.app_id}: {sorted(remaining)}"
    )
  return tuple(selected)


def _parse_package_version(output: str) -> tuple[str, str]:
  code = VERSION_CODE.search(output)
  name = VERSION_NAME.search(output)
  return (
      name.group(1).strip() if name else "",
      code.group(1) if code else "",
  )


def _parse_pm_paths(output: str) -> tuple[str, ...]:
  paths: list[str] = []
  for raw_line in output.splitlines():
    match = PM_PATH_LINE.fullmatch(raw_line.strip())
    if match:
      paths.append(match.group(1))
  if not paths or len(paths) != len(set(paths)):
    raise ProvisionError(f"Invalid or duplicate pm path output: {output!r}")
  return tuple(paths)


def _apk_evidence(apk: ApkFile, *, remote_path: str = "") -> dict[str, Any]:
  return {
      "split_id": apk.split_id,
      "remote_path": remote_path,
      "sha256": apk.sha256,
      "size_bytes": apk.path.stat().st_size,
      "package_name": apk.package_name,
      "version_name": apk.version_name,
      "version_code": apk.version_code,
      "signer_leaf_certificate_sha256": list(
          apk.signer_leaf_certificate_sha256
      ),
      "verified_schemes": dict(apk.verified_schemes),
  }


def collect_installed_app(
    app: PreparedApp,
    selected_apks: Sequence[ApkFile],
    *,
    profile: DeviceProfile,
    adb_path: Path,
    aapt_path: Path,
    apksigner_path: Path,
    pull_root: Path,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
  dumpsys = _adb(
      adb_path,
      profile.serial,
      profile.adb_server_port,
      ["shell", "dumpsys", "package", app.pin.package_name],
      timeout=30,
      run_command=run_command,
  )
  actual_name, actual_code = _parse_package_version(dumpsys.stdout or "")
  errors: list[str] = []
  if dumpsys.returncode != 0 or not actual_name or not actual_code:
    errors.append("package_missing_or_unreadable")
  if actual_name and actual_name != app.pin.version_name:
    errors.append("version_name_mismatch")
  if actual_code and actual_code != app.pin.version_code:
    errors.append("version_code_mismatch")
  path_result = _adb(
      adb_path,
      profile.serial,
      profile.adb_server_port,
      ["shell", "pm", "path", app.pin.package_name],
      timeout=30,
      run_command=run_command,
  )
  try:
    remote_paths = _parse_pm_paths(path_result.stdout or "")
  except ProvisionError:
    remote_paths = ()
    errors.append("active_apk_paths_missing_or_invalid")
  if path_result.returncode != 0 and "active_apk_paths_missing_or_invalid" not in errors:
    errors.append("active_apk_paths_missing_or_invalid")

  installed: list[tuple[str, ApkFile]] = []
  app_pull_root = pull_root / app.pin.app_id
  app_pull_root.mkdir(parents=True, exist_ok=True)
  for index, remote_path in enumerate(remote_paths):
    local_path = app_pull_root / f"{index:03d}.apk"
    pulled = _adb(
        adb_path,
        profile.serial,
        profile.adb_server_port,
        ["pull", remote_path, str(local_path)],
        timeout=300,
        run_command=run_command,
    )
    if pulled.returncode != 0 or not local_path.is_file():
      errors.append(f"apk_pull_failed:{index}")
      continue
    try:
      observed = _inspect_apk(
          local_path,
          aapt_path=aapt_path,
          apksigner_path=apksigner_path,
          expected_signers=app.pin.expected_signers,
          allow_split_resource_manifest_fallback=len(selected_apks) > 1,
          run_command=run_command,
      )
    except ProvisionError as exc:
      errors.append(f"installed_apk_verification_failed:{index}:{exc}")
      continue
    installed.append((remote_path, observed))

  expected_by_split = {apk.split_id: apk.sha256 for apk in selected_apks}
  installed_by_split = {apk.split_id: apk.sha256 for _, apk in installed}
  if len(installed_by_split) != len(installed):
    errors.append("duplicate_installed_split_id")
  if installed_by_split != expected_by_split:
    errors.append("installed_apk_set_differs_from_selected_artifact_bytes")
  base_hashes = [apk.sha256 for _, apk in installed if apk.split_id == "base"]
  if base_hashes != [app.pin.apk_sha256]:
    errors.append("installed_base_apk_differs_from_pin")

  apk_rows = [
      _apk_evidence(apk, remote_path=remote_path)
      for remote_path, apk in sorted(installed, key=lambda item: item[1].split_id)
  ]
  installed_hashes = sorted({apk.sha256 for _, apk in installed})
  signer_hashes = sorted({
      signer
      for _, apk in installed
      for signer in apk.signer_leaf_certificate_sha256
  })
  return {
      "actual_version_name": actual_name,
      "actual_version_code": actual_code,
      "active_apks": apk_rows,
      "installed_apk_sha256": installed_hashes,
      "signer_leaf_certificate_sha256": signer_hashes,
      "installed_bytes_evidence_sha256": _sha256_json([
          {
              "split_id": row["split_id"],
              "remote_path": row["remote_path"],
              "sha256": row["sha256"],
              "size_bytes": row["size_bytes"],
          }
          for row in apk_rows
      ]),
      "signature_verification_evidence_sha256": _sha256_json([
          {
              "split_id": row["split_id"],
              "sha256": row["sha256"],
              "signers": row["signer_leaf_certificate_sha256"],
              "verified_schemes": row["verified_schemes"],
          }
          for row in apk_rows
      ]),
      "signature_verification_status": (
          "fully_cryptographically_verified" if not errors else "invalid"
      ),
      "valid": not errors,
      "errors": errors,
  }


def install_app(
    app: PreparedApp,
    selected_apks: Sequence[ApkFile],
    *,
    profile: DeviceProfile,
    adb_path: Path,
    run_command: RunCommand = subprocess.run,
) -> str:
  base = _adb_base(adb_path, profile.serial, profile.adb_server_port)
  apk_paths = [str(apk.path) for apk in selected_apks]
  if len(apk_paths) == 1:
    command = [*base, "install", "-r", "-d", "-t", "-g", apk_paths[0]]
  else:
    command = [
        *base,
        "install-multiple",
        "-r",
        "-d",
        "-t",
        "-g",
        *apk_paths,
    ]
  result = _run(command, timeout=600, run_command=run_command)
  output = (result.stdout + "\n" + result.stderr).strip()
  if result.returncode != 0 or "success" not in output.lower():
    raise ProvisionError(f"Install failed for {app.pin.app_id}: {output}")
  return output


def _prepared_evidence(
    app: PreparedApp, selected: Sequence[ApkFile]
) -> dict[str, Any]:
  return {
      "artifact_path": str(app.pin.artifact_path),
      "artifact_sha256": app.artifact_sha256,
      "pinned_base_resolution_scope": app.artifact_scope,
      "pinned_base_member": app.artifact_member,
      "all_artifact_apks": [_apk_evidence(apk) for apk in app.apks],
      "selected_device_apks": [_apk_evidence(apk) for apk in selected],
      "selected_device_apk_set_sha256": _sha256_json([
          {"split_id": apk.split_id, "sha256": apk.sha256}
          for apk in selected
      ]),
  }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
  resolved = path.expanduser().resolve()
  resolved.parent.mkdir(parents=True, exist_ok=True)
  if resolved.exists():
    raise ProvisionError(f"Refusing to overwrite output that already exists: {resolved}")
  temporary = resolved.with_name(resolved.name + ".tmp")
  if temporary.exists():
    raise ProvisionError(f"Refusing to overwrite stale temporary output: {temporary}")
  payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
  try:
    with temporary.open("xb") as handle:
      handle.write(payload)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, resolved)
  finally:
    try:
      temporary.unlink()
    except FileNotFoundError:
      pass


def _validate_new_output(path: Path) -> None:
  resolved = path.expanduser().resolve()
  temporary = resolved.with_name(resolved.name + ".tmp")
  if resolved.exists():
    raise ProvisionError(f"Refusing to overwrite output that already exists: {resolved}")
  if temporary.exists():
    raise ProvisionError(f"Refusing to overwrite stale temporary output: {temporary}")


def run_provision_and_attest(
    *,
    mode: str,
    cohort_path: Path,
    pins_path: Path,
    apps_path: Path,
    signer_audit_path: Path,
    artifact_root: Path,
    output_path: Path,
    adb_path: Path,
    aapt_path: Path,
    apksigner_path: Path,
    serial: str,
    adb_server_port: int | None,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
  if mode not in {"provision", "attest"}:
    raise ProvisionError(f"Unsupported mode: {mode}")
  # Reject an occupied evidence destination before any device-side effect.
  _validate_new_output(output_path)
  started_at = dt.datetime.now(dt.UTC).isoformat()
  (
      cohort,
      cohort_sha256,
      pins_sha256,
      apps_sha256,
      signer_audit_sha256,
      pins,
  ) = load_frozen_inputs(
      cohort_path,
      pins_path,
      apps_path,
      signer_audit_path,
      artifact_root,
  )
  profile = collect_device_profile(
      adb_path=adb_path,
      serial=serial,
      adb_server_port=adb_server_port,
      run_command=run_command,
  )
  device_fields = dataclasses.asdict(profile)
  device_fields["abi_list"] = list(profile.abi_list)
  device_fields["device_identity_sha256"] = _sha256_json(device_fields)
  rows: list[dict[str, Any]] = []
  with tempfile.TemporaryDirectory(prefix="catbench_app_provision_") as tmpdir:
    temp_root = Path(tmpdir)
    # Complete local preflight precedes the first install, so no malformed,
    # mismatched, unsigned, or partial roster can cause a device mutation.
    prepared = prepare_local_apps(
        pins,
        workspace=temp_root / "prepared",
        aapt_path=aapt_path,
        apksigner_path=apksigner_path,
        run_command=run_command,
    )
    selections = tuple(select_device_apks(app, profile) for app in prepared)
    for app_index, (app, selected) in enumerate(zip(prepared, selections)):
      before = collect_installed_app(
          app,
          selected,
          profile=profile,
          adb_path=adb_path,
          aapt_path=aapt_path,
          apksigner_path=apksigner_path,
          pull_root=temp_root / "before" / f"{app_index:02d}",
          run_command=run_command,
      )
      operation = "already_exact" if before["valid"] else "attest_only"
      install_output = ""
      install_error = ""
      if mode == "provision" and not before["valid"]:
        operation = "installed_exact_artifact_set"
        try:
          install_output = install_app(
              app,
              selected,
              profile=profile,
              adb_path=adb_path,
              run_command=run_command,
          )
        except ProvisionError as exc:
          install_error = str(exc)
          operation = "install_failed"
      after = collect_installed_app(
          app,
          selected,
          profile=profile,
          adb_path=adb_path,
          aapt_path=aapt_path,
          apksigner_path=apksigner_path,
          pull_root=temp_root / "after" / f"{app_index:02d}",
          run_command=run_command,
      )
      errors = list(after["errors"])
      if install_error:
        errors.insert(0, install_error)
      valid = after["valid"] and not install_error
      rows.append({
          "app_id": app.pin.app_id,
          "category": app.pin.category,
          "package_name": app.pin.package_name,
          "version_name": app.pin.version_name,
          "version_code": app.pin.version_code,
          "pinned_artifact_sha256": app.pin.apk_sha256,
          "expected_signer_leaf_certificate_sha256": list(
              app.pin.expected_signers
          ),
          "operation": operation,
          "local_artifact": _prepared_evidence(app, selected),
          "before": before,
          "actual_version_name": after["actual_version_name"],
          "actual_version_code": after["actual_version_code"],
          "active_apks": after["active_apks"],
          "installed_apk_sha256": after["installed_apk_sha256"],
          "signer_leaf_certificate_sha256": after[
              "signer_leaf_certificate_sha256"
          ],
          "installed_bytes_evidence_sha256": after[
              "installed_bytes_evidence_sha256"
          ],
          "signature_verification_evidence_sha256": after[
              "signature_verification_evidence_sha256"
          ],
          "verification_tool_sha256": _sha256_path(apksigner_path),
          "signature_verification_status": after[
              "signature_verification_status"
          ],
          "install_output": install_output,
          "valid": valid,
          "errors": errors,
      })
  valid_apps = sum(bool(row["valid"]) for row in rows)
  report: dict[str, Any] = {
      "schema_version": 1,
      "evidence_type": "catbench_live_device_app_provision_and_attestation",
      "approval_status": "not_an_approved_release_attestation",
      "claim": (
          "machine_collected_device_evidence_only; independent signer review, "
          "base-snapshot binding, and release approval are required"
      ),
      "mode": mode,
      "release_id": cohort.get("release_id"),
      "cohort_sha256": cohort_sha256,
      "app_pins_sha256": pins_sha256,
      "app_inventory_sha256": apps_sha256,
      "signer_audit_sha256": signer_audit_sha256,
      "artifact_root": str(artifact_root.expanduser().resolve()),
      "collection_tool": str(Path(__file__).resolve()),
      "collection_tool_sha256": _sha256_path(Path(__file__).resolve()),
      "adb_tool_sha256": _sha256_path(adb_path),
      "aapt_tool_sha256": _sha256_path(aapt_path),
      "verification_tool_sha256": _sha256_path(apksigner_path),
      "started_at": started_at,
      "completed_at": dt.datetime.now(dt.UTC).isoformat(),
      "device": device_fields,
      "expected_apps": len(pins),
      "valid_apps": valid_apps,
      "invalid_apps": len(pins) - valid_apps,
      "valid": valid_apps == len(pins),
      "apps": rows,
  }
  _atomic_json(output_path, report)
  return report


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", required=True, choices=("provision", "attest"))
  parser.add_argument("--cohort_manifest", default=str(DEFAULT_COHORT))
  parser.add_argument("--pins", default=str(DEFAULT_PINS))
  parser.add_argument("--apps", default=str(DEFAULT_APPS))
  parser.add_argument("--signer_audit", default=str(DEFAULT_SIGNER_AUDIT))
  parser.add_argument("--artifact_root", default=str(DEFAULT_ARTIFACT_ROOT))
  parser.add_argument("--output", required=True)
  parser.add_argument("--adb", default="")
  parser.add_argument("--aapt", default="")
  parser.add_argument("--apksigner", default="")
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb_server_port", type=int)
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)
  try:
    adb_path = _tool_path(args.adb, ("adb",), "adb")
    aapt_path = _tool_path(args.aapt, ("aapt2", "aapt"), "aapt/aapt2")
    apksigner_path = _tool_path(args.apksigner, ("apksigner",), "apksigner")
    if args.adb_server_port is not None and not (
        1024 <= args.adb_server_port <= 65535
    ):
      raise ProvisionError("adb_server_port must be in [1024, 65535]")
    report = run_provision_and_attest(
        mode=args.mode,
        cohort_path=Path(args.cohort_manifest),
        pins_path=Path(args.pins),
        apps_path=Path(args.apps),
        signer_audit_path=Path(args.signer_audit),
        artifact_root=Path(args.artifact_root),
        output_path=Path(args.output),
        adb_path=adb_path,
        aapt_path=aapt_path,
        apksigner_path=apksigner_path,
        serial=args.serial,
        adb_server_port=args.adb_server_port,
    )
  except ProvisionError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 2
  print(json.dumps({
      "mode": report["mode"],
      "serial": report["device"]["serial"],
      "expected_apps": report["expected_apps"],
      "valid_apps": report["valid_apps"],
      "invalid_apps": report["invalid_apps"],
      "valid": report["valid"],
      "output": str(Path(args.output).expanduser().resolve()),
  }, indent=2))
  return 0 if report["valid"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
