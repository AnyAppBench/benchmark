#!/usr/bin/env python3
"""Audit signer certificates for the exact frozen CATBench APK artifacts.

This is an observational, read-only audit.  It first resolves an APK by the
already-pinned APK SHA-256 (including APK members nested in XAPK files).  It
never treats an APK filename or a same-package, different-version artifact as
identity evidence.  Certificate extraction and cryptographic signature
verification are deliberately reported as separate facts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
from typing import Any, BinaryIO, Iterable
import warnings
import zipfile

try:
  from cryptography import x509
  from cryptography.hazmat.primitives import serialization
  from cryptography.hazmat.primitives.serialization import pkcs7
except ImportError:  # pragma: no cover - exercised only in a reduced runtime.
  x509 = None
  serialization = None
  pkcs7 = None


BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COHORT = BENCHMARK_ROOT / "configs" / "catbench_5cat_primary_cohort.json"
DEFAULT_PINS = BENCHMARK_ROOT / "configs" / "app_versions_pinned.csv"
DEFAULT_APPS = BENCHMARK_ROOT / "app_generalization_apps.csv"
DEFAULT_ARTIFACT_ROOT = Path(
    "$HOME/anyappbench_apks"
)

APK_SIG_BLOCK_MAGIC = b"APK Sig Block 42"
APK_SIGNATURE_SCHEMES = {
    0x7109871A: "v2",
    0xF05368C0: "v3",
    0x1B93AD61: "v3.1",
}
ZIP_EOCD_MAGIC = b"PK\x05\x06"
MAX_EOCD_SIZE = 22 + 65535


class SignerAuditError(ValueError):
  """Raised when an APK signing structure is malformed or unsupported."""


def _sha256_stream(handle: BinaryIO) -> str:
  digest = hashlib.sha256()
  for chunk in iter(lambda: handle.read(1024 * 1024), b""):
    digest.update(chunk)
  return digest.hexdigest()


def _sha256_path(path: Path) -> str:
  with path.open("rb") as handle:
    return _sha256_stream(handle)


def _csv_index(path: Path, key: str) -> dict[str, dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    return {str(row[key]): dict(row) for row in csv.DictReader(handle)}


def _read_uint32(data: bytes, offset: int) -> tuple[int, int]:
  if offset + 4 > len(data):
    raise SignerAuditError("truncated uint32 length/value")
  return struct.unpack_from("<I", data, offset)[0], offset + 4


def _read_length_prefixed(data: bytes, offset: int) -> tuple[bytes, int]:
  length, offset = _read_uint32(data, offset)
  end = offset + length
  if end > len(data):
    raise SignerAuditError(
        f"length-prefixed field ends at {end}, beyond buffer size {len(data)}"
    )
  return data[offset:end], end


def _iter_length_prefixed(data: bytes) -> Iterable[bytes]:
  offset = 0
  while offset < len(data):
    value, offset = _read_length_prefixed(data, offset)
    yield value
  if offset != len(data):  # Defensive; the loop currently makes this impossible.
    raise SignerAuditError("length-prefixed sequence has trailing bytes")


def _scheme_certificates(value: bytes) -> list[list[bytes]]:
  """Returns one DER certificate chain per v2/v3 signer.

  Android APK Signature Scheme v2, v3, and v3.1 all start each signer with a
  length-prefixed signed-data record.  The first two records inside signed data
  are the digest sequence and certificate sequence respectively.  Later
  scheme-specific fields are irrelevant to certificate extraction.
  """
  signers_blob, end = _read_length_prefixed(value, 0)
  if end != len(value):
    raise SignerAuditError("signer sequence has trailing bytes")
  chains: list[list[bytes]] = []
  for signer_blob in _iter_length_prefixed(signers_blob):
    signed_data, _ = _read_length_prefixed(signer_blob, 0)
    _, offset = _read_length_prefixed(signed_data, 0)  # digests
    certificates_blob, _ = _read_length_prefixed(signed_data, offset)
    chain = list(_iter_length_prefixed(certificates_blob))
    if not chain:
      raise SignerAuditError("APK signer has an empty certificate sequence")
    chains.append(chain)
  if not chains:
    raise SignerAuditError("APK signature scheme has an empty signer sequence")
  return chains


def _central_directory_offset(handle: BinaryIO) -> int:
  handle.seek(0, os.SEEK_END)
  size = handle.tell()
  tail_size = min(size, MAX_EOCD_SIZE)
  handle.seek(size - tail_size)
  tail = handle.read(tail_size)
  search_end = len(tail)
  while True:
    index = tail.rfind(ZIP_EOCD_MAGIC, 0, search_end)
    if index < 0:
      raise SignerAuditError("ZIP end-of-central-directory record not found")
    if index + 22 <= len(tail):
      comment_length = struct.unpack_from("<H", tail, index + 20)[0]
      if index + 22 + comment_length == len(tail):
        central_offset = struct.unpack_from("<I", tail, index + 16)[0]
        if central_offset == 0xFFFFFFFF:
          raise SignerAuditError("ZIP64 APK is not supported by this parser")
        return central_offset
    search_end = index


def _apk_signing_block_entries(handle: BinaryIO) -> dict[int, list[bytes]]:
  central_offset = _central_directory_offset(handle)
  if central_offset < 32:
    return {}
  handle.seek(central_offset - 24)
  footer = handle.read(24)
  if len(footer) != 24 or footer[8:] != APK_SIG_BLOCK_MAGIC:
    return {}
  footer_size = struct.unpack_from("<Q", footer, 0)[0]
  total_size = footer_size + 8
  block_start = central_offset - total_size
  if block_start < 0 or total_size < 32:
    raise SignerAuditError("invalid APK Signing Block size")
  handle.seek(block_start)
  header = handle.read(8)
  if len(header) != 8 or struct.unpack("<Q", header)[0] != footer_size:
    raise SignerAuditError("APK Signing Block header/footer size mismatch")
  pairs_size = total_size - 32
  pairs = handle.read(pairs_size)
  if len(pairs) != pairs_size:
    raise SignerAuditError("truncated APK Signing Block pairs")
  entries: dict[int, list[bytes]] = {}
  offset = 0
  while offset < len(pairs):
    if offset + 8 > len(pairs):
      raise SignerAuditError("truncated APK Signing Block pair size")
    pair_size = struct.unpack_from("<Q", pairs, offset)[0]
    offset += 8
    if pair_size < 4 or offset + pair_size > len(pairs):
      raise SignerAuditError("invalid APK Signing Block pair size")
    entry_id = struct.unpack_from("<I", pairs, offset)[0]
    value_start = offset + 4
    value_end = offset + pair_size
    entries.setdefault(entry_id, []).append(pairs[value_start:value_end])
    offset = value_end
  return entries


def _certificate_metadata(der: bytes) -> dict[str, Any]:
  fingerprint = hashlib.sha256(der).hexdigest()
  result: dict[str, Any] = {
      "certificate_sha256": fingerprint,
      "der_length": len(der),
  }
  if x509 is None:
    result["metadata_status"] = "cryptography_unavailable"
    return result
  try:
    certificate = x509.load_der_x509_certificate(der)
  except ValueError as exc:
    result["metadata_status"] = "invalid_der_certificate"
    result["metadata_error"] = str(exc)
    return result
  with warnings.catch_warnings(record=True) as metadata_warnings:
    warnings.simplefilter("always")
    subject = certificate.subject.rfc4514_string()
    issuer = certificate.issuer.rfc4514_string()
  result.update({
      "metadata_status": "parsed",
      "subject": subject,
      "issuer": issuer,
      "serial_number_hex": format(certificate.serial_number, "x"),
      "not_valid_before": certificate.not_valid_before_utc.isoformat(),
      "not_valid_after": certificate.not_valid_after_utc.isoformat(),
  })
  if metadata_warnings:
    result["metadata_warnings"] = sorted({
        str(warning.message) for warning in metadata_warnings
    })
  return result


def _extract_v1_certificates(archive: zipfile.ZipFile) -> tuple[list[bytes], list[str]]:
  certificates: list[bytes] = []
  errors: list[str] = []
  signature_blocks = [
      name
      for name in archive.namelist()
      if name.upper().startswith("META-INF/")
      and name.count("/") == 1
      and name.upper().endswith((".RSA", ".DSA", ".EC"))
  ]
  if not signature_blocks:
    return certificates, errors
  if pkcs7 is None or serialization is None:
    return [], ["cryptography PKCS#7 support unavailable"]
  for name in sorted(signature_blocks):
    try:
      payload = archive.read(name)
      try:
        parsed = pkcs7.load_der_pkcs7_certificates(payload)
      except ValueError:
        parsed = pkcs7.load_pem_pkcs7_certificates(payload)
      for certificate in parsed:
        certificates.append(
            certificate.public_bytes(serialization.Encoding.DER)
        )
    except (KeyError, RuntimeError, ValueError) as exc:
      errors.append(f"{name}: {exc}")
  return certificates, errors


def extract_certificates(apk_path: Path) -> dict[str, Any]:
  """Extracts embedded certificates without claiming signature verification."""
  observations: dict[str, dict[str, Any]] = {}
  errors: list[str] = []

  def observe(der: bytes, source: dict[str, Any]) -> None:
    fingerprint = hashlib.sha256(der).hexdigest()
    if fingerprint not in observations:
      observations[fingerprint] = {
          **_certificate_metadata(der),
          "sources": [],
      }
    observations[fingerprint]["sources"].append(source)

  try:
    with apk_path.open("rb") as handle:
      entries = _apk_signing_block_entries(handle)
    for entry_id, scheme in APK_SIGNATURE_SCHEMES.items():
      for entry_index, value in enumerate(entries.get(entry_id, [])):
        try:
          for signer_index, chain in enumerate(_scheme_certificates(value)):
            for chain_index, der in enumerate(chain):
              observe(der, {
                  "scheme": scheme,
                  "entry_index": entry_index,
                  "signer_index": signer_index,
                  "chain_index": chain_index,
                  "role": (
                      "signer_leaf" if chain_index == 0 else "chain_certificate"
                  ),
              })
        except SignerAuditError as exc:
          errors.append(f"{scheme} entry {entry_index}: {exc}")
  except (OSError, SignerAuditError) as exc:
    errors.append(f"APK Signing Block: {exc}")

  try:
    with zipfile.ZipFile(apk_path) as archive:
      v1_certificates, v1_errors = _extract_v1_certificates(archive)
      errors.extend(f"v1: {error}" for error in v1_errors)
      for certificate_index, der in enumerate(v1_certificates):
        observe(der, {
            "scheme": "v1",
            "certificate_index": certificate_index,
            "role": "pkcs7_certificate_role_unresolved",
        })
  except (OSError, zipfile.BadZipFile) as exc:
    errors.append(f"ZIP/JAR: {exc}")

  schemes = sorted({
      source["scheme"]
      for certificate in observations.values()
      for source in certificate["sources"]
  })
  leaf_fingerprints = sorted({
      fingerprint
      for fingerprint, certificate in observations.items()
      if any(
          source["role"] == "signer_leaf"
          for source in certificate["sources"]
      )
  })
  if observations:
    status = "extracted"
  elif errors:
    status = "error"
  else:
    status = "no_certificate_found"
  return {
      "status": status,
      "claim": "embedded_certificate_extraction_only_not_signature_verification",
      "schemes_with_extracted_certificates": schemes,
      "signer_leaf_certificate_sha256": leaf_fingerprints,
      "all_embedded_certificate_sha256": sorted(observations),
      "certificates": [observations[key] for key in sorted(observations)],
      "errors": errors,
  }


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command,
      check=False,
      capture_output=True,
      text=True,
      timeout=180,
  )


def _verification_tool_metadata(
    tool: str, path: str | None
) -> dict[str, Any]:
  if not path:
    return {"tool": tool, "available": False, "path": ""}
  resolved = Path(path).expanduser().resolve()
  result: dict[str, Any] = {
      "tool": tool,
      "available": resolved.is_file(),
      "path": str(resolved),
  }
  if not resolved.is_file():
    return result
  result["executable_sha256"] = _sha256_path(resolved)
  version_command = (
      [str(resolved), "version"]
      if tool == "apksigner"
      else [str(resolved), "-J-version"]
  )
  try:
    completed = _run_command(version_command)
    result["version_return_code"] = completed.returncode
    result["version_output"] = (
        completed.stdout + "\n" + completed.stderr
    ).strip()
  except (OSError, subprocess.TimeoutExpired) as exc:
    result["version_error"] = str(exc)
  return result


def _parse_apksigner_schemes(output: str) -> dict[str, bool]:
  result: dict[str, bool] = {}
  labels = {
      "v1 scheme (jar signing)": "v1",
      "v2 scheme (apk signature scheme v2)": "v2",
      "v3 scheme (apk signature scheme v3)": "v3",
      "v3.1 scheme (apk signature scheme v3.1)": "v3.1",
      "v4 scheme (apk signature scheme v4)": "v4",
  }
  for line in output.splitlines():
    lowered = line.strip().lower()
    if not lowered.startswith("verified using ") or ":" not in lowered:
      continue
    key, value = lowered.split(":", 1)
    key = key.removeprefix("verified using ")
    if key in labels:
      result[labels[key]] = value.strip() == "true"
  return result


def _parse_apksigner_certificate_sha256(output: str) -> list[str]:
  fingerprints: set[str] = set()
  # `--print-certs` may also print a distinct Source Stamp signer.  A source
  # stamp authenticates distribution metadata; it is not an APK signer and is
  # not present in the v2/v3 signer certificate chains parsed above.  Anchor
  # this expression to apksigner's actual APK-signer labels so a valid source
  # stamp does not create a false certificate-consistency failure.
  pattern = re.compile(
      r"^\s*Signer(?: #\d+| \([^\n)]*\))? certificate SHA-256 digest:"
      r"\s*([0-9a-fA-F:]{64,95})\s*$",
      flags=re.IGNORECASE | re.MULTILINE,
  )
  for match in pattern.finditer(output):
    fingerprint = match.group(1).replace(":", "").lower()
    if len(fingerprint) == 64:
      fingerprints.add(fingerprint)
  return sorted(fingerprints)


def verify_signature(
    apk_path: Path,
    extraction: dict[str, Any],
    apksigner_path: str | None = None,
    jarsigner_path: str | None = None,
) -> dict[str, Any]:
  """Runs available verifiers and describes their exact coverage."""
  apksigner = apksigner_path or shutil.which("apksigner")
  if apksigner:
    try:
      completed = _run_command([
          apksigner,
          "verify",
          "--verbose",
          "--print-certs",
          str(apk_path),
      ])
    except (OSError, subprocess.TimeoutExpired) as exc:
      return {
          "status": "verification_tool_error",
          "fully_cryptographically_verified": False,
          "tool": "apksigner",
          "tool_path": apksigner,
          "coverage": "android_apk_signature_schemes",
          "error": str(exc),
      }
    combined = completed.stdout + "\n" + completed.stderr
    verified_fingerprints = _parse_apksigner_certificate_sha256(combined)
    extracted_fingerprints = extraction.get(
        "signer_leaf_certificate_sha256", []
    )
    return {
        "status": "verified" if completed.returncode == 0 else "failed",
        "fully_cryptographically_verified": completed.returncode == 0,
        "tool": "apksigner",
        "tool_path": apksigner,
        "coverage": "android_apk_signature_schemes_supported_by_tool",
        "return_code": completed.returncode,
        "verified_schemes": _parse_apksigner_schemes(combined),
        "verified_signer_certificate_sha256": verified_fingerprints,
        "certificate_fingerprints_consistent_with_extraction": (
            bool(verified_fingerprints)
            and set(verified_fingerprints) == set(extracted_fingerprints)
        ),
        "output": combined.strip(),
    }

  has_v1 = "v1" in extraction.get("schemes_with_extracted_certificates", [])
  jarsigner = jarsigner_path or shutil.which("jarsigner")
  if has_v1 and jarsigner:
    try:
      completed = _run_command([
          jarsigner,
          "-verify",
          str(apk_path),
      ])
    except (OSError, subprocess.TimeoutExpired) as exc:
      return {
          "status": "verification_tool_error",
          "fully_cryptographically_verified": False,
          "tool": "jarsigner",
          "tool_path": jarsigner,
          "coverage": "v1_jar_signatures_only",
          "error": str(exc),
      }
    combined = completed.stdout + "\n" + completed.stderr
    jar_verified = (
        completed.returncode == 0 and "jar verified." in combined.lower()
    )
    blocked_by_policy = "treated as unsigned" in combined.lower()
    return {
        "status": (
            "partial_v1_verified"
            if jar_verified
            else (
                "v1_verification_blocked_by_local_crypto_policy"
                if blocked_by_policy
                else "v1_verification_failed"
            )
        ),
        "fully_cryptographically_verified": False,
        "tool": "jarsigner",
        "tool_path": jarsigner,
        "coverage": "v1_jar_signatures_only_not_v2_v3",
        "return_code": completed.returncode,
        "v1_jar_cryptographically_verified": jar_verified,
        "blocked_by_local_crypto_policy": blocked_by_policy,
        "output": combined.strip(),
    }

  return {
      "status": "not_performed",
      "fully_cryptographically_verified": False,
      "tool": "none",
      "coverage": "none",
      "reason": (
          "apksigner unavailable; jarsigner cannot verify v2/v3 signatures"
          if not has_v1
          else "no APK signature verification tool available"
      ),
  }


def _copy_zip_member(
    artifact_path: Path, member_index: int, destination: Path
) -> None:
  with zipfile.ZipFile(artifact_path) as archive:
    info = archive.infolist()[member_index]
    with archive.open(info) as source, destination.open("wb") as target:
      shutil.copyfileobj(source, target, length=1024 * 1024)


def _matching_apk_candidates(
    artifact_path: Path, pinned_sha256: str
) -> tuple[str, list[dict[str, str]]]:
  """Finds exact content matches; member names are labels, never identity."""
  if not artifact_path.is_file():
    return "artifact_missing", []
  candidates: list[dict[str, str]] = []
  artifact_sha256 = _sha256_path(artifact_path)
  if artifact_sha256 == pinned_sha256:
    candidates.append({
        "scope": "artifact_file",
        "member": "",
        "sha256": artifact_sha256,
    })
  if zipfile.is_zipfile(artifact_path):
    with zipfile.ZipFile(artifact_path) as archive:
      for member_index, info in enumerate(archive.infolist()):
        # Hash every regular member.  Neither an extension nor a member name is
        # accepted as evidence that the member is the pinned APK.
        if info.is_dir():
          continue
        with archive.open(info) as handle:
          member_sha256 = _sha256_stream(handle)
        if member_sha256 == pinned_sha256:
          candidates.append({
              "scope": "zip_member_exact_hash",
              "member": info.filename,
              "member_index": str(member_index),
              "sha256": member_sha256,
          })
  if not candidates:
    return "pinned_hash_not_found_in_local_artifact_or_apk_members", []
  return "", candidates


def audit(
    cohort_path: Path,
    pins_path: Path,
    apps_path: Path,
    artifact_root: Path,
    apksigner_path: str | None = None,
    jarsigner_path: str | None = None,
) -> dict[str, Any]:
  cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
  pins = _csv_index(pins_path, "app_id")
  apps = _csv_index(apps_path, "app_id")
  frozen_app_ids = [
      app_id
      for category in cohort["categories"].values()
      for app_id in category["app_ids"]
  ]
  resolved_apksigner = apksigner_path or shutil.which("apksigner")
  resolved_jarsigner = jarsigner_path or shutil.which("jarsigner")
  rows: list[dict[str, Any]] = []
  with tempfile.TemporaryDirectory(prefix="catbench_signer_audit_") as tmpdir:
    temporary_root = Path(tmpdir)
    for app_index, app_id in enumerate(frozen_app_ids):
      pin = pins.get(app_id, {})
      inventory = apps.get(app_id, {})
      artifact_path = artifact_root / str(inventory.get("apk_filename") or "")
      pinned_sha256 = str(pin.get("apk_sha256") or "")
      identity_error, candidates = _matching_apk_candidates(
          artifact_path, pinned_sha256
      )
      row: dict[str, Any] = {
          "app_id": app_id,
          "category": str(pin.get("category") or ""),
          "package_name": str(pin.get("package_name") or ""),
          "pinned_version_name": str(pin.get("version_name") or ""),
          "pinned_version_code": str(pin.get("version_code") or ""),
          "pinned_apk_sha256": pinned_sha256,
          "artifact_path": str(artifact_path),
          "artifact_identity": {
              "status": "matched" if candidates else "unresolved",
              "valid": bool(candidates),
              "matches": candidates,
              "error": identity_error,
              "rule": "exact_pinned_sha256_only",
          },
      }
      if not candidates:
        row["certificate_extraction"] = {
            "status": "not_attempted_artifact_identity_unresolved",
            "claim": "signer_unknown",
            "schemes_with_extracted_certificates": [],
            "signer_leaf_certificate_sha256": [],
            "all_embedded_certificate_sha256": [],
            "certificates": [],
            "errors": [],
        }
        row["signature_verification"] = {
            "status": "not_attempted_artifact_identity_unresolved",
            "fully_cryptographically_verified": False,
            "tool": "none",
            "coverage": "none",
        }
        row["signer_identity_known"] = False
        row["fully_cryptographically_verified"] = False
        rows.append(row)
        continue

      candidate_reports: list[dict[str, Any]] = []
      for candidate_index, candidate in enumerate(candidates):
        if candidate["scope"] == "artifact_file":
          candidate_path = artifact_path
        else:
          candidate_path = temporary_root / f"{app_index}_{candidate_index}.apk"
          _copy_zip_member(
              artifact_path, int(candidate["member_index"]), candidate_path
          )
          if _sha256_path(candidate_path) != pinned_sha256:
            raise SignerAuditError(
                f"copied XAPK member hash changed for {app_id}"
            )
        extraction = extract_certificates(candidate_path)
        verification = verify_signature(
            candidate_path,
            extraction,
            apksigner_path=resolved_apksigner,
            jarsigner_path=resolved_jarsigner,
        )
        candidate_reports.append({
            "match": candidate,
            "certificate_extraction": extraction,
            "signature_verification": verification,
        })

      fingerprint_sets = {
          tuple(report["certificate_extraction"][
              "all_embedded_certificate_sha256"
          ])
          for report in candidate_reports
      }
      row["matching_apk_reports"] = candidate_reports
      row["certificate_extraction"] = candidate_reports[0][
          "certificate_extraction"
      ]
      row["signature_verification"] = candidate_reports[0][
          "signature_verification"
      ]
      row["duplicate_matches_certificate_consistent"] = (
          len(fingerprint_sets) == 1
      )
      row["signer_identity_known"] = bool(
          row["certificate_extraction"][
              "signer_leaf_certificate_sha256"
          ]
      ) and len(fingerprint_sets) == 1
      row["fully_cryptographically_verified"] = all(
          report["signature_verification"][
              "fully_cryptographically_verified"
          ]
          and report["signature_verification"].get(
              "certificate_fingerprints_consistent_with_extraction", True
          )
          for report in candidate_reports
      )
      rows.append(row)

  artifact_valid = sum(row["artifact_identity"]["valid"] for row in rows)
  signer_known = sum(row["signer_identity_known"] for row in rows)
  fully_verified = sum(
      row["fully_cryptographically_verified"] for row in rows
  )
  return {
      "audit_type": "frozen_real_app_signing_certificate_observation",
      "pinning_status": "observational_report_not_an_approved_signer_pinset",
      "auditor_script": str(Path(__file__).resolve()),
      "auditor_script_sha256": _sha256_path(Path(__file__).resolve()),
      "cohort_release_id": cohort["release_id"],
      "cohort_manifest": str(cohort_path),
      "cohort_manifest_sha256": _sha256_path(cohort_path),
      "pins_file": str(pins_path),
      "pins_file_sha256": _sha256_path(pins_path),
      "app_inventory_file": str(apps_path),
      "app_inventory_file_sha256": _sha256_path(apps_path),
      "artifact_root": str(artifact_root),
      "verification_tools": [
          _verification_tool_metadata("apksigner", resolved_apksigner),
          _verification_tool_metadata("jarsigner", resolved_jarsigner),
      ],
      "method": {
          "artifact_resolution": (
              "exact pinned APK SHA-256 over the artifact file or any regular "
              "member nested in an XAPK; filenames, extensions, and package "
              "labels are not identity evidence"
          ),
          "certificate_extraction": (
              "parse v2/v3/v3.1 APK Signing Blocks and v1 PKCS#7 blocks; "
              "this does not cryptographically verify a signature"
          ),
          "signature_verification": (
              "use apksigner when available; otherwise jarsigner may verify "
              "only v1 JAR signatures and is explicitly marked partial"
          ),
      },
      "expected_apps": len(frozen_app_ids),
      "artifact_identity_valid_apps": artifact_valid,
      "artifact_identity_unresolved_apps": len(rows) - artifact_valid,
      "signer_identity_known_apps": signer_known,
      "signer_identity_unknown_apps": len(rows) - signer_known,
      "fully_cryptographically_verified_apps": fully_verified,
      "release_gate_valid": (
          artifact_valid == len(rows)
          and signer_known == len(rows)
          and fully_verified == len(rows)
      ),
      "apps": rows,
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cohort_manifest", default=str(DEFAULT_COHORT))
  parser.add_argument("--pins", default=str(DEFAULT_PINS))
  parser.add_argument("--apps", default=str(DEFAULT_APPS))
  parser.add_argument("--artifact_root", default=str(DEFAULT_ARTIFACT_ROOT))
  parser.add_argument(
      "--apksigner",
      default="",
      help="Optional explicit Android SDK apksigner executable.",
  )
  parser.add_argument(
      "--jarsigner",
      default="",
      help="Optional explicit jarsigner executable for partial v1 verification.",
  )
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  report = audit(
      Path(args.cohort_manifest).expanduser().resolve(),
      Path(args.pins).expanduser().resolve(),
      Path(args.apps).expanduser().resolve(),
      Path(args.artifact_root).expanduser().resolve(),
      apksigner_path=args.apksigner or None,
      jarsigner_path=args.jarsigner or None,
  )
  output = Path(args.output).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(
      json.dumps(report, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )
  print(json.dumps({
      "expected_apps": report["expected_apps"],
      "artifact_identity_valid_apps": report[
          "artifact_identity_valid_apps"
      ],
      "signer_identity_known_apps": report["signer_identity_known_apps"],
      "fully_cryptographically_verified_apps": report[
          "fully_cryptographically_verified_apps"
      ],
      "release_gate_valid": report["release_gate_valid"],
      "unresolved_app_ids": [
          row["app_id"]
          for row in report["apps"]
          if not row["artifact_identity"]["valid"]
      ],
  }, indent=2))
  return 0 if report["release_gate_valid"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
