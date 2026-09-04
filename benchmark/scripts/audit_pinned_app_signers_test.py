"""Tests for exact-artifact APK signing-certificate auditing."""

from __future__ import annotations

import datetime
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from unittest import mock
import zipfile

from absl.testing import absltest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import pkcs7
from cryptography.x509.oid import NameOID

import audit_pinned_app_signers as signer_audit


def _lp(value: bytes) -> bytes:
  return struct.pack("<I", len(value)) + value


def _scheme_value(*certificate_chains: tuple[bytes, ...]) -> bytes:
  signers = b""
  for chain in certificate_chains:
    certificates = b"".join(_lp(certificate) for certificate in chain)
    signed_data = _lp(b"") + _lp(certificates) + _lp(b"")
    signers += _lp(_lp(signed_data))
  return _lp(signers)


def _inject_signing_block(zip_bytes: bytes, entry_id: int, value: bytes) -> bytes:
  eocd = zip_bytes.rfind(signer_audit.ZIP_EOCD_MAGIC)
  if eocd < 0:
    raise AssertionError("test ZIP has no EOCD")
  central_offset = struct.unpack_from("<I", zip_bytes, eocd + 16)[0]
  pair = struct.pack("<Q", 4 + len(value)) + struct.pack("<I", entry_id) + value
  block_size_without_header = len(pair) + 24
  signing_block = (
      struct.pack("<Q", block_size_without_header)
      + pair
      + struct.pack("<Q", block_size_without_header)
      + signer_audit.APK_SIG_BLOCK_MAGIC
  )
  result = bytearray(
      zip_bytes[:central_offset] + signing_block + zip_bytes[central_offset:]
  )
  new_eocd = eocd + len(signing_block)
  struct.pack_into(
      "<I", result, new_eocd + 16, central_offset + len(signing_block)
  )
  return bytes(result)


def _minimal_zip() -> bytes:
  output = io.BytesIO()
  with zipfile.ZipFile(output, "w") as archive:
    archive.writestr("AndroidManifest.xml", b"parser fixture")
  return output.getvalue()


def _v1_pkcs7_fixture() -> tuple[bytes, str]:
  key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
  name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Parser fixture")])
  now = datetime.datetime.now(datetime.timezone.utc)
  certificate = (
      x509.CertificateBuilder()
      .subject_name(name)
      .issuer_name(name)
      .public_key(key.public_key())
      .serial_number(1)
      .not_valid_before(now - datetime.timedelta(days=1))
      .not_valid_after(now + datetime.timedelta(days=1))
      .sign(key, hashes.SHA256())
  )
  signature_block = (
      pkcs7.PKCS7SignatureBuilder()
      .set_data(b"parser fixture")
      .add_signer(certificate, key, hashes.SHA256())
      .sign(serialization.Encoding.DER, [pkcs7.PKCS7Options.Binary])
  )
  output = io.BytesIO()
  with zipfile.ZipFile(output, "w") as archive:
    archive.writestr("AndroidManifest.xml", b"parser fixture")
    archive.writestr("META-INF/CERT.RSA", signature_block)
  der = certificate.public_bytes(serialization.Encoding.DER)
  return output.getvalue(), hashlib.sha256(der).hexdigest()


class SignerAuditTest(absltest.TestCase):

  def test_checked_real_report_has_exact_artifacts_and_verified_signers(self):
    report_path = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "audits"
        / "pinned_app_signer_audit.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    self.assertEqual(report["expected_apps"], 23)
    self.assertEqual(report["artifact_identity_valid_apps"], 23)
    self.assertEqual(report["signer_identity_known_apps"], 23)
    self.assertEqual(report["fully_cryptographically_verified_apps"], 23)
    self.assertTrue(report["release_gate_valid"])
    self.assertTrue(all(row["artifact_identity"]["valid"] for row in report["apps"]))
    self.assertTrue(all(row["signer_identity_known"] for row in report["apps"]))
    self.assertTrue(
        all(row["fully_cryptographically_verified"] for row in report["apps"])
    )

  def test_apksigner_fingerprint_parser_excludes_source_stamp(self):
    apk_signer = "d4" * 32
    source_stamp = "32" * 32
    output = (
        "Signer #1 certificate SHA-256 digest: " + apk_signer + "\n"
        "Source Stamp Signer certificate SHA-256 digest: " + source_stamp + "\n"
    )

    self.assertEqual(
        signer_audit._parse_apksigner_certificate_sha256(output),  # pylint: disable=protected-access
        [apk_signer],
    )

  def test_v2_signer_parser_preserves_leaf_and_chain_order(self):
    value = _scheme_value((b"leaf-a", b"chain-a"), (b"leaf-b",))

    self.assertEqual(
        signer_audit._scheme_certificates(value),  # pylint: disable=protected-access
        [[b"leaf-a", b"chain-a"], [b"leaf-b"]],
    )

  def test_apk_signing_block_parser_reads_scheme_by_numeric_id(self):
    value = _scheme_value((b"certificate",))
    apk_bytes = _inject_signing_block(
        _minimal_zip(), 0x7109871A, value
    )

    entries = signer_audit._apk_signing_block_entries(  # pylint: disable=protected-access
        io.BytesIO(apk_bytes)
    )

    self.assertEqual(entries, {0x7109871A: [value]})

  def test_extracts_v2_leaf_fingerprint_without_verification_claim(self):
    certificate = b"minimal DER parser fixture"
    apk_bytes = _inject_signing_block(
        _minimal_zip(), 0x7109871A, _scheme_value((certificate,))
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "fixture.bin"
      path.write_bytes(apk_bytes)

      result = signer_audit.extract_certificates(path)

    fingerprint = hashlib.sha256(certificate).hexdigest()
    self.assertEqual(result["status"], "extracted")
    self.assertEqual(result["schemes_with_extracted_certificates"], ["v2"])
    self.assertEqual(result["signer_leaf_certificate_sha256"], [fingerprint])
    self.assertIn("not_signature_verification", result["claim"])

  def test_extracts_real_x509_certificate_from_v1_pkcs7_fixture(self):
    apk_bytes, expected_fingerprint = _v1_pkcs7_fixture()
    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "fixture.zip"
      path.write_bytes(apk_bytes)

      result = signer_audit.extract_certificates(path)

    self.assertEqual(result["status"], "extracted")
    self.assertEqual(result["schemes_with_extracted_certificates"], ["v1"])
    self.assertEqual(
        result["all_embedded_certificate_sha256"], [expected_fingerprint]
    )
    self.assertEmpty(result["signer_leaf_certificate_sha256"])
    self.assertEqual(result["certificates"][0]["metadata_status"], "parsed")

  def test_xapk_member_is_selected_by_hash_not_member_name(self):
    apk_bytes, _ = _v1_pkcs7_fixture()
    pinned_hash = hashlib.sha256(apk_bytes).hexdigest()
    with tempfile.TemporaryDirectory() as tmpdir:
      xapk_path = Path(tmpdir) / "clock_you.apk"
      with zipfile.ZipFile(xapk_path, "w") as archive:
        archive.writestr("misleading-name.apk", apk_bytes)
        archive.writestr("clock_you.apk", b"different content")

      error, candidates = signer_audit._matching_apk_candidates(  # pylint: disable=protected-access
          xapk_path, pinned_hash
      )

    self.assertEmpty(error)
    self.assertEqual(
        candidates,
        [{
            "scope": "zip_member_exact_hash",
            "member": "misleading-name.apk",
            "member_index": "0",
            "sha256": pinned_hash,
        }],
    )

  def test_missing_pinned_build_never_inherits_newer_artifact_signer(self):
    apk_bytes, _ = _v1_pkcs7_fixture()
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      cohort = root / "cohort.json"
      pins = root / "pins.csv"
      apps = root / "apps.csv"
      artifact_root = root / "artifacts"
      artifact_root.mkdir()
      cohort.write_text(
          '{"release_id":"test","categories":{"clock":{"app_ids":'
          '["clock_clockyou"]}}}',
          encoding="utf-8",
      )
      pins.write_text(
          "category,app_id,package_name,version_name,version_code,apk_sha256\n"
          "clock,clock_clockyou,com.bnyro.clock,9.1,19,"
          + "0" * 64
          + "\n",
          encoding="utf-8",
      )
      apps.write_text(
          "app_id,apk_filename\nclock_clockyou,clock_you.apk\n",
          encoding="utf-8",
      )
      (artifact_root / "clock_you.apk").write_bytes(apk_bytes)

      report = signer_audit.audit(
          cohort, pins, apps, artifact_root
      )

    row = report["apps"][0]
    self.assertFalse(row["artifact_identity"]["valid"])
    self.assertEqual(
        row["certificate_extraction"]["status"],
        "not_attempted_artifact_identity_unresolved",
    )
    self.assertFalse(row["signer_identity_known"])

  def test_v1_jar_verification_remains_partial(self):
    extraction = {"schemes_with_extracted_certificates": ["v1", "v2"]}
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="jar verified.\n", stderr=""
    )
    with mock.patch.object(
        signer_audit, "_run_command", return_value=completed
    ):
      result = signer_audit.verify_signature(
          Path("fixture.apk"), extraction, jarsigner_path="jarsigner"
      )

    self.assertEqual(result["status"], "partial_v1_verified")
    self.assertTrue(result["v1_jar_cryptographically_verified"])
    self.assertFalse(result["fully_cryptographically_verified"])
    self.assertEqual(result["coverage"], "v1_jar_signatures_only_not_v2_v3")


if __name__ == "__main__":
  absltest.main()
