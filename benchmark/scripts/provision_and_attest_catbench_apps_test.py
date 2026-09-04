#!/usr/bin/env python3
"""Focused tests for exact CATBench device app provisioning/attestation.

Fixtures use a real frozen CATBench app identity (Clock You).  Synthetic byte
payloads and mocked ADB responses exercise fail-closed control flow only; they
make no claim about an emulator run, app behavior, task success, or result.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import provision_and_attest_catbench_apps as provision


RELEASE_ID = "catbench_acl_revision_5cat_v1"
APP_ID = "clock_clockyou"
PACKAGE_NAME = "com.bnyro.clock"
VERSION_NAME = "9.1"
VERSION_CODE = "19"
CLOCK_YOU_SIGNER_SHA256 = (
    "b3bc73b117df5dfe38130c6c2b946852ae7088557fe8e433f0d9983a6b55cc95"
)


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def _minimal_release(root: Path) -> dict[str, Path | str]:
  artifact_root = root / "artifacts"
  artifact_root.mkdir()
  artifact = artifact_root / "clock_you.apk"
  artifact.write_bytes(b"synthetic-clock-you-unit-fixture-not-an-apk")
  apk_sha = _sha(artifact)
  cohort = root / "cohort.json"
  cohort.write_text(json.dumps({
      "release_id": RELEASE_ID,
      "categories": {"clock": {"app_ids": [APP_ID]}},
  }), encoding="utf-8")
  pins = root / "pins.csv"
  _write_csv(
      pins,
      [
          "category", "app_id", "package_name", "version_name",
          "version_code", "apk_sha256",
      ],
      [{
          "category": "clock",
          "app_id": APP_ID,
          "package_name": PACKAGE_NAME,
          "version_name": VERSION_NAME,
          "version_code": VERSION_CODE,
          "apk_sha256": apk_sha,
      }],
  )
  apps = root / "apps.csv"
  _write_csv(
      apps,
      ["app_id", "package_name", "apk_filename"],
      [{
          "app_id": APP_ID,
          "package_name": PACKAGE_NAME,
          "apk_filename": "clock_you.apk",
      }],
  )
  signer = CLOCK_YOU_SIGNER_SHA256
  audit = root / "signers.json"
  audit.write_text(json.dumps({
      "audit_type": "frozen_real_app_signing_certificate_observation",
      "pinning_status": "observational_report_not_an_approved_signer_pinset",
      "auditor_script_sha256": _sha(Path(provision.signer_audit.__file__).resolve()),
      "cohort_release_id": RELEASE_ID,
      "cohort_manifest_sha256": _sha(cohort),
      "pins_file_sha256": _sha(pins),
      "app_inventory_file_sha256": _sha(apps),
      "expected_apps": 1,
      "artifact_identity_valid_apps": 1,
      "signer_identity_known_apps": 1,
      "fully_cryptographically_verified_apps": 1,
      "release_gate_valid": True,
      "apps": [{
          "app_id": APP_ID,
          "category": "clock",
          "package_name": PACKAGE_NAME,
          "pinned_version_name": VERSION_NAME,
          "pinned_version_code": VERSION_CODE,
          "pinned_apk_sha256": apk_sha,
          "artifact_identity": {"valid": True},
          "certificate_extraction": {
              "signer_leaf_certificate_sha256": [signer],
          },
          "signature_verification": {
              "status": "verified",
              "fully_cryptographically_verified": True,
              "certificate_fingerprints_consistent_with_extraction": True,
              "verified_signer_certificate_sha256": [signer],
          },
          "signer_identity_known": True,
          "fully_cryptographically_verified": True,
      }],
  }), encoding="utf-8")
  return {
      "artifact_root": artifact_root,
      "artifact": artifact,
      "apk_sha": apk_sha,
      "cohort": cohort,
      "pins": pins,
      "apps": apps,
      "audit": audit,
      "signer": signer,
  }


def _apk(
    root: Path,
    split_id: str,
    *,
    payload: bytes | None = None,
    signers: tuple[str, ...] = (CLOCK_YOU_SIGNER_SHA256,),
) -> provision.ApkFile:
  path = root / ("base.apk" if split_id == "base" else f"{split_id}.apk")
  path.write_bytes(payload or split_id.encode())
  return provision.ApkFile(
      split_id=split_id,
      path=path,
      sha256=_sha(path),
      package_name=PACKAGE_NAME,
      version_name=VERSION_NAME,
      version_code=VERSION_CODE,
      signer_leaf_certificate_sha256=signers,
      verified_schemes={"v2": True},
  )


class FrozenInputTest(unittest.TestCase):

  def test_complete_exact_release_is_loaded(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      release = _minimal_release(Path(tmpdir))
      _, _, _, _, _, pins = provision.load_frozen_inputs(
          release["cohort"],
          release["pins"],
          release["apps"],
          release["audit"],
          release["artifact_root"],
      )
      self.assertEqual([pin.app_id for pin in pins], [APP_ID])
      self.assertEqual(pins[0].apk_sha256, release["apk_sha"])
      self.assertEqual(pins[0].expected_signers, (release["signer"],))

  def test_stale_signer_audit_is_rejected_before_provisioning(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      release = _minimal_release(Path(tmpdir))
      audit = json.loads(release["audit"].read_text(encoding="utf-8"))
      audit["cohort_manifest_sha256"] = "f" * 64
      release["audit"].write_text(json.dumps(audit), encoding="utf-8")
      with self.assertRaisesRegex(
          provision.ProvisionError, "cohort_manifest_sha256"
      ):
        provision.load_frozen_inputs(
            release["cohort"],
            release["pins"],
            release["apps"],
            release["audit"],
            release["artifact_root"],
        )

  def test_duplicate_json_keys_are_rejected(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      release = _minimal_release(Path(tmpdir))
      release["cohort"].write_text(
          f'{{"release_id":"{RELEASE_ID}",'
          f'"release_id":"{RELEASE_ID}",'
          f'"categories":{{"clock":{{"app_ids":["{APP_ID}"]}}}}}}',
          encoding="utf-8",
      )
      with self.assertRaisesRegex(provision.ProvisionError, "Duplicate JSON key"):
        provision.load_frozen_inputs(
            release["cohort"],
            release["pins"],
            release["apps"],
            release["audit"],
            release["artifact_root"],
        )

  def test_inventory_path_traversal_is_rejected(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      release = _minimal_release(Path(tmpdir))
      _write_csv(
          release["apps"],
          ["app_id", "package_name", "apk_filename"],
          [{
              "app_id": APP_ID,
              "package_name": PACKAGE_NAME,
              "apk_filename": "../clock_you.apk",
          }],
      )
      audit = json.loads(release["audit"].read_text(encoding="utf-8"))
      audit["app_inventory_file_sha256"] = _sha(release["apps"])
      release["audit"].write_text(json.dumps(audit), encoding="utf-8")
      with self.assertRaisesRegex(provision.ProvisionError, "unsafe apk_filename"):
        provision.load_frozen_inputs(
            release["cohort"],
            release["pins"],
            release["apps"],
            release["audit"],
            release["artifact_root"],
        )


class BundleResolutionTest(unittest.TestCase):

  def test_xapk_resolves_one_exact_base_and_all_declared_splits(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      artifact = root / "clock_you.xapk"
      base = b"synthetic-clock-you-base-unit-fixture"
      manifest = {
          "package_name": PACKAGE_NAME,
          "version_name": VERSION_NAME,
          "version_code": VERSION_CODE,
          "split_apks": [
              {"file": "clock_you.apk", "id": "base"},
              {"file": "config.x86_64.apk", "id": "config.x86_64"},
          ],
      }
      with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("clock_you.apk", base)
        archive.writestr("config.x86_64.apk", b"split")
        archive.writestr("manifest.json", json.dumps(manifest))
      pin = provision.AppPin(
          app_id=APP_ID,
          category="clock",
          package_name=PACKAGE_NAME,
          version_name=VERSION_NAME,
          version_code=VERSION_CODE,
          apk_sha256=hashlib.sha256(base).hexdigest(),
          artifact_path=artifact,
          expected_signers=(CLOCK_YOU_SIGNER_SHA256,),
      )
      destination = root / "out"
      destination.mkdir()
      scope, member, extracted = provision._prepare_bundle(  # pylint: disable=protected-access
          pin, destination
      )
      self.assertEqual(scope, "zip_member_exact_hash")
      self.assertEqual(member, "clock_you.apk")
      self.assertEqual([split for split, _ in extracted], ["base", "config.x86_64"])
      self.assertTrue(all(path.is_file() for _, path in extracted))

  def test_xapk_rejects_an_unpinned_base(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      artifact = root / "clock_you.xapk"
      with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("clock_you.apk", b"wrong")
        archive.writestr("manifest.json", json.dumps({
            "package_name": PACKAGE_NAME,
            "version_name": VERSION_NAME,
            "version_code": VERSION_CODE,
            "split_apks": [{"file": "clock_you.apk", "id": "base"}],
        }))
      pin = provision.AppPin(
          APP_ID, "clock", PACKAGE_NAME, VERSION_NAME, VERSION_CODE,
          "f" * 64, artifact, (CLOCK_YOU_SIGNER_SHA256,),
      )
      destination = root / "out"
      destination.mkdir()
      with self.assertRaisesRegex(provision.ProvisionError, "exactly one pinned"):
        provision._prepare_bundle(pin, destination)  # pylint: disable=protected-access


class MetadataParsingTest(unittest.TestCase):

  def test_configuration_split_may_omit_version_name(self):
    parsed = provision._parse_badging(  # pylint: disable=protected-access
        "package: name='com.bnyro.clock' versionCode='19' "
        "versionName='' split='config.en'\n",
        "Clock You language split",
    )
    self.assertEqual(parsed, ("com.bnyro.clock", "", "19", "config.en"))

  def test_base_apk_must_have_nonempty_version_name(self):
    with self.assertRaisesRegex(
        provision.ProvisionError, "incomplete package metadata"
    ):
      provision._parse_badging(  # pylint: disable=protected-access
          "package: name='com.bnyro.clock' versionCode='19' versionName=''\n",
          "Clock You base APK",
      )

  def test_xmltree_parses_aapt_hex_base_metadata(self):
    parsed = provision._parse_manifest_xmltree(  # pylint: disable=protected-access
        "N: android=http://schemas.android.com/apk/res/android\n"
        "  E: manifest (line=2)\n"
        "    A: android:versionCode(0x0101021b)=(type 0x10)0xad0d\n"
        "    A: android:versionName(0x0101021c)=\"4.43.01\" "
        "(Raw: \"4.43.01\")\n"
        "    A: package=\"com.lonelycatgames.Xplore\" "
        "(Raw: \"com.lonelycatgames.Xplore\")\n"
        "    E: application (line=140)\n"
        "      A: android:name(0x01010003)=\"irrelevant\"\n",
        "X-plore base APK",
    )
    self.assertEqual(
        parsed,
        ("com.lonelycatgames.Xplore", "4.43.01", "44301", "base"),
    )

  def test_xmltree_parses_aapt2_configuration_split_metadata(self):
    parsed = provision._parse_manifest_xmltree(  # pylint: disable=protected-access
        "N: android=http://schemas.android.com/apk/res/android (line=0)\n"
        "  E: manifest (line=0)\n"
        "    A: http://schemas.android.com/apk/res/android:versionCode"
        "(0x0101021b)=44301\n"
        "    A: package=\"com.lonelycatgames.Xplore\" "
        "(Raw: \"com.lonelycatgames.Xplore\")\n"
        "    A: split=\"config.x86_64\" (Raw: \"config.x86_64\")\n"
        "      E: application (line=0)\n",
        "X-plore ABI split",
    )
    self.assertEqual(
        parsed,
        ("com.lonelycatgames.Xplore", "", "44301", "config.x86_64"),
    )


class SplitResourceBadgingFallbackTest(unittest.TestCase):

  @staticmethod
  def _badging(package_name: str = "com.lonelycatgames.Xplore") -> str:
    return (
        f"package: name='{package_name}' versionCode='44301' "
        "versionName='4.43.01' platformBuildVersionName='15'\n"
    )

  @staticmethod
  def _xmltree(package_name: str = "com.lonelycatgames.Xplore") -> str:
    return (
        "N: android=http://schemas.android.com/apk/res/android\n"
        "  E: manifest (line=2)\n"
        "    A: android:versionCode(0x0101021b)=(type 0x10)0xad0d\n"
        "    A: android:versionName(0x0101021c)=\"4.43.01\" "
        "(Raw: \"4.43.01\")\n"
        f"    A: package=\"{package_name}\" (Raw: \"{package_name}\")\n"
        "    E: application (line=140)\n"
    )

  @staticmethod
  def _missing_icon_error() -> str:
    return (
        "AndroidManifest.xml:435: error: ERROR getting 'android:icon' "
        "attribute: attribute value reference does not exist\n"
    )

  def _inspect(
      self,
      root: Path,
      runner,
      *,
      observed_signer: str = CLOCK_YOU_SIGNER_SHA256,
  ) -> provision.ApkFile:
    path = root / "xplore-base.apk"
    path.write_bytes(b"signed-X-plore-base-fixture")
    extraction = {
        "signer_leaf_certificate_sha256": [observed_signer],
    }
    verification = {
        "status": "verified",
        "fully_cryptographically_verified": True,
        "certificate_fingerprints_consistent_with_extraction": True,
        "verified_signer_certificate_sha256": [observed_signer],
        "verified_schemes": {"v2": True},
    }
    with (
        mock.patch.object(
            provision.signer_audit,
            "extract_certificates",
            return_value=extraction,
        ),
        mock.patch.object(
            provision.signer_audit,
            "verify_signature",
            return_value=verification,
        ),
    ):
      return provision._inspect_apk(  # pylint: disable=protected-access
          path,
          aapt_path=Path("/sdk/aapt"),
          apksigner_path=Path("/sdk/apksigner"),
          expected_signers=(CLOCK_YOU_SIGNER_SHA256,),
          allow_split_resource_manifest_fallback=True,
          run_command=runner,
      )

  def test_pinned_xplore_missing_split_icon_uses_manifest_metadata(self):
    calls: list[list[str]] = []

    def runner(command, **kwargs):
      del kwargs
      calls.append(command)
      if command[1:3] == ["dump", "badging"]:
        return subprocess.CompletedProcess(
            command, 1, self._badging(), self._missing_icon_error()
        )
      if command[1:3] == ["dump", "xmltree"]:
        return subprocess.CompletedProcess(command, 0, self._xmltree(), "")
      raise AssertionError(command)

    with tempfile.TemporaryDirectory() as tmpdir:
      apk = self._inspect(Path(tmpdir), runner)
    self.assertEqual(
        (apk.package_name, apk.version_name, apk.version_code, apk.split_id),
        ("com.lonelycatgames.Xplore", "4.43.01", "44301", "base"),
    )
    self.assertEqual(calls[1][1:], [
        "dump", "xmltree", calls[1][3], "AndroidManifest.xml",
    ])

  def test_nonzero_badging_is_not_accepted_for_a_standalone_apk(self):
    def runner(command, **kwargs):
      del kwargs
      return subprocess.CompletedProcess(
          command, 1, self._badging(), self._missing_icon_error()
      )

    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "standalone.apk"
      path.write_bytes(b"standalone")
      with self.assertRaisesRegex(provision.ProvisionError, "aapt rejected"):
        provision._inspect_apk(  # pylint: disable=protected-access
            path,
            aapt_path=Path("/sdk/aapt"),
            apksigner_path=Path("/sdk/apksigner"),
            expected_signers=(CLOCK_YOU_SIGNER_SHA256,),
            run_command=runner,
        )

  def test_unrelated_badging_error_cannot_use_manifest_fallback(self):
    def runner(command, **kwargs):
      del kwargs
      return subprocess.CompletedProcess(
          command, 1, self._badging(), "ERROR: corrupt resources.arsc\n"
      )

    with tempfile.TemporaryDirectory() as tmpdir:
      with self.assertRaisesRegex(provision.ProvisionError, "aapt rejected"):
        self._inspect(Path(tmpdir), runner)

  def test_badging_and_manifest_identity_must_match(self):
    def runner(command, **kwargs):
      del kwargs
      if command[1:3] == ["dump", "badging"]:
        return subprocess.CompletedProcess(
            command, 1, self._badging(), self._missing_icon_error()
        )
      return subprocess.CompletedProcess(
          command, 0, self._xmltree("com.example.substituted"), ""
      )

    with tempfile.TemporaryDirectory() as tmpdir:
      with self.assertRaisesRegex(
          provision.ProvisionError, "badging/xmltree metadata mismatch"
      ):
        self._inspect(Path(tmpdir), runner)

  def test_manifest_fallback_does_not_bypass_signer_gate(self):
    def runner(command, **kwargs):
      del kwargs
      if command[1:3] == ["dump", "badging"]:
        return subprocess.CompletedProcess(
            command, 1, self._badging(), self._missing_icon_error()
        )
      return subprocess.CompletedProcess(command, 0, self._xmltree(), "")

    with tempfile.TemporaryDirectory() as tmpdir:
      with self.assertRaisesRegex(
          provision.ProvisionError, "signer verification failed or mismatched"
      ):
        self._inspect(Path(tmpdir), runner, observed_signer="f" * 64)


class SplitSelectionTest(unittest.TestCase):

  def _profile(self) -> provision.DeviceProfile:
    return provision.DeviceProfile(
        serial="emulator-5576",
        adb_server_port=5051,
        build_fingerprint="google/test",
        api_level="33",
        abi_list=("x86_64", "x86"),
        density=420,
        locale="en-US",
        boot_id="boot-id",
    )

  def test_selects_only_matching_abi_and_nearest_density(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      rows = tuple(_apk(root, split) for split in (
          "base", "config.x86", "config.x86_64", "config.arm64_v8a",
          "config.xhdpi", "config.xxhdpi", "config.fr",
      ))
      pin = provision.AppPin(
          APP_ID, "clock", PACKAGE_NAME, VERSION_NAME, VERSION_CODE,
          rows[0].sha256, rows[0].path, (CLOCK_YOU_SIGNER_SHA256,),
      )
      app = provision.PreparedApp(pin, rows[0].sha256, "artifact_file", "", rows)
      selected = provision.select_device_apks(app, self._profile())
      self.assertEqual(
          [apk.split_id for apk in selected],
          ["base", "config.x86_64", "config.xxhdpi"],
      )

  def test_unknown_split_qualifier_fails_closed(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      base = _apk(root, "base")
      unknown = _apk(root, "config.feature-unknown")
      pin = provision.AppPin(
          APP_ID, "clock", PACKAGE_NAME, VERSION_NAME, VERSION_CODE,
          base.sha256, base.path, (CLOCK_YOU_SIGNER_SHA256,),
      )
      app = provision.PreparedApp(
          pin, base.sha256, "artifact_file", "", (base, unknown)
      )
      with self.assertRaisesRegex(provision.ProvisionError, "Unsupported split"):
        provision.select_device_apks(app, self._profile())


class AdbAndInstallTest(unittest.TestCase):

  def test_adb_server_port_precedes_device_serial(self):
    self.assertEqual(
        provision._adb_base(  # pylint: disable=protected-access
            Path("/sdk/adb"), "emulator-5576", 5051
        ),
        ["/sdk/adb", "-P", "5051", "-s", "emulator-5576"],
    )

  def test_install_multiple_contains_only_selected_apks(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      base = _apk(root, "base")
      split = _apk(root, "config.x86_64")
      unused = _apk(root, "config.arm64_v8a")
      pin = provision.AppPin(
          APP_ID, "clock", PACKAGE_NAME, VERSION_NAME, VERSION_CODE,
          base.sha256, base.path, (CLOCK_YOU_SIGNER_SHA256,),
      )
      app = provision.PreparedApp(
          pin, base.sha256, "artifact_file", "", (base, split, unused)
      )
      profile = provision.DeviceProfile(
          "emulator-5576", 5051, "fingerprint", "33", ("x86_64",),
          420, "en-US", "boot-id",
      )
      calls: list[list[str]] = []

      def runner(command, **kwargs):
        del kwargs
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "Success\n", "")

      provision.install_app(
          app,
          (base, split),
          profile=profile,
          adb_path=Path("/sdk/adb"),
          run_command=runner,
      )
      self.assertEqual(calls[0][0:6], [
          "/sdk/adb", "-P", "5051", "-s", "emulator-5576",
          "install-multiple",
      ])
      self.assertIn(str(base.path), calls[0])
      self.assertIn(str(split.path), calls[0])
      self.assertNotIn(str(unused.path), calls[0])

  def test_package_and_pm_path_parsers_are_exact(self):
    self.assertEqual(
        provision._parse_package_version(  # pylint: disable=protected-access
            f"  versionCode={VERSION_CODE} minSdk=24 targetSdk=35\n"
            f"  versionName={VERSION_NAME}\n"
        ),
        (VERSION_NAME, VERSION_CODE),
    )
    self.assertEqual(
        provision._parse_pm_paths(  # pylint: disable=protected-access
            "package:/data/app/base.apk\npackage:/data/app/config.x86_64.apk\n"
        ),
        ("/data/app/base.apk", "/data/app/config.x86_64.apk"),
    )
    with self.assertRaises(provision.ProvisionError):
      provision._parse_pm_paths("package:/data/app/base.apk\npackage:/data/app/base.apk\n")

  def test_attestation_repulls_and_matches_the_exact_selected_split_set(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      base = _apk(root, "base", payload=b"base-bytes")
      split = _apk(root, "config.x86_64", payload=b"split-bytes")
      pin = provision.AppPin(
          APP_ID, "clock", PACKAGE_NAME, VERSION_NAME, VERSION_CODE,
          base.sha256, base.path, (CLOCK_YOU_SIGNER_SHA256,),
      )
      app = provision.PreparedApp(
          pin, base.sha256, "artifact_file", "", (base, split)
      )
      profile = provision.DeviceProfile(
          "emulator-5576", 5051, "fingerprint", "33", ("x86_64",),
          420, "en-US", "boot-id",
      )

      def runner(command, **kwargs):
        del kwargs
        if command[-4:] == ["shell", "dumpsys", "package", PACKAGE_NAME]:
          return subprocess.CompletedProcess(
              command, 0,
              f"versionCode={VERSION_CODE} minSdk=24\n"
              f"versionName={VERSION_NAME}\n", "",
          )
        if command[-4:] == ["shell", "pm", "path", PACKAGE_NAME]:
          return subprocess.CompletedProcess(
              command, 0,
              "package:/data/app/base.apk\n"
              "package:/data/app/config.x86_64.apk\n", "",
          )
        if "pull" in command:
          payload = b"base-bytes" if command[-2].endswith("base.apk") else b"split-bytes"
          Path(command[-1]).write_bytes(payload)
          return subprocess.CompletedProcess(command, 0, "pulled", "")
        raise AssertionError(command)

      def inspect(path, **kwargs):
        del kwargs
        payload = path.read_bytes()
        source = base if payload == b"base-bytes" else split
        return provision.dataclasses.replace(source, path=path)

      with mock.patch.object(provision, "_inspect_apk", side_effect=inspect):
        evidence = provision.collect_installed_app(
            app,
            (base, split),
            profile=profile,
            adb_path=Path("/sdk/adb"),
            aapt_path=Path("/sdk/aapt2"),
            apksigner_path=Path("/sdk/apksigner"),
            pull_root=root / "pull",
            run_command=runner,
        )
      self.assertTrue(evidence["valid"])
      self.assertEqual(evidence["errors"], [])
      self.assertEqual(
          evidence["installed_apk_sha256"], sorted([base.sha256, split.sha256])
      )


class EvidenceSafetyTest(unittest.TestCase):

  def test_output_is_not_silently_overwritten(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      output = Path(tmpdir) / "evidence.json"
      output.write_text("existing", encoding="utf-8")
      with self.assertRaisesRegex(provision.ProvisionError, "already exists"):
        provision._atomic_json(output, {"valid": True})  # pylint: disable=protected-access

  def test_badging_parser_preserves_split_identity(self):
    output = (
        f"package: name='{PACKAGE_NAME}' versionCode='{VERSION_CODE}' "
        f"versionName='{VERSION_NAME}' split='config.x86_64' "
        "platformBuildVersionName=''\n"
    )
    self.assertEqual(
        provision._parse_badging(output, "split.apk"),  # pylint: disable=protected-access
        (PACKAGE_NAME, VERSION_NAME, VERSION_CODE, "config.x86_64"),
    )


if __name__ == "__main__":
  unittest.main()
