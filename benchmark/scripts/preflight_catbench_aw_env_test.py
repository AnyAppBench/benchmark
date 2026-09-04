from __future__ import annotations

import subprocess
from unittest import mock

import preflight_catbench_aw_env as preflight


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
  return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def test_device_runtime_health_accepts_booted_root_device_without_dialog() -> None:
  with mock.patch.object(
      preflight,
      "_adb",
      side_effect=[
          _completed("1\n"),
          _completed("0\n"),
          _completed("mCurrentFocus=Window{1 u0 launcher}\n"),
      ],
  ):
    health = preflight._device_runtime_health("emulator-5800")
  assert health["valid"] is True
  assert health["window_fault_markers"] == []


def test_device_runtime_health_rejects_system_ui_anr() -> None:
  with mock.patch.object(
      preflight,
      "_adb",
      side_effect=[
          _completed("1\n"),
          _completed("0\n"),
          _completed(
              "mCurrentFocus=Window{1 u0 Application Not Responding: "
              "com.android.systemui}\n"
          ),
      ],
  ):
    health = preflight._device_runtime_health("emulator-5800")
  assert health["valid"] is False
  assert health["window_fault_markers"] == ["Application Not Responding:"]


def test_device_runtime_health_rejects_crash_dialog_case_insensitively() -> None:
  with mock.patch.object(
      preflight,
      "_adb",
      side_effect=[
          _completed("1\n"),
          _completed("0\n"),
          _completed("mCurrentFocus=Window{1 u0 APP KEEPS STOPPING}\n"),
      ],
  ):
    health = preflight._device_runtime_health("emulator-5800")
  assert health["valid"] is False
  assert health["window_fault_markers"] == ["keeps stopping"]


def test_comaps_resource_series_matches_pinned_apk_series() -> None:
  comaps = next(
      group for group in preflight.MAP_RESOURCE_GROUPS
      if group["package"] == "app.comaps.fdroid"
  )
  assert comaps["base_url"] == (
      "https://mapgen-fi-1.comaps.app/maps/260405"
  )
  assert comaps["cache_dir"] == "comaps_260405"
  assert str(comaps["remote_dir"]).endswith("/files/260405")
  assert str(comaps["internal_dir"]).endswith("/files/260405")
  assert comaps["files"] == {
      "WorldCoasts.mwm": 8_492_865,
      "World.mwm": 53_104_836,
      "Liechtenstein.mwm": 3_993_906,
      "Switzerland_Eastern.mwm": 108_002_034,
      "US_California_Chico.mwm": 60_436_041,
  }
  assert comaps["sha256"] == {
      "WorldCoasts.mwm": "5a1b573696057250e148afa21b6b01324281322b443d1f584943289a82a05850",
      "World.mwm": "6ee0f7be132895b2cbb350a569ceddfb1eb9f6cf8045bf06b3f9dc0835de6ac2",
      "Liechtenstein.mwm": "11ab91df1671e96ba63e4f2990be6fa15317acc17677d419e450a9c2347ceaf8",
      "Switzerland_Eastern.mwm": "201af2dc4734f0f3b83d71ef1005290bc16f34922b9da111c4f32a8d83d9f0b0",
      "US_California_Chico.mwm": "7f0930feef88ed7ddc7de05e951218bdd0335c7f1396d202b2d395bbea6b254d",
  }


def test_every_offline_resource_has_exact_sha256_pin() -> None:
  assert len(preflight.OSMAND_MAP_SHA256) == 64
  for group in preflight.MAP_RESOURCE_GROUPS:
    assert set(group["files"]) == set(group["sha256"])
    assert all(
        len(digest) == 64 and int(digest, 16) >= 0
        for digest in group["sha256"].values()
    )


def test_sha256_file_reads_all_bytes(tmp_path) -> None:
  path = tmp_path / "resource.bin"
  path.write_bytes(b"CATBench map bytes")
  assert preflight._sha256_file(path) == (
      "4c6784aa9a8c7a345b7154e2fcb73aa6b952f75d09893df6818c2079e66c6cf6"
  )
