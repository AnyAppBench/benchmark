"""No-launch tests for the CATBench emulator resource/image contracts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / 'benchmark/docker_setup/start_catbench_emu_headless.sh'
DOCKER_MANAGER = REPO_ROOT / 'benchmark/scripts/manage_catbench_docker_pool.sh'
PODMAN_MANAGER = (
    REPO_ROOT / 'benchmark/scripts/manage_catbench_podman_58xx_pool.sh'
)
FAKE_DIGEST = 'sha256:' + 'a' * 64


def _run(
    script: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
  full_env = os.environ.copy()
  if env:
    full_env.update(env)
  return subprocess.run(
      ['bash', str(script), *args],
      check=False,
      capture_output=True,
      text=True,
      timeout=10,
      env=full_env,
  )


class CatbenchEmulatorLaunchContractTest(unittest.TestCase):

  def test_wrapper_reports_pinned_resource_contract_without_launch(self):
    result = _run(
        WRAPPER,
        env={'CATBENCH_PRINT_EMULATOR_RESOURCE_CONTRACT': '1'},
    )

    self.assertEqual(0, result.returncode, result.stderr)
    self.assertEqual(
        'CATBENCH_EMULATOR_RESOURCE_CONTRACT memory_mb=4096 cores=2',
        result.stdout.strip(),
    )

  def test_wrapper_rejects_legacy_resource_values(self):
    result = _run(
        WRAPPER,
        env={
            'CATBENCH_PRINT_EMULATOR_RESOURCE_CONTRACT': '1',
            'CATBENCH_EMULATOR_MEMORY_MB': '2048',
            'CATBENCH_EMULATOR_CORES': '1',
        },
    )

    self.assertEqual(64, result.returncode)
    self.assertIn('memory drift', result.stderr)

  def test_wrapper_rewrites_legacy_launcher_flags(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      output = root / 'args.txt'
      emulator = root / 'emulator'
      emulator.write_text(
          '#!/usr/bin/env bash\n'
          'printf "%s\\n" "$@" > "${CATBENCH_FAKE_EMULATOR_ARGS}"\n',
          encoding='utf-8',
      )
      emulator.chmod(0o755)
      base = root / 'base.sh'
      base.write_text(
          '#!/usr/bin/env bash\n'
          ': "${HW_ACCEL_OVERRIDE}"\n'
          'nohup emulator @test -memory 2048 -cores 1 -no-window &\n'
          'pid=$!\n'
          'wait "${pid}"\n',
          encoding='utf-8',
      )
      base.chmod(0o755)

      result = _run(
          WRAPPER,
          env={
              'PATH': f'{root}:{os.environ["PATH"]}',
              'CATBENCH_BASE_EMULATOR_LAUNCHER': str(base),
              'CATBENCH_FAKE_EMULATOR_ARGS': str(output),
          },
      )

      self.assertEqual(0, result.returncode, result.stderr)
      self.assertEqual(
          ['@test', '-memory', '4096', '-cores', '2', '-no-window'],
          output.read_text(encoding='utf-8').splitlines(),
      )

  def test_production_manager_contract_is_digest_and_resource_pinned(self):
    result = _run(DOCKER_MANAGER, 'contract')

    self.assertEqual(0, result.returncode, result.stderr)
    self.assertIn('memory_mb=4096 cores=2', result.stdout)
    self.assertIn(
        'RAM_MB="${EMULATOR_MEMORY_MB}"',
        PODMAN_MANAGER.read_text(encoding='utf-8'),
    )

  def test_production_manager_rejects_mutable_image_reference(self):
    result = _run(DOCKER_MANAGER, 'contract', env={'IMAGE': 'android_world:latest'})

    self.assertNotEqual(0, result.returncode)
    self.assertIn('immutable repository digest', result.stderr)

  def test_podman_manager_requires_explicit_diagnostic_immutable_id(self):
    missing = _run(PODMAN_MANAGER, 'contract')
    mutable = _run(
        PODMAN_MANAGER,
        'contract',
        env={
            'CATBENCH_PODMAN_RUNTIME_DISPOSITION': 'diagnostic_only',
            'CATBENCH_PODMAN_EMULATOR_IMAGE': 'local/image:latest',
        },
    )
    explicit = _run(
        PODMAN_MANAGER,
        'contract',
        env={
            'CATBENCH_PODMAN_RUNTIME_DISPOSITION': 'diagnostic_only',
            'CATBENCH_PODMAN_EMULATOR_IMAGE': FAKE_DIGEST,
        },
    )

    self.assertNotEqual(0, missing.returncode)
    self.assertIn('diagnostic_only', missing.stderr)
    self.assertNotEqual(0, mutable.returncode)
    self.assertIn('exact local sha256 image ID', mutable.stderr)
    self.assertEqual(0, explicit.returncode, explicit.stderr)
    self.assertIn('analysis_eligible=false', explicit.stdout)

  def test_podman_manager_has_no_primary_release_path(self):
    result = _run(
        PODMAN_MANAGER,
        'contract',
        env={
            'CATBENCH_PODMAN_RUNTIME_DISPOSITION': 'primary',
            'CATBENCH_PODMAN_EMULATOR_IMAGE': FAKE_DIGEST,
        },
    )

    self.assertNotEqual(0, result.returncode)
    self.assertIn('no primary-release approval path', result.stderr)

  def test_podman_bare_inspect_id_is_normalized_before_comparison(self):
    source = PODMAN_MANAGER.read_text(encoding='utf-8')
    self.assertIn(
        'actual_image_id="sha256:${actual_image_id#sha256:}"', source
    )
    self.assertIn(
        'candidate_id="sha256:${candidate_id#sha256:}"', source
    )

  def test_podman_workers_use_independent_adb_servers(self):
    result = _run(
        PODMAN_MANAGER,
        'specs',
        env={
            'CATBENCH_PODMAN_POOL_SIZE': '4',
            'CATBENCH_PODMAN_FIRST_ADB_SERVER_PORT': '5041',
        },
    )

    self.assertEqual(0, result.returncode, result.stderr)
    self.assertEqual(
        'CATBENCH_EMULATORS='
        '5800:8800:-:5041,5802:8801:-:5042,'
        '5804:8802:-:5043,5806:8803:-:5044',
        result.stdout.strip(),
    )
    source = PODMAN_MANAGER.read_text(encoding='utf-8')
    self.assertIn(
        'WORKER_ADB_SERVER_PORT="$((FIRST_ADB_SERVER_PORT + index))"',
        source,
    )


if __name__ == '__main__':
  unittest.main()
