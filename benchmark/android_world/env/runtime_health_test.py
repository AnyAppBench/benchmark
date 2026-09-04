"""Tests for in-episode CATBench emulator runtime checks."""

from __future__ import annotations

from unittest import mock

from absl.testing import absltest
from android_env.proto import adb_pb2
from android_world.agents import episode_exceptions
from android_world.env import runtime_health


def _response(output: str, *, ok: bool = True) -> adb_pb2.AdbResponse:
  response = adb_pb2.AdbResponse()
  response.status = (
      adb_pb2.AdbResponse.Status.OK
      if ok
      else adb_pb2.AdbResponse.Status.ADB_ERROR
  )
  response.generic.output = output.encode('utf-8')
  return response


def _healthy_output(windows: str = 'mCurrentFocus=Window{app}') -> str:
  return (
      'CATBENCH_BOOT=1\n'
      'CATBENCH_UID=0\n'
      'CATBENCH_WINDOWS_BEGIN\n'
      f'{windows}\n'
      'CATBENCH_WINDOWS_END\n'
  )


class RuntimeHealthTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = mock.MagicMock()
    self.controller = self.env.controller

  @mock.patch.object(runtime_health.adb_utils, 'issue_generic_request')
  def test_healthy_device_passes(self, issue_request):
    issue_request.return_value = _response(_healthy_output())

    runtime_health.assert_device_runtime_healthy(self.env)

    issue_request.assert_called_once_with(
        ['shell', runtime_health._HEALTH_SCRIPT],
        self.controller,
        timeout_sec=10.0,
    )
    self.assertIn('grep -E -i', runtime_health._HEALTH_SCRIPT)
    self.assertIn('Window #0 ', runtime_health._HEALTH_SCRIPT)

  @mock.patch.object(runtime_health.adb_utils, 'issue_generic_request')
  def test_visible_anr_raises_typed_environment_error(self, issue_request):
    issue_request.return_value = _response(
        _healthy_output('Application Not Responding: com.android.systemui')
    )

    with self.assertRaisesRegex(
        episode_exceptions.EmulatorRuntimeHealthError,
        'ANR/crash dialog',
    ):
      runtime_health.assert_device_runtime_healthy(self.env)

  @mock.patch.object(runtime_health.adb_utils, 'issue_generic_request')
  def test_visible_crash_raises_typed_environment_error(self, issue_request):
    issue_request.return_value = _response(
        _healthy_output('App keeps stopping')
    )

    with self.assertRaisesRegex(
        episode_exceptions.EmulatorRuntimeHealthError,
        'ANR/crash dialog',
    ):
      runtime_health.assert_device_runtime_healthy(self.env)

  @mock.patch.object(runtime_health.adb_utils, 'issue_generic_request')
  def test_lost_boot_or_adb_fails_closed(self, issue_request):
    issue_request.return_value = _response(
        _healthy_output().replace('CATBENCH_BOOT=1', 'CATBENCH_BOOT=')
    )

    with self.assertRaisesRegex(
        episode_exceptions.EmulatorRuntimeHealthError,
        'not boot-complete',
    ):
      runtime_health.assert_device_runtime_healthy(self.env)

  @mock.patch.object(runtime_health.adb_utils, 'issue_generic_request')
  def test_empty_window_manager_response_fails_closed(self, issue_request):
    issue_request.return_value = _response(_healthy_output(''))

    with self.assertRaisesRegex(
        episode_exceptions.EmulatorRuntimeHealthError,
        'window-manager response is empty',
    ):
      runtime_health.assert_device_runtime_healthy(self.env)


if __name__ == '__main__':
  absltest.main()
