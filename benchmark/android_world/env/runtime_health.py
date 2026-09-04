"""Fail-closed runtime health checks for CATBench Android episodes."""

from __future__ import annotations

import re
from typing import Any

from android_env.proto import adb_pb2
from android_world.agents import episode_exceptions
from android_world.env import adb_utils
from android_world.env import interface


_HEALTH_SCRIPT = (
    'printf "CATBENCH_BOOT="; getprop sys.boot_completed; '
    'printf "CATBENCH_UID="; id -u; '
    'printf "CATBENCH_WINDOWS_BEGIN\\n"; '
    'dumpsys window windows | grep -E -i '
    "'Window #0 |mCurrentFocus|mFocusedApp|Application Not Responding:|"
    "Application Error:|isn.t responding|is not responding|has stopped|"
    "keeps stopping' || true; "
    'printf "CATBENCH_WINDOWS_END\\n"'
)
_FAULT_PATTERNS = (
    re.compile(r'Application Not Responding:', re.IGNORECASE),
    re.compile(r'Application Error:', re.IGNORECASE),
    re.compile(r"\bisn['’]?t responding\b", re.IGNORECASE),
    re.compile(r'\bis not responding\b', re.IGNORECASE),
    re.compile(r'\bhas stopped\b', re.IGNORECASE),
    re.compile(r'\bkeeps stopping\b', re.IGNORECASE),
)


def _response_output(response: Any) -> str:
  try:
    return response.generic.output.decode('utf-8', errors='replace')
  except AttributeError as error:
    raise episode_exceptions.EmulatorRuntimeHealthError(
        'Runtime-health ADB response did not contain generic output.'
    ) from error


def assert_device_runtime_healthy(env: interface.AsyncEnv) -> None:
  """Raises if a CATBench device rebooted, lost root, or shows a fault dialog.

  One device-specific ADB shell transaction is used so this check can run at
  every episode boundary without multiplying ADB round trips. Only focus and
  crash/ANR lines are emitted from the current window dump: an unfiltered dump
  can exceed the controller's response limit and lose its end sentinel.
  """
  controller = getattr(env, 'controller', None)
  if controller is None:
    raise episode_exceptions.EmulatorRuntimeHealthError(
        'CATBench runtime-health check requires an Android controller.'
    )
  try:
    response = adb_utils.issue_generic_request(
        ['shell', _HEALTH_SCRIPT],
        controller,
        timeout_sec=10.0,
    )
  except Exception as error:
    raise episode_exceptions.EmulatorRuntimeHealthError(
        f'Runtime-health ADB transaction failed: {error}'
    ) from error
  if response.status != adb_pb2.AdbResponse.Status.OK:
    raise episode_exceptions.EmulatorRuntimeHealthError(
        'Runtime-health ADB transaction returned non-OK status '
        f'{response.status}: {_response_output(response).strip()}'
    )

  output = _response_output(response).replace('\r', '')
  boot = re.search(r'^CATBENCH_BOOT=(.*)$', output, re.MULTILINE)
  uid = re.search(r'^CATBENCH_UID=(.*)$', output, re.MULTILINE)
  begin = output.find('CATBENCH_WINDOWS_BEGIN\n')
  end = output.find('CATBENCH_WINDOWS_END\n')
  if boot is None or uid is None or begin < 0 or end <= begin:
    raise episode_exceptions.EmulatorRuntimeHealthError(
        'Runtime-health response is missing required sentinels.'
    )
  if boot.group(1).strip() != '1':
    raise episode_exceptions.EmulatorRuntimeHealthError(
        f'Emulator is not boot-complete (sys.boot_completed={boot.group(1)!r}).'
    )
  if uid.group(1).strip() != '0':
    raise episode_exceptions.EmulatorRuntimeHealthError(
        f'Emulator lost required root ADB (uid={uid.group(1)!r}).'
    )

  windows = output[begin + len('CATBENCH_WINDOWS_BEGIN\n'):end]
  if not windows.strip():
    raise episode_exceptions.EmulatorRuntimeHealthError(
        'Runtime-health window-manager response is empty.'
    )
  faults = sorted({
      pattern.pattern
      for pattern in _FAULT_PATTERNS
      if pattern.search(windows)
  })
  if faults:
    raise episode_exceptions.EmulatorRuntimeHealthError(
        'Visible Android ANR/crash dialog detected by window manager: '
        + ', '.join(faults)
    )
