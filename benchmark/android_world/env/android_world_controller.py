# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Controller for Android that adds UI tree information to the observation."""

import contextlib
import enum
import os
import time
import urllib.request  # pylint: disable=unused-import
from typing import Any
from typing import cast
from typing import Optional
from absl import logging
from android_env import env_interface
from android_env import loader
from android_env.components import adb_controller as adb_controller_lib
from android_env.components import config_classes
from android_env.components.simulators.emulator import emulator_simulator as emulator_simulator_lib
from android_env.proto.a11y import android_accessibility_forest_pb2
from android_env.wrappers import a11y_grpc_wrapper
from android_env.wrappers import base_wrapper
from android_world.env import adb_utils
from android_world.env import representation_utils
from android_world.utils import file_utils
import dm_env


def _has_wrapper(
    env: env_interface.AndroidEnvInterface,
    target_wrapper: Any,
) -> bool:
  """Checks recursively if an environment object has a certain wrapper.

  Args:
    env: The environment object potentially wrapped.
    target_wrapper: The wrapper type to search for.

  Returns:
    True if the target_wrapper is found, otherwise False.
  """
  if isinstance(env, target_wrapper):
    return True
  elif hasattr(env, '_env'):
    return _has_wrapper(env._env, target_wrapper)  # pylint: disable=protected-access
  else:
    return False


def get_a11y_tree(
    env: env_interface.AndroidEnvInterface,
    max_retries: int = 5,
    sleep_duration: float = 1.0,
) -> android_accessibility_forest_pb2.AndroidAccessibilityForest:
  """Gets a11y tree.

  Args:
    env: AndroidEnv.
    max_retries: Maximum number of retries to get a11y tree.
    sleep_duration: Time to sleep between each retry in seconds.

  Returns:
    A11y tree.

  Raises:
    RuntimeError: If the a11y tree was not able to be retrieved.
  """
  if not _has_wrapper(env, a11y_grpc_wrapper.A11yGrpcWrapper):
    raise ValueError(
        'Must use a11y_grpc_wrapper.A11yGrpcWrapper to get the a11y tree.'
    )
  env = cast(a11y_grpc_wrapper.A11yGrpcWrapper, env)
  if adb_utils.retry(3)(adb_utils.check_airplane_mode)(env):
    logging.warning(
        'Airplane mode is on -- cannot retrieve a11y tree via gRPC. Turning'
        ' it off...'
    )
    logging.info('Enabling networking...')
    env.attempt_enable_networking()
    time.sleep(1.0)

  forest: Optional[
      android_accessibility_forest_pb2.AndroidAccessibilityForest
  ] = None
  for _ in range(max_retries):
    try:
      forest = env.accumulate_new_extras()['accessibility_tree'][-1]  # pytype:disable=attribute-error
      return forest
    except (KeyError, IndexError, TypeError):
      logging.warning('Could not get a11y tree, retrying.')
    time.sleep(sleep_duration)

  if forest is None:
    raise RuntimeError('Could not get a11y tree.')
  return forest


_TASK_PATH = os.environ.get('ANDROID_WORLD_TASK_PROTO_PATH') or (
    file_utils.convert_to_posix_path(
        file_utils.get_local_tmp_directory(),
        f'default_{os.getuid()}_{os.getpid()}.textproto',
    )
)
DEFAULT_ADB_PATH = '~/Android/Sdk/platform-tools/adb'
DEFAULT_ADB_SERVER_PORT = 5037


def _patch_emulator_device_name_for_explicit_serials() -> None:
  """Stops AndroidEnv from replacing an explicitly configured ADB serial."""
  simulator_class = emulator_simulator_lib.EmulatorSimulator
  original = simulator_class.adb_device_name
  if getattr(original, '_catbench_explicit_serial', False):
    return

  def adb_device_name(self) -> str:
    configured = str(  # pylint: disable=protected-access
        self._config.adb_controller.device_name or ''
    )
    return configured or original(self)

  adb_device_name._catbench_explicit_serial = True
  simulator_class.adb_device_name = adb_device_name


def _patch_adb_restart_for_tcp_serials() -> None:
  """Reconnects an explicit TCP device after AndroidEnv restarts ADB.

  AndroidEnv restarts its configured ADB server after a command timeout. A TCP
  transport is not retained across that restart, so the next retry would fail
  with ``device not found`` unless it is connected again. This process-local
  patch leaves ordinary ``emulator-NNNN`` devices unchanged.
  """
  controller_class = adb_controller_lib.AdbController
  original = (  # pylint: disable=protected-access
      controller_class._restart_server
  )
  if getattr(original, '_catbench_tcp_reconnect', False):
    return

  def restart_and_reconnect(self, timeout: float | None = None):
    original(self, timeout)
    device_name = str(  # pylint: disable=protected-access
        self._config.device_name or ''
    )
    if ':' not in device_name:
      return
    configured_timeout = float(
        timeout
        if timeout is not None
        else self._config.default_timeout  # pylint: disable=protected-access
    )
    self.execute_command(
        ['connect', device_name],
        timeout=max(configured_timeout, 10.0),
        device_specific=False,
    )

  restart_and_reconnect._catbench_tcp_reconnect = True
  controller_class._restart_server = (  # pylint: disable=protected-access
      restart_and_reconnect
  )


_patch_emulator_device_name_for_explicit_serials()
_patch_adb_restart_for_tcp_serials()


def _get_adb_server_port() -> int:
  raw_port = os.environ.get('ANDROID_ADB_SERVER_PORT') or os.environ.get(
      'ADB_SERVER_PORT'
  )
  if not raw_port:
    return DEFAULT_ADB_SERVER_PORT
  try:
    return int(raw_port)
  except ValueError:
    logging.warning(
        'Invalid ADB server port %r; falling back to %d.',
        raw_port,
        DEFAULT_ADB_SERVER_PORT,
    )
    return DEFAULT_ADB_SERVER_PORT


# UI tree-specific keys that are added to observations:

# The forest is essentially a comprehensive snapshot of all user interface
# elements currently displayed on an Android device's screen. Each 'tree' in
# this 'forest' represents the accessibility details of a different window or
# screen section, providing structured information. The tree's origin is from
# the AccessibilityService. Please see the following for more detail:
# https://developer.android.com/reference/android/accessibilityservice/AccessibilityService

OBSERVATION_KEY_FOREST = 'forest'
# UI elements are specific nodes extracted from forest. See
# representation_utils.forest_to_ui_elements for details.
OBSERVATION_KEY_UI_ELEMENTS = 'ui_elements'

_A11Y_FORWARDER_PACKAGE = 'com.google.androidenv.accessibilityforwarder'
_A11Y_FORWARDER_RECEIVER = (
    f'{_A11Y_FORWARDER_PACKAGE}/'
    f'{_A11Y_FORWARDER_PACKAGE}.FlagsBroadcastReceiver'
)
_A11Y_FORWARDER_SERVICE = (
    f'{_A11Y_FORWARDER_PACKAGE}/'
    f'{_A11Y_FORWARDER_PACKAGE}.AccessibilityForwarder'
)
_ACTION_ENABLE_ACCESSIBILITY_TREE_LOGS = (
    'accessibility_forwarder.intent.action.ENABLE_ACCESSIBILITY_TREE_LOGS'
)
_ACTION_ENABLE_GRPC = 'accessibility_forwarder.intent.action.ENABLE_GRPC'
_ACTION_SET_GRPC = 'accessibility_forwarder.intent.action.SET_GRPC'
_A11Y_GRPC_HOST = '10.0.2.2'


class A11yMethod(enum.Enum):
  """Method to get a11y tree."""

  # Custom gRPC wrapper that uses a11y forwarder app.
  A11Y_FORWARDER_APP = 'a11y_forwarder_app'

  # From `uiautomator dump``.
  UIAUTOMATOR = 'uiautomator'

  # No A11y tree retrieval
  NONE = 'none'


def apply_a11y_forwarder_app_wrapper(
    env: env_interface.AndroidEnvInterface, install_a11y_forwarding_app: bool
) -> env_interface.AndroidEnvInterface:
  return a11y_grpc_wrapper.A11yGrpcWrapper(
      env,
      install_a11y_forwarding=install_a11y_forwarding_app,
      start_a11y_service=True,
      enable_a11y_tree_info=True,
      latest_a11y_info_only=True,
  )


def _adb_output_text(response: Any) -> str:
  """Best-effort ADB generic output decoding."""
  try:
    return response.generic.output.decode('utf-8').strip()
  except (AttributeError, UnicodeDecodeError):
    return ''


def _a11y_refresh_command(
    env: env_interface.AndroidEnvInterface,
    args: list[str],
    timeout_sec: float = 3.0,
) -> None:
  """Runs a best-effort ADB command during a11y recovery."""
  try:
    adb_utils.issue_generic_request(args, env, timeout_sec=timeout_sec)
  except Exception:  # pylint: disable=broad-exception-caught
    logging.exception('A11y recovery command failed: %s', args)


def _refresh_a11y_forwarder(env: env_interface.AndroidEnvInterface) -> None:
  """Best-effort nudge for a stale AndroidWorld accessibility stream.

  The a11y gRPC wrapper can temporarily stop producing tree extras even though
  the emulator is otherwise alive. This refresh keeps the running app intact:
  it wakes the device, ensures the forwarder accessibility service is enabled,
  and re-sends the forwarder's gRPC endpoint and logging flags.
  """
  logging.warning('Refreshing AndroidWorld accessibility forwarder.')
  attempt_enable_networking = getattr(env, 'attempt_enable_networking', None)
  if callable(attempt_enable_networking):
    try:
      attempt_enable_networking()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception('Could not refresh emulator networking for a11y.')

  _a11y_refresh_command(env, ['shell', 'input', 'keyevent', 'KEYCODE_WAKEUP'])
  _a11y_refresh_command(
      env, ['shell', 'settings', 'put', 'secure', 'accessibility_enabled', '1']
  )

  services_response = None
  try:
    services_response = adb_utils.issue_generic_request(
        ['shell', 'settings', 'get', 'secure', 'enabled_accessibility_services'],
        env,
        timeout_sec=3.0,
    )
  except Exception:  # pylint: disable=broad-exception-caught
    logging.exception('Could not read enabled accessibility services.')

  enabled_services = _adb_output_text(services_response) if services_response else ''
  if enabled_services.lower() in ('', 'null'):
    service_entries: list[str] = []
  else:
    service_entries = [
        item for item in enabled_services.split(':') if item and item != 'null'
    ]
  if _A11Y_FORWARDER_SERVICE not in service_entries:
    service_entries.append(_A11Y_FORWARDER_SERVICE)
    _a11y_refresh_command(
        env,
        [
            'shell',
            'settings',
            'put',
            'secure',
            'enabled_accessibility_services',
            ':'.join(service_entries),
        ],
    )

  grpc_port: Optional[int] = None
  get_port = getattr(env, 'get_port', None)
  if callable(get_port):
    try:
      grpc_port = int(get_port())
    except (TypeError, ValueError):
      logging.warning('A11y wrapper returned an invalid gRPC port.')
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception('Could not read a11y wrapper gRPC port.')

  if grpc_port and grpc_port > 0:
    _a11y_refresh_command(
        env,
        [
            'shell',
            'settings',
            'put',
            'global',
            'no_proxy',
            f'{_A11Y_GRPC_HOST}:{grpc_port}',
        ],
    )
    _a11y_refresh_command(
        env,
        [
            'shell',
            'am',
            'broadcast',
            '-n',
            _A11Y_FORWARDER_RECEIVER,
            '-a',
            _ACTION_SET_GRPC,
            '--ei',
            'port',
            str(grpc_port),
        ],
    )
  else:
    logging.warning('Skipping a11y SET_GRPC refresh because no port is known.')

  for action in (_ACTION_ENABLE_ACCESSIBILITY_TREE_LOGS, _ACTION_ENABLE_GRPC):
    _a11y_refresh_command(
        env,
        ['shell', 'am', 'broadcast', '-n', _A11Y_FORWARDER_RECEIVER, '-a', action],
    )


class AndroidWorldController(base_wrapper.BaseWrapper):
  """Controller for an Android instance that adds accessibility tree data.

  The Accessibility Tree in Android is a tree-based structure, originally for
  for assisting accessibility services. It provides information about UI
  elements (like text, buttons, and images) in a hierarchical format. The tree
  includes details such as the properties and actions available for each
  element.
  """

  def __init__(
      self,
      env: env_interface.AndroidEnvInterface,
      a11y_method: A11yMethod = A11yMethod.A11Y_FORWARDER_APP,
      install_a11y_forwarding_app: bool = True,
  ):
    self._original_env = env
    if a11y_method == A11yMethod.A11Y_FORWARDER_APP:
      self._env = apply_a11y_forwarder_app_wrapper(
          env, install_a11y_forwarding_app
      )
      self._env.reset()  # Initializes required server services in a11y wrapper.
      try:
        # Warm up a11y stream once to reduce first-step retrieval failures.
        get_a11y_tree(self._env, max_retries=10, sleep_duration=1.0)
      except Exception:  # pylint: disable=broad-exception-caught
        logging.warning(
            'A11y service was not ready after initialization; continuing and'
            ' retrying on demand.',
        )
    else:
      self._env = env
    self._a11y_method = a11y_method

  @property
  def device_screen_size(self) -> tuple[int, int]:
    """Returns the physical screen size of the device: (width, height)."""
    return adb_utils.get_screen_size(self._env)

  @property
  def logical_screen_size(self) -> tuple[int, int]:
    """Returns the logical screen size of the device.

    This will be different with the physical size if orientation or resolution
    is changed.
    """
    return adb_utils.get_logical_screen_size(self._env)

  @property
  def env(self) -> env_interface.AndroidEnvInterface:
    return self._env

  def refresh_env(self):
    # pylint: disable=protected-access
    # pytype: disable=attribute-error
    # Reconnect to emulator and reload a11y wrapper in case we lose connection.
    self._env = get_controller(
        console_port=self.env._coordinator._simulator._config.emulator_launcher.emulator_console_port,
        adb_path=self.env._coordinator._simulator._config.adb_controller.adb_path,
        grpc_port=self.env._coordinator._simulator._config.emulator_launcher.grpc_port,
    ).env
    # pylint: enable=protected-access
    # pytype: enable=attribute-error

  def _get_a11y_forest(
      self,
  ) -> android_accessibility_forest_pb2.AndroidAccessibilityForest:
    try:
      return get_a11y_tree(self._env)
    except RuntimeError:
      _refresh_a11y_forwarder(self._env)
      return get_a11y_tree(self._env, max_retries=8, sleep_duration=1.0)

  def get_a11y_forest(
      self,
  ) -> android_accessibility_forest_pb2.AndroidAccessibilityForest:
    """Returns the most recent a11y forest from the device."""
    try:
      return self._get_a11y_forest()
    except RuntimeError:
      print(
          'Could not get a11y tree. Reconnecting to Android, reinitializing'
          ' AndroidEnv, and restarting a11y forwarding.'
      )
      try:
        self.refresh_env()
        return self._get_a11y_forest()
      except RuntimeError:
        raise
      except Exception as reconnect_error:
        raise RuntimeError(
            'Could not recover a11y tree after reconnect.'
        ) from reconnect_error

  def get_ui_elements(self) -> list[representation_utils.UIElement]:
    """Returns the most recent UI elements from the device."""
    if self._a11y_method == A11yMethod.A11Y_FORWARDER_APP:
      try:
        return representation_utils.forest_to_ui_elements(
            self.get_a11y_forest(),
            exclude_invisible_elements=True,
        )
      except RuntimeError:
        logging.warning(
            'Falling back to uiautomator dump after a11y gRPC failure.'
        )
        return self._get_uiautomator_ui_elements()
    elif self._a11y_method == A11yMethod.UIAUTOMATOR:
      return self._get_uiautomator_ui_elements()
    else:
      return []

  def _get_uiautomator_ui_elements(
      self,
  ) -> list[representation_utils.UIElement]:
    try:
      return representation_utils.xml_dump_to_ui_elements(
          adb_utils.uiautomator_dump(self._env)
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception('Could not retrieve UI elements via uiautomator.')
      return []

  def _process_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Adds a11y tree info to the observation."""
    if self._a11y_method == A11yMethod.A11Y_FORWARDER_APP:
      try:
        forest = self.get_a11y_forest()
        ui_elements = representation_utils.forest_to_ui_elements(
            forest,
            exclude_invisible_elements=True,
        )
      except RuntimeError:
        logging.warning(
            'A11y gRPC unavailable after recovery; using uiautomator fallback.'
        )
        forest = None
        ui_elements = self._get_uiautomator_ui_elements()
    else:
      forest = None
      ui_elements = self.get_ui_elements()
    timestep.observation[OBSERVATION_KEY_FOREST] = forest
    timestep.observation[OBSERVATION_KEY_UI_ELEMENTS] = ui_elements
    return timestep

  def pull_file(
      self, remote_db_file_path: str, timeout_sec: Optional[float] = None
  ) -> contextlib._GeneratorContextManager[str]:
    """Pulls a file from the device to a temporary directory.

    The directory will be deleted when the context manager exits.
    Args:
      remote_db_file_path: The path to the file on the device.
      timeout_sec: Timeout in seconds for the adb calls.

    Returns:
      The path to the temporary directory containing the file.
    """
    remote_db_directory = os.path.dirname(remote_db_file_path)
    return file_utils.tmp_directory_from_device(
        remote_db_directory, self.env, timeout_sec
    )

  def push_file(
      self,
      local_db_file_path: str,
      remote_db_file_path: str,
      timeout_sec: Optional[float] = None,
  ) -> None:
    """Pushes a local file to the device."""

    remote_db_directory = os.path.dirname(remote_db_file_path)

    # First delete old .db, .db-wal, and .db-shm files.
    file_utils.clear_directory(remote_db_directory, self)
    file_utils.copy_data_to_device(
        local_db_file_path,
        remote_db_file_path,
        self.env,
        timeout_sec,
    )


def _write_default_task_proto() -> str:
  with open(_TASK_PATH, 'w') as f:
    f.write("""\
id: "default"

name: "Default task for device control."
description: "Empty task"

max_episode_sec: 7200  # Prevent infinite episodes.
  """)
  return _TASK_PATH


def get_controller(
    console_port: int = 5554,
    adb_path: str = DEFAULT_ADB_PATH,
    grpc_port: int = 8554,
    a11y_method: A11yMethod = A11yMethod.A11Y_FORWARDER_APP,
    install_a11y_forwarding_app: bool = True,
) -> AndroidWorldController:
  """Creates a controller by connecting to an existing Android environment."""

  adb_server_port = _get_adb_server_port()
  device_name = (
      os.environ.get('ANDROID_SERIAL') or f'emulator-{console_port}'
  )
  config = config_classes.AndroidEnvConfig(
      task=config_classes.FilesystemTaskConfig(
          path=_write_default_task_proto()
      ),
      simulator=config_classes.EmulatorConfig(
          emulator_launcher=config_classes.EmulatorLauncherConfig(
              emulator_console_port=console_port,
              adb_port=console_port + 1,
              grpc_port=grpc_port,
          ),
          adb_controller=config_classes.AdbControllerConfig(
              adb_path=adb_path,
              adb_server_port=adb_server_port,
              device_name=device_name,
          ),
      ),
  )
  android_env_instance = loader.load(config)
  # Retain the requested serial on the shared config object as a defense against
  # future AndroidEnv loader changes.
  config.simulator.adb_controller.device_name = device_name
  logging.info('Setting up AndroidWorldController.')
  return AndroidWorldController(
      android_env_instance,
      a11y_method=a11y_method,
      install_a11y_forwarding_app=install_a11y_forwarding_app,
  )
