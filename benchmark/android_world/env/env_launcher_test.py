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

import os
from unittest import mock

from absl.testing import absltest
from android_env import env_interface
from android_env import loader
from android_env.components import config_classes
from android_env.components.simulators.emulator import emulator_simulator
from android_world.env import android_world_controller
from android_world.env import env_launcher
from android_world.env import interface


class EnvLauncherTest(absltest.TestCase):

  def test_androidenv_preserves_explicit_tcp_serial(self):
    config = config_classes.EmulatorConfig(
        emulator_launcher=config_classes.EmulatorLauncherConfig(
            emulator_console_port=5800,
            adb_port=5801,
            grpc_port=8800,
        ),
        adb_controller=config_classes.AdbControllerConfig(
            device_name="localhost:5801"
        ),
    )
    simulator = object.__new__(emulator_simulator.EmulatorSimulator)
    simulator._config = config  # pylint: disable=protected-access

    self.assertEqual("localhost:5801", simulator.adb_device_name())

  @mock.patch.object(interface, "AsyncAndroidEnv", autospec=True)
  @mock.patch.object(
      android_world_controller, "AndroidWorldController", autospec=True
  )
  @mock.patch.object(loader, "load", autospec=True)
  def test_get_env(
      self,
      mock_loader,
      mock_controller,
      mock_async_android_env,
  ):
    mock_android_env = mock.create_autospec(env_interface.AndroidEnvInterface)
    mock_loader.return_value = mock_android_env

    env_launcher._get_env(5556, "some_adb_path", 8554)

    mock_loader.assert_called_with(
        config=config_classes.AndroidEnvConfig(
            task=config_classes.FilesystemTaskConfig(
                path=android_world_controller._TASK_PATH
            ),
            simulator=config_classes.EmulatorConfig(
                emulator_launcher=config_classes.EmulatorLauncherConfig(
                    emulator_console_port=5556, adb_port=5557, grpc_port=8554
                ),
                adb_controller=config_classes.AdbControllerConfig(
                    adb_path="some_adb_path",
                    device_name="emulator-5556",
                ),
            ),
        )
    )
    mock_controller.assert_called_with(
        mock_android_env,
        a11y_method=android_world_controller.A11yMethod.A11Y_FORWARDER_APP,
        install_a11y_forwarding_app=True,
    )
    mock_async_android_env.assert_called_with(mock_controller.return_value)

  @mock.patch.object(interface, "AsyncAndroidEnv", autospec=True)
  @mock.patch.object(
      android_world_controller, "AndroidWorldController", autospec=True
  )
  @mock.patch.object(loader, "load", autospec=True)
  def test_get_env_restores_explicit_tcp_serial_after_loader(
      self,
      mock_loader,
      mock_controller,
      mock_async_android_env,
  ):
    mock_android_env = mock.create_autospec(env_interface.AndroidEnvInterface)

    def emulate_emulator_simulator(config):
      config.simulator.adb_controller.device_name = "emulator-5556"
      return mock_android_env

    mock_loader.side_effect = emulate_emulator_simulator
    with mock.patch.dict(
        os.environ, {"ANDROID_SERIAL": "localhost:5803"}, clear=False
    ):
      env_launcher._get_env(5556, "some_adb_path", 8554)

    config = mock_loader.call_args.args[0]
    self.assertEqual(
        "localhost:5803", config.simulator.adb_controller.device_name
    )
    mock_controller.assert_called_once()
    mock_async_android_env.assert_called_with(mock_controller.return_value)


if __name__ == "__main__":
  absltest.main()
