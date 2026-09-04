from unittest import mock

from absl.testing import absltest
from PIL import Image

from android_world.agents import episode_exceptions
from android_world.agents import ui_voyager
from android_world.utils import test_utils


_VALID_CLICK_RESPONSE = """Thought: Tap the visible control.
Action: Tap the control.
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 500]}}
</tool_call>"""


class UIVoyagerFailureBoundaryTest(absltest.TestCase):

  def _agent(self, env=None, *, max_retries=2):
    if env is None:
      env = test_utils.FakeAsyncEnv()
    agent = ui_voyager.UIVoyagerAgent(
        env,
        endpoint_url="http://127.0.0.1:8000/v1",
        model_name="fake-model",
        max_retries=max_retries,
        wait_after_action_seconds=0,
    )
    agent.transition_pause = 0
    return agent

  def test_endpoint_failure_raises_model_endpoint_error_after_retries(self):
    agent = self._agent(max_retries=2)

    with mock.patch.object(
        ui_voyager.requests,
        "post",
        side_effect=ui_voyager.requests.ConnectionError("offline"),
    ), mock.patch.object(ui_voyager.time, "sleep") as sleep_mock:
      with self.assertRaises(episode_exceptions.ModelEndpointError):
        agent._call_llm("do something", Image.new("RGB", (10, 10)))

    self.assertEqual(sleep_mock.call_count, 1)

  def test_default_generation_budget_leaves_room_for_multimodal_prompt(self):
    agent = self._agent(max_retries=1)
    response = mock.MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": _VALID_CLICK_RESPONSE}}]
    }
    with mock.patch.object(
        ui_voyager.requests, "post", return_value=response
    ) as post:
      agent._call_llm("do something", Image.new("RGB", (10, 10)))

    self.assertEqual(512, post.call_args.kwargs["json"]["max_tokens"])

  def test_nonpositive_generation_budget_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "max_new_tokens"):
      ui_voyager.UIVoyagerAgent(
          test_utils.FakeAsyncEnv(), max_new_tokens=0
      )

  def test_malformed_output_raises_declared_action_error(self):
    agent = self._agent()
    malformed = (
        "<tool_call>"
        '{"name":"mobile_use","arguments":{"action":"click"}}'
        "</tool_call>"
    )

    with mock.patch.object(
        agent, "_call_llm", return_value=(malformed, "system", "user")
    ), self.assertRaises(episode_exceptions.MalformedActionError):
      agent.step("do something")

  def test_converter_bug_propagates(self):
    agent = self._agent()

    with mock.patch.object(
        agent,
        "_call_llm",
        return_value=(_VALID_CLICK_RESPONSE, "system", "user"),
    ), mock.patch.object(
        ui_voyager,
        "_tool_call_to_json_action",
        side_effect=RuntimeError("converter bug"),
    ), self.assertRaisesRegex(RuntimeError, "converter bug"):
      agent.step("do something")

  def test_actuation_fault_propagates(self):
    env = test_utils.FakeAsyncEnv()
    agent = self._agent(env)

    with mock.patch.object(
        agent,
        "_call_llm",
        return_value=(_VALID_CLICK_RESPONSE, "system", "user"),
    ), mock.patch.object(
        env, "execute_action", side_effect=RuntimeError("adb failed")
    ), self.assertRaisesRegex(RuntimeError, "adb failed"):
      agent.step("do something")


if __name__ == "__main__":
  absltest.main()
