from unittest import mock

from absl.testing import absltest

from android_world.agents import episode_exceptions
from android_world.agents import openai_python_action
from android_world.env import actuation
from android_world.env import representation_utils
from android_world.utils import test_utils


class OpenAIPythonActionParserTest(absltest.TestCase):

  def test_incomplete_do_token_is_not_silently_converted_to_wait(self):
    with self.assertRaises(ValueError):
      openai_python_action._parse_function_call("do")

  def test_do_without_action_is_explicitly_malformed(self):
    parsed = openai_python_action._parse_function_call("do()")

    self.assertEqual(parsed.action, "")
    self.assertEqual(parsed.params, {})

  def test_step_rejects_incomplete_do_token(self):
    env = test_utils.FakeAsyncEnv()
    agent = openai_python_action.OpenAIPythonActionAgent(
        env,
        endpoint_url="http://127.0.0.1:8000/v1",
        model_name="fake-model",
        wait_after_action_seconds=0,
    )

    agent.transition_pause = 0
    with mock.patch.object(
        agent,
        "_call_model",
        return_value=("do", "", "task", 1.0, 1.0, (100, 100)),
    ), self.assertRaises(episode_exceptions.ActionParseError):
      agent.step("do something")

  def test_parse_response_extracts_embedded_complete_call(self):
    agent = openai_python_action.OpenAIPythonActionAgent(
        test_utils.FakeAsyncEnv(),
        endpoint_url="http://127.0.0.1:8000/v1",
        model_name="fake-model",
    )

    _, action_text, _, parsed = agent._parse_response(
        'I should open it. do(action="Launch", app="Messages") '
        'Then maybe tap later.'
    )

    self.assertEqual(action_text, 'do(action="Launch", app="Messages")')
    self.assertEqual(parsed.action, "Launch")
    self.assertEqual(parsed.params["app"], "Messages")

  def test_parse_response_truncated_call_raises_declared_parse_error(self):
    agent = openai_python_action.OpenAIPythonActionAgent(
        test_utils.FakeAsyncEnv(),
        endpoint_url="http://127.0.0.1:8000/v1",
        model_name="fake-model",
    )

    with self.assertRaises(episode_exceptions.ActionParseError):
      agent._parse_response('do(action="Tap')

  def test_parse_response_skips_malformed_tap_element(self):
    agent = openai_python_action.OpenAIPythonActionAgent(
        test_utils.FakeAsyncEnv(),
        endpoint_url="http://127.0.0.1:8000/v1",
        model_name="fake-model",
    )

    _, action_text, _, parsed = agent._parse_response(
        'do(action="Tap", element=[769-453-2360])\n'
        'do(action="Tap", element=[100,200,150,250])'
    )

    self.assertEqual(action_text, 'do(action="Tap", element=[100,200,150,250])')
    self.assertEqual(parsed.action, "Tap")
    self.assertEqual(parsed.params["element"], [100, 200, 150, 250])

  def test_parse_response_malformed_tap_element_is_declared_malformed(self):
    agent = openai_python_action.OpenAIPythonActionAgent(
        test_utils.FakeAsyncEnv(),
        endpoint_url="http://127.0.0.1:8000/v1",
        model_name="fake-model",
    )

    with self.assertRaises(episode_exceptions.MalformedActionError):
      agent._parse_response('do(action="Tap", element=[769-453-2360])')

  def test_endpoint_failure_raises_model_endpoint_error(self):
    agent = openai_python_action.OpenAIPythonActionAgent(
        test_utils.FakeAsyncEnv(),
        endpoint_url="http://127.0.0.1:8000/v1",
        model_name="fake-model",
        wait_after_action_seconds=0,
    )
    agent.transition_pause = 0

    with mock.patch.object(
        openai_python_action.requests,
        "post",
        side_effect=openai_python_action.requests.ConnectionError("offline"),
    ) as post_mock, mock.patch.object(openai_python_action.time, "sleep"):
      with self.assertRaises(episode_exceptions.ModelEndpointError):
        agent.step("do something")

    self.assertEqual(post_mock.call_count, 3)

  def test_converter_bug_propagates(self):
    agent = openai_python_action.OpenAIPythonActionAgent(
        test_utils.FakeAsyncEnv(),
        endpoint_url="http://127.0.0.1:8000/v1",
        model_name="fake-model",
        wait_after_action_seconds=0,
    )
    agent.transition_pause = 0

    with mock.patch.object(
        agent,
        "_call_model",
        return_value=("Wait()", "", "task", 1.0, 1.0, (100, 100)),
    ), mock.patch.object(
        agent, "_json_action_for", side_effect=RuntimeError("converter bug")
    ), self.assertRaisesRegex(RuntimeError, "converter bug"):
      agent.step("do something")

  def test_actuation_fault_propagates(self):
    agent = openai_python_action.OpenAIPythonActionAgent(
        test_utils.FakeAsyncEnv(),
        endpoint_url="http://127.0.0.1:8000/v1",
        model_name="fake-model",
        wait_after_action_seconds=0,
    )
    agent.transition_pause = 0

    with mock.patch.object(
        agent,
        "_call_model",
        return_value=("Wait()", "", "task", 1.0, 1.0, (100, 100)),
    ), mock.patch.object(
        actuation,
        "execute_adb_action",
        side_effect=RuntimeError("adb failed"),
    ), self.assertRaisesRegex(RuntimeError, "adb failed"):
      agent.step("do something")

  def test_mobilerl_generic_file_manager_launch_uses_goal_app(self):
    parsed = openai_python_action.ParsedAction(
        "Launch", {"app": "file manager"}, 'do(action="Launch", app="file manager")'
    )

    normalized = openai_python_action._normalize_mobilerl_launch_action(
        parsed,
        "Using the Amaze File Manager app, create a new folder in CATBench.",
    )

    self.assertEqual(normalized.params["app"], "Amaze File Manager")

  def test_mobilerl_generic_messages_launch_uses_goal_app(self):
    parsed = openai_python_action.ParsedAction(
        "Launch", {"app": "messages"}, 'do(action="Launch", app="messages")'
    )

    normalized = openai_python_action._normalize_mobilerl_launch_action(
        parsed,
        "Using the Fossify Messages app, send an SMS.",
    )

    self.assertEqual(normalized.params["app"], "Fossify Messages")

  def test_mobilerl_generic_clock_launch_prefers_specific_goal_app(self):
    parsed = openai_python_action.ParsedAction(
        "Launch", {"app": "Clock"}, 'do(action="Launch", app="Clock")'
    )

    normalized = openai_python_action._normalize_mobilerl_launch_action(
        parsed,
        "In the Google Clock app, add Sydney to your world clocks.",
    )

    self.assertEqual(normalized.params["app"], "Google Clock")

  def test_mobilerl_compact_clock_launch_uses_spaced_goal_app(self):
    parsed = openai_python_action.ParsedAction(
        "Launch", {"app": "ClockYou"}, 'do(action="Launch", app="ClockYou")'
    )

    normalized = openai_python_action._normalize_mobilerl_launch_action(
        parsed,
        "In the Clock You app, add London to your world clocks.",
    )

    self.assertEqual(normalized.params["app"], "Clock You")

  def test_mobilerl_relative_boxes_scale_from_0_999_screen_range(self):
    parsed = openai_python_action.ParsedAction(
        "Tap", {"element": [628, 54, 719, 144]}, ""
    )

    scaled = openai_python_action._scale_parsed_action(
        parsed, 1080 / 999, 1920 / 999, relative_coord_base=999
    )

    self.assertEqual(scaled.params["element"], [679, 104, 777, 277])

  def test_mobilerl_absolute_boxes_are_not_scaled_again(self):
    parsed = openai_python_action.ParsedAction(
        "Tap", {"element": [1431, 123, 1638, 328]}, ""
    )

    scaled = openai_python_action._scale_parsed_action(
        parsed, 1080 / 999, 1920 / 999, relative_coord_base=999
    )

    self.assertEqual(scaled.params["element"], [1431, 123, 1638, 328])

  def test_mobilerl_ui_context_formats_relative_bounds(self):
    element = representation_utils.UIElement(
        text="Allow",
        class_name="android.widget.Button",
        bbox_pixels=representation_utils.BoundingBox(100, 300, 200, 500),
        is_clickable=True,
        is_enabled=True,
        is_visible=True,
    )

    context = openai_python_action._format_mobilerl_ui_context(
        [element], (1000, 2000)
    )

    self.assertIn("The screenshot's size is 999x999", context)
    self.assertIn('text="Allow"', context)
    self.assertIn("class=Button", context)
    self.assertIn("bounds=[100,100,300,250]", context)
    self.assertIn("clickable=true", context)

  def test_mobilerl_messages_include_ui_context_but_store_compact_history(self):
    agent = openai_python_action.OpenAIPythonActionAgent(
        test_utils.FakeAsyncEnv(),
        endpoint_url="http://127.0.0.1:8000/v1",
        model_name="fake-model",
        prompt_style="mobilerl_point_think",
    )

    messages, system_prompt, prompt_user = agent._messages(
        "Create a contact",
        "data:image/png;base64,abc",
        "The tree structure description of the current screenshot is shown:",
    )

    self.assertIn("choose one element from the current state", system_prompt)
    self.assertIn("Never put text, phone numbers", system_prompt)
    self.assertIn("Create a contact", prompt_user)
    self.assertIn("tree structure description", prompt_user)
    self.assertLen(
        [
            part
            for message in messages
            for part in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if part.get("type") == "image_url"
        ],
        1,
    )
    self.assertEqual(
        agent._mobilerl_current_history_user["content"],
        "Create a contact",
    )
    self.assertIn(
        "data:image/png;base64,abc",
        agent._mobilerl_current_image_user["content"][0]["image_url"]["url"],
    )

  def test_mobilerl_picture_round_two_replays_previous_image(self):
    agent = openai_python_action.OpenAIPythonActionAgent(
        test_utils.FakeAsyncEnv(),
        endpoint_url="http://127.0.0.1:8000/v1",
        model_name="fake-model",
        prompt_style="mobilerl_point_think",
    )

    agent._messages("Create a contact", "data:image/png;base64,first", "ctx")
    agent._mobilerl_history.append(agent._mobilerl_current_history_user)
    agent._mobilerl_history.append({
        "role": "assistant",
        "content": (
            "<think>\nopen contacts\n</think>\n"
            "<answer>\ndo(action=\"Launch\", app=\"Contacts\")\n</answer>"
        ),
    })
    agent._mobilerl_last_image_user = agent._mobilerl_current_image_user
    agent.history.append('do(action="Launch", app="Contacts")')

    messages, _, _ = agent._messages(
        "Create a contact", "data:image/png;base64,second", "ctx"
    )

    image_urls = [
        part["image_url"]["url"]
        for message in messages
        for part in (
            message["content"] if isinstance(message["content"], list) else []
        )
        if part.get("type") == "image_url"
    ]
    self.assertEqual(
        image_urls,
        ["data:image/png;base64,first", "data:image/png;base64,second"],
    )


if __name__ == "__main__":
  absltest.main()
