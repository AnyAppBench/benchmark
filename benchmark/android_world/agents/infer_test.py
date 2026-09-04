from unittest import mock

from absl.testing import absltest

from android_world.agents import infer


class GeminiGcpWrapperTest(absltest.TestCase):

  def test_gemini_api_key_alias_is_accepted(self):
    with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "key"}, clear=True):
      wrapper = infer.GeminiGcpWrapper()

    self.assertEqual(wrapper.model_name, infer.DEFAULT_GEMINI_MODEL)
    self.assertEqual(wrapper._base_generation_config["max_output_tokens"], 4096)

  def test_empty_gemini_env_values_are_not_credentials(self):
    with mock.patch.dict(
        "os.environ",
        {"GCP_API_KEY": "", "GEMINI_API_KEY": ""},
        clear=True,
    ):
      with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
        infer.GeminiGcpWrapper()


class ClaudeProxyWrapperTest(absltest.TestCase):

  def test_empty_claude_env_values_are_not_credentials(self):
    with mock.patch.dict(
        "os.environ",
        {"CLAUDE_PROXY_API_KEY": "", "ANTHROPIC_API_KEY": ""},
        clear=True,
    ):
      with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
        infer.ClaudeProxyWrapper()

  def test_nonempty_anthropic_key_allows_direct_api_wrapper(self):
    with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}, clear=True):
      wrapper = infer.ClaudeProxyWrapper()

    self.assertEqual(wrapper.api_key, "key")
    self.assertEqual(wrapper.base_url, "")
    self.assertEqual(wrapper.model, infer.DEFAULT_CLAUDE_MODEL)


if __name__ == "__main__":
  absltest.main()
