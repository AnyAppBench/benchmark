"""Failure-attribution tests for primary OpenAI-compatible wrappers."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from android_world.agents import episode_exceptions
from android_world.agents import infer_ma3


def _failing_bot(error: Exception):
  return SimpleNamespace(
      chat=SimpleNamespace(
          completions=SimpleNamespace(create=mock.Mock(side_effect=error))
      )
  )


class InferMA3FailureTest(unittest.TestCase):

  def _openai_wrapper(self, wrapper_type):
    wrapper = object.__new__(wrapper_type)
    wrapper.max_retry = 1
    wrapper.temperature = 0.0
    wrapper.model = "pinned-model"
    wrapper.bot = _failing_bot(RuntimeError("endpoint unavailable"))
    return wrapper

  @mock.patch.object(infer_ma3.time, "sleep", autospec=True)
  def test_gui_owl_endpoint_exhaustion_propagates(self, unused_sleep):
    wrapper = self._openai_wrapper(infer_ma3.GUIOwlWrapper)

    with self.assertRaises(episode_exceptions.ModelEndpointError):
      wrapper.predict_mm("goal", [np.zeros((8, 8, 3), dtype=np.uint8)])

  @mock.patch.object(infer_ma3.time, "sleep", autospec=True)
  def test_qwen_endpoint_exhaustion_propagates(self, unused_sleep):
    wrapper = self._openai_wrapper(infer_ma3.Qwen3VLWrapper)

    with self.assertRaises(episode_exceptions.ModelEndpointError):
      wrapper.predict_mm("goal", [np.zeros((8, 8, 3), dtype=np.uint8)])

  @mock.patch.object(infer_ma3.time, "sleep", autospec=True)
  @mock.patch.object(infer_ma3.requests, "post", autospec=True)
  def test_qwen_predict_endpoint_exhaustion_propagates(
      self, post, unused_sleep
  ):
    post.side_effect = RuntimeError("endpoint unavailable")
    wrapper = object.__new__(infer_ma3.Qwen3VLPredictWrapper)
    wrapper.max_retry = 1
    wrapper.endpoint_url = "http://127.0.0.1:1/predict"
    wrapper.timeout = 1
    wrapper.max_tokens = 1

    with self.assertRaises(episode_exceptions.ModelEndpointError):
      wrapper.predict_mm("goal", [np.zeros((8, 8, 3), dtype=np.uint8)])


if __name__ == "__main__":
  unittest.main()
