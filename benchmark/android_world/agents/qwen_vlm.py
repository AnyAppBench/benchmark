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

"""Local Qwen-compatible fallback used by CATBench runners.

This module provides a lightweight compatibility layer so scripts expecting
`android_world.agents.qwen_vlm.QwenVlmAgent` continue to work in repos that do
not ship the historical implementation.

The implementation routes through MobileAgentV3 with an OpenAI-compatible
endpoint client (for example vLLM).
"""

from __future__ import annotations

import os

from android_world.agents import infer_ma3
from android_world.agents import mobile_agent_v3
from android_world.env import interface


DEFAULT_MODEL_ID = "xuyifan/MobileRL-9B"


def _normalize_openai_base_url(endpoint_url: str) -> str:
  """Normalizes endpoint URLs for OpenAI-compatible clients."""
  base = endpoint_url.rstrip("/")
  if base.endswith("/v1"):
    return base
  return f"{base}/v1"


def _resolve_api_key() -> str:
  # Local vLLM setups typically accept a placeholder key.
  return (
      os.environ.get("VLLM_API_KEY")
      or os.environ.get("OPENAI_API_KEY")
      or "EMPTY"
  )


class QwenVlmAgent(mobile_agent_v3.MobileAgentV3_M3A):
  """Compatibility agent for legacy `qwen_vlm` entry points."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      model_id: str = DEFAULT_MODEL_ID,
      device: str = "cuda:0",
      save_failed_tasks: bool = True,
      mode: str = "local",
      endpoint_url: str = "http://127.0.0.1:8000",
      debug_verbose: bool = False,
      debug_history: bool = False,
      allow_infeasible: bool = False,
  ):
    # `device` is intentionally not used here: GPU assignment is handled
    # upstream in run_uitars._launch_local_vllm() via CUDA_VISIBLE_DEVICES
    # before the vLLM server subprocess is started.  Accepting the parameter
    # keeps this constructor signature compatible with run_uitars.py which
    # passes it through.
    #
    # `debug_verbose`, `debug_history`, and `allow_infeasible` are accepted
    # for forward-compatibility but are not yet wired into MobileAgentV3.
    del device, debug_verbose, debug_history, allow_infeasible

    # Both 'local' and 'endpoint' modes reach the model through an
    # OpenAI-compatible HTTP endpoint (vLLM).  In 'local' mode the caller
    # (run_uitars._main) is responsible for launching the vLLM subprocess
    # before constructing this agent.
    if mode not in {"local", "endpoint"}:
      raise ValueError(f"Unsupported mode={mode!r}. Expected 'local' or 'endpoint'.")

    llm = infer_ma3.Qwen3VLWrapper(
        api_key=_resolve_api_key(),
        base_url=_normalize_openai_base_url(endpoint_url),
        model_name=model_id,
    )

    failed_tasks_dir = ""
    if save_failed_tasks:
      runs_root = os.path.expanduser(
          os.environ.get("CATBENCH_RUNS_DIR", "~/catbench_runs")
      )
      failed_tasks_dir = os.path.join(runs_root, "qwen_vlm_failed_tasks")
      os.makedirs(failed_tasks_dir, exist_ok=True)

    super().__init__(
        env,
        vllm=llm,
        name="qwen_vlm",
        output_path=failed_tasks_dir,
    )

    # suite_utils expects this attribute on the agent instance.
    self.max_steps = None