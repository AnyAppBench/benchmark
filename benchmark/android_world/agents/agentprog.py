"""AgentProg adapter for CATBench/AndroidWorld.

This wraps MobileLLM/AgentProg's official AndroidWorld integration. AgentProg
runs its full program-generation and program-execution loop inside one
AndroidWorld step, then AndroidWorld evaluates the final device state.

Adapted from AgentProg/eval/android_world/android_world/agents/agentprog.py.
The wrapper preserves the upstream pipeline call: it builds AgentProgConfig
with the same fields the upstream agent passes, then invokes
agentprog_pipeline_core(...) and forwards any "answer" via JSONAction.

CATBench-specific (evaluation-only) additions:
  * Resolve the AgentProg checkout (`agentprog_root`) so we don't require a
    pip install.
  * Allow CATBench to redirect AgentProg's UI-TARS endpoint via env vars.
  * Place AgentProg's per-task workdir under CATBench's `output_path` instead
    of upstream's hardcoded `agentprog/scripts/agentprog/<EXP_NAME>` path.
  * Capture the workflow / log / screenshot paths in the per-step `step_data`
    payload that CATBench's checkpointer serialises.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
import traceback
from typing import Any

from PIL import Image

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action


def _slug(value: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "task"


def _candidate_agentprog_roots(agentprog_root: str | None) -> list[Path]:
  candidates: list[Path] = []
  if agentprog_root:
    candidates.append(Path(agentprog_root).expanduser())
  if os.environ.get("AGENTPROG_ROOT"):
    candidates.append(Path(os.environ["AGENTPROG_ROOT"]).expanduser())

  here = Path(__file__).resolve()
  if len(here.parents) >= 5:
    candidates.append(here.parents[4] / "AgentProg")
  candidates.append(Path("/tmp/catbench_agentprog_source"))

  deduped: list[Path] = []
  for candidate in candidates:
    if candidate not in deduped:
      deduped.append(candidate)
  return deduped


def _ensure_agentprog_importable(agentprog_root: str | None) -> Path | None:
  if importlib.util.find_spec("agentprog") is not None:
    return None

  for candidate in _candidate_agentprog_roots(agentprog_root):
    if (candidate / "agentprog" / "__init__.py").exists():
      sys.path.insert(0, str(candidate))
      if importlib.util.find_spec("agentprog") is not None:
        return candidate

  searched = ", ".join(str(p) for p in _candidate_agentprog_roots(agentprog_root))
  raise RuntimeError(
      "Could not import the AgentProg package. Install it with `pip install -e "
      "/path/to/AgentProg` or pass --agentprog_root /path/to/AgentProg. "
      f"Searched: {searched}"
  )


def _copy_nonempty_env(name: str, value: str | None) -> None:
  if value:
    os.environ[name] = value


def _patch_ui_tars_endpoint() -> None:
  """Let CATBench AGENTPROG_UI_TARS_* override AgentProg's Ark defaults."""
  from agentprog.all_utils import ui_tars_utils

  if getattr(ui_tars_utils, "_catbench_endpoint_patch", False):
    return

  original_init = ui_tars_utils.init_get_ui_tars_response

  def init_get_ui_tars_response(
      base_url: str | None = None,
      api_key: str | None = None,
      model: str = "doubao-seed-1-8-251228",
      init_response_args=None,
  ):
    override_model = os.environ.get("AGENTPROG_UI_TARS_MODEL")
    override_base_url = os.environ.get("AGENTPROG_UI_TARS_BASE_URL")
    override_api_key = os.environ.get("AGENTPROG_UI_TARS_API_KEY")

    if init_response_args is not None:
      if override_model:
        init_response_args.model = override_model
      if override_base_url:
        init_response_args.base_url = override_base_url
      if override_api_key:
        init_response_args.api_key = override_api_key
    else:
      model = override_model or model
      base_url = override_base_url or base_url
      api_key = override_api_key or api_key

    return original_init(
        base_url=base_url,
        api_key=api_key,
        model=model,
        init_response_args=init_response_args,
    )

  ui_tars_utils.init_get_ui_tars_response = init_get_ui_tars_response
  ui_tars_utils._catbench_endpoint_patch = True


def _patch_workflow_limits(max_retry_time: int | None, max_loop_time: int | None) -> None:
  """Keep AgentProg's internal loop budget practical for benchmark runs."""
  from agentprog.plan import workflow_utils

  if max_retry_time and max_retry_time > 0:
    workflow_utils.MAX_RETRY_TIME = int(max_retry_time)
  if max_loop_time and max_loop_time > 0:
    workflow_utils.MAX_LOOP_TIME = int(max_loop_time)


class _AgentProgPipelineTimeoutError(BaseException):
  """Raised when one AgentProg pipeline call exceeds the benchmark budget."""


class AgentProg(base_agent.EnvironmentInteractingAgent):
  """AndroidWorld agent wrapper around MobileLLM/AgentProg."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      name: str = "agentprog",
      output_path: str = "",
      agentprog_root: str | None = None,
      console_port: int = 5554,
      grpc_port: int = 8554,
      exp_name: str = "catbench",
      tool_set: str = "mobile",
      model: str = "",
      api_key: str = "",
      base_url: str = "",
      ui_tars_model: str = "",
      ui_tars_api_key: str = "",
      ui_tars_base_url: str = "",
      use_belief_state: bool = True,
      use_aw_locator: bool = False,
      cache_mode: bool = False,
      show_dashboard: bool = False,
      fold_dashboard: bool = True,
      transition_pause: float | None = 1.0,
      max_retry_time: int | None = 12,
      max_loop_time: int | None = 12,
      step_timeout_seconds: int | None = 1200,
  ):
    super().__init__(env, name=name, transition_pause=transition_pause)
    loaded_root = _ensure_agentprog_importable(agentprog_root)
    self.agentprog_root = str(loaded_root or agentprog_root or "")

    from agentprog.plan.agentprog_utils import RequestMode, ToolSet
    from agentprog.plan.code_exec.workflow.config.core_config import (
        AgentProgConfig,
    )
    from agentprog.plan.code_exec.workflow.pipeline import (
        agentprog_pipeline_core,
    )

    self._request_mode = RequestMode.api.name
    self._tool_set_cls = ToolSet
    self._config_cls = AgentProgConfig
    self._pipeline = agentprog_pipeline_core
    _patch_ui_tars_endpoint()

    if tool_set not in ToolSet._member_names_:
      raise ValueError(
          f"Unsupported AgentProg tool_set={tool_set!r}. "
          f"Valid values: {ToolSet._member_names_}"
      )

    # Mirror upstream env-var contract: AgentProgConfig.__post_init__ reads
    # MODEL / GEMINI_API_KEY / GEMINI_BASE_URL when building default model
    # args, and the UI-TARS path reads ARK_API_KEY / DOUBAO_BASE_URL.
    _copy_nonempty_env("MODEL", model)
    _copy_nonempty_env("GEMINI_API_KEY", api_key)
    _copy_nonempty_env("GEMINI_BASE_URL", base_url)
    _copy_nonempty_env("ARK_API_KEY", ui_tars_api_key)
    _copy_nonempty_env("DOUBAO_BASE_URL", ui_tars_base_url)
    _copy_nonempty_env("AGENTPROG_UI_TARS_MODEL", ui_tars_model)
    os.environ.setdefault("EXP_NAME", exp_name)
    os.environ.setdefault("TOOL_SET", tool_set)
    os.environ.setdefault("WEBSOCKET_PORT", "6666")

    self.console_port = console_port
    self.grpc_port = grpc_port
    self.exp_name = exp_name
    self.tool_set = tool_set
    self.model = model or os.environ.get("MODEL") or "gemini/gemini-2.5-pro"
    self.api_key = api_key
    self.base_url = base_url
    self.ui_tars_model = ui_tars_model
    self.use_belief_state = use_belief_state
    self.use_aw_locator = use_aw_locator
    self.cache_mode = cache_mode
    self.show_dashboard = show_dashboard
    self.fold_dashboard = fold_dashboard
    self.max_retry_time = max_retry_time
    self.max_loop_time = max_loop_time
    self.step_timeout_seconds = step_timeout_seconds
    self.task_name: dict[str, str] = {}
    self.output_path = Path(output_path).expanduser() if output_path else Path(
        os.path.expanduser("~/catbench_runs/agentprog")
    )
    self.base_path = self.output_path / "agentprog" / self.exp_name

  def _run_pipeline_with_timeout(self, config):
    timeout = int(self.step_timeout_seconds or 0)
    if timeout <= 0 or threading.current_thread() is not threading.main_thread():
      return self._pipeline(config)

    def _handle_timeout(signum, frame):  # pylint: disable=unused-argument
      raise _AgentProgPipelineTimeoutError(
          f"AgentProg pipeline exceeded {timeout} seconds"
      )

    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    effective_timeout = timeout
    if previous_timer[0] > 0:
      effective_timeout = min(timeout, max(1, int(previous_timer[0])))
    signal.setitimer(signal.ITIMER_REAL, effective_timeout)
    signal.signal(signal.SIGALRM, _handle_timeout)
    try:
      return self._pipeline(config)
    finally:
      signal.setitimer(signal.ITIMER_REAL, 0)
      signal.signal(signal.SIGALRM, previous_handler)
      if previous_timer[0] > 0:
        remaining = max(0, previous_timer[0] - (time.monotonic() - started))
        if remaining > 0:
          signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])

  def get_task_name(self, suite) -> None:
    for name, instances in suite.items():
      if instances:
        self.task_name[instances[0].goal] = name

  def _task_dir(self, goal: str) -> Path:
    task_id = self.task_name.get(goal, _slug(goal)[:80])
    task_dir = self.base_path / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir

  def _list_screenshots(self, image_dir: Path) -> list[str]:
    if not image_dir.exists():
      return []
    return [str(path) for path in sorted(image_dir.glob("*.png"))]

  def step(
      self, goal: str, step_numb: bool = False
  ) -> base_agent.AgentInteractionResult:
    # `step_numb` is an unused CATBench base_agent compatibility parameter.
    del step_numb
    task_dir = self._task_dir(goal)
    image_dir = task_dir / "images"
    meta_info_dir = task_dir / "meta_info"
    workflow_path = task_dir / f"{task_dir.name}.ap"
    logging_path = meta_info_dir / "log.txt"
    tensorboard_log_dir = meta_info_dir / "tensorboard"

    step_data: dict[str, Any] = {
        "raw_screenshot": None,
        "prompt_user": goal,
        "response": None,
        "thought": None,
        "action_desc": "AgentProg full-pipeline execution",
        "action": None,
        "agentprog_task_dir": str(task_dir),
        "agentprog_workflow_path": str(workflow_path),
        "agentprog_image_dir": str(image_dir),
        "agentprog_meta_info_dir": str(meta_info_dir),
        "agentprog_log_path": str(logging_path),
        "agentprog_screenshots": [],
    }

    try:
      state = self.env.get_state(wait_to_stabilize=False)
      step_data["raw_screenshot"] = state.pixels.copy()
      Image.fromarray(state.pixels).save(task_dir / "androidworld_start.png")
    except Exception as exc:  # pylint: disable=broad-except
      step_data["screenshot_error"] = str(exc)

    try:
      # Match upstream agentprog.AgentProg.step: build AgentProgConfig with
      # the same set of fields and let __post_init__ build the workflow /
      # executor model args from MODEL / GEMINI_API_KEY / GEMINI_BASE_URL.
      _patch_workflow_limits(self.max_retry_time, self.max_loop_time)
      config = self._config_cls(
          task_description=goal,
          workflow_path=str(workflow_path),
          tool_set=self.tool_set,
          request_mode=self._request_mode,
          image_dir=str(image_dir),
          meta_info_dir=str(meta_info_dir),
          serial=f"emulator-{self.console_port}",
          serial_port=str(self.console_port),
          cache_mode=self.cache_mode,
          use_belief_state=self.use_belief_state,
          use_aw_locator=self.use_aw_locator,
          tensorboard_log_dir=str(tensorboard_log_dir),
          logging_path=str(logging_path),
          show_dashboard=self.show_dashboard,
          fold_dashboard=self.fold_dashboard,
      )
      workflow_result = self._run_pipeline_with_timeout(config)
      answer = getattr(workflow_result, "global_variables", {}).get(
          "answer", None
      )
      if answer is not None:
        self.env.execute_action(
            json_action.JSONAction(
                action_type=json_action.ANSWER, text=str(answer)
            )
        )
      step_data.update({
          "response": "AgentProg pipeline completed.",
          "thought": "See AgentProg log/workflow files for generated program and execution reasoning.",
          "action": "agentprog_pipeline",
          "agentprog_answer": answer,
          "agentprog_workflow": (
              workflow_path.read_text(encoding="utf-8")
              if workflow_path.exists() else ""
          ),
          "agentprog_global_variables": json.dumps(
              getattr(workflow_result, "global_variables", {}),
              ensure_ascii=False,
              default=str,
          ),
          "agentprog_screenshots": self._list_screenshots(image_dir),
      })
      return base_agent.AgentInteractionResult(True, step_data)
    except Exception as exc:  # pylint: disable=broad-except
      step_data["response"] = str(exc)
      step_data["traceback"] = traceback.format_exc()
      step_data["agentprog_screenshots"] = self._list_screenshots(image_dir)
      traceback.print_exc()
      return base_agent.AgentInteractionResult(True, step_data)
