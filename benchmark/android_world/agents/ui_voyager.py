"""UI-Voyager (MarsXL/UI-Voyager, 4B) agent for AndroidWorld.

UI-Voyager is a Qwen3-VL-4B-Instruct fine-tune. It speaks the Qwen3-VL
`mobile_use` tool-calling interface with a 999x999 virtual coordinate frame.
This agent talks to an OpenAI-compatible endpoint (e.g. vLLM) serving the
UI-Voyager checkpoint, converts the tool_call into a
`json_action.JSONAction`, rescales coordinates to the device screen, and
executes the action through the AndroidWorld environment.

Serve example:

    vllm serve MarsXL/UI-Voyager \
        --host 0.0.0.0 --port 8000 \
        --served-model-name ui-voyager \
        --limit-mm-per-prompt '{"image": 1}' \
        --max-model-len 196608

If startup fails with KV-cache memory errors, reduce `--max-model-len`
further (for example to 131072).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image

from android_world.agents import base_agent
from android_world.agents import episode_exceptions
from android_world.env import interface
from android_world.env import json_action


MIN_PIXELS = 3136
MAX_PIXELS = 4 * 1024 * 1024  # 4M pixels, within the 4B model's comfort zone.

SYSTEM_PROMPT = (
    Path(__file__).with_name("prompts") / "ui_voyager_qwen3vl_instruct.md"
).read_text(encoding="utf-8")

USER_TEMPLATE = (
    "The user query: {goal}\n"
    "{history_section}"
    "\n"
    "Current Screenshot: <image>\n"
    "\n"
    "Please analyze the current screenshot and history to generate the next step."
)


def _smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
  """Fallback of qwen_vl_utils.smart_resize so we do not hard-require it."""
  try:
    from qwen_vl_utils import smart_resize  # type: ignore
  except ImportError:
    smart_resize = None
  if smart_resize is not None:
    return smart_resize(
        height, width, factor=factor,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )

  def _round_by_factor(n: float, f: int) -> int:
    return max(f, int(round(n / f)) * f)

  h = _round_by_factor(height, factor)
  w = _round_by_factor(width, factor)
  if h * w > max_pixels:
    beta = ((h * w) / max_pixels) ** 0.5
    h = _round_by_factor(height / beta, factor)
    w = _round_by_factor(width / beta, factor)
  if h * w < min_pixels:
    beta = (min_pixels / (h * w)) ** 0.5
    h = _round_by_factor(height * beta, factor)
    w = _round_by_factor(width * beta, factor)
  return h, w


def _pil_to_b64_url(image: Image.Image, fmt: str = "PNG") -> str:
  buf = io.BytesIO()
  image.save(buf, format=fmt)
  return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


def _content_from_image_placeholder(
    text_prompt: str, image_url: str,
) -> list[dict[str, Any]]:
  """Builds OpenAI multimodal content using UI-Voyager's <image> marker."""
  if "<image>" not in text_prompt:
    return [
        {"type": "text", "text": text_prompt},
        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
    ]
  content: list[dict[str, Any]] = []
  parts = text_prompt.split("<image>")
  for idx, part in enumerate(parts):
    if part:
      content.append({"type": "text", "text": part})
    if idx < len(parts) - 1:
      content.append(
          {
              "type": "image_url",
              "image_url": {"url": image_url, "detail": "high"},
          }
      )
  return content


def _extract_tag(tag: str, text: str) -> Optional[str]:
  m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
  return m.group(1).strip() if m else None


def _rescale_999_to_pixels(x: float, y: float, width: int, height: int) -> tuple[int, int]:
  return int(round(x / 999 * width)), int(round(y / 999 * height))


_SYSTEM_BUTTON_MAP = {
    "back": json_action.NAVIGATE_BACK,
    "home": json_action.NAVIGATE_HOME,
    "enter": json_action.KEYBOARD_ENTER,
}


def _tool_call_to_json_action(
    tool_call: dict,
    screen_width: int,
    screen_height: int,
) -> json_action.JSONAction:
  """Converts a Qwen3-VL `mobile_use` tool_call into a JSONAction."""
  args = tool_call.get("arguments", {}) or {}
  action_name = (args.get("action") or "").lower()
  if action_name == "tap":
    action_name = "click"

  def _point(coord):
    x, y = coord
    return _rescale_999_to_pixels(x, y, screen_width, screen_height)

  if action_name in ("click", "long_press"):
    x, y = _point(args["coordinate"])
    return json_action.JSONAction(
        action_type=json_action.CLICK if action_name == "click"
        else json_action.LONG_PRESS,
        x=x, y=y,
    )
  if action_name == "swipe":
    x1, y1 = _point(args["coordinate"])
    x2, y2 = _point(args["coordinate2"])
    return json_action.JSONAction(
        action_type=json_action.SWIPE, x=x1, y=y1, x_=x2, y_=y2,
    )
  if action_name == "type":
    return json_action.JSONAction(
        action_type=json_action.INPUT_TEXT, text=args.get("text", ""),
    )
  if action_name == "answer":
    return json_action.JSONAction(
        action_type=json_action.ANSWER, text=args.get("text", ""),
    )
  if action_name == "open_app":
    return json_action.JSONAction(
        action_type=json_action.OPEN_APP,
        app_name=(args.get("text") or args.get("app_name") or "").lower(),
    )
  if action_name == "wait":
    return json_action.JSONAction(action_type=json_action.WAIT)
  if action_name == "system_button":
    mapped = _SYSTEM_BUTTON_MAP.get((args.get("button") or "").lower())
    if mapped is None:
      return json_action.JSONAction(action_type=json_action.UNKNOWN)
    return json_action.JSONAction(action_type=mapped)
  if action_name == "terminate":
    return json_action.JSONAction(
        action_type=json_action.STATUS,
        goal_status=(
            "complete" if args.get("status", "success") == "success"
            else "infeasible"
        ),
    )
  return json_action.JSONAction(action_type=json_action.UNKNOWN)


def _validate_tool_call(tool_call: Any) -> dict[str, Any]:
  """Validates UI-Voyager's declared ``mobile_use`` action schema.

  This boundary is deliberately separate from action conversion.  A schema
  violation can therefore be attributed to the model, while an unexpected
  converter exception remains an infrastructure/runtime failure.
  """
  if not isinstance(tool_call, dict):
    raise episode_exceptions.MalformedActionError(
        "<tool_call> must decode to a JSON object."
    )
  if tool_call.get("name") != "mobile_use":
    raise episode_exceptions.MalformedActionError(
        "<tool_call> must name the mobile_use function."
    )
  args = tool_call.get("arguments")
  if not isinstance(args, dict):
    raise episode_exceptions.MalformedActionError(
        "mobile_use arguments must be a JSON object."
    )

  action = args.get("action")
  if not isinstance(action, str) or not action.strip():
    raise episode_exceptions.MalformedActionError(
        "mobile_use arguments must include a non-empty action."
    )
  action = action.strip().lower()
  if action == "tap":
    action = "click"
  supported_actions = {
      "click",
      "long_press",
      "swipe",
      "type",
      "answer",
      "open_app",
      "wait",
      "system_button",
      "terminate",
  }
  if action not in supported_actions:
    raise episode_exceptions.MalformedActionError(
        f"Unsupported mobile_use action: {action!r}."
    )

  def _validate_point(field: str) -> None:
    point = args.get(field)
    if not isinstance(point, (list, tuple)) or len(point) != 2:
      raise episode_exceptions.MalformedActionError(
          f"{action} requires {field}=[x, y]."
      )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in point
    ):
      raise episode_exceptions.MalformedActionError(
          f"{field} must contain two numeric coordinates."
      )
    if any(not np.isfinite(value) for value in point):
      raise episode_exceptions.MalformedActionError(
          f"{field} coordinates must be finite."
      )

  if action in {"click", "long_press", "swipe"}:
    _validate_point("coordinate")
  if action == "swipe":
    _validate_point("coordinate2")
  if action in {"type", "answer"} and not isinstance(args.get("text"), str):
    raise episode_exceptions.MalformedActionError(
        f"{action} requires a text string."
    )
  if action == "open_app":
    app_name = args.get("text", args.get("app_name"))
    if not isinstance(app_name, str) or not app_name.strip():
      raise episode_exceptions.MalformedActionError(
          "open_app requires a non-empty text or app_name string."
      )
  if action == "system_button":
    button = args.get("button")
    if not isinstance(button, str) or button.lower() not in {
        "back",
        "home",
        "menu",
        "enter",
    }:
      raise episode_exceptions.MalformedActionError(
          f"Unsupported system_button value: {button!r}."
      )
  if action == "terminate" and args.get("status", "success") not in {
      "success",
      "failure",
  }:
    raise episode_exceptions.MalformedActionError(
        f"Unsupported terminate status: {args.get('status')!r}."
    )
  return tool_call


class UIVoyagerAgent(base_agent.EnvironmentInteractingAgent):
  """UI-Voyager agent that calls an OpenAI-compatible endpoint."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      endpoint_url: str = "http://127.0.0.1:8000/v1",
      model_name: str = "ui-voyager",
      api_key: str = "EMPTY",
      max_new_tokens: int = 512,
      temperature: float = 0.7,
      top_p: float = 0.8,
      request_timeout: int = 300,
      max_retries: int = 3,
      wait_after_action_seconds: float = 1.5,
      history_len: int = 30,
      output_path: str = "",
      name: str = "UI-Voyager",
  ):
    super().__init__(env, name)
    self.endpoint_url = endpoint_url.rstrip("/")
    if not self.endpoint_url.endswith("/v1"):
      self.endpoint_url = f"{self.endpoint_url}/v1"
    self.model_name = model_name
    self.api_key = api_key
    if max_new_tokens <= 0:
      raise ValueError("max_new_tokens must be positive.")
    self.max_new_tokens = max_new_tokens
    self.temperature = temperature
    self.top_p = top_p
    self.request_timeout = request_timeout
    self.max_retries = max_retries
    self.wait_after_action_seconds = wait_after_action_seconds
    self.history_len = history_len
    self.output_path = output_path
    self.history: list[str] = []
    self.raw_responses: list[str] = []

  def reset(self, go_home_on_reset: bool = False) -> None:
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history.clear()
    self.raw_responses.clear()

  def _construct_prompt(self, goal: str) -> str:
    recent_history = (
        self.history[-self.history_len:]
        if len(self.history) > self.history_len
        else self.history
    )
    history_section = ""
    if recent_history:
      start_step = len(self.history) - len(recent_history) + 1
      lines = [
          "Task progress (You have done the following operations on the current device):"
      ]
      for idx, action_text in enumerate(recent_history):
        cleaned = re.sub(r"\s+", " ", action_text).strip().strip("\"'")
        lines.append(f"Step{start_step + idx}: {cleaned}")
      history_section = "\n" + "\n".join(lines) + "\n"
    return USER_TEMPLATE.format(goal=goal, history_section=history_section)

  def _call_llm(
      self, goal: str, screenshot: Image.Image,
  ) -> tuple[str, str, str]:
    """Calls the OpenAI-compatible endpoint.

    Returns:
      (response_text, system_prompt, user_text)

    The two trailing strings are the exact prompt segments sent to the model,
    so the runner can persist them alongside the model's reasoning. The
    image bytes are intentionally NOT returned -- the per-step ``raw_screenshot``
    field already captures them.
    """
    image_url = _pil_to_b64_url(screenshot.convert("RGB"), fmt="JPEG")
    user_text = self._construct_prompt(goal)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _content_from_image_placeholder(user_text, image_url),
        },
    ]

    payload: dict[str, Any] = {
        "model": self.model_name,
        "messages": messages,
        "max_tokens": self.max_new_tokens,
        "temperature": self.temperature,
        "top_p": self.top_p,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}",
    }

    last_err: Optional[Exception] = None
    for attempt in range(self.max_retries):
      try:
        resp = requests.post(
            f"{self.endpoint_url}/chat/completions",
            headers=headers, json=payload, timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
          raise TypeError("Endpoint response content must be a non-empty string.")
        return content, SYSTEM_PROMPT, user_text
      except (
          requests.RequestException,
          json.JSONDecodeError,
          KeyError,
          IndexError,
          TypeError,
      ) as exc:
        last_err = exc
        print(f"[UI-Voyager] endpoint call failed (attempt {attempt + 1}): {exc}")
        if attempt + 1 < self.max_retries:
          time.sleep(2 ** attempt)
    raise episode_exceptions.ModelEndpointError(
        f"UI-Voyager endpoint failed after {self.max_retries} attempts: "
        f"{last_err}"
    ) from last_err


  def _parse_response(
      self, response_text: str,
  ) -> tuple[str, str, dict]:
    """Splits the response into (thought, action_summary, tool_call_json)."""
    thought, action_desc = "", ""
    if "Action:" in response_text:
      head, tail = response_text.split("Action:", 1)
      thought = head.replace("Thought:", "").strip()
      action_desc = tail.split("<tool_call>", 1)[0].strip()
    tool_call_str = _extract_tag("tool_call", response_text)
    if not tool_call_str:
      raise episode_exceptions.ActionParseError(
          "No <tool_call> block in model response."
      )
    try:
      tool_call = json.loads(tool_call_str)
    except json.JSONDecodeError as exc:
      raise episode_exceptions.ActionParseError(
          "The model's <tool_call> block is not valid JSON."
      ) from exc
    return thought, action_desc, _validate_tool_call(tool_call)

  def step(
      self, goal: str, step_numb: bool = False,
  ) -> base_agent.AgentInteractionResult:
    # ``prompt_system`` and ``prompt_user`` are the exact strings the model
    # received this step (image is in ``raw_screenshot``). ``response`` is the
    # raw model output; ``thought`` / ``action_desc`` are parsed reasoning.
    # The reporter ingests these to fulfil the "save what prompt the agent
    # received and what they reasoned" requirement.
    step_data: dict[str, Any] = {
        "raw_screenshot": None,
        "prompt_system": None,
        "prompt_user": None,
        "response": None,
        "thought": None,
        "action_desc": None,
        "tool_call": None,
        "action": None,
    }

    state = self.get_post_transition_state()
    step_data["raw_screenshot"] = state.pixels.copy()
    screenshot = Image.fromarray(state.pixels)
    screen_width, screen_height = screenshot.size

    response_text, system_prompt, user_text = self._call_llm(
        goal, screenshot,
    )

    step_data["prompt_system"] = system_prompt
    step_data["prompt_user"] = user_text
    step_data["response"] = response_text
    self.raw_responses.append(response_text)
    print(f"\n========== UI-Voyager response ==========\n{response_text}\n")

    thought, action_desc, tool_call = self._parse_response(response_text)

    step_data["thought"] = thought
    step_data["action_desc"] = action_desc
    step_data["tool_call"] = tool_call

    action = _tool_call_to_json_action(
        tool_call, screen_width, screen_height,
    )

    step_data["action"] = action
    print(f"[UI-Voyager] action: {action}")

    if action.action_type == json_action.STATUS:
      self.history.append(action_desc or "terminate")
      return base_agent.AgentInteractionResult(True, step_data)

    if action.action_type == json_action.ANSWER:
      self.env.execute_action(action)
      self.history.append(action_desc or str(action.action_type))
      return base_agent.AgentInteractionResult(True, step_data)

    if action.action_type == json_action.UNKNOWN:
      raise RuntimeError(
          "Validated UI-Voyager tool call converted to UNKNOWN action."
      )

    self.env.execute_action(action)

    time.sleep(self.wait_after_action_seconds)
    self.history.append(action_desc or str(action.action_type))

    if self.output_path:
      os.makedirs(self.output_path, exist_ok=True)
      idx = len(self.history)
      screenshot.save(os.path.join(self.output_path, f"step_{idx:03d}.png"))
      with open(
          os.path.join(self.output_path, "trace.jsonl"), "a", encoding="utf-8",
      ) as f:
        # Include the full prompt + raw response so the trace is enough to
        # reconstruct what the agent saw and how it reasoned, without
        # needing the gzip-pickled episode file.
        f.write(json.dumps({
            "step": idx,
            "goal": goal,
            "prompt_system": system_prompt,
            "prompt_user": user_text,
            "response": response_text,
            "thought": thought,
            "action_desc": action_desc,
            "tool_call": tool_call,
        }) + "\n")

    return base_agent.AgentInteractionResult(False, step_data)
