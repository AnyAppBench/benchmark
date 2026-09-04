"""Hybrid planner + UI-Venus grounding agent for AndroidWorld.

Architecture:
    * Planner (closed-source, multimodal): GPT-5.1 via OpenAI chat API, or
      Gemini 3 Pro via GCP. Takes the current screenshot + action history
      and outputs one of the `m3a`-style JSON actions. For click /
      long_press / input_text / scroll the action carries an *element
      description* instead of a numeric index.
    * Grounder: inclusionAI/UI-Venus-Ground-72B served behind an
      OpenAI-compatible endpoint. Given a description + screenshot it
      returns `[x1, y1, x2, y2]` pixel coordinates. The agent uses the
      center point to actuate.

Serve UI-Venus (example, two-node tensor-parallel vLLM):

    vllm serve inclusionAI/UI-Venus-Ground-72B \\
        --host 0.0.0.0 --port 8001 \\
        --served-model-name ui-venus-gd \\
        --tensor-parallel-size 4 \\
        --limit-mm-per-prompt '{"image": 1}'

The planner prompt is kept compatible with the rest of the codebase and
mirrors the m3a action schema, with `element` replacing `index`.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import traceback
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image
from openai import OpenAI

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import m3a_utils
from android_world.env import actuation
from android_world.env import interface
from android_world.env import json_action

# `infer` pulls in `google.genai` at import time, which is an optional dep.
# Only require it lazily so this module can be imported for the grounder
# alone (e.g. during a standalone grounding sanity check).
try:
  from android_world.agents import infer  # type: ignore
except Exception:  # pylint: disable=broad-exception-caught
  infer = None  # type: ignore


PROMPT_PREFIX = (
    "You are an agent who can operate an Android phone on behalf of a user."
    " Based on user's goal/request, you may\n"
    "- Answer back if the request/goal is a question (or a chat message).\n"
    "- Complete some tasks described in the requests/goals by performing"
    " actions (step by step) on the phone.\n\n"
    "When given a user request, you will try to complete it step by step."
    " At each step, you will be given the current screenshot and a history"
    " of what you have done (in text). Based on these pieces of information"
    " and the goal, you must choose to perform one of the actions in the"
    " following list (action description followed by the JSON format) by"
    " outputting the action in the correct JSON format.\n"
    "- If you think the task has been completed, finish the task by using"
    " the status action with complete as goal_status:"
    ' `{{"action_type": "status", "goal_status": "complete"}}`\n'
    "- If you think the task is not feasible (including cases like you"
    " don't have enough information or cannot perform some necessary"
    " actions), finish by using the `status` action with infeasible as"
    ' goal_status: `{{"action_type": "status", "goal_status":'
    ' "infeasible"}}`\n'
    '- Answer user\'s question: `{{"action_type": "answer", "text":'
    ' "<answer_text>"}}`\n'
    "- Click/tap on an element. Provide a concise natural-language"
    " description that uniquely identifies the target on the current"
    ' screenshot: `{{"action_type": "click", "element": "<description>"}}`\n'
    "- Long press on an element (same description rule as click):"
    ' `{{"action_type": "long_press", "element": "<description>"}}`\n'
    "- Type text into a text field (includes clicking the field, clearing"
    " it if needed, typing, and pressing enter). Provide the description"
    ' of the target text field: `{{"action_type": "input_text", "text":'
    ' "<text_input>", "element": "<description>"}}`\n'
    '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
    '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
    '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
    "- Scroll the whole screen in one of the four directions (direction is"
    " the direction to move the content, opposite of swipe):"
    ' `{{"action_type": "scroll", "direction": "<up|down|left|right>"}}`\n'
    "- Open an app (always prefer this over launching from the app"
    ' drawer): `{{"action_type": "open_app", "app_name": "<name>"}}`\n'
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
)


GUIDANCE = (
    "Here are some useful guidelines you must follow:\n"
    "General:\n"
    "- Carefully examine the current screenshot before choosing an"
    " action. The summarized history may not be fully reliable.\n"
    "- Usually there are multiple ways to complete a task; pick the"
    " easiest one. If something does not work, a simple retry may help,"
    " but avoid looping on the same failing action.\n"
    "- If the desired state is already achieved, just use the `status`"
    " action with `complete`.\n\n"
    "Action Related:\n"
    "- ALWAYS use the `open_app` action to open an app when possible,"
    " rather than hunting the app drawer.\n"
    "- Use `input_text` to type text, not one key at a time. Clear any"
    " default text first if it would interfere.\n"
    "- For `click`, `long_press`, and `input_text`, the `element` field"
    " must describe something actually visible on the current"
    " screenshot. Make the description specific enough to be unique —"
    " use visible labels, position hints (e.g. 'top-right icon'), and"
    " role where helpful. Do not repeat the same description twice if it"
    " didn't locate the target last time; try a different wording.\n"
    "- `scroll` direction is opposite to swipe: to see content below,"
    " scroll `down`.\n"
)


ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + "\nThe current user goal/request is: {goal}\n\n"
    "Here is a history of what you have done so far:\n{history}\n\n"
    "The current screenshot is also given to you.\n"
    + GUIDANCE
    + "{additional_guidelines}"
    + "\nNow output an action from the above list in the correct JSON"
    " format, following the reason why you do that. Your answer should"
    " look like:\nReason: ...\nAction: {{\"action_type\": ...}}\n\n"
    "Your Answer:\n"
)


SUMMARY_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + "\nThe (overall) user goal/request is: {goal}\n"
    "Now I want you to summarize the latest step.\n"
    "You will be given the screenshot before the action (labelled"
    " 'before') and after the action (labelled 'after'; a red dot marks"
    " the location the grounding model resolved if the action had a"
    " target element).\n\n"
    "This is the action you picked: {action}\n"
    "Based on the reason: {reason}\n\n"
    "By comparing the two screenshots and the action performed, give a"
    " brief (<100 words, single line) summary of this step. Be honest"
    " about whether the red dot actually points at the intended target"
    " and whether the action produced the expected effect, so future"
    " steps can correct course.\n\nSummary of this step: "
)


def _action_selection_prompt(
    goal: str,
    history: list[str],
    additional_guidelines: list[str] | None = None,
) -> str:
  history_str = "\n".join(history) if history else (
      "You just started, no action has been performed yet."
  )
  extra = ""
  if additional_guidelines:
    extra = "For The Current Task:\n"
    for g in additional_guidelines:
      extra += f"- {g}\n"
  return ACTION_SELECTION_PROMPT_TEMPLATE.format(
      goal=goal, history=history_str, additional_guidelines=extra,
  )


def _summary_prompt(action: str, reason: str, goal: str) -> str:
  return SUMMARY_PROMPT_TEMPLATE.format(
      goal=goal, action=action, reason=reason,
  )


# ---- UI-Venus grounding client --------------------------------------------

UI_VENUS_PROMPT = (
    "Outline the position corresponding to the instruction: {desc}."
    " The output should be only [x1,y1,x2,y2]."
)

_BBOX_RE = re.compile(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*"
                      r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]")
_GOAL_APP_RE = re.compile(r"\bIn the (?P<app>[^,.]+?) app\b", re.IGNORECASE)
_GENERIC_CLOCK_APP_NAMES = {"clock", "clock app"}


def _target_app_from_goal(goal: str) -> str:
  match = _GOAL_APP_RE.search(goal)
  return match.group("app").strip() if match else ""


def _normalize_open_app_name(app_name: str, goal: str) -> str:
  normalized = app_name.strip().lower()
  target_app = _target_app_from_goal(goal)
  if normalized in _GENERIC_CLOCK_APP_NAMES and target_app.lower().endswith("clock"):
    return target_app.lower()
  return normalized


def _grounder_headers_from_env() -> dict[str, str]:
  """Reads optional HTTP headers for private UI-Venus grounder endpoints."""
  headers: dict[str, str] = {}
  authorization = os.environ.get("UI_VENUS_GROUNDER_AUTHORIZATION")
  if authorization:
    headers["Authorization"] = authorization
  if os.environ.get("UI_VENUS_GROUNDER_NGROK_SKIP_WARNING"):
    headers["ngrok-skip-browser-warning"] = (
        os.environ["UI_VENUS_GROUNDER_NGROK_SKIP_WARNING"]
    )
  raw_json = os.environ.get("UI_VENUS_GROUNDER_HEADERS_JSON")
  if raw_json:
    try:
      parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
      raise ValueError("UI_VENUS_GROUNDER_HEADERS_JSON must be valid JSON.") from exc
    if not isinstance(parsed, dict):
      raise ValueError("UI_VENUS_GROUNDER_HEADERS_JSON must be a JSON object.")
    headers.update({str(key): str(value) for key, value in parsed.items()})
  return headers


def _normalize_predict_endpoint(endpoint_url: str) -> str:
  endpoint_url = endpoint_url.rstrip("/")
  if endpoint_url.endswith("/health"):
    endpoint_url = endpoint_url[: -len("/health")]
  if endpoint_url.endswith("/predict"):
    return endpoint_url
  return f"{endpoint_url}/predict"


def _format_predict_response(data: Any) -> str:
  if isinstance(data, str):
    return data
  if not isinstance(data, dict):
    return json.dumps(data, ensure_ascii=False)
  choices = data.get("choices")
  if isinstance(choices, list) and choices:
    choice = choices[0]
    if isinstance(choice, dict):
      message = choice.get("message")
      if isinstance(message, dict) and message.get("content"):
        return str(message["content"])
      if choice.get("text"):
        return str(choice["text"])
  for key in ("raw_response", "response", "output", "text", "generated_text"):
    value = str(data.get(key) or "").strip()
    if value:
      return value
  parts = []
  for tag in ("think", "action", "conclusion"):
    value = str(data.get(tag) or "").strip()
    if value:
      parts.append(f"<{tag}>{value}</{tag}>")
  return "\n".join(parts) or json.dumps(data, ensure_ascii=False)


class UIVenusGrounder:
  """Client for UI-Venus-Ground-72B served via OpenAI or /predict endpoint."""

  def __init__(
      self,
      base_url: str = "http://127.0.0.1:8001/v1",
      model_name: str = "ui-venus-gd",
      api_key: str = "EMPTY",
      endpoint_format: str = "openai",
      min_pixels: int = 2_000_000,
      max_pixels: int = 4_800_000,
      request_timeout: int = 300,
      default_headers: dict[str, str] | None = None,
  ):
    self.endpoint_format = endpoint_format
    self.model_name = model_name
    self.min_pixels = min_pixels
    self.max_pixels = max_pixels
    self.request_timeout = request_timeout
    self.default_headers = dict(default_headers or {})
    self.default_headers.update(_grounder_headers_from_env())
    if endpoint_format == "predict":
      self.predict_url = _normalize_predict_endpoint(base_url)
      self.client = None
      return
    if endpoint_format != "openai":
      raise ValueError(
          "Unsupported UI-Venus grounder endpoint_format: "
          f"{endpoint_format}"
      )
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
      base_url = f"{base_url}/v1"
    self.client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=self.default_headers or None,
    )

  def _resize_for_venus(self, img: Image.Image) -> Image.Image:
    w, h = img.size
    pixels = w * h
    if pixels <= self.max_pixels and pixels >= self.min_pixels:
      return img
    if pixels > self.max_pixels:
      scale = (self.max_pixels / pixels) ** 0.5
    else:
      scale = (self.min_pixels / pixels) ** 0.5
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size)

  def locate(
      self, image: np.ndarray, description: str,
  ) -> Optional[tuple[int, int]]:
    """Returns (x, y) pixel coords on the *original* image, or None."""
    pil_img = Image.fromarray(image)
    if pil_img.mode == "RGBA":
      pil_img = pil_img.convert("RGB")
    orig_w, orig_h = pil_img.size

    sent_img = self._resize_for_venus(pil_img)
    sent_w, sent_h = sent_img.size

    buf = io.BytesIO()
    sent_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_url = f"data:image/png;base64,{b64}"
    prompt = UI_VENUS_PROMPT.format(desc=description)

    if self.endpoint_format == "predict":
      payload = {
          "image": data_url,
          "instruction": prompt,
          "user_task": prompt,
          "prompt": prompt,
          "max_tokens": 128,
          "max_new_tokens": 128,
      }
      try:
        response = requests.post(
            self.predict_url,
            headers={
                "Content-Type": "application/json",
                **self.default_headers,
            },
            json=payload,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        raw = _format_predict_response(response.json()).strip()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[UI-Venus] /predict grounding call failed: {exc}")
        return None
      return self._parse_point(raw, orig_w, orig_h, sent_w, sent_h)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ],
    }]

    try:
      resp = self.client.chat.completions.create(
          model=self.model_name, messages=messages, temperature=0,
          max_tokens=128,
      )
      raw = resp.choices[0].message.content.strip()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"[UI-Venus] grounding call failed: {exc}")
      return None

    return self._parse_point(raw, orig_w, orig_h, sent_w, sent_h)

  def _parse_point(
      self,
      raw: str,
      orig_w: int,
      orig_h: int,
      sent_w: int,
      sent_h: int,
  ) -> Optional[tuple[int, int]]:
    match = _BBOX_RE.search(raw)
    if not match:
      print(f"[UI-Venus] could not parse bbox from: {raw!r}")
      return None
    x1, y1, x2, y2 = (float(match.group(i)) for i in range(1, 5))
    cx_sent = (x1 + x2) / 2.0
    cy_sent = (y1 + y2) / 2.0
    cx = int(round(cx_sent * orig_w / sent_w))
    cy = int(round(cy_sent * orig_h / sent_h))
    return cx, cy


# ---- Agent ----------------------------------------------------------------


class M3AVenusAgent(base_agent.EnvironmentInteractingAgent):
  """M3A-style planner + UI-Venus grounding agent."""

  # Actions that require resolving an element description to (x, y).
  _ELEMENT_ACTIONS = (
      json_action.CLICK,
      json_action.LONG_PRESS,
      json_action.INPUT_TEXT,
  )

  def __init__(
      self,
      env: interface.AsyncEnv,
      planner_llm: "infer.MultimodalLlmWrapper",
      grounder: UIVenusGrounder,
      name: str = "M3AVenus",
      wait_after_action_seconds: float = 2.0,
      additional_guidelines: list[str] | None = None,
  ):
    super().__init__(env, name)
    self.llm = planner_llm
    self.grounder = grounder
    self.history: list[dict[str, Any]] = []
    self.additional_guidelines = additional_guidelines
    self.wait_after_action_seconds = wait_after_action_seconds

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False) -> None:
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history = []

  # -- plan ---------------------------------------------------------------

  def _plan(self, goal: str, screenshot: np.ndarray) -> tuple[str, str, str]:
    prompt = _action_selection_prompt(
        goal,
        [f"Step {i + 1}- {h['summary']}" for i, h in enumerate(self.history)],
        self.additional_guidelines,
    )
    action_output, is_safe, raw_response = self.llm.predict_mm(
        prompt, [screenshot],
    )
    if is_safe is False:
      action_output = (
          f"Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}\n"
          'Action: {"action_type": "status", "goal_status": "infeasible"}'
      )
    if not raw_response:
      raise RuntimeError("Planner LLM returned no response.")
    return prompt, action_output, str(raw_response)

  # -- act ----------------------------------------------------------------

  def _resolve_element(
      self, converted: json_action.JSONAction, element_desc: str,
      screenshot: np.ndarray,
  ) -> bool:
    """Fills converted.x/y from the element description via UI-Venus."""
    if not element_desc:
      print("[M3AVenus] Missing 'element' for a grounding-required action.")
      return False
    point = self.grounder.locate(screenshot, element_desc)
    if point is None:
      return False
    converted.x, converted.y = point
    return True

  # -- step ---------------------------------------------------------------

  def step(
      self, goal: str, step_numb: bool = False,
  ) -> base_agent.AgentInteractionResult:
    step_data: dict[str, Any] = {
        "raw_screenshot": None,
        "action_prompt": None,
        "action_output": None,
        "action_output_json": None,
        "action_reason": None,
        "action_raw_response": None,
        "grounded_point": None,
        "summary_prompt": None,
        "summary": None,
        "summary_raw_response": None,
    }
    print(f"----------step {len(self.history) + 1}")

    state = self.get_post_transition_state()
    raw_screenshot = state.pixels.copy()
    step_data["raw_screenshot"] = raw_screenshot

    try:
      prompt, action_output, raw_response = self._plan(goal, raw_screenshot)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"[M3AVenus] planner error: {exc}")
      traceback.print_exc()
      step_data["summary"] = f"Planner LLM error: {exc}"
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    step_data["action_prompt"] = prompt
    step_data["action_output"] = action_output
    step_data["action_raw_response"] = raw_response

    reason, action_str = m3a_utils.parse_reason_action_output(action_output)
    if not reason or not action_str:
      print("[M3AVenus] could not parse Reason/Action from planner output.")
      step_data["summary"] = (
          "Output for action selection is not in the correct format, so no"
          " action is performed."
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    print(f"Reason: {reason}")
    print(f"Action: {action_str}")
    step_data["action_reason"] = reason

    # Extract JSON, then separate `element` out (not a JSONAction field).
    try:
      action_dict = agent_utils.extract_json(action_str)
      if action_dict is None:
        raise ValueError("extract_json returned None")
      element_desc = action_dict.pop("element", None)
      if action_dict.get("action_type") == json_action.OPEN_APP:
        name = action_dict.get("app_name", "")
        if isinstance(name, str):
          action_dict["app_name"] = _normalize_open_app_name(name, goal)
      if action_dict.get("action_type") == json_action.INPUT_TEXT:
        action_dict.setdefault("clear_text", True)
      converted = json_action.JSONAction(**action_dict)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"[M3AVenus] failed to build JSONAction: {exc}")
      traceback.print_exc()
      step_data["summary"] = (
          "Can not parse the output to a valid action. Make sure to pick"
          " the action from the list with required parameters (if any) in"
          " the correct JSON format."
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    step_data["action_output_json"] = converted

    # Grounding step if needed.
    if converted.action_type in self._ELEMENT_ACTIONS:
      ok = self._resolve_element(converted, element_desc, raw_screenshot)
      if not ok:
        step_data["summary"] = (
            "Grounding failed for element description"
            f" {element_desc!r}; action skipped."
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)
      step_data["grounded_point"] = (converted.x, converted.y)
      print(f"[M3AVenus] grounded {element_desc!r} -> ({converted.x}, {converted.y})")

    # Terminal / non-actuated actions.
    if converted.action_type == json_action.STATUS:
      if converted.goal_status == "infeasible":
        print("[M3AVenus] planner declared task infeasible.")
      step_data["summary"] = "Agent thinks the request has been completed."
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(True, step_data)

    if converted.action_type == json_action.ANSWER:
      print(f"[M3AVenus] answered: {converted.text}")

    # Execute on the device.
    try:
      if not step_numb:
        actuation.execute_adb_action(
            converted, [], self.env.logical_screen_size, self.env.controller,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"[M3AVenus] actuation error: {exc}")
      traceback.print_exc()
      step_data["summary"] = (
          "Can not execute the action; make sure required parameters are"
          " present in the correct JSON format."
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    time.sleep(self.wait_after_action_seconds)

    # Post-action state + summary.
    after_state = self.env.get_state(wait_to_stabilize=False)
    after_screenshot = after_state.pixels.copy()

    before_annotated = raw_screenshot.copy()
    if converted.x is not None and converted.y is not None:
      try:
        from android_world.agents import m3a_utils_gd
        m3a_utils_gd.add_ui_element_dot(
            before_annotated,
            target_element=[int(round(converted.x)), int(round(converted.y))],
        )
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    try:
      m3a_utils.add_screenshot_label(before_annotated, "before")
      m3a_utils.add_screenshot_label(after_screenshot, "after")
    except Exception:  # pylint: disable=broad-exception-caught
      pass

    summary_prompt = _summary_prompt(action_str, reason, goal)
    try:
      summary, is_safe, raw_sum = self.llm.predict_mm(
          summary_prompt, [before_annotated, after_screenshot],
      )
      if is_safe is False:
        summary = "Summary triggered LLM safety classifier."
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"[M3AVenus] summarizer error: {exc}")
      summary = f"Summary LLM error: {exc}"
      raw_sum = None

    step_data["summary_prompt"] = summary_prompt
    step_data["summary"] = f"Action selected: {action_str}. {summary}"
    step_data["summary_raw_response"] = raw_sum
    print(f"Summary: {summary}")

    self.history.append(step_data)
    return base_agent.AgentInteractionResult(False, step_data)
