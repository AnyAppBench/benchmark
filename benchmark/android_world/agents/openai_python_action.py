"""OpenAI-compatible screenshot agents that emit Python-style UI actions."""

from __future__ import annotations

import ast
import base64
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests
from PIL import Image

from android_world.agents import base_agent
from android_world.agents import episode_exceptions
from android_world.env import actuation
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils


UI_VENUS_NAVI_PROMPT_TEMPLATE = """**You are a GUI Agent.**
Your task is to analyze a given user task, review current screenshot and previous actions, and determine the next action to complete the task.

### User Task
{user_task}

### Previous Actions
{previous_actions}
### Available Actions
Click(box=(x1, y1))
Drag(start=(x1, y1), end=(x2, y2))
Scroll(start=(x1, y1), end=(x2, y2), direction='down/up/right/left')
Type(content='')
Launch(app='')
Wait()
Finished(content='')
CallUser(content='')
LongPress(box=(x1, y1))
PressBack()
PressHome()
PressEnter()
PressRecent()
### Instruction
- Make sure you understand the task goal to avoid wrong actions.
- Examine the screenshot carefully. History may be unreliable.
- For user questions, reply with `CallUser`, then `Finished` if done.
- Explore screen content using scroll in different directions.
- Copy text: select -> click `copy`.
- Paste text: long press text box -> click `paste`.
- First reason inside <think>, then provide <action>, then summarize in <conclusion>.
"""


MOBILERL_POINT_THINK_SYSTEM_PROMPT = """# Setup
You are a professional Android operation agent assistant that can fulfill the user's high-level instructions. Given a screenshot of the Android interface at each step, you first analyze the situation, then plan the best course of action using Python-style pseudo-code.

# More details about the code
Your response format must be structured as follows:

Think first: Use <think>...</think> to analyze the current screen, identify key elements, and determine the most efficient action.
Provide the action: Use <answer>...</answer> to return a single line of pseudo-code representing the operation.

Your output should STRICTLY follow the format:
<think>
[Your throught]
</think>
<answer>
[Your operation code]
</answer>

- **Tap**
  Perform a tap action on a specified screen area. The element is a list of 4 integers, representing the coordinates of the top-left and bottom-right corners of the rectangle. You must choose one element from the current state.
  Never put text, phone numbers, labels, indexes, or resource ids in `element`; only copy a 4-integer `bounds=[x1,y1,x2,y2]` box from the current state.
  **Example**:
  <answer>
  do(action="Tap", element=[100, 200, 150, 250])
  </answer>
- **Type**
  Enter text into the currently focused input field.
  **Example**:
  <answer>
  do(action="Type", text="Hello World")
  </answer>
- **Swipe**
  Perform a swipe action in a specified direction (`"up"`, `"down"`, `"left"`, `"right"`).
  The swipe distance can be `"long"`, `"medium"` (default), or `"short"`.
  You can add the element to the action to specify the swipe area. The element is a list of 4 integers, representing the coordinates of the top-left and bottom-right corners of the rectangle. You must choose one element from the current state.
  Never put text, phone numbers, labels, indexes, or resource ids in `element`; only copy a 4-integer `bounds=[x1,y1,x2,y2]` box from the current state.
  **Examples**:
  <answer>
  do(action="Swipe", direction="up", dist="long", element=[100, 200, 150, 250])
  </answer>
- **Long Press**
  Perform a long press action on a specified screen area.
  You can add the element to the action to specify the long press area. The element is a list of 4 integers, representing the coordinates of the top-left and bottom-right corners of the rectangle. You must choose one element from the current state.
  Never put text, phone numbers, labels, indexes, or resource ids in `element`; only copy a 4-integer `bounds=[x1,y1,x2,y2]` box from the current state.
  **Example**:
  <answer>
  do(action="Long Press", element=[200, 300, 250, 350])
  </answer>
- **Launch**
  Launch an app. Try to use launch action when you need to launch an app. Check the instruction to choose the right app before you use this action.
  **Example**:
  <answer>
  do(action="Launch", app="Settings")
  </answer>
- **Back**
  Press the Back button to navigate to the previous screen.
  **Example**:
  <answer>
  do(action="Back")
  </answer>
- **Finish**
  Terminate the program and optionally print a message.
  **Example**:
  <answer>
  finish(message="Task completed.")
  </answer>


REMEMBER:
- Think before you act: Always analyze the current UI and the best course of action before executing any step, and output in <think> part.
- Only ONE LINE of action in <answer> part per response: Each step must contain exactly one line of executable code.
- Generate execution code strictly according to format requirements.
- A screenshot of the current page and the UI location information will be provided at the same time. If your action involves location information, try to choose the known UI location information.
- For Tap, Long Press, or Swipe element, `element` must be a 4-integer coordinate box copied from `bounds=[...]`.
"""


APPAGENT_V2_LITE_PROMPT_TEMPLATE = """You are a mobile GUI agent. Use the screenshot, task, previous actions, and visible UI structure to choose exactly one next action.

Task:
{user_task}

Previous actions:
{previous_actions}

Use one action in this format inside <action>...</action>:
Click(box=(x, y))
LongPress(box=(x, y))
Scroll(start=(x1, y1), end=(x2, y2), direction='down/up/right/left')
Type(content='')
Launch(app='')
PressBack()
PressHome()
PressEnter()
Wait()
Finished(content='')

Think briefly inside <think>...</think>, write exactly one action inside <action>...</action>, and summarize inside <conclusion>...</conclusion>.
"""


@dataclass(frozen=True)
class ParsedAction:
  action: str
  params: dict[str, Any]
  raw: str


def _pil_to_b64_url(image: Image.Image) -> str:
  buffer = io.BytesIO()
  image.save(buffer, format="PNG")
  return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _resize_to_max_pixels(image: Image.Image, max_pixels: int | None) -> Image.Image:
  if not max_pixels or max_pixels <= 0:
    return image
  pixels = image.width * image.height
  if pixels <= max_pixels:
    return image
  scale = (max_pixels / pixels) ** 0.5
  width = max(1, int(image.width * scale))
  height = max(1, int(image.height * scale))
  return image.resize((width, height))


def _normalize_openai_endpoint(endpoint_url: str) -> str:
  endpoint_url = endpoint_url.rstrip("/")
  return endpoint_url if endpoint_url.endswith("/v1") else f"{endpoint_url}/v1"


def _normalize_venus_predict_endpoint(endpoint_url: str) -> str:
  endpoint_url = endpoint_url.rstrip("/")
  if endpoint_url.endswith("/health"):
    endpoint_url = endpoint_url[: -len("/health")]
  if endpoint_url.endswith("/predict"):
    return endpoint_url
  return f"{endpoint_url}/predict"


def _format_venus_predict_response(data: dict[str, Any]) -> str:
  if isinstance(data, str):
    return data
  if not isinstance(data, dict):
    return ""
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
  return "\n".join(parts)


def _post_json_with_retries(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    attempts: int = 3,
    headers: dict[str, str] | None = None,
) -> requests.Response:
  last_response: requests.Response | None = None
  last_exc: requests.RequestException | None = None
  for attempt in range(1, attempts + 1):
    try:
      response = requests.post(
          url,
          headers=headers or {"Content-Type": "application/json"},
          json=payload,
          timeout=timeout,
      )
      if response.ok or response.status_code < 500 or attempt == attempts:
        return response
      last_response = response
    except requests.RequestException as exc:
      last_exc = exc
      if attempt == attempts:
        raise
    time.sleep(min(2 ** (attempt - 1), 8))
  if last_response is not None:
    return last_response
  assert last_exc is not None
  raise last_exc


def _checked_endpoint_json(
    response: requests.Response, endpoint_url: str
) -> Any:
  """Returns endpoint JSON or raises a typed service failure."""
  if not response.ok:
    raise episode_exceptions.ModelEndpointError(
        f"Model endpoint {endpoint_url} returned HTTP "
        f"{response.status_code}: {response.text[:1000]}"
    )
  try:
    return response.json()
  except ValueError as exc:
    raise episode_exceptions.ModelEndpointError(
        f"Model endpoint {endpoint_url} returned invalid JSON."
    ) from exc


def _openai_message_content(data: Any, endpoint_url: str) -> str:
  """Validates the service envelope without classifying model action text."""
  if not isinstance(data, dict):
    raise episode_exceptions.ModelEndpointError(
        f"Model endpoint {endpoint_url} returned a non-object JSON envelope."
    )
  choices = data.get("choices")
  if not isinstance(choices, list) or not choices:
    raise episode_exceptions.ModelEndpointError(
        f"Model endpoint {endpoint_url} returned no choices."
    )
  first_choice = choices[0]
  if not isinstance(first_choice, dict):
    raise episode_exceptions.ModelEndpointError(
        f"Model endpoint {endpoint_url} returned an invalid first choice."
    )
  message = first_choice.get("message")
  if not isinstance(message, dict):
    raise episode_exceptions.ModelEndpointError(
        f"Model endpoint {endpoint_url} returned no message object."
    )
  content = message.get("content")
  if not isinstance(content, str) or not content.strip():
    raise episode_exceptions.ModelEndpointError(
        f"Model endpoint {endpoint_url} returned empty message content."
    )
  return content


def _shorten_for_prompt(value: Any, limit: int = 80) -> str:
  text = str(value or "").replace("\n", " ").strip()
  text = re.sub(r"\s+", " ", text)
  if len(text) <= limit:
    return text
  return text[: limit - 3].rstrip() + "..."


def _format_prompt_value(name: str, value: Any, limit: int = 80) -> str:
  text = _shorten_for_prompt(value, limit)
  if not text:
    return ""
  text = text.replace('"', '\\"')
  return f'{name}="{text}"'


def _relative_bbox_from_element(
    element: representation_utils.UIElement,
    screen_width: int,
    screen_height: int,
    relative_base: int = 999,
) -> list[int] | None:
  bbox = element.bbox_pixels or element.bbox
  if bbox is None or screen_width <= 0 or screen_height <= 0:
    return None

  x_min = float(bbox.x_min)
  x_max = float(bbox.x_max)
  y_min = float(bbox.y_min)
  y_max = float(bbox.y_max)

  # Some AndroidWorld elements expose normalized boxes, while XML-derived
  # elements expose pixel boxes. Keep both paths deterministic.
  if max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)) <= 1.0:
    x_min *= screen_width
    x_max *= screen_width
    y_min *= screen_height
    y_max *= screen_height

  if x_max <= x_min or y_max <= y_min:
    return None

  x_min = min(max(x_min, 0.0), float(screen_width))
  x_max = min(max(x_max, 0.0), float(screen_width))
  y_min = min(max(y_min, 0.0), float(screen_height))
  y_max = min(max(y_max, 0.0), float(screen_height))
  if x_max <= x_min or y_max <= y_min:
    return None

  return [
      int(round(x_min * relative_base / screen_width)),
      int(round(y_min * relative_base / screen_height)),
      int(round(x_max * relative_base / screen_width)),
      int(round(y_max * relative_base / screen_height)),
  ]


def _format_mobilerl_ui_context(
    ui_elements: list[representation_utils.UIElement],
    screen_size: tuple[int, int],
    relative_base: int = 999,
    max_elements: int | None = None,
) -> str:
  """Formats AndroidWorld UI elements in MobileRL's bbox-observation style."""
  if max_elements is None:
    max_elements = int(os.environ.get("MOBILERL_MAX_UI_ELEMENTS", "120"))

  screen_width, screen_height = screen_size
  lines = [
      (
          f"The screenshot's size is {relative_base}x{relative_base}. "
          "The value in bounds is relative to the screenshot's size."
      ),
      "The tree structure description of the current screenshot is shown:",
  ]

  count = 0
  for element in ui_elements:
    if element.is_visible is False:
      continue
    bbox = _relative_bbox_from_element(
        element, screen_width, screen_height, relative_base
    )
    if bbox is None:
      continue

    pieces = [
        _format_prompt_value("text", element.text),
        _format_prompt_value("content_desc", element.content_description),
        _format_prompt_value("hint", element.hint_text),
    ]
    class_name = _shorten_for_prompt(element.class_name, 60)
    if class_name:
      pieces.append(f"class={class_name.split('.')[-1]}")
    resource = _format_prompt_value(
        "resource", element.resource_id or element.resource_name, 80
    )
    if resource:
      pieces.append(resource)

    flags = []
    for name, value in (
        ("clickable", element.is_clickable),
        ("editable", element.is_editable),
        ("scrollable", element.is_scrollable),
        ("checkable", element.is_checkable),
        ("checked", element.is_checked),
        ("selected", element.is_selected),
        ("focused", element.is_focused),
        ("enabled", element.is_enabled),
    ):
      if value is True:
        flags.append(f"{name}=true")

    pieces = [piece for piece in pieces if piece]
    if not pieces and not flags:
      continue

    count += 1
    lines.append(
        f"{count}. "
        + "; ".join([
            *(pieces or ["node"]),
            f"bounds=[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]",
            *flags,
        ])
    )
    if count >= max_elements:
      break

  if count == 0:
    lines.append("No visible UI elements were exposed by accessibility.")
  return "\n".join(lines)


def _extract_latest_tag(text: str, tag: str) -> str:
  matches = list(re.finditer(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL))
  return matches[-1].group(1).strip() if matches else ""


def _strip_quotes(value: str) -> str:
  value = value.strip()
  if (value.startswith("'") and value.endswith("'")) or (
      value.startswith('"') and value.endswith('"')
  ):
    return value[1:-1]
  return value


def _split_params(params: str) -> list[str]:
  pieces = []
  start = 0
  depth = 0
  quote = ""
  for idx, char in enumerate(params):
    if quote:
      if char == quote and params[idx - 1:idx] != "\\":
        quote = ""
      continue
    if char in {"'", '"'}:
      quote = char
      continue
    if char in "([{":
      depth += 1
    elif char in ")]}":
      depth = max(0, depth - 1)
    elif char == "," and depth == 0:
      pieces.append(params[start:idx].strip())
      start = idx + 1
  tail = params[start:].strip()
  if tail:
    pieces.append(tail)
  return pieces


def _literal(value: str) -> Any:
  try:
    return ast.literal_eval(value)
  except (SyntaxError, ValueError):
    return _strip_quotes(value)


def _parse_function_call(code: str) -> ParsedAction:
  code = code.strip().strip("`").strip()
  if code.startswith("finish("):
    return ParsedAction("Finished", {}, code)
  match = re.match(r"^(\w+)\((.*)\)$", code, re.DOTALL)
  if not match:
    raise ValueError(f"Could not parse action call: {code}")
  name, params_raw = match.group(1), match.group(2).strip()
  params: dict[str, Any] = {}
  for piece in _split_params(params_raw):
    if "=" not in piece:
      params[piece] = None
      continue
    key, value = piece.split("=", 1)
    params[key.strip()] = _literal(value.strip())
  if name == "do":
    action = str(params.pop("action", "")).strip()
  else:
    action = name
  return ParsedAction(action, params, code)


def _extract_complete_function_calls(text: str) -> list[str]:
  call_start = re.compile(
      r"\b(finish|do|Click|Drag|Scroll|Type|Launch|Wait|Finished|CallUser|"
      r"LongPress|PressBack|PressHome|PressEnter|PressRecent)\s*\("
  )
  calls = []
  for match in call_start.finditer(text):
    start = match.start()
    depth = 0
    quote = ""
    escaped = False
    for idx in range(match.start(), len(text)):
      char = text[idx]
      if quote:
        if escaped:
          escaped = False
        elif char == "\\":
          escaped = True
        elif char == quote:
          quote = ""
        continue
      if char in {"'", '"'}:
        quote = char
      elif char == "(":
        depth += 1
      elif char == ")":
        depth -= 1
        if depth == 0:
          calls.append(text[start:idx + 1].strip())
          break
  return calls


def _extract_complete_function_call(text: str) -> str:
  calls = _extract_complete_function_calls(text)
  return calls[0] if calls else ""


def _is_coordinate_list(value: Any, allowed_lengths: tuple[int, ...]) -> bool:
  if isinstance(value, str):
    value = _literal(value)
  if not isinstance(value, (list, tuple)):
    return False
  if len(value) not in allowed_lengths:
    return False
  try:
    coords = [float(coord) for coord in value]
  except (TypeError, ValueError):
    return False
  return all(math.isfinite(coord) for coord in coords)


def _has_coordinate_param(
    parsed: ParsedAction,
    keys: tuple[str, ...],
    allowed_lengths: tuple[int, ...],
) -> bool:
  return any(
      key in parsed.params
      and _is_coordinate_list(parsed.params[key], allowed_lengths)
      for key in keys
  )


def _validate_executable_action(parsed: ParsedAction) -> ParsedAction:
  action = parsed.action.strip().lower().replace("_", " ")
  supported_actions = {
      "tap",
      "click",
      "long press",
      "longpress",
      "type",
      "launch",
      "back",
      "pressback",
      "home",
      "presshome",
      "enter",
      "pressenter",
      "wait",
      "finished",
      "finish",
      "calluser",
      "pressrecent",
      "recent",
      "swipe",
      "drag",
      "scroll",
  }
  if action not in supported_actions:
    raise ValueError(f"Unsupported action: {parsed.raw}")
  if action in {"tap", "click", "long press", "longpress"}:
    if not _has_coordinate_param(parsed, ("element", "box"), (2, 4)):
      raise ValueError(f"{parsed.action} requires a coordinate box: {parsed.raw}")
  if action == "type":
    text = parsed.params.get("text", parsed.params.get("content"))
    if not isinstance(text, str):
      raise ValueError(f"Type requires text content: {parsed.raw}")
  if action == "launch":
    app = parsed.params.get("app")
    if not isinstance(app, str) or not app.strip():
      raise ValueError(f"Launch requires a non-empty app name: {parsed.raw}")
  if action in {"swipe", "drag", "scroll"}:
    if ("start" in parsed.params or "end" in parsed.params) and not (
        _has_coordinate_param(parsed, ("start",), (2,))
        and _has_coordinate_param(parsed, ("end",), (2,))
    ):
      raise ValueError(f"Swipe start/end must be points: {parsed.raw}")
    if ("element" in parsed.params or "box" in parsed.params) and not (
        _has_coordinate_param(parsed, ("element", "box"), (2, 4))
    ):
      raise ValueError(f"Swipe element must be a coordinate box: {parsed.raw}")
    direction = parsed.params.get("direction")
    if direction is not None and str(direction).strip("'\"").lower() not in {
        "up",
        "down",
        "left",
        "right",
    }:
      raise ValueError(f"Unsupported swipe direction: {parsed.raw}")
    dist = parsed.params.get("dist")
    if isinstance(dist, str) and dist.strip("'\"").lower() not in {
        "short",
        "medium",
        "long",
    }:
      raise ValueError(f"Unsupported swipe distance: {parsed.raw}")
    if dist is not None and not isinstance(dist, str):
      try:
        finite_dist = math.isfinite(float(dist))
      except (TypeError, ValueError):
        finite_dist = False
      if not finite_dist:
        raise ValueError(f"Swipe distance must be finite: {parsed.raw}")
    if action == "drag" and not (
        _has_coordinate_param(parsed, ("start",), (2,))
        and _has_coordinate_param(parsed, ("end",), (2,))
    ):
      raise ValueError(f"Drag requires start and end points: {parsed.raw}")
  return parsed


_CATBENCH_APP_NAMES = (
    "Simple SMS Messenger",
    "Fossify Messages",
    "QUIK SMS",
    "Messages",
    "Material Files",
    "Amaze File Manager",
    "Fossify File Manager",
    "Total Commander",
    "X-plore File Manager",
    "OsmAnd~",
    "Organic Maps",
    "Google Maps",
    "CoMaps",
    "MAPS.ME",
    "Google Contacts",
    "Fossify Contacts",
    "Connect You",
    "Simple Contacts Pro SE",
    "Right Contact",
    "Clock",
    "Simple Clock",
    "Google Clock",
    "Clock You",
    "Chrono",
    "Fossify Clock",
)

_GENERIC_APP_ALIASES = {
    "file": ("Material Files", "Amaze File Manager", "Fossify File Manager",
             "Total Commander", "X-plore File Manager"),
    "file app": ("Material Files", "Amaze File Manager", "Fossify File Manager",
                 "Total Commander", "X-plore File Manager"),
    "files": ("Material Files", "Amaze File Manager", "Fossify File Manager",
              "Total Commander", "X-plore File Manager"),
    "file manager": ("Material Files", "Amaze File Manager",
                     "Fossify File Manager", "Total Commander",
                     "X-plore File Manager"),
    "sms": ("Simple SMS Messenger", "Fossify Messages", "QUIK SMS",
            "Messages"),
    "message": ("Simple SMS Messenger", "Fossify Messages", "QUIK SMS",
                "Messages"),
    "messages": ("Simple SMS Messenger", "Fossify Messages", "QUIK SMS",
                 "Messages"),
    "contacts": ("Google Contacts", "Fossify Contacts", "Connect You",
                 "Simple Contacts Pro SE", "Right Contact"),
    "contact": ("Google Contacts", "Fossify Contacts", "Connect You",
                "Simple Contacts Pro SE", "Right Contact"),
    "maps": ("OsmAnd~", "Organic Maps", "Google Maps", "CoMaps", "MAPS.ME"),
    "map": ("OsmAnd~", "Organic Maps", "Google Maps", "CoMaps", "MAPS.ME"),
    "clock": ("Clock", "Simple Clock", "Google Clock", "Clock You",
              "Chrono", "Fossify Clock"),
}


def _resolve_mobilerl_launch_app(app_name: Any, goal: str) -> str:
  app = _strip_quotes(str(app_name or "")).strip()
  if not app:
    return app
  goal_lower = goal.lower()
  app_lower = app.lower()
  app_compact = re.sub(r"[^a-z0-9]+", "", app_lower)

  if app_lower in _GENERIC_APP_ALIASES:
    for known in sorted(
        _GENERIC_APP_ALIASES[app_lower], key=len, reverse=True
    ):
      if known.lower() in goal_lower:
        return known

  for known in sorted(_CATBENCH_APP_NAMES, key=len, reverse=True):
    known_lower = known.lower()
    known_compact = re.sub(r"[^a-z0-9]+", "", known_lower)
    if known_lower == app_lower or known_compact == app_compact:
      return known

  for known in sorted(_CATBENCH_APP_NAMES, key=len, reverse=True):
    known_lower = known.lower()
    if known_lower in goal_lower:
      return known

  for known in sorted(_CATBENCH_APP_NAMES, key=len, reverse=True):
    known_lower = known.lower()
    known_compact = re.sub(r"[^a-z0-9]+", "", known_lower)
    if (
        app_lower in known_lower
        or known_lower in app_lower
        or app_compact in known_compact
        or known_compact in app_compact
    ):
      return known
  return app


def _normalize_mobilerl_launch_action(
    parsed: ParsedAction, goal: str
) -> ParsedAction:
  action = parsed.action.strip().lower().replace("_", " ")
  if action != "launch":
    return parsed
  params = dict(parsed.params)
  params["app"] = _resolve_mobilerl_launch_app(params.get("app", ""), goal)
  return ParsedAction(parsed.action, params, parsed.raw)


def _center(value: Any) -> tuple[int, int]:
  if isinstance(value, str):
    value = _literal(value)
  if not isinstance(value, (list, tuple)):
    raise ValueError(f"Expected coordinate list, got {value!r}")
  coords = [float(v) for v in value]
  if len(coords) == 2:
    return int(round(coords[0])), int(round(coords[1]))
  if len(coords) == 4:
    return int(round((coords[0] + coords[2]) / 2)), int(
        round((coords[1] + coords[3]) / 2)
    )
  raise ValueError(f"Expected 2 or 4 coordinates, got {value!r}")


def _point(value: Any) -> tuple[int, int]:
  if isinstance(value, str):
    value = _literal(value)
  coords = [float(v) for v in value]
  if len(coords) != 2:
    raise ValueError(f"Expected point coordinate, got {value!r}")
  return int(round(coords[0])), int(round(coords[1]))


def _scale_coords(
    value: Any,
    scale_x: float,
    scale_y: float,
    relative_coord_base: float | None = None,
) -> Any:
  if value is None:
    return None
  if isinstance(value, str):
    value = _literal(value)
  if not isinstance(value, (list, tuple)):
    return value
  coords = [float(v) for v in value]
  if relative_coord_base is not None and any(
      coord > relative_coord_base or coord < 0 for coord in coords
  ):
    return [int(round(coord)) for coord in coords]
  if len(coords) == 2:
    return [int(round(coords[0] * scale_x)), int(round(coords[1] * scale_y))]
  if len(coords) == 4:
    return [
        int(round(coords[0] * scale_x)),
        int(round(coords[1] * scale_y)),
        int(round(coords[2] * scale_x)),
        int(round(coords[3] * scale_y)),
    ]
  return value


def _scale_parsed_action(
    parsed: ParsedAction,
    scale_x: float,
    scale_y: float,
    relative_coord_base: float | None = None,
) -> ParsedAction:
  if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
    return parsed
  params = dict(parsed.params)
  for key in ("element", "box", "start", "end"):
    if key in params:
      params[key] = _scale_coords(
          params[key], scale_x, scale_y, relative_coord_base
      )
  return ParsedAction(parsed.action, params, parsed.raw)


def _action_repetition_key(parsed: ParsedAction) -> str:
  return json.dumps(
      {"action": parsed.action, "params": parsed.params},
      sort_keys=True,
      default=str,
  )


class OpenAIPythonActionAgent(base_agent.EnvironmentInteractingAgent):
  """Screenshot agent for models that produce Python-like UI actions."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      endpoint_url: str,
      model_name: str,
      api_key: str = "EMPTY",
      endpoint_format: str = "openai",
      prompt_style: str = "ui_venus_navi",
      max_new_tokens: int = 2048,
      temperature: float = 0.0,
      image_max_pixels: int | None = None,
      wait_after_action_seconds: float = 2.0,
      request_timeout: int = 300,
      output_path: str = "",
      name: str = "openai_python_action",
  ):
    super().__init__(env, name)
    if endpoint_format == "openai":
      self.endpoint_url = _normalize_openai_endpoint(endpoint_url)
    elif endpoint_format in {"venus_predict", "gui_proxy_predict"}:
      self.endpoint_url = _normalize_venus_predict_endpoint(endpoint_url)
    else:
      raise ValueError(f"Unsupported endpoint_format: {endpoint_format}")
    self.endpoint_format = endpoint_format
    self.model_name = model_name
    self.api_key = api_key or "EMPTY"
    self.prompt_style = prompt_style
    self.max_new_tokens = max_new_tokens
    self.temperature = temperature
    self.image_max_pixels = image_max_pixels
    self.wait_after_action_seconds = wait_after_action_seconds
    self.request_timeout = request_timeout
    self.output_path = output_path
    self.history: list[str] = []
    self._mobilerl_history: list[dict[str, Any]] = []
    self._mobilerl_current_history_user: dict[str, Any] | None = None
    self._mobilerl_current_image_user: dict[str, Any] | None = None
    self._mobilerl_last_image_user: dict[str, Any] | None = None
    self._mobilerl_recent_action_keys: list[str] = []
    self._mobilerl_max_history_messages = int(
        os.environ.get("MOBILERL_MAX_HISTORY_MESSAGES", "4")
    )
    self._mobilerl_picture_round = max(
        1, int(os.environ.get("MOBILERL_PICTURE_ROUND", "2"))
    )

  def reset(self, go_home_on_reset: bool = False) -> None:
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history.clear()
    self._mobilerl_history.clear()
    self._mobilerl_current_history_user = None
    self._mobilerl_current_image_user = None
    self._mobilerl_last_image_user = None
    self._mobilerl_recent_action_keys.clear()

  def _previous_actions(self) -> str:
    return "\n".join(
        f"{idx + 1}. {entry}" for idx, entry in enumerate(self.history)
    )

  def _trim_mobilerl_history(self) -> None:
    if self._mobilerl_max_history_messages <= 0:
      self._mobilerl_history.clear()
      return
    if len(self._mobilerl_history) > self._mobilerl_max_history_messages:
      self._mobilerl_history = self._mobilerl_history[
          -self._mobilerl_max_history_messages:
      ]

  def _mobilerl_user_message(
      self, image_url: str, text: str
  ) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": text},
        ],
    }

  def _messages(
      self, goal: str, image_url: str, ui_context: str = ""
  ) -> tuple[list[dict[str, Any]], str, str]:
    previous = self._previous_actions()
    if self.prompt_style == "mobilerl_point_think":
      self._trim_mobilerl_history()
      base_text = goal if not self.history else f"Task: {goal}\n** Screen Info **"
      text = f"{base_text}\n{ui_context}" if ui_context else base_text
      current_message = self._mobilerl_user_message(image_url, text)
      self._mobilerl_current_history_user = {
          "role": "user",
          "content": base_text,
      }
      self._mobilerl_current_image_user = self._mobilerl_user_message(
          image_url, base_text
      )
      history = list(self._mobilerl_history)
      if (
          self._mobilerl_picture_round >= 2
          and self._mobilerl_last_image_user is not None
          and len(history) >= 2
      ):
        history = [
            *history[:-2],
            self._mobilerl_last_image_user,
            history[-1],
        ]
      return (
          [
              {"role": "system", "content": MOBILERL_POINT_THINK_SYSTEM_PROMPT},
              *history,
              current_message,
          ],
          MOBILERL_POINT_THINK_SYSTEM_PROMPT,
          text,
      )
    if self.prompt_style == "appagent_v2_lite":
      text = APPAGENT_V2_LITE_PROMPT_TEMPLATE.format(
          user_task=goal, previous_actions=previous
      )
    else:
      text = UI_VENUS_NAVI_PROMPT_TEMPLATE.format(
          user_task=goal, previous_actions=previous
      )
    return (
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "",
        text,
    )

  def _call_model(
      self, goal: str, screenshot: Image.Image, ui_context: str = ""
  ) -> tuple[str, str, str, float, float, tuple[int, int]]:
    image = _resize_to_max_pixels(screenshot.convert("RGB"), self.image_max_pixels)
    image_url = _pil_to_b64_url(image)
    scale_x = screenshot.width / image.width
    scale_y = screenshot.height / image.height
    if self.endpoint_format == "venus_predict":
      previous_actions = self._previous_actions() or "No previous actions."
      payload = {
          "user_task": goal,
          "previous_actions": previous_actions,
          "image": image_url,
          "max_new_tokens": self.max_new_tokens,
      }
      try:
        response = _post_json_with_retries(
            self.endpoint_url,
            payload,
            timeout=self.request_timeout,
        )
      except requests.RequestException as exc:
        raise episode_exceptions.ModelEndpointError(
            f"Model endpoint {self.endpoint_url} failed after retries: {exc}"
        ) from exc
      response_data = _checked_endpoint_json(response, self.endpoint_url)
      response_text = _format_venus_predict_response(response_data)
      if not response_text.strip():
        raise episode_exceptions.ModelEndpointError(
            f"Model endpoint {self.endpoint_url} returned no usable output."
        )
      return (
          response_text,
          "",
          (
              f"user_task:\n{goal}\n\n"
              f"previous_actions:\n{previous_actions}\n\n"
              f"endpoint_format: venus_predict"
          ),
          scale_x,
          scale_y,
          (image.width, image.height),
      )
    if self.endpoint_format == "gui_proxy_predict":
      previous_actions = self._previous_actions()
      previous_actions_text = previous_actions or "No previous actions."
      instruction = goal
      if previous_actions:
        instruction = f"{goal}\n\nPrevious actions:\n{previous_actions}"
      payload = {
          "instruction": instruction,
          "user_task": goal,
          "previous_actions": previous_actions_text,
          "image": image_url,
          "max_new_tokens": self.max_new_tokens,
      }
      try:
        response = _post_json_with_retries(
            self.endpoint_url,
            payload,
            timeout=self.request_timeout,
        )
      except requests.RequestException as exc:
        raise episode_exceptions.ModelEndpointError(
            f"Model endpoint {self.endpoint_url} failed after retries: {exc}"
        ) from exc
      response_data = _checked_endpoint_json(response, self.endpoint_url)
      response_text = _format_venus_predict_response(response_data)
      if not response_text.strip():
        raise episode_exceptions.ModelEndpointError(
            f"Model endpoint {self.endpoint_url} returned no usable output."
        )
      return (
          response_text,
          "",
          (
              f"instruction:\n{instruction}\n\n"
              f"user_task:\n{goal}\n\n"
              f"previous_actions:\n{previous_actions_text}\n\n"
              f"endpoint_format: gui_proxy_predict"
          ),
          scale_x,
          scale_y,
          (image.width, image.height),
      )

    messages, prompt_system, prompt_user = self._messages(
        goal, image_url, ui_context
    )
    payload = {
        "model": self.model_name,
        "messages": messages,
        "max_tokens": self.max_new_tokens,
        "temperature": self.temperature,
    }
    if self.prompt_style == "mobilerl_point_think":
      payload["top_p"] = 0.95
      payload["seed"] = 34
      if os.environ.get("MOBILERL_STOP_AFTER_ANSWER", "1") != "0":
        payload["stop"] = ["<|user|>", "<|observation|>", "</answer>"]
    chat_endpoint = f"{self.endpoint_url}/chat/completions"
    try:
      response = _post_json_with_retries(
          chat_endpoint,
          payload,
          timeout=self.request_timeout,
          headers={
              "Content-Type": "application/json",
              "Authorization": f"Bearer {self.api_key}",
          },
      )
    except requests.RequestException as exc:
      raise episode_exceptions.ModelEndpointError(
          f"Model endpoint {chat_endpoint} failed after retries: {exc}"
      ) from exc
    data = _checked_endpoint_json(response, chat_endpoint)
    return (
        _openai_message_content(data, chat_endpoint),
        prompt_system,
        prompt_user,
        scale_x,
        scale_y,
        (image.width, image.height),
    )

  def _parse_response(self, response: str) -> tuple[str, str, str, ParsedAction]:
    think = _extract_latest_tag(response, "think")
    conclusion = _extract_latest_tag(response, "conclusion")
    action_text = _extract_latest_tag(response, "action")
    if not action_text:
      action_text = _extract_latest_tag(response, "answer")
    candidates = []
    if action_text:
      candidates.append(action_text)
      candidates.extend(_extract_complete_function_calls(action_text))
    candidates.extend(_extract_complete_function_calls(response))
    if not candidates:
      raise episode_exceptions.ActionParseError(
          "Model response contains no complete Python-style action call."
      )

    first_action_text = candidates[0]
    seen = set()
    parsed_but_malformed = False
    for candidate in candidates:
      candidate = candidate.strip()
      if not candidate or candidate in seen:
        continue
      seen.add(candidate)
      try:
        parsed = _parse_function_call(candidate)
      except ValueError:
        continue
      try:
        parsed = _validate_executable_action(parsed)
      except ValueError:
        parsed_but_malformed = True
        continue
      action_text = candidate
      break
    else:
      if parsed_but_malformed:
        raise episode_exceptions.MalformedActionError(
            f"Model action violates the executable schema: {first_action_text}"
        )
      raise episode_exceptions.ActionParseError(
          f"Could not parse model action: {first_action_text}"
      )
    return think, action_text, conclusion, parsed

  def _json_action_for(self, parsed: ParsedAction) -> json_action.JSONAction:
    action = parsed.action.strip().lower().replace("_", " ")
    params = parsed.params
    if action in {"tap", "click"}:
      x, y = _center(params.get("element", params.get("box")))
      return json_action.JSONAction(action_type=json_action.CLICK, x=x, y=y)
    if action in {"long press", "longpress"}:
      x, y = _center(params.get("element", params.get("box")))
      return json_action.JSONAction(
          action_type=json_action.LONG_PRESS, x=x, y=y
      )
    if action == "type":
      return json_action.JSONAction(
          action_type=json_action.INPUT_TEXT,
          text=str(params.get("text", params.get("content", ""))),
      )
    if action == "launch":
      return json_action.JSONAction(
          action_type=json_action.OPEN_APP,
          app_name=str(params.get("app", "")).lower(),
      )
    if action in {"back", "pressback"}:
      return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)
    if action in {"home", "presshome"}:
      return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME)
    if action in {"enter", "pressenter"}:
      return json_action.JSONAction(action_type=json_action.KEYBOARD_ENTER)
    if action in {"wait"}:
      return json_action.JSONAction(action_type=json_action.WAIT)
    if action in {"finished", "finish", "calluser"}:
      return json_action.JSONAction(
          action_type=json_action.STATUS, goal_status="complete"
      )
    raise ValueError(f"Unsupported action: {parsed.raw}")

  def _execute_swipe_if_needed(self, parsed: ParsedAction) -> bool:
    action = parsed.action.strip().lower().replace("_", " ")
    if action in {"pressrecent", "recent"}:
      adb_utils.issue_generic_request(
          ["shell", "input", "keyevent", "KEYCODE_APP_SWITCH"],
          self.env.controller,
      )
      return True
    if action not in {"swipe", "drag", "scroll"}:
      return False
    start = parsed.params.get("start")
    end = parsed.params.get("end")
    if start is not None and end is not None:
      x1, y1 = _point(start)
      x2, y2 = _point(end)
      request = adb_utils.generate_swipe_command(x1, y1, x2, y2, 500)
      adb_utils.issue_generic_request(request, self.env.controller)
      return True
    direction = str(parsed.params.get("direction", "up")).strip("'\"").lower()
    screen_width, screen_height = self.env.logical_screen_size
    element = parsed.params.get("element", parsed.params.get("box"))
    if element is None:
      start_x, start_y = screen_width // 2, screen_height // 2
    else:
      start_x, start_y = _center(element)
    dist = parsed.params.get("dist", "medium")
    if isinstance(dist, str):
      dist = dist.strip("'\"").lower()
      unit_dist = int(screen_width / 10)
      if dist == "long":
        unit_dist *= 10
      elif dist == "medium":
        unit_dist *= 2
    else:
      unit_dist = int(dist)
    if direction == "up":
      end_x, end_y = start_x, start_y - (2 * unit_dist)
    elif direction == "down":
      end_x, end_y = start_x, start_y + (2 * unit_dist)
    elif direction == "left":
      end_x, end_y = start_x - int(unit_dist * 2.5), start_y
    elif direction == "right":
      end_x, end_y = start_x + int(unit_dist * 2.5), start_y
    else:
      synthetic = json_action.JSONAction(
          action_type=json_action.SCROLL, direction=direction
      )
      actuation.execute_adb_action(
          synthetic, [], self.env.logical_screen_size, self.env.controller
      )
      return True
    request = adb_utils.generate_swipe_command(
        start_x, start_y, end_x, end_y, 500
    )
    adb_utils.issue_generic_request(request, self.env.controller)
    return True

  def step(
      self, goal: str, step_numb: bool = False
  ) -> base_agent.AgentInteractionResult:
    del step_numb
    step_data: dict[str, Any] = {
        "raw_screenshot": None,
        "prompt_system": None,
        "prompt_user": None,
        "response": None,
        "thought": None,
        "action_desc": None,
        "conclusion": None,
        "action": None,
        "action_raw": None,
        "action_scaled": None,
        "sent_image_size": None,
        "coordinate_scale": None,
    }
    state = self.get_post_transition_state()
    step_data["raw_screenshot"] = state.pixels.copy()
    screenshot = Image.fromarray(state.pixels)
    ui_context = ""
    if self.prompt_style == "mobilerl_point_think":
      ui_context = _format_mobilerl_ui_context(
          state.ui_elements, (screenshot.width, screenshot.height)
      )
    (
        response,
        prompt_system,
        prompt_user,
        scale_x,
        scale_y,
        sent_image_size,
    ) = self._call_model(goal, screenshot, ui_context)
    think, action_text, conclusion, parsed = self._parse_response(response)
    if self.prompt_style == "mobilerl_point_think":
      parsed = _normalize_mobilerl_launch_action(parsed, goal)
    relative_coord_base = None
    exec_scale_x, exec_scale_y = scale_x, scale_y
    if self.prompt_style == "mobilerl_point_think":
      # MobileRL's AndroidWorld coordinate prompt uses 0..999 UI boxes.
      # The model can also emit absolute boxes; those are left unscaled.
      relative_coord_base = 999.0
      exec_scale_x = screenshot.width / relative_coord_base
      exec_scale_y = screenshot.height / relative_coord_base
    parsed_for_execution = _scale_parsed_action(
        parsed, exec_scale_x, exec_scale_y, relative_coord_base
    )
    if self.prompt_style == "mobilerl_point_think":
      if self._mobilerl_current_history_user is not None:
        self._mobilerl_history.append(self._mobilerl_current_history_user)
      assistant_history = (
          f"<think>\n{think}\n</think>\n<answer>\n{action_text}\n</answer>"
      )
      self._mobilerl_history.append(
          {"role": "assistant", "content": assistant_history}
      )
      self._mobilerl_last_image_user = self._mobilerl_current_image_user
      self._trim_mobilerl_history()

    step_data.update({
        "prompt_system": prompt_system,
        "prompt_user": prompt_user,
        "response": response,
        "thought": think,
        "action_desc": action_text,
        "conclusion": conclusion,
        "action": parsed_for_execution.raw,
        "action_raw": parsed.raw,
        "action_scaled": {
            "name": parsed_for_execution.action,
            "params": parsed_for_execution.params,
        },
        "sent_image_size": sent_image_size,
        "coordinate_scale": [exec_scale_x, exec_scale_y],
    })

    if self.output_path:
      os.makedirs(self.output_path, exist_ok=True)
      with open(
          os.path.join(self.output_path, "trace.jsonl"), "a", encoding="utf-8"
      ) as handle:
        handle.write(json.dumps({
            "goal": goal,
            "prompt_system": prompt_system,
            "prompt_user": prompt_user,
            "response": response,
            "thought": think,
            "action_raw": parsed.raw,
            "action_scaled": {
                "name": parsed_for_execution.action,
                "params": parsed_for_execution.params,
            },
            "conclusion": conclusion,
            "sent_image_size": sent_image_size,
            "coordinate_scale": [exec_scale_x, exec_scale_y],
        }) + "\n")

    if parsed_for_execution.action.strip().lower() in {
        "finished", "finish", "calluser",
    }:
      self.history.append(parsed.raw)
      return base_agent.AgentInteractionResult(True, step_data)
    if not self._execute_swipe_if_needed(parsed_for_execution):
      action = self._json_action_for(parsed_for_execution)
      step_data["action"] = action
      actuation.execute_adb_action(
          action, [], self.env.logical_screen_size, self.env.controller
      )

    time.sleep(self.wait_after_action_seconds)
    self.history.append(parsed.raw)
    done = False
    if self.prompt_style == "mobilerl_point_think":
      self._mobilerl_recent_action_keys.append(
          _action_repetition_key(parsed_for_execution)
      )
      self._mobilerl_recent_action_keys = self._mobilerl_recent_action_keys[-5:]
      done = (
          len(self._mobilerl_recent_action_keys) == 5
          and len(set(self._mobilerl_recent_action_keys)) == 1
      )
      if done:
        step_data["mobilerl_auto_stop"] = "same_action_5_times"
    return base_agent.AgentInteractionResult(done, step_data)
