"""Functions to create Android automation actions.

These functions are designed to be used as tool calls by an LLM to generate
properly formatted actions for various Android interactions.
"""

from typing import Optional

from android_world.env.json_action import JSONAction

# Global scale factor for coordinate scaling
SCALE = 0.4
TARGET_WIDTH: Optional[int] = None
TARGET_HEIGHT: Optional[int] = None
LOGICAL_WIDTH: Optional[int] = None
LOGICAL_HEIGHT: Optional[int] = None


def configure_screen(
    *,
    scale: float,
    target_width: int,
    target_height: int,
    logical_width: int,
    logical_height: int,
) -> None:
    """Configure coordinate conversion for the current emulator screen."""
    global SCALE, TARGET_WIDTH, TARGET_HEIGHT, LOGICAL_WIDTH, LOGICAL_HEIGHT
    SCALE = scale
    TARGET_WIDTH = target_width
    TARGET_HEIGHT = target_height
    LOGICAL_WIDTH = logical_width
    LOGICAL_HEIGHT = logical_height


def _to_int(value: int | float | str) -> int:
    return int(round(float(value)))


def _clamp(value: int, limit: Optional[int]) -> int:
    if limit is None or limit <= 0:
        return value
    return max(0, min(value, limit - 1))


def _scale_axis(value: int, target_limit: Optional[int], logical_limit: Optional[int]) -> int:
    """Scale one axis, accepting occasional already-logical coordinates.

    The executor prompt asks for resized-screenshot coordinates. Some providers
    occasionally mix in full-device coordinates on one axis, so each axis is
    interpreted independently and clamped to the logical device bounds.
    """
    if (
        target_limit is not None
        and logical_limit is not None
        and target_limit > 0
        and logical_limit > 0
        and target_limit <= value < logical_limit
    ):
        return _clamp(value, logical_limit)
    return _clamp(_to_int(value / SCALE), logical_limit)


def _scale_point(x: int | float | str, y: int | float | str) -> tuple[int, int]:
    raw_x = _to_int(x)
    raw_y = _to_int(y)
    return (
        _scale_axis(raw_x, TARGET_WIDTH, LOGICAL_WIDTH),
        _scale_axis(raw_y, TARGET_HEIGHT, LOGICAL_HEIGHT),
    )


def _scale_optional_point(
    x: Optional[int | float | str], y: Optional[int | float | str]
) -> tuple[Optional[int], Optional[int]]:
    if x is not None and y is not None:
        return _scale_point(x, y)
    scaled_x = _scale_axis(_to_int(x), TARGET_WIDTH, LOGICAL_WIDTH) if x is not None else None
    scaled_y = _scale_axis(_to_int(y), TARGET_HEIGHT, LOGICAL_HEIGHT) if y is not None else None
    return scaled_x, scaled_y


def _scale_path(
    start_x: int | float | str,
    start_y: int | float | str,
    end_x: int | float | str,
    end_y: int | float | str,
) -> tuple[int, int, int, int]:
    raw_start_x, raw_start_y, raw_end_x, raw_end_y = (
        _to_int(v) for v in (start_x, start_y, end_x, end_y)
    )
    return (
        _scale_axis(raw_start_x, TARGET_WIDTH, LOGICAL_WIDTH),
        _scale_axis(raw_start_y, TARGET_HEIGHT, LOGICAL_HEIGHT),
        _scale_axis(raw_end_x, TARGET_WIDTH, LOGICAL_WIDTH),
        _scale_axis(raw_end_y, TARGET_HEIGHT, LOGICAL_HEIGHT),
    )


def click(x: int, y: int) -> JSONAction:
    """Create a click action at the specified coordinates.

    Args:
        x: The x-coordinate of the click position on the screen.
        y: The y-coordinate of the click position on the screen.

    Example:
        >>> click(100, 200)
        # Performs a click at position (100, 200)
    """
    scaled_x, scaled_y = _scale_point(x, y)
    return JSONAction(action_type="click", x=scaled_x, y=scaled_y)


def double_tap(x: int, y: int) -> JSONAction:
    """Create a double-tap action at the specified coordinates.
    You may use this to perform a zoom in gesture on a map or an image.

    Args:
        x: The x-coordinate of the double-tap position on the screen.
        y: The y-coordinate of the double-tap position on the screen.

    Example:
        >>> double_tap(150, 300)
        # Performs a double-tap at position (150, 300)
    """
    scaled_x, scaled_y = _scale_point(x, y)
    return JSONAction(action_type="double_tap", x=scaled_x, y=scaled_y)


def long_press(x: int, y: int) -> JSONAction:
    """Create a long-press action at the specified coordinates.

    Args:
        x: The x-coordinate of the long-press position on the screen.
        y: The y-coordinate of the long-press position on the screen.

    Example:
        >>> long_press(200, 400)
        # Performs a long-press at position (200, 400)
    """
    scaled_x, scaled_y = _scale_point(x, y)
    return JSONAction(action_type="long_press", x=scaled_x, y=scaled_y)


def scroll(
    direction: str, x: Optional[int] = None, y: Optional[int] = None
) -> JSONAction:
    """Create a scroll action in the specified direction.
    In order to scroll a tiny bit, use swipe in the inverse direction instead.
    Args:
        direction: The scroll direction. Must be one of: 'up', 'down', 'left', 'right'.
        x: Optional x-coordinate for the scroll origin. If not provided, scrolls from center.
        y: Optional y-coordinate for the scroll origin. If not provided, scrolls from center.





    Raises:
        ValueError: If direction is not one of the valid scroll directions.

    Example:
        >>> scroll('down')
        # Scrolls down from center of screen

        >>> scroll('up', x=300, y=500)
        # Scrolls up from position (300, 500)
    """
    valid_directions = ("up", "down", "left", "right")
    if direction not in valid_directions:
        raise ValueError(
            f"Direction must be one of {valid_directions}, got: {direction}"
        )

    scaled_x, scaled_y = _scale_optional_point(x, y)
    return JSONAction(action_type="scroll", direction=direction, x=scaled_x, y=scaled_y)


def swipe(
    direction: str, x: Optional[int] = None, y: Optional[int] = None
    ) -> JSONAction:
    """Create a swipe action in the specified direction.
    You may scroll tiny amounts by using swipe instead.
    For example, to scroll down, swipe the screen up from the upper half of the screen.
    Args:
        direction: The swipe direction. Must be one of: 'up', 'down', 'left', 'right'.
        x: Optional x-coordinate for the swipe origin. If not provided, swipes from center.
        y: Optional y-coordinate for the swipe origin. If not provided, swipes from center.

    Raises:
        ValueError: If direction is not one of the valid swipe directions.

    Example:
        >>> swipe('left')
        # Swipes left from center of screen

        >>> swipe('right', x=100, y=200)
        # Swipes right from position (100, 200)
    """
    valid_directions = ("up", "down", "left", "right")
    if direction not in valid_directions:
        raise ValueError(
            f"Direction must be one of {valid_directions}, got: {direction}"
        )
    
    scaled_x, scaled_y = _scale_optional_point(x, y)
    return JSONAction(action_type="swipe", direction=direction, x=scaled_x, y=scaled_y)


def swipe_coords(
    start_x: int, start_y: int, end_x: int, end_y: int
) -> JSONAction:
    """Create a swipe action with explicit start and end coordinates.
    
    This gives precise control over the swipe path. The coordinates are in logical
    screen space and will be scaled appropriately.
    
    Args:
        start_x: X coordinate where the swipe starts
        start_y: Y coordinate where the swipe starts
        end_x: X coordinate where the swipe ends
        end_y: Y coordinate where the swipe ends
    
    Example:
        >>> swipe_coords(100, 1200, 800, 1200)
        # Swipes horizontally from (100, 1200) to (800, 1200)
    """
    scaled_start_x, scaled_start_y, scaled_end_x, scaled_end_y = _scale_path(
        start_x, start_y, end_x, end_y
    )
    
    action = JSONAction(
        action_type="swipe", 
        x=scaled_start_x,
        y=scaled_start_y,
        x_=scaled_end_x,
        y_=scaled_end_y,
    )
    return action


def input_text(
    text: str,
    x: Optional[int] = None,
    y: Optional[int] = None,
    clear_text: bool = False,
) -> JSONAction:
    """Create an input text action to type text into a field.

    Args:
        text: The text string to input.
        x: Optional x-coordinate of the text field to click before typing.
        y: Optional y-coordinate of the text field to click before typing.
        clear_text: Whether to clear existing text in the field before typing.
                   Defaults to False.

    Example:
        >>> input_text("Hello World")
        # Types "Hello World" into the currently focused field

        >>> input_text("user@example.com", x=200, y=300, clear_text=True)
        # Clicks at (200, 300), clears the field, then types the email
    """
    scaled_x, scaled_y = _scale_optional_point(x, y)
    return JSONAction(
        action_type="input_text",
        text=text,
        x=scaled_x,
        y=scaled_y,
        clear_text=clear_text if clear_text else None,
    )


def keyboard_enter() -> JSONAction:
    """Create a keyboard enter/return key press action.

    Example:
        >>> keyboard_enter()
        # Presses the enter key
    """
    return JSONAction(action_type="keyboard_enter")


def navigate_back() -> JSONAction:
    """Create a navigate back action (Android back button).

    Example:
        >>> navigate_back()
        # Presses the Android back button
    """
    return JSONAction(action_type="navigate_back")


def navigate_home() -> JSONAction:
    """Create a navigate home action (Android home button).

    Example:
        >>> navigate_home()
        # Presses the Android home button to go to home screen
    """
    return JSONAction(action_type="navigate_home")


def open_app(app_name: str) -> JSONAction:
    """Create an open app action to launch a specific application.

    Args:
        app_name: The name of the application to open (case-insensitive).

    Example:
        >>> open_app("Settings")
        # Opens the Settings app

        >>> open_app("Google Chrome")
        # Opens Google Chrome browser
    """
    return JSONAction(action_type="open_app", app_name=app_name)


def wait() -> JSONAction:
    """Create a wait action to pause execution.

    This is useful for waiting for UI elements to load or animations to complete.

    Example:
        >>> wait()
        # Pauses execution temporarily
    """
    return JSONAction(action_type="wait")


def tap(x: int, y: int) -> JSONAction:
    """Alias for click() - create a tap action at the specified coordinates.

    Args:
        x: The x-coordinate of the tap position.
        y: The y-coordinate of the tap position.

    Example:
        >>> tap(250, 500)
        # Taps at position (250, 500)
    """
    return click(x, y)


def type_text(text: str, clear_first: bool = False) -> JSONAction:
    """Simplified text input without coordinates.

    Use this when the text field is already focused or selected.

    Args:
        text: The text to type.
        clear_first: Whether to clear existing text before typing.

    Example:
        >>> type_text("Hello")
        # Types "Hello" into the currently focused field
    """
    return input_text(text, clear_text=clear_first)


def swipe_up(x: Optional[int] = None, y: Optional[int] = None) -> JSONAction:
    """Convenience function for swiping up.

    Args:
        x: Optional x-coordinate for swipe origin.
        y: Optional y-coordinate for swipe origin.
    """
    return swipe("up", x, y)


def swipe_down(x: Optional[int] = None, y: Optional[int] = None) -> JSONAction:
    """Convenience function for swiping down.

    Args:
        x: Optional x-coordinate for swipe origin.
        y: Optional y-coordinate for swipe origin.
    """
    return swipe("down", x, y)


def extracted_data(data: str):
    """
    This function is a callback to return extracted data.
    If you have been tasked with extracting data from a screen, you may use this function to return that data.
    Please send the data in a structured format that can be effectively used.
    """


def transcribe_screen() -> str:
    """Transcribe all text and UI elements visible on the current screen.
    
    This tool provides a complete transcription of the current screen, including:
    - All visible text content
    - UI elements (buttons, icons, text fields, etc.)
    - Labels and descriptions
    - Any other readable content
    
    Use this when you need to:
    - Read file content
    - Extract list items
    - Read form fields, search results, or any text on screen
    - Find UI elements and their labels (buttons, icons, text fields)
    - Understand the current screen state
    
    Returns:
        A complete transcription of the current screen as a string.
    """
    pass


def report(notes: str):
    """Reports what it has achieved of current conversation objectives.

    This function serves as a callback or notification mechanism to indicate
    the overall status and summary of the performed actions on the originally stated goal.

    It will help the planner in setting future objectives, and tracking the status
    of the goals.

    Use this when you have achieved what you set out to achieve or don't have a way to proceed.

    Args:
        notes: Succint notes on everything that was done and observed in this flow.

    """
