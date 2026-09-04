"""Screen transcription utility using Haiku for pre-transcribing screens."""

import base64
import os
from io import BytesIO
from typing import Optional

import litellm
import numpy as np
from PIL import Image


def _normalize_openai_base_url(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return None
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _completion_kwargs(
    model: str,
    messages: list[dict],
    max_tokens: int,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict:
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    normalized_base_url = _normalize_openai_base_url(base_url)
    if api_key and (api_key.upper() != "EMPTY" or normalized_base_url):
        kwargs["api_key"] = api_key
    if normalized_base_url:
        kwargs["api_base"] = normalized_base_url
    elif model.startswith("openai/"):
        env_base_url = (
            os.environ.get("AUTODEV_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        normalized_base_url = _normalize_openai_base_url(env_base_url)
        if normalized_base_url:
            kwargs["api_base"] = normalized_base_url
    return kwargs


def transcribe_screen(
    screenshot: np.ndarray,
    model: str = "openai/gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """Transcribe text content from a screenshot using Haiku.
    
    NOTE: This function is currently not used in the agent workflow.
    It is kept for potential future use or reference.
    
    Args:
        screenshot: Screenshot as numpy array
        model: Model to use for transcription (default: Haiku)
    
    Returns:
        Transcribed text content, or None if transcription fails
    """
    try:
        # Convert numpy array to PIL Image
        if screenshot.dtype != np.uint8:
            screenshot = (screenshot * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(screenshot)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create messages for transcription
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe all text visible on this screen. Include all UI elements, labels, buttons, text fields, and any other readable content and mention them as icons , buttons, text fields, etc. Be thorough and complete."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ]
        
        # Call Haiku for transcription
        response = litellm.completion(
            **_completion_kwargs(model, messages, 2000, api_key, base_url)
        )
        
        transcription = response.choices[0].message.content
        return transcription.strip() if transcription else None
        
    except Exception as e:
        print(f"⚠️  Transcription error: {e}")
        return None


def get_ui_elements(
    screenshot: np.ndarray,
    focus: Optional[str] = None,
    model: str = "openai/gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """List and transcribe content from a screenshot.
    
    Use this when you need to find UI elements like buttons, icons, text fields, etc.
    This is focused on UI elements, not data extraction.
    
    Args:
        screenshot: Screenshot as numpy array
        focus: Optional focus area (e.g., "action bar", "bottom navigation", "menu items")
        model: Model to use (default: Haiku)
    
    Returns:
        Description of UI elements, buttons, icons, or None if fails
    """
    try:
        # Convert numpy array to PIL Image
        if screenshot.dtype != np.uint8:
            screenshot = (screenshot * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(screenshot)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        
        focus_text = f" Focus on: {focus}." if focus else ""
        
        # Create messages for UI element detection
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Analyze this screenshot and identify all UI elements, buttons, icons, and interactive elements.{focus_text}

                        IMPORTANT: Focus ONLY on UI elements that can be interacted with:
                        - Buttons (with labels or icons)
                        - Icons (especially action bar icons, menu icons, navigation icons)
                        - Text fields (input fields, search bars)
                        - Interactive elements (checkboxes, radio buttons, switches, sliders)
                        - Navigation elements (tabs, menu items, back buttons)
                        - Action bar buttons (especially rename, edit, delete, save icons)

                        For each element, describe:
                        - What it is (button, icon, text field, etc.)
                        - Its label/text if visible
                        - Its approximate location (top, bottom, left, right, center)
                        - Its function if clear from context (e.g., "AI or pencil icon button - likely rename function")

                        DO NOT include:
                        - Regular text content (use extract_data for that)
                        - Data from lists (use extract_data for that)
                        - File content (use extract_data for that)

                        Return a clear, structured list of all interactive UI elements visible on screen."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ]
        
        # Call model for UI element detection
        response = litellm.completion(
            **_completion_kwargs(model, messages, 2000, api_key, base_url)
        )
        
        ui_elements = response.choices[0].message.content
        return ui_elements.strip() if ui_elements else None
        
    except Exception as e:
        print(f"⚠️  UI elements detection error: {e}")
        return None


def extract_data_from_screen(
    screenshot: np.ndarray, 
    extraction_instruction: str,
    model: str = "openai/gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """Extract and list visible content from a screenshot based on extraction instructions.
    
    This function lists/transcribes visible content without filtering. The planner handles filtering logic.
    
    Args:
        screenshot: Screenshot as numpy array
        extraction_instruction: Instruction on what to list/extract (e.g., 
            "list all recipes on screen",
            "extract whole content of file",
            "list all items in file that can be seen on screen",
            "list all activities on screen")
        model: Model to use for extraction (default: Haiku)
    
    Returns:
        Listed/transcribed content as structured text, or None if extraction fails
    """
    try:
        # Convert numpy array to PIL Image
        if screenshot.dtype != np.uint8:
            screenshot = (screenshot * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(screenshot)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create extraction prompt focused on listing visible content
        extraction_prompt = f"""List and transcribe all visible content based on the instruction. DO NOT filter by criteria - just list everything that is visible.

EXTRACTION INSTRUCTION: {extraction_instruction}

IMPORTANT RULES:
1. List ALL visible items/content that match the instruction - do not filter by criteria (dates, categories, etc.)
2. If instruction says "list all recipes" → List ALL recipes visible on screen with their details (name, prep time, ingredients, etc.)
3. If instruction says "extract whole content of file" → Transcribe ALL text content visible in the file
4. If instruction says "list all items in file" → List ALL items visible in the file
5. If instruction says "list all activities" → List ALL activities visible with their details
6. Include all visible details for each item (names, descriptions, dates, categories, etc.)
7. Return data in a structured format (JSON array, bullet list, or structured text)
8. DO NOT include UI elements, buttons, or navigation - only the actual content/data
9. Be complete - list everything visible that matches the instruction
10. **CRITICAL - End of List Detection**: Analyze if the end of the list/screen is visible:
    - If the end of list is clearly visible, add a note at the end: "END OF LIST REACHED - No more items below"

Return the listed content in a clear, structured format that can be easily parsed."""
        
        # Create messages for extraction
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": extraction_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ]
        
        # Call model for extraction
        response = litellm.completion(
            **_completion_kwargs(model, messages, 2000, api_key, base_url)
        )
        
        extracted_data = response.choices[0].message.content
        return extracted_data.strip() if extracted_data else None
        
    except Exception as e:
        print(f"⚠️  Data extraction error: {e}")
        return None
