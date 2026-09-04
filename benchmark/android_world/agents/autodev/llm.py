"""Simple LiteLLM wrapper for AutoDev agent."""

import base64
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, TypedDict, Union

import litellm
import numpy as np
from PIL import Image

from .scratchpad import Scratchpad
from .todo_list import TodoList


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is optional at import time.
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def _normalize_openai_base_url(base_url: Optional[str]) -> Optional[str]:
    """Normalize OpenAI-compatible endpoints for LiteLLM."""
    if not base_url:
        return None
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


class ToolCall(TypedDict):
    """Type for tool call structure."""

    id: str
    type: str
    function: Dict[str, Any]  # Contains 'name' and 'arguments'


class ChatResponse(TypedDict):
    """Response from chat method containing content and tool calls."""

    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]


class CacheStats(TypedDict):
    """Statistics for prompt cache usage."""

    cache_read_tokens: int
    cache_creation_tokens: int
    total_requests: int


class AutoDevLLM:
    """Simple LLM wrapper with conversation history and tool calls."""

    def __init__(
        self,
        model: str = "gpt-4",
        system_prompt: str = "",
        todo_list_enabled: bool = False,
        scratchpad: Optional[Scratchpad] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: float = 2.0,
    ) -> None:
        self.is_anthropic: bool = model.startswith("anthropic")
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        if system_prompt:
            if self.is_anthropic:
                self.messages.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt,
                                "cache_control": {
                                    "type": "ephemeral",
                                    "ttl": "5m"},
                            }
                        ],
                    }
                )
            else:
                self.messages.append({"role": "system", "content": system_prompt})
        self.todo_list_enabled = todo_list_enabled
        self.todo_list = TodoList()
        self.scratchpad = scratchpad if scratchpad is not None else Scratchpad()
        self.api_key = api_key
        self.base_url = _normalize_openai_base_url(base_url)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Cache statistics tracking
        self._cache_read_tokens: int = 0
        self._cache_creation_tokens: int = 0
        self._total_requests: int = 0

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable (transient failures).

        IMPORTANT: 400 Bad Request errors are NOT retryable - they indicate
        malformed requests (e.g., missing tool_result blocks).
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # 400 Bad Request is NOT retryable - it's a malformed request
        if "400" in error_str or "bad request" in error_str:
            return False

        # Check for retryable error patterns
        retryable_patterns = [
            "503",  # Service unavailable
            "429",  # Rate limit
            "500",  # Internal server error
            "502",  # Bad gateway
            "504",  # Gateway timeout
            "overloaded",
            "unavailable",
            "timeout",
            "rate limit",
            "too many requests",
            "connection",
            "network",
            "internal server error",
            "bad gateway",
            "service unavailable",
        ]

        # Check error string
        if any(pattern in error_str for pattern in retryable_patterns):
            return True

        # Check for specific LiteLLM exception types
        if hasattr(litellm, "exceptions"):
            if isinstance(error, litellm.exceptions.BadRequestError):
                # BadRequestError (400) is NOT retryable
                return False
            if isinstance(error, litellm.exceptions.InternalServerError):
                return True
            if isinstance(error, litellm.exceptions.RateLimitError):
                return True
            if isinstance(error, litellm.exceptions.APIConnectionError):
                return True
            if isinstance(error, litellm.exceptions.APIError):
                # Check status code if available
                if hasattr(error, "status_code"):
                    if error.status_code == 400:
                        return False  # 400 is NOT retryable
                    if error.status_code in [429, 500, 502, 503, 504]:
                        return True

        return False

    def chat(
        self,
        user_message: Optional[str],
        screenshot: np.ndarray,
        tools: Optional[List[Dict[str, Any]]] = None,
        transcription: Optional[str] = None,
    ) -> ChatResponse:
        """Send a message and get response, optionally with tool calls.
        
        Args:
            user_message: Optional user message/goal
            screenshot: Screenshot as numpy array
            tools: Optional list of tools available
            transcription: Optional pre-transcribed text from the screen
        """
        # Prepare user message content
        tools_for_call = list(tools) if tools is not None else []

        parts: List[Dict[str, Any]] = []

        if user_message:  # only add if not None/empty
            parts.append({"type": "text", "text": user_message})

        # Add transcription if available
        if transcription:
            parts.append({
                "type": "text",
                "text": f"<screen_transcription>\n{transcription}\n</screen_transcription>"
            })

        # Add cache_control to the last text block BEFORE the image for Anthropic models
        # This ensures the cached prefix doesn't include the image, which gets removed
        # from history later via _remove_image_blocks_from_history(). If we cached
        # after the image, the cache would be invalidated when the image is stripped.
        if self.is_anthropic:
            # Remove cache_control from previous user messages
            for msg in self.messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                    for part in msg["content"]:
                        if "cache_control" in part:
                            del part["cache_control"]

            # Add cache_control to the last text part before image (if any)
            if parts:
                parts[-1]["cache_control"] = {"type": "ephemeral"}

        # Add the image (not part of cached prefix since it gets removed from history)
        image_data = self._encode_image(screenshot)
        parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}",
                },
            }
        )

        # Add remaining text parts after image
        if self.todo_list_enabled:
            parts.append({"type": "text", "text": self.todo_list.get_system_reminder()})
        parts.append({"type": "text", "text": self.scratchpad.get_system_reminder()})

        self.messages.append({"role": "user", "content": parts})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "num_retries": self.max_retries,  # Use LiteLLM's built-in retry mechanism
        }
        if self.api_key and (self.api_key.upper() != "EMPTY" or self.base_url):
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        elif self.model.startswith("openai/"):
            env_base_url = (
                os.environ.get("AUTODEV_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE")
            )
            normalized_base_url = _normalize_openai_base_url(env_base_url)
            if normalized_base_url:
                kwargs["api_base"] = normalized_base_url
        if self.todo_list_enabled:
            tools_for_call.append(TodoList.get_tool())
        # Always add scratchpad tools
        tools_for_call.append(Scratchpad.get_create_tool())
        tools_for_call.append(Scratchpad.get_fetch_tool())
        if tools_for_call:
            wrapped_tools = []
            for t in tools_for_call:
                wrapped_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get(
                                "parameters", {"type": "object", "properties": {}}
                            ),
                        },
                    }
                )
            tools_for_call = wrapped_tools  # replace original
            kwargs["tools"] = tools_for_call
            kwargs["tool_choice"] = "auto"

        retry_count = 0
        current_delay = self.retry_delay
        last_exception = None

        while retry_count <= self.max_retries:
            try:
                response = litellm.completion(**kwargs)
                assistant_message = response.choices[0].message

                # Track cache statistics from response (Anthropic-specific)
                self._total_requests += 1
                self._log_cache_stats(response)

                content = assistant_message.content

                tool_calls = getattr(assistant_message, "tool_calls", None)

                self.messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )

                self._remove_image_blocks_from_history()

                return ChatResponse(content=content, tool_calls=tool_calls)

            except Exception as e:
                last_exception = e

                if not self._is_retryable_error(e):
                    raise

                if retry_count >= self.max_retries:
                    print(
                        f"❌ LLM API error (max retries {self.max_retries} reached): {e}"
                    )
                    raise RuntimeError(
                        f"LLM API call failed after {self.max_retries} retries. Last error: {last_exception}"
                    ) from last_exception

                retry_count += 1
                print(
                    f"⚠️  LLM API error (attempt {retry_count}/{self.max_retries}): {type(e).__name__}: {str(e)[:100]}"
                )
                print(f"   Retrying in {current_delay:.1f} seconds...")
                time.sleep(current_delay)

                current_delay = min(current_delay * 2, 60.0)

        raise RuntimeError(
            f"LLM API call failed after {self.max_retries} retries. Last error: {last_exception}"
        ) from last_exception

    def add_tool_result(self, tool_call_id: str, result: str) -> None:
        """Add tool execution result to conversation."""
        self.messages.append(
            {
                "role": "tool",
                "type": "function_call_output",
                "tool_call_id": tool_call_id,
                "content": result,
            }
        )

    def _remove_image_blocks_from_history(self) -> None:
        """Remove image blocks from message history to save memory."""
        for message in self.messages:
            if message.get("role") == "user" and isinstance(
                message.get("content"), list
            ):
                # Filter out image_url blocks, keep only text blocks
                message["content"] = [
                    part
                    for part in message["content"]
                    if part.get("type") != "image_url"
                ]

    def _log_cache_stats(self, response: Any) -> None:
        """Extract and log cache statistics from the LLM response.

        For Anthropic models, the response includes:
        - cache_creation_input_tokens: tokens written to cache
        - cache_read_input_tokens: tokens read from cache
        """
        if not self.is_anthropic:
            return

        usage = getattr(response, "usage", None)
        if not usage:
            return

        # Extract cache tokens from usage (LiteLLM passes these through)
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

        self._cache_read_tokens += cache_read
        self._cache_creation_tokens += cache_creation

        # Log cache stats for this request
        if cache_read > 0 or cache_creation > 0:
            print(
                f"📦 Cache: read={cache_read:,} tokens, "
                f"created={cache_creation:,} tokens"
            )
        else:
            print("📦 Cache: no cache hit (tokens not in cache or caching disabled)")

    def get_cache_stats(self) -> CacheStats:
        """Get cumulative cache statistics for this session.

        Returns:
            CacheStats with total cache_read_tokens, cache_creation_tokens,
            and total_requests.
        """
        return CacheStats(
            cache_read_tokens=self._cache_read_tokens,
            cache_creation_tokens=self._cache_creation_tokens,
            total_requests=self._total_requests,
        )

    def print_cache_summary(self) -> None:
        """Print a summary of cache usage for the session."""
        stats = self.get_cache_stats()
        print("\n" + "=" * 50)
        print("📊 Prompt Cache Summary")
        print("=" * 50)
        print(f"Total requests:        {stats['total_requests']}")
        print(f"Total tokens read:     {stats['cache_read_tokens']:,}")
        print(f"Total tokens created:  {stats['cache_creation_tokens']:,}")
        if stats["cache_read_tokens"] > 0:
            # Anthropic charges 10% for cache reads vs 100% for cache creation
            savings_pct = (stats["cache_read_tokens"] * 0.9) / (
                stats["cache_read_tokens"] + stats["cache_creation_tokens"]
            ) * 100 if (stats["cache_read_tokens"] + stats["cache_creation_tokens"]) > 0 else 0
            print(f"Estimated savings:     ~{savings_pct:.1f}% on cached tokens")
        print("=" * 50 + "\n")

    def clear_history(self) -> None:
        """Clear conversation history but keep system prompt."""
        system_msg = None
        if self.messages and self.messages[0]["role"] == "system":
            system_msg = self.messages[0]
        self.messages = [system_msg] if system_msg else []

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode numpy array image to base64 string for LLM API.

        Args:
            image: Image as numpy array

        Returns:
            Base64 encoded image string
        """
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        # Convert to bytes
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # Encode to base64
        return base64.b64encode(image_bytes).decode("utf-8")
