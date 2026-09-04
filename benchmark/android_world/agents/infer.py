# Copyright 2024 The android_world Authors.
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

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import time
from typing import Any, Optional
from google import genai
from google.genai import types as genai_types
import numpy as np
from PIL import Image
import requests


ERROR_CALLING_LLM = 'Error calling LLM'


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
  """Converts a numpy array into a byte string for a JPEG image."""
  image = Image.fromarray(image)
  return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
  in_mem_file = io.BytesIO()
  image.save(in_mem_file, format='JPEG')
  # Reset file pointer to start
  in_mem_file.seek(0)
  img_bytes = in_mem_file.read()
  return img_bytes


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """


SAFETY_SETTINGS_BLOCK_NONE = [
  genai_types.SafetySetting(
    category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
  ),
  genai_types.SafetySetting(
    category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
  ),
  genai_types.SafetySetting(
    category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
  ),
  genai_types.SafetySetting(
    category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
  ),
]

DEFAULT_GEMINI_MODEL = 'gemini-3.1-pro-preview'
DEFAULT_CLAUDE_MODEL = 'claude-sonnet-4-6'


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Gemini GCP interface."""

  def __init__(
      self,
      model_name: str | None = None,
      max_retry: int = 2,
      temperature: float = 0.0,
      top_p: float = 0.95,
      enable_safety_checks: bool = True,
  ):
    api_key = os.environ.get('GCP_API_KEY') or os.environ.get('GEMINI_API_KEY')
    if not api_key:
      raise RuntimeError('GCP_API_KEY or GEMINI_API_KEY must be non-empty.')
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name or DEFAULT_GEMINI_MODEL
    self._base_generation_config = {
        'temperature': temperature,
        'top_p': top_p,
        'max_output_tokens': 4096,
    }
    self._default_enable_safety_checks = enable_safety_checks
    if max_retry <= 0:
      max_retry = 5
      print('Max_retry must be positive. Reset it to 5')
    self.max_retry = max_retry

  def _normalize_generation_config(self, generation_config: Any) -> dict[str, Any]:
    if generation_config is None:
      return {}
    if isinstance(generation_config, dict):
      return dict(generation_config)
    if hasattr(generation_config, 'model_dump'):
      return generation_config.model_dump(exclude_none=True)
    return {}

  def _build_generation_config(
      self,
      generation_config: Any = None,
      enable_safety_checks: bool = True,
      safety_settings: list[genai_types.SafetySetting] | None = None,
  ) -> genai_types.GenerateContentConfig:
    config_dict = dict(self._base_generation_config)
    config_dict.update(self._normalize_generation_config(generation_config))
    if safety_settings is not None:
      config_dict['safety_settings'] = safety_settings
    elif not enable_safety_checks:
      config_dict['safety_settings'] = SAFETY_SETTINGS_BLOCK_NONE
    return genai_types.GenerateContentConfig(**config_dict)

  def predict(
      self,
      text_prompt: str,
      enable_safety_checks: bool | None = None,
      generation_config: Any | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    if enable_safety_checks is None:
      enable_safety_checks = self._default_enable_safety_checks
    return self.predict_mm(
        text_prompt, [], enable_safety_checks, generation_config
    )

  def is_safe(self, raw_response):
    try:
      finish_reason = raw_response.candidates[0].finish_reason
      if finish_reason is None:
        return True
      finish_reason_value = getattr(finish_reason, 'value', finish_reason)
      return str(finish_reason_value).upper() != 'SAFETY'
    except Exception:  # pylint: disable=broad-exception-caught
      #  Assume safe if the response is None or doesn't have candidates.
      return True

  def predict_mm(
      self,
      text_prompt: str,
      images: list[np.ndarray],
      enable_safety_checks: bool | None = None,
      generation_config: Any | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    if enable_safety_checks is None:
      enable_safety_checks = self._default_enable_safety_checks
    counter = self.max_retry
    retry_delay = 2.0
    output = None
    while counter > 0:
      try:
        output = self.client.models.generate_content(
            model=self.model_name,
            contents=[text_prompt] + [Image.fromarray(image) for image in images],
            config=self._build_generation_config(
                generation_config=generation_config,
                enable_safety_checks=enable_safety_checks,
            ),
        )
        output_text = getattr(output, 'text', None)
        if output_text:
          return output_text, True, output
        if not self.is_safe(output):
          return ERROR_CALLING_LLM, False, output
        raise RuntimeError('Gemini response did not include text output.')
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print(f'Error calling LLM, will retry in {retry_delay} seconds', flush=True)
        print(e, flush=True)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2

    if (output is not None) and (not self.is_safe(output)):
      return ERROR_CALLING_LLM, False, output
    return ERROR_CALLING_LLM, None, None

  def generate(
      self,
      contents: Any,
      safety_settings: list[genai_types.SafetySetting] | None = None,
      generation_config: Any | None = None,
  ) -> tuple[str, Any]:
    """Exposes the generate_content API.

    Args:
      contents: The input to the LLM.
      safety_settings: Safety settings.
      generation_config: Generation config.

    Returns:
      The output text and the raw response.
    Raises:
      RuntimeError:
    """
    counter = self.max_retry
    retry_delay = 1.0
    response = None
    if isinstance(contents, list):
      contents = self.convert_content(contents)
    while counter > 0:
      try:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=self._build_generation_config(
                generation_config=generation_config,
                enable_safety_checks=self._default_enable_safety_checks,
                safety_settings=safety_settings,
            ),
        )
        output_text = getattr(response, 'text', None)
        if output_text:
          return output_text, response
        if not self.is_safe(response):
          raise RuntimeError('Gemini response blocked by safety settings.')
        raise RuntimeError('Gemini response did not include text output.')
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2
    raise RuntimeError(f'Error calling LLM. {response}.')

  def convert_content(
      self,
      contents: list[str | np.ndarray | Image.Image],
  ) -> list[str | Image.Image]:
    """Converts a list of contents to Gemini-compatible inputs."""
    converted: list[str | Image.Image] = []
    for item in contents:
      if isinstance(item, str):
        converted.append(item)
      elif isinstance(item, np.ndarray):
        converted.append(Image.fromarray(item))
      elif isinstance(item, Image.Image):
        converted.append(item)
    return converted


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI GPT4 wrapper."""

  RETRY_WAITING_SECONDS = 30
  MODELS_WITH_FIXED_TEMPERATURE = {'gpt-5', 'gpt-5-turbo'}

  def __init__(
      self,
      model_name: str,
      max_retry: int = 5,
      temperature: float = 0.0,
      base_url: str = 'https://api.openai.com/v1/chat/completions',
      api_key: str = None,
      use_stream: bool = False,
  ):
    if api_key:
      self.openai_api_key = api_key
    elif 'OPENAI_API_KEY' in os.environ:
      self.openai_api_key = os.environ['OPENAI_API_KEY']
    else:
      raise RuntimeError('OpenAI API key not set.')
    if max_retry <= 0:
      max_retry = 5
      print('Max_retry must be positive. Reset it to 5')
    self.max_retry = max_retry
    
    # 检查模型是否支持自定义temperature
    self.model = model_name
    if any(fixed_model in model_name.lower() for fixed_model in self.MODELS_WITH_FIXED_TEMPERATURE):
      self.temperature = None  # 不使用自定义temperature
      print(f'Note: {model_name} does not support custom temperature. Using default value (1).')
    else:
      self.temperature = temperature
    
    self.base_url = base_url
    self.use_stream = use_stream

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """文本模式预测（不包含图像）"""
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """多模态预测（包含文本和图像）"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.openai_api_key}',
    }

    payload = {
        'model': self.model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
        'max_completion_tokens': 4096,
    }
    
    # 只有在temperature不为None时才添加到payload
    if self.temperature is not None:
      payload['temperature'] = self.temperature
    
    if self.use_stream:
      payload['stream'] = True

    # Gpt-4v supports multiple images
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=300,
            stream=self.use_stream,
        )
        
        if self.use_stream and response.ok:
          # Collect all streaming chunks
          full_response = []
          for chunk in response.iter_lines():
            if chunk:
              full_response.append(chunk.decode('utf-8'))
          
          response_text = '\n'.join(full_response)
          content = self._parse_sse_response(response_text)
          
          if content:
            return (content, None, response)
          else:
            print(f'No content parsed from response: {response_text[:500]}')
        elif response.ok and 'choices' in response.json():
          return (
              response.json()['choices'][0]['message']['content'],
              None,
              response,
          )
        else:
          try:
            error_msg = response.json().get('error', {}).get('message', 'Unknown error')
            print(f'Error calling OpenAI API with error message: {error_msg}')
          except Exception:
            print(f'API returned status {response.status_code}: {response.text[:500]}')
        
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
      except Exception as e:  # pylint: disable=broad-exception-caught
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None

  def _parse_sse_response(self, response_text: str) -> str:
    """Parse SSE (Server-Sent Events) response and extract content."""
    full_content = []
    
    for line in response_text.split('\n'):
      line = line.strip()
      if not line:
        continue
      # Handle SSE format: data:{"choices":...}
      if line.startswith('data:'):
        json_str = line[5:].strip()  # Remove 'data:' prefix
        if json_str == '[DONE]':
          break
        try:
          data = json.loads(json_str)
          if 'choices' in data and len(data['choices']) > 0:
            choice = data['choices'][0]
            # Handle streaming delta format
            if 'delta' in choice and 'content' in choice['delta']:
              full_content.append(choice['delta']['content'])
            # Handle non-streaming message format
            elif 'message' in choice and 'content' in choice['message']:
              full_content.append(choice['message']['content'])
            # Handle text field directly
            elif 'text' in choice:
              full_content.append(choice['text'])
        except Exception:  # pylint: disable=broad-exception-caught
          continue
    
    return ''.join(full_content)


class GeminiProxyWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Gemini wrapper using OpenAI-compatible proxy API (e.g., Aliyun proxy)."""

  RETRY_WAITING_SECONDS = 30

  def __init__(
      self,
      model_name: str = DEFAULT_GEMINI_MODEL,
      max_retry: int = 5,
      temperature: float = 0.0,
      base_url: str = '',
      api_key: str = None,
  ):
    if api_key:
      self.api_key = api_key
    elif 'GEMINI_PROXY_API_KEY' in os.environ:
      self.api_key = os.environ['GEMINI_PROXY_API_KEY']
    else:
      raise RuntimeError('GEMINI_PROXY_API_KEY not set.')
    if max_retry <= 0:
      max_retry = 5
      print('Max_retry must be positive. Reset it to 5')
    self.max_retry = max_retry
    self.temperature = temperature
    self.model = model_name
    self.base_url = base_url

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def _parse_sse_response(self, response_text: str) -> str:
    """Parse SSE (Server-Sent Events) response and extract content."""
    import json
    full_content = []
    
    for line in response_text.split('\n'):
      line = line.strip()
      if not line:
        continue
      # Handle SSE format: data:{"choices":...}
      if line.startswith('data:'):
        json_str = line[5:].strip()  # Remove 'data:' prefix
        if json_str == '[DONE]':
          break
        try:
          data = json.loads(json_str)
          if 'choices' in data and len(data['choices']) > 0:
            choice = data['choices'][0]
            # Handle streaming delta format
            if 'delta' in choice and 'content' in choice['delta']:
              full_content.append(choice['delta']['content'])
            # Handle non-streaming message format
            elif 'message' in choice and 'content' in choice['message']:
              full_content.append(choice['message']['content'])
            # Handle text field directly
            elif 'text' in choice:
              full_content.append(choice['text'])
        except json.JSONDecodeError:
          continue
    
    return ''.join(full_content)

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.api_key}',
    }

    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
        'max_tokens': 16384,  # Increased for reasoning models like Gemini 3
        'stream': True,  # Enable streaming for SSE response
    }

    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=300,
            stream=True,
        )
        
        if response.ok:
          # Collect all streaming chunks
          full_response = []
          for chunk in response.iter_lines():
            if chunk:
              full_response.append(chunk.decode('utf-8'))
          
          response_text = '\n'.join(full_response)
          content = self._parse_sse_response(response_text)
          
          if content:
            return (content, None, response)
          else:
            print(f'No content parsed from response: {response_text[:500]}')
        else:
          print(f'API returned status {response.status_code}: {response.text[:500]}')
          # Handle rate limiting (429) with longer wait
          if response.status_code == 429:
            wait_seconds = max(wait_seconds, 60)  # Wait at least 60 seconds for rate limiting
            print(f'Rate limited (429), waiting {wait_seconds} seconds before retry...')
        
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
      except Exception as e:
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None


class ClaudeProxyWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Claude wrapper using OpenAI-compatible proxy API (e.g., MatrixLLM)."""

  RETRY_WAITING_SECONDS = 5

  def __init__(
      self,
      model_name: str = DEFAULT_CLAUDE_MODEL,
      max_retry: int = 2,
      temperature: float = 0.0,
      base_url: str = '',
      api_key: str = None,
      use_stream: bool = True,
      request_timeout: int = 60,
  ):
    claude_proxy_key = os.environ.get('CLAUDE_PROXY_API_KEY') or None
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY') or None
    if api_key:
      self.api_key = api_key
    elif base_url:
      if claude_proxy_key:
        self.api_key = claude_proxy_key
      elif anthropic_key:
        self.api_key = anthropic_key
      else:
        raise RuntimeError(
            'CLAUDE_PROXY_API_KEY or ANTHROPIC_API_KEY must be non-empty.'
        )
    elif anthropic_key:
      self.api_key = anthropic_key
    elif claude_proxy_key:
      self.api_key = claude_proxy_key
    else:
      raise RuntimeError(
          'ANTHROPIC_API_KEY or CLAUDE_PROXY_API_KEY must be non-empty.'
      )
    if max_retry <= 0:
      max_retry = 5
      print('Max_retry must be positive. Reset it to 5')
    self.max_retry = max_retry
    self.temperature = temperature
    self.model = model_name
    self.base_url = base_url
    self.use_stream = use_stream
    self.request_timeout = request_timeout

  def _predict_anthropic_messages(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    content = [{'type': 'text', 'text': text_prompt}]
    for image in images:
      content.append({
          'type': 'image',
          'source': {
              'type': 'base64',
              'media_type': 'image/jpeg',
              'data': self.encode_image(image),
          },
      })
    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': 4096,
    }
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': self.api_key,
        'anthropic-version': '2023-06-01',
    }
    response = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers=headers,
        json=payload,
        timeout=self.request_timeout,
    )
    if not response.ok:
      try:
        error = response.json().get('error', {})
        message = error.get('message', response.text[:500])
      except Exception:  # pylint: disable=broad-exception-caught
        message = response.text[:500]
      raise requests.HTTPError(
          f'{response.status_code} Claude Messages API: {message}',
          response=response,
      )
    data = response.json()
    text_parts = [
        item.get('text', '')
        for item in data.get('content', [])
        if item.get('type') == 'text'
    ]
    return ''.join(text_parts), None, response

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def _parse_sse_response(self, response_text: str) -> str:
    """Parse SSE (Server-Sent Events) response and extract content."""
    import json
    full_content = []
    
    for line in response_text.split('\n'):
      line = line.strip()
      if not line:
        continue
      # Handle SSE format: data:{"choices":...}
      if line.startswith('data:'):
        json_str = line[5:].strip()  # Remove 'data:' prefix
        if json_str == '[DONE]':
          break
        try:
          data = json.loads(json_str)
          if 'choices' in data and len(data['choices']) > 0:
            choice = data['choices'][0]
            # Handle streaming delta format
            if 'delta' in choice and 'content' in choice['delta']:
              full_content.append(choice['delta']['content'])
            # Handle non-streaming message format
            elif 'message' in choice and 'content' in choice['message']:
              full_content.append(choice['message']['content'])
            # Handle text field directly
            elif 'text' in choice:
              full_content.append(choice['text'])
        except json.JSONDecodeError:
          continue
    
    return ''.join(full_content)

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    if not self.base_url:
      counter = self.max_retry
      wait_seconds = self.RETRY_WAITING_SECONDS
      while counter > 0:
        try:
          return self._predict_anthropic_messages(text_prompt, images)
        except Exception as e:  # pylint: disable=broad-exception-caught
          time.sleep(wait_seconds)
          wait_seconds *= 2
          counter -= 1
          print('Error calling Claude Messages API, will retry soon...')
          print(e, flush=True)
      return ERROR_CALLING_LLM, None, None

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.api_key}',
    }

    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
        'max_tokens': 4096,
    }
    
    if self.use_stream:
      payload['stream'] = True

    # Claude supports multiple images in the content list
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.request_timeout,
            stream=self.use_stream,
        )
        
        if self.use_stream and response.ok:
          # Collect all streaming chunks
          full_response = []
          for chunk in response.iter_lines():
            if chunk:
              full_response.append(chunk.decode('utf-8'))
          
          response_text = '\n'.join(full_response)
          content = self._parse_sse_response(response_text)
          
          if content:
            return (content, None, response)
          else:
            print(f'No content parsed from response: {response_text[:500]}')
        elif response.ok and 'choices' in response.json():
          return (
              response.json()['choices'][0]['message']['content'],
              None,
              response,
          )
        else:
          try:
            error_msg = response.json().get('error', {}).get('message', 'Unknown error')
            print(f'Error calling Claude API with error message: {error_msg}')
          except:
            print(f'API returned status {response.status_code}: {response.text[:500]}')
        
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling Claude LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None
