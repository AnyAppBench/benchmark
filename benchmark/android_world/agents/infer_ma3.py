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
import os
import time
from typing import Any, Optional
import numpy as np
from PIL import Image
from openai import OpenAI
from qwen_vl_utils import smart_resize
from io import BytesIO
import base64
import requests

from android_world.agents import episode_exceptions

ERROR_CALLING_LLM = 'Error calling LLM'

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG") 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def _as_pil_image(image):
  if isinstance(image, Image.Image):
    return image.copy()
  if isinstance(image, (str, bytes, os.PathLike)):
    return Image.open(image)
  return Image.fromarray(np.asarray(image))

def _combine_images(images):
  pil_images = [_as_pil_image(image).convert("RGB") for image in images]
  if len(pil_images) <= 1:
    return pil_images[0] if pil_images else None
  padding = 16
  width = max(image.width for image in pil_images)
  height = sum(image.height for image in pil_images) + padding * (len(pil_images) - 1)
  combined = Image.new("RGB", (width, height), (255, 255, 255))
  y_offset = 0
  for image in pil_images:
    combined.paste(image, (0, y_offset))
    y_offset += image.height + padding
  return combined

def image_to_base64(image_path):
  dummy_image = _as_pil_image(image_path)
  MIN_PIXELS=3136
  MAX_PIXELS=10035200
  resized_height, resized_width  = smart_resize(dummy_image.height,
      dummy_image.width,
      factor=28,
      min_pixels=MIN_PIXELS,
      max_pixels=MAX_PIXELS,)
  dummy_image = dummy_image.resize((resized_width, resized_height))
  return f"data:image/png;base64,{pil_to_base64(dummy_image)}"


def image_to_base64_qwen3vl(image_path):
  dummy_image = _as_pil_image(image_path)
  MIN_PIXELS=3136
  MAX_PIXELS=10035200
  resized_height, resized_width  = smart_resize(dummy_image.height,
      dummy_image.width,
      factor=32,
      min_pixels=MIN_PIXELS,
      max_pixels=MAX_PIXELS,)
  dummy_image = dummy_image.resize((resized_width, resized_height))
  return f"data:image/png;base64,{pil_to_base64(dummy_image)}"

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
      self, text_prompt: str, images: list[np.ndarray], messages = None
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """

class GUIOwlWrapper(LlmWrapper, MultimodalLlmWrapper):

    RETRY_WAITING_SECONDS = 20

    def __init__(
            self,
            api_key: str,
            base_url: str,
            model_name: str,
            max_retry: int = 10,
            temperature: float = 0.0,
    ):
        if max_retry <= 0:
            max_retry = 10
            print('Max_retry must be positive. Reset it to 3')
        self.max_retry = min(max_retry, 10)
        self.temperature = temperature
        self.model = model_name
        # 300 s gives a 9B model plenty of time to generate long reasoning
        # chains.  The original 30 s caused timeouts on complex prompts.
        self.bot = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300
        )

    def convert_messages_format_to_openaiurl(self, messages):
      converted_messages = []
      for message in messages:
          new_content = []
          for item in message['content']:
              if list(item.keys())[0] == 'text':
                  new_content.append({'type': 'text', 'text': item['text']})
              elif list(item.keys())[0] == 'image':
                new_content.append({'type': 'image_url', 'image_url': {'url': image_to_base64(item['image'])}})
          converted_messages.append({'role': message['role'], 'content': new_content})

      return converted_messages
    
    def predict(
            self,
            text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
            self, text_prompt: str, images: list[np.ndarray], messages = None
    ) -> tuple[str, Optional[bool], Any]:
        # import pdb; pdb.set_trace()
        if messages is None:
          payload = [
              {
                  "role": "user",
                  "content": [
                      {"text": text_prompt},
                  ]
              }
          ]

          image_payload = images
          if len(images) > 1:
            image_payload = [_combine_images(images)]
          for image in image_payload:
            payload[0]['content'].append({
                'image': image
            })
        else:
          payload = messages
            
        payload = self.convert_messages_format_to_openaiurl(payload)

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        last_error = None
        while counter > 0:
            try:
              chat_completion_from_url = self.bot.chat.completions.create(model=self.model, messages=payload, max_tokens=4096, temperature=self.temperature)
              # print('messages: ', messages)
              # print('chat_completion_from_url: ', chat_completion_from_url)
              
              return (chat_completion_from_url.choices[0].message.content, payload, chat_completion_from_url)
            except Exception as error:
                last_error = error
                time.sleep(wait_seconds)
                wait_seconds *= 1
                counter -= 1
                print('Error calling LLM, will retry soon...')
                print(error)
        raise episode_exceptions.ModelEndpointError(
            f"GUI-Owl endpoint failed after {self.max_retry} attempts."
        ) from last_error
    

class Qwen3VLWrapper(LlmWrapper, MultimodalLlmWrapper):

    RETRY_WAITING_SECONDS = 20

    def __init__(
            self,
            api_key: str,
            base_url: str,
            model_name: str,
            max_retry: int = 10,
            temperature: float = 0.0,
    ):
        if max_retry <= 0:
            max_retry = 10
            print('Max_retry must be positive. Reset it to 3')
        self.max_retry = min(max_retry, 10)
        self.temperature = temperature
        self.model = model_name
        # 300 s gives a 9B model plenty of time to generate long reasoning
        # chains.  The original 30 s caused timeouts on complex prompts.
        self.bot = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300
        )

    def convert_messages_format_to_openaiurl(self, messages):
      converted_messages = []
      for message in messages:
          new_content = []
          for item in message['content']:
              if list(item.keys())[0] == 'text':
                  new_content.append({'type': 'text', 'text': item['text']})
              elif list(item.keys())[0] == 'image':
                new_content.append({'type': 'image_url', 'image_url': {'url': image_to_base64_qwen3vl(item['image'])}})
          converted_messages.append({'role': message['role'], 'content': new_content})

      return converted_messages
    
    def predict(
            self,
            text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
            self, text_prompt: str, images: list[np.ndarray], messages = None
    ) -> tuple[str, Optional[bool], Any]:
        # import pdb; pdb.set_trace()
        if messages is None:
          payload = [
              {
                  "role": "user",
                  "content": [
                      {"text": text_prompt},
                  ]
              }
          ]

          image_payload = images
          if len(images) > 1:
            image_payload = [_combine_images(images)]
          for image in image_payload:
            payload[0]['content'].append({
                'image': image
            })
        else:
          payload = messages
            
        payload = self.convert_messages_format_to_openaiurl(payload)
        # import pdb; pdb.set_trace()
        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        last_error = None
        while counter > 0:
            try:
              chat_completion_from_url = self.bot.chat.completions.create(model=self.model, messages=payload, max_tokens=4096, temperature=self.temperature)
              # print('payload: ', payload)
              # print('chat_completion_from_url: ', chat_completion_from_url)

              return (chat_completion_from_url.choices[0].message.content, payload, chat_completion_from_url)
            except Exception as error:
                last_error = error
                time.sleep(wait_seconds)
                wait_seconds *= 1
                counter -= 1
                print('Error calling LLM, will retry soon...')
                print(error)
        raise episode_exceptions.ModelEndpointError(
            f"Qwen3-VL endpoint failed after {self.max_retry} attempts."
        ) from last_error


def _normalize_predict_endpoint(endpoint_url: str) -> str:
    endpoint_url = endpoint_url.rstrip('/')
    if endpoint_url.endswith('/health'):
      endpoint_url = endpoint_url[:-len('/health')]
    if endpoint_url.endswith('/predict'):
      return endpoint_url
    return f'{endpoint_url}/predict'


def _format_predict_response(data: Any) -> str:
    if isinstance(data, str):
      return data
    if not isinstance(data, dict):
      return str(data)
    choices = data.get('choices')
    if isinstance(choices, list) and choices:
      choice = choices[0]
      if isinstance(choice, dict):
        message = choice.get('message')
        if isinstance(message, dict) and message.get('content'):
          return str(message['content'])
        if choice.get('text'):
          return str(choice['text'])
    for key in ('raw_response', 'response', 'output', 'text', 'generated_text'):
      value = str(data.get(key) or '').strip()
      if value:
        return value
    if data.get('tool_call'):
      tool_call = str(data['tool_call']).strip()
      think = str(data.get('think') or data.get('thinking') or '').strip()
      action = str(data.get('action') or '').strip()
      return (
          f"Thought: {think}\n"
          f"Action: {action}\n"
          f"<tool_call>\n{tool_call}\n</tool_call>"
      )
    return str(data)


def _extract_qwen3vl_task_and_history(messages: list[dict[str, Any]]) -> tuple[str, str]:
    user_task = ''
    previous_actions = ''
    marker = '\nTask progress (You have done the following operation on the current device): '
    for message in messages:
      if message.get('role') != 'user':
        continue
      for item in message.get('content', []):
        if 'text' not in item:
          continue
        text = str(item['text'])
        if marker not in text:
          user_task = user_task or text.strip()
          continue
        raw_task, raw_history = text.split(marker, 1)
        if raw_task.startswith('The user query: '):
          raw_task = raw_task[len('The user query: '):]
        user_task = raw_task.strip()
        if user_task.endswith('.'):
          user_task = user_task[:-1].strip()
        previous_actions = raw_history.strip()
        if previous_actions.endswith('.'):
          previous_actions = previous_actions[:-1].strip()
        if not previous_actions:
          previous_actions = 'No previous actions.'
    return user_task, previous_actions or 'No previous actions.'


def _first_message_image(messages: list[dict[str, Any]]) -> str:
    for message in messages:
      for item in message.get('content', []):
        image_value = item.get('image')
        if image_value:
          image_text = str(image_value)
          if image_text.startswith('data:image/'):
            return image_text
          return image_to_base64_qwen3vl(image_value)
        image_url = item.get('image_url')
        if isinstance(image_url, dict) and image_url.get('url'):
          return str(image_url['url'])
    raise ValueError('No image found in Qwen3-VL predict payload.')


class Qwen3VLPredictWrapper(LlmWrapper, MultimodalLlmWrapper):

    RETRY_WAITING_SECONDS = 20

    def __init__(
            self,
            endpoint_url: str,
            max_retry: int = 10,
            timeout: int = 300,
            max_tokens: int = 512,
    ):
        if max_retry <= 0:
            max_retry = 10
            print('Max_retry must be positive. Reset it to 10')
        self.max_retry = min(max_retry, 10)
        self.endpoint_url = _normalize_predict_endpoint(endpoint_url)
        self.timeout = timeout
        self.max_tokens = max_tokens

    def predict(
            self,
            text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
            self, text_prompt: str, images: list[np.ndarray], messages = None
    ) -> tuple[str, Optional[bool], Any]:
        if messages is None:
          image_payload = images
          if len(images) > 1:
            image_payload = [_combine_images(images)]
          if not image_payload:
            raise ValueError('Qwen3-VL /predict endpoint requires an image.')
          payload = {
              'image': image_to_base64_qwen3vl(image_payload[0]),
              'user_task': text_prompt,
              'previous_actions': 'No previous actions.',
              'max_tokens': self.max_tokens,
              'max_new_tokens': self.max_tokens,
          }
        else:
          user_task, previous_actions = _extract_qwen3vl_task_and_history(messages)
          payload = {
              'image': _first_message_image(messages),
              'user_task': user_task,
              'previous_actions': previous_actions,
              'max_tokens': self.max_tokens,
              'max_new_tokens': self.max_tokens,
          }

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        last_error = None
        while counter > 0:
            try:
              response = requests.post(
                  self.endpoint_url,
                  headers={'Content-Type': 'application/json'},
                  json=payload,
                  timeout=self.timeout,
              )
              response.raise_for_status()
              data = response.json()
              return (_format_predict_response(data), payload, data)
            except Exception as error:
                last_error = error
                time.sleep(wait_seconds)
                counter -= 1
                print('Error calling Qwen3-VL /predict endpoint, will retry soon...')
                print(error)
        raise episode_exceptions.ModelEndpointError(
            "Qwen3-VL /predict endpoint failed after "
            f"{self.max_retry} attempts."
        ) from last_error
