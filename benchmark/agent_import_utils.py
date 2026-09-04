"""Helpers for importing external agent adapter packages."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any


def _iter_search_roots(project_root: str) -> list[str]:
  roots: list[str] = []

  def _add(path: str) -> None:
    norm = os.path.abspath(os.path.expanduser(path))
    if norm not in roots:
      roots.append(norm)

  _add(project_root)

  workspace_root = os.path.dirname(project_root)
  _add(workspace_root)

  env_roots = os.environ.get("CATBENCH_AGENT_ROOT", "")
  if env_roots:
    _add(env_roots)

  env_roots_multi = os.environ.get("CATBENCH_AGENT_ROOTS", "")
  if env_roots_multi:
    for item in env_roots_multi.split(os.pathsep):
      if item.strip():
        _add(item.strip())

  for parent in (project_root, workspace_root):
    if not os.path.isdir(parent):
      continue
    for child in os.listdir(parent):
      child_path = os.path.join(parent, child)
      if os.path.isdir(os.path.join(child_path, "agents")):
        _add(child_path)

  return roots


def import_optional_agent(project_root: str, module_name: str, class_name: str):
  """Imports an agent adapter after extending sys.path with likely roots."""
  searched_roots = _iter_search_roots(project_root)

  for root in searched_roots:
    if root not in sys.path:
      sys.path.insert(0, root)

  try:
    module = importlib.import_module(module_name)
  except ModuleNotFoundError as exc:
    searched = "\n".join(f"  - {root}" for root in searched_roots)
    raise ModuleNotFoundError(
        f"Missing local adapter module: {module_name}\n"
        "Place the external agent checkout so it exposes agents/... and either:\n"
        "1. set CATBENCH_AGENT_ROOT=/path/to/repo, or\n"
        "2. set CATBENCH_AGENT_ROOTS to multiple repo roots separated by your OS path separator, or\n"
        "3. add the repo root to PYTHONPATH.\n"
        "Searched roots:\n"
        f"{searched}"
    ) from exc

  return getattr(module, class_name)


def resolve_uitars_agent_class(project_root: str) -> tuple[type[Any], str]:
  """Resolves a UI-TARS-compatible agent class.

  Priority:
  1) External adapter class at agents.uitars.adapters.android_world
  2) Local in-repo fallback at android_world.agents.qwen_vlm.QwenVlmAgent

  Returns:
    Tuple of (agent_class, source_label).
  """
  try:
    return (
        import_optional_agent(
            project_root,
            'agents.uitars.adapters.android_world',
            'AndroidWorldUITARSAgent',
        ),
        'external:agents.uitars.adapters.android_world',
    )
  except ModuleNotFoundError as external_exc:
    try:
      from android_world.agents import qwen_vlm as qwen_module
    except (ModuleNotFoundError, ImportError) as local_exc:
      raise ModuleNotFoundError(
          'Could not resolve a UI-TARS-compatible agent backend.\n\n'
          'External adapter error:\n'
          f'{external_exc}\n\n'
          'Local fallback error:\n'
          f'{local_exc}\n\n'
          'Expected one of:\n'
          '1. External adapter module '
          'agents.uitars.adapters.android_world, or\n'
          '2. Local in-repo agent dependencies for '
          'android_world.agents.qwen_vlm.'
      ) from local_exc

  def _looks_like_maiui_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return 'mai-ui' in lowered or lowered.startswith('tongyi-mai/')

  def _normalize_openai_base_url(endpoint_url: str) -> str:
    base = endpoint_url.rstrip('/')
    return base if base.endswith('/v1') else f'{base}/v1'

  class AndroidWorldUITARSAgent:
    """Compatibility wrapper for environments without external UI-TARS adapter."""

    def __new__(
        cls,
        env,
        model_name: str = qwen_module.DEFAULT_MODEL_ID,
        device: str = 'cuda:0',
        save_failed_tasks: bool = True,
        mode: str = 'local',
        endpoint_url: str = 'http://127.0.0.1:8000',
        debug_verbose: bool = False,
        debug_history: bool = False,
    ):
      if _looks_like_maiui_model(model_name):
        if mode != 'endpoint':
          raise ValueError(
              'MAI-UI backend requires endpoint mode. '
              'Use --mode=endpoint --endpoint_url=http://127.0.0.1:8000'
          )

        try:
          from android_world.agents import maiui as maiui_module
        except Exception as maiui_exc:  # pylint: disable=broad-exception-caught
          raise ModuleNotFoundError(
              'Failed to import local MAI-UI backend '
              '(android_world.agents.maiui).\n'
              'Install MAI-UI optional dependencies and retry.\n'
              'Suggested: pip install qwen-agent soundfile'
          ) from maiui_exc

        return maiui_module.MAIUI(
            env=env,
            llm_base_url=_normalize_openai_base_url(endpoint_url),
            src_format='abs_resized',
            name=model_name,
        )

      return qwen_module.QwenVlmAgent(
          env,
          model_id=model_name,
          device=device,
          save_failed_tasks=save_failed_tasks,
          mode=mode,
          endpoint_url=endpoint_url,
          debug_verbose=debug_verbose,
          debug_history=debug_history,
      )

    def __init__(
        self,
        env,
        model_name: str = qwen_module.DEFAULT_MODEL_ID,
        device: str = 'cuda:0',
        save_failed_tasks: bool = True,
        mode: str = 'local',
        endpoint_url: str = 'http://127.0.0.1:8000',
        debug_verbose: bool = False,
        debug_history: bool = False,
    ):
      del env, model_name, device, save_failed_tasks, mode
      del endpoint_url, debug_verbose, debug_history

  return AndroidWorldUITARSAgent, 'local:android_world.agents.{maiui,qwen_vlm}'
