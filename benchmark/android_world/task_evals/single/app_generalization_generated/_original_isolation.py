"""Adds cross-category app isolation to upstream AndroidWorld tasks.

The cross-app ports under ``app_generalization_generated`` already disable
every sibling app in the same category before a task runs (see
``PackageAppEval`` in ``_cross_app_base``). The original AndroidWorld tasks
(Markor, Tasks.org / Clock / SimpleCalendar / ContactsAddContact /
SimpleSms*) live in upstream files and do **not** carry that isolation.

That asymmetry breaks the "model | AW-original | new-installed | delta"
comparison the user asked for: when the upstream Markor task runs without
isolation, the Markor agent sees Joplin, NotallyX, Notesnook, etc. all
installed, and the cross-app port runs against a pruned environment. The
delta would conflate "model is worse on third-party apps" with "the
environment differs between cells".

This module provides ``with_category_isolation(cls, package_name)`` -- a
class wrapper that subclasses ``cls`` and adds the same enable/disable
sibling behaviour around ``initialize_task`` / ``tear_down``. The wrapped
class is registered under the **same name** as the original so the rest of
the suite is oblivious. We never touch the upstream files.

Usage (in registry.py)::

    isolation.with_category_isolation(
        markor.MarkorAddNoteHeader, "net.gsantner.markor",
    )
"""

from __future__ import annotations

from typing import Type, TypeVar

from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.single.app_generalization_generated import (
    _cross_app_base as base,
)


_T = TypeVar("_T", bound=task_eval.TaskEval)


def with_category_isolation(
    cls: Type[_T], package_name: str
) -> Type[_T]:
  """Returns a subclass of ``cls`` that isolates ``package_name``'s category.

  Behaviour mirrors ``PackageAppEval``:

    1. ``initialize_task``: re-enable every package in the category (heal a
       prior crash), then disable every sibling.
    2. ``tear_down``: re-enable every sibling.

  If ``package_name`` is not registered in
  ``app_generalization_apps.csv`` (i.e. has no siblings), the wrapper is a
  no-op and the original class is returned unchanged. That keeps the wrapper
  cheap to apply blanketly from the registry.
  """
  siblings = base.sibling_packages(package_name)
  if not siblings:
    return cls

  class _Isolated(cls):  # type: ignore[misc, valid-type]
    """Auto-generated isolation wrapper. See ``with_category_isolation``."""

    def initialize_task(self, env: interface.AsyncEnv) -> None:
      base.isolate_package_category(
          env,
          package_name,
          task_name=type(self).__name__,
      )
      super().initialize_task(env)

    def tear_down(self, env: interface.AsyncEnv) -> None:
      try:
        super().tear_down(env)
      finally:
        base.restore_package_category(env, package_name)

  _Isolated.__name__ = cls.__name__
  _Isolated.__qualname__ = cls.__qualname__
  _Isolated.__module__ = cls.__module__
  _Isolated.__doc__ = cls.__doc__
  return _Isolated  # type: ignore[return-value]
