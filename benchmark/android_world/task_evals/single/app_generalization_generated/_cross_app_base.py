"""Shared base classes and helpers for app-generalization task ports.

All cross-app task modules (clock, calendar, contacts, ...) use this module to
stay on a consistent template: a package-launched `TaskEval` subclass with
shared text-matching helpers, lifecycle hooks, and **category isolation** so
that exactly one app from a category is enabled while a task runs.

Category isolation
------------------

The (package -> category) mapping is loaded once from
``app_generalization_apps.csv`` at import time. The category for each row is
derived from the prefix of ``app_id`` (e.g. ``clock_chrono`` -> ``clock``).

Before each task starts, ``PackageAppEval.initialize_task`` runs::

    pm enable      --user 0 <every package in this category>   # heal prior crash
    pm disable-user --user 0 <every sibling package>            # leave only target

On teardown the siblings are re-enabled. Both calls are best-effort and never
raise -- if a device build doesn't allow disabling a system app, the test
still proceeds, just without isolation. Set ``isolate_category = False`` on a
task class to opt out (e.g. some SMS flows that need the system default-handler
to be a different app).
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import subprocess
import time
from typing import Any, Iterable

from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import representation_utils
from android_world.task_evals import task_eval


# -----------------------------------------------------------------------------
# Category catalog -- loaded once from app_generalization_apps.csv.
# -----------------------------------------------------------------------------


_CSV_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..",
        "app_generalization_apps.csv",
    )
)

_PACKAGE_TO_CATEGORY: dict[str, str] = {}
_CATEGORY_TO_PACKAGES: dict[str, frozenset[str]] = {}
_INSTALLED_PACKAGES: set[str] | None = None
_PACKAGE_PERMISSIONS: dict[str, frozenset[str] | None] = {}
_TRUE_ENV_VALUES = {"1", "true", "yes", "required", "strict"}
_ISOLATE_CATEGORY_SCRIPT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..",
        "scripts",
        "isolate_category.sh",
    )
)

_CATEGORY_RUNTIME_PERMISSIONS: dict[str, tuple[str, ...]] = {
    "clock": (
        "android.permission.POST_NOTIFICATIONS",
    ),
    "contacts": (
        "android.permission.READ_CONTACTS",
        "android.permission.WRITE_CONTACTS",
        "android.permission.GET_ACCOUNTS",
        "android.permission.CALL_PHONE",
        "android.permission.POST_NOTIFICATIONS",
    ),
    "files": (
        "android.permission.READ_EXTERNAL_STORAGE",
        "android.permission.WRITE_EXTERNAL_STORAGE",
        "android.permission.READ_MEDIA_AUDIO",
        "android.permission.READ_MEDIA_IMAGES",
        "android.permission.READ_MEDIA_VIDEO",
        "android.permission.POST_NOTIFICATIONS",
    ),
    "maps": (
        "android.permission.ACCESS_COARSE_LOCATION",
        "android.permission.ACCESS_FINE_LOCATION",
        "android.permission.POST_NOTIFICATIONS",
    ),
    "sms": (
        "android.permission.READ_CONTACTS",
        "android.permission.READ_SMS",
        "android.permission.SEND_SMS",
        "android.permission.RECEIVE_SMS",
        "android.permission.POST_NOTIFICATIONS",
    ),
}


def _load_catalog() -> None:
  """Populate the package <-> category mappings from the CSV."""
  if not os.path.isfile(_CSV_PATH):
    return
  by_category: dict[str, set[str]] = {}
  try:
    with open(_CSV_PATH, newline="", encoding="utf-8") as fh:
      reader = csv.DictReader(fh)
      for row in reader:
        app_id = (row.get("app_id") or "").strip()
        package = (row.get("package_name") or "").strip()
        if not app_id or not package:
          continue
        category = app_id.split("_", 1)[0]
        _PACKAGE_TO_CATEGORY[package] = category
        by_category.setdefault(category, set()).add(package)
  except OSError:
    return
  for category, packages in by_category.items():
    _CATEGORY_TO_PACKAGES[category] = frozenset(packages)


_load_catalog()


def category_packages(package_name: str) -> tuple[str, ...]:
  """Every package registered in the same CSV category as ``package_name``."""
  category = _PACKAGE_TO_CATEGORY.get(package_name)
  if not category:
    return ()
  return tuple(sorted(_CATEGORY_TO_PACKAGES.get(category, frozenset())))


def category_for_package(package_name: str) -> str:
  """CATBench category for ``package_name``, or empty string if unknown."""
  return _PACKAGE_TO_CATEGORY.get(package_name, "")


def sibling_packages(package_name: str) -> tuple[str, ...]:
  """Every OTHER package in ``package_name``'s category."""
  return tuple(p for p in category_packages(package_name) if p != package_name)


def _env_flag(name: str) -> bool:
  return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def _pm_timeout_sec() -> float:
  try:
    return float(os.environ.get("CATBENCH_PM_TIMEOUT_SEC", 45))
  except ValueError:
    return 45.0


def _pm(env: interface.AsyncEnv, action: str, package: str) -> None:
  """Best-effort ``pm`` invocation; never raise."""
  if not _is_installed(env, package):
    return
  try:
    adb_utils.issue_generic_request(
        ["shell", f"pm {action} --user 0 {package}"],
        env.controller,
        timeout_sec=_pm_timeout_sec(),
    )
  except Exception:  # pylint: disable=broad-except
    pass


def _grant_runtime_permissions(env: interface.AsyncEnv, package: str) -> None:
  """Best-effort runtime permission grants for the app's CATBench category.

  Always targets ``--user 0`` to stay consistent with ``_pm`` above and to
  avoid silent failures on multi-user device builds.
  """
  category = _PACKAGE_TO_CATEGORY.get(package)
  if not category:
    return
  for permission in _CATEGORY_RUNTIME_PERMISSIONS.get(category, ()):
    declared_permissions = _declared_permissions(env, package)
    if declared_permissions is not None and permission not in declared_permissions:
      continue
    try:
      adb_utils.issue_generic_request(
          ["shell", f"pm grant --user 0 {package} {permission}"],
          env.controller,
          timeout_sec=_pm_timeout_sec(),
      )
    except Exception:  # pylint: disable=broad-except
      pass


def _declared_permissions(
    env: interface.AsyncEnv, package: str
) -> frozenset[str] | None:
  """Runtime permissions declared by ``package``, or ``None`` if unknown.

  ``pm grant`` logs and retries loudly when a package never requested the
  permission.  Querying the package dump first keeps the grant helper genuinely
  best-effort instead of producing false alarming setup logs.
  """
  if package not in _PACKAGE_PERMISSIONS:
    try:
      response = adb_utils.issue_generic_request(
          ["shell", f"dumpsys package {package}"],
          env.controller,
          timeout_sec=_pm_timeout_sec(),
      )
      output = response.generic.output.decode("utf-8", errors="ignore")
      _PACKAGE_PERMISSIONS[package] = frozenset(
          re.findall(r"android\.permission\.[A-Z0-9_]+", output)
      )
    except Exception:  # pylint: disable=broad-except
      _PACKAGE_PERMISSIONS[package] = None
  return _PACKAGE_PERMISSIONS[package]


def _is_installed(env: interface.AsyncEnv, package: str) -> bool:
  """Returns whether ``package`` exists on the current device."""
  global _INSTALLED_PACKAGES
  if _INSTALLED_PACKAGES is None:
    try:
      response = adb_utils.issue_generic_request(
          ["shell", "pm", "list", "packages"],
          env.controller,
          timeout_sec=_pm_timeout_sec(),
      )
      output = response.generic.output.decode("utf-8", errors="ignore")
      _INSTALLED_PACKAGES = {
          line.removeprefix("package:").strip()
          for line in output.splitlines()
          if line.strip().startswith("package:")
      }
    except Exception:  # pylint: disable=broad-except
      _INSTALLED_PACKAGES = set()
  return package in _INSTALLED_PACKAGES


def installed_packages(env: interface.AsyncEnv) -> frozenset[str]:
  """All packages currently installed on the device (cached per-process)."""
  global _INSTALLED_PACKAGES
  if _INSTALLED_PACKAGES is None:
    _is_installed(env, "")  # populates cache
  return frozenset(_INSTALLED_PACKAGES or set())


def reset_installed_cache() -> None:
  """Drop the cached installed-packages set (call after install/uninstall)."""
  global _INSTALLED_PACKAGES
  _INSTALLED_PACKAGES = None
  _PACKAGE_PERMISSIONS.clear()


class TaskAppNotInstalled(RuntimeError):
  """Raised by ``initialize_task`` when the target package is missing.

  The suite runner should catch this and report the task as ``skipped`` rather
  than ``failed``. This is the signal we use to honour the user's requirement
    that running a task for an uninstalled app must not be counted as a failure
    in the model-vs-AW comparison table.
  """


class VerifierStateReadError(RuntimeError):
  """Native verifier evidence could not be read reliably.

  Evaluators must raise this error instead of converting ADB, provider, SQLite,
  or filesystem read failures into a semantic ``False`` result. The suite
  runner records verifier-stage exceptions as invalid infrastructure attempts,
  which keeps environment faults out of the model success denominator.
  """


def disable_packages(
    env: interface.AsyncEnv, packages: Iterable[str]
) -> None:
  """Disable each package for user 0 (best-effort, swallows errors)."""
  for pkg in packages:
    _pm(env, "disable-user", pkg)


def enable_packages(
    env: interface.AsyncEnv, packages: Iterable[str]
) -> None:
  """Re-enable each package for user 0 (best-effort, swallows errors)."""
  for pkg in packages:
    _pm(env, "enable", pkg)


def _adb_output_text(response: Any) -> str:
  try:
    return response.generic.output.decode("utf-8", errors="ignore")
  except Exception:  # pylint: disable=broad-except
    return ""


def _disabled_packages(env: interface.AsyncEnv) -> frozenset[str]:
  try:
    response = adb_utils.issue_generic_request(
        ["shell", "pm", "list", "packages", "-d"],
        env.controller,
        timeout_sec=_pm_timeout_sec(),
    )
  except Exception:  # pylint: disable=broad-except
    return frozenset()
  disabled = set()
  for line in _adb_output_text(response).splitlines():
    line = line.strip().removeprefix("package:").strip()
    if line:
      disabled.add(line)
  return frozenset(disabled)


def _write_isolation_event(record: dict[str, Any]) -> None:
  log_path = os.environ.get("CATBENCH_CATEGORY_ISOLATION_LOG", "").strip()
  if not log_path:
    return
  record = {
      "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "pid": os.getpid(),
      **record,
  }
  try:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
      handle.write(json.dumps(record, sort_keys=True) + "\n")
  except Exception:  # pylint: disable=broad-except
    pass


def _controller_adb_env(env: interface.AsyncEnv) -> dict[str, str]:
  """Environment for invoking the external isolation script on this emulator."""
  adb_env = dict(os.environ)
  try:
    raw_env = env.controller.env
    config = raw_env._coordinator._simulator._config  # pylint: disable=protected-access
    launcher = config.emulator_launcher
    adb_controller = config.adb_controller
    console_port = getattr(launcher, "emulator_console_port", None)
    if console_port and not (
        adb_env.get("ANDROID_SERIAL") or adb_env.get("ADB_SERIAL")
    ):
      adb_env["ANDROID_SERIAL"] = f"emulator-{console_port}"
    adb_path = getattr(adb_controller, "adb_path", None)
    if adb_path:
      adb_env["ADB_BIN"] = adb_path
    adb_server_port = getattr(adb_controller, "adb_server_port", None)
    if adb_server_port:
      adb_env["ADB_SERVER_PORT"] = str(adb_server_port)
  except Exception:  # pylint: disable=broad-except
    pass
  return adb_env


def _isolate_category_with_script(
    env: interface.AsyncEnv,
    package_name: str,
    task_name: str,
) -> bool:
  """Run scripts/isolate_category.sh for this task target."""
  category = category_for_package(package_name)
  if not category:
    return False
  if not os.path.isfile(_ISOLATE_CATEGORY_SCRIPT):
    if _env_flag("CATBENCH_STRICT_CATEGORY_ISOLATION"):
      raise RuntimeError(
          f"isolate_category.sh not found: {_ISOLATE_CATEGORY_SCRIPT}"
      )
    return False

  completed = subprocess.run(
      [_ISOLATE_CATEGORY_SCRIPT, category, package_name],
      check=False,
      env=_controller_adb_env(env),
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      timeout=90,
  )
  _write_isolation_event(
      {
          "event": "isolate_category_script",
          "task": task_name,
          "category": category,
          "target_package": package_name,
          "returncode": completed.returncode,
          "output": completed.stdout[-4000:],
      }
  )
  if completed.returncode != 0:
    if _env_flag("CATBENCH_STRICT_CATEGORY_ISOLATION"):
      raise RuntimeError(
          "isolate_category.sh failed for "
          f"{task_name} / {package_name}: {completed.stdout}"
      )
    return False
  return True


def _audit_category_isolation(
    env: interface.AsyncEnv,
    package_name: str,
    task_name: str,
    mode: str,
) -> None:
  """Log and optionally enforce that only the target app is enabled."""
  packages = category_packages(package_name)
  if not packages:
    return
  installed = installed_packages(env)
  disabled = _disabled_packages(env)
  siblings = sibling_packages(package_name)
  missing_disabled = tuple(
      pkg for pkg in siblings if pkg in installed and pkg not in disabled
  )
  target_disabled = package_name in disabled
  record = {
      "event": "category_isolation_audit",
      "task": task_name,
      "mode": mode,
      "category": category_for_package(package_name),
      "target_package": package_name,
      "category_packages": list(packages),
      "disabled_category_packages": [
          pkg for pkg in packages if pkg in disabled
      ],
      "missing_disabled_siblings": list(missing_disabled),
      "target_disabled": target_disabled,
      "strict": _env_flag("CATBENCH_STRICT_CATEGORY_ISOLATION"),
  }
  _write_isolation_event(record)
  if _env_flag("CATBENCH_STRICT_CATEGORY_ISOLATION") and (
      target_disabled or missing_disabled
  ):
    raise RuntimeError(
        "Category isolation failed for "
        f"{task_name} / {package_name}: "
        f"target_disabled={target_disabled}, "
        f"missing_disabled_siblings={list(missing_disabled)}"
    )


def isolate_package_category(
    env: interface.AsyncEnv,
    package_name: str,
    task_name: str = "",
) -> None:
  """Enable the target category, disable siblings, then audit the result."""
  task_name = task_name or package_name
  used_script = False
  if _env_flag("CATBENCH_USE_ISOLATE_CATEGORY_SCRIPT"):
    used_script = _isolate_category_with_script(env, package_name, task_name)
  if not used_script:
    enable_packages(env, category_packages(package_name))
    disable_packages(env, sibling_packages(package_name))
  _audit_category_isolation(
      env,
      package_name,
      task_name,
      mode="isolate_category.sh" if used_script else "internal",
  )


def restore_package_category(
    env: interface.AsyncEnv,
    package_name: str,
) -> None:
  """Restore sibling apps after a task completes."""
  enable_packages(env, sibling_packages(package_name))


# -----------------------------------------------------------------------------
# UI-text helpers (shared by every cross-app module).
# -----------------------------------------------------------------------------


def close_app(package_name: str, env: interface.AsyncEnv) -> None:
  """Clears app data for the given package."""
  adb_utils.clear_app_data(package_name, env.controller)


def matches_any_text(
    element: representation_utils.UIElement,
    candidates: Iterable[str],
) -> bool:
  """Returns True if `text` or `content_description` contains any candidate.

  Matching is case-insensitive substring.
  """
  fields = (element.text or "", element.content_description or "")
  lowered_fields = tuple(field.lower() for field in fields if field)
  lowered_candidates = tuple(candidate.lower() for candidate in candidates)
  return any(
      candidate in field
      for field in lowered_fields
      for candidate in lowered_candidates
  )


def element_text_contains(
    ui_elements: list[representation_utils.UIElement],
    candidates: Iterable[str],
) -> bool:
  """Returns True if any UI element matches any candidate."""
  candidates = tuple(candidates)
  return any(matches_any_text(el, candidates) for el in ui_elements)


def matches_any_word(
    element: representation_utils.UIElement,
    candidates: Iterable[str],
) -> bool:
  """Word-boundary aware variant of ``matches_any_text``.

  Use for control-button names like "start", "pause", "reset" where the
  substring "start" must NOT match "Restart". A control label is matched
  only when it appears as a whole word in either text or content_description.
  """
  fields = (element.text or "", element.content_description or "")
  lowered_fields = tuple(field.lower() for field in fields if field)
  if not lowered_fields:
    return False
  patterns = tuple(
      re.compile(rf"\b{re.escape(candidate.lower())}\b")
      for candidate in candidates
  )
  return any(
      pattern.search(field)
      for field in lowered_fields
      for pattern in patterns
  )


def element_text_contains_word(
    ui_elements: list[representation_utils.UIElement],
    candidates: Iterable[str],
) -> bool:
  """Returns True iff any UI element has a candidate as a whole word."""
  candidates = tuple(candidates)
  return any(matches_any_word(el, candidates) for el in ui_elements)


_NETWORK_ERROR_MARKERS: tuple[str, ...] = (
    "something went wrong",
    "no internet connection",
    "you are offline",
    "couldn't connect",
    "could not connect",
    "check your connection",
    "no network",
    "connection failed",
    "service unavailable",
    "internet connection",
)


class _EnvironmentNetworkError(RuntimeError):
  """Raised when the validator can prove the failure is environmental.

  The runner converts this to ``exception_info`` so the episode is excluded
  from the success-rate denominator instead of being counted as an agent
  failure. Currently only Google Maps surfaces these markers in CATBench.
  """


def network_error_visible(
    ui_elements: list[representation_utils.UIElement],
) -> bool:
  """Returns True iff a network/connectivity error dialog is on screen.

  This is heuristic but the markers below are unambiguous in the apps we
  ship: Google Maps in particular shows "Something went wrong" with an
  internet hint whenever it cannot reach its backends, and the agent has no
  way to recover from inside the benchmark image.
  """
  return any(
      matches_any_text(element, _NETWORK_ERROR_MARKERS)
      for element in ui_elements
  )


def raise_if_network_error(
    ui_elements: list[representation_utils.UIElement],
    package_name: str,
) -> None:
  """Raise ``_EnvironmentNetworkError`` when a known network dialog is up.

  Validators should call this at the top of ``is_successful`` so the runner
  can bucket the episode out of the agent-failure count.
  """
  if network_error_visible(ui_elements):
    raise _EnvironmentNetworkError(
        f"network/connectivity error dialog visible in {package_name}"
    )


# -----------------------------------------------------------------------------
# Base class.
# -----------------------------------------------------------------------------


class PackageAppEval(task_eval.TaskEval):
  """Base class for cross-app ports that launch a target package.

  Subclasses must set ``app_names`` and ``package_name``.  Optional overrides:
    - ``clear_data_on_init`` (default ``True``) -- wipes app data before launch.
    - ``clear_data_on_teardown`` (default ``True``).
    - ``isolate_category`` (default ``True``) -- disables every sibling app in
      the same CSV category before launch and re-enables them on teardown.

  Lifecycle:
    1. ``initialize_task``: re-enables every package in the category (heals
       any previous crash that left a sibling disabled), then disables every
       sibling, then clears data on the target and launches it.
    2. ``tear_down``: closes recents, optionally clears app data, re-enables
       every sibling.
  """

  app_names: tuple[str, ...] = ()
  package_name: str = ""
  clear_data_on_init: bool = True
  clear_data_on_teardown: bool = True
  isolate_category: bool = True
  # Packages within this category that do NOT support this task's feature.
  # Per-app generator loops skip (task, app) pairs where ``app`` is listed
  # here, instead of synthesising a class that is guaranteed to fail.
  excluded_packages: tuple[str, ...] = ()

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    if not self.package_name:
      raise ValueError(
          f"{type(self).__name__}.package_name must be set before use."
      )
    if not _is_installed(env, self.package_name):
      # Surface the skip BEFORE we mark the task initialized so tear_down is a
      # no-op and the runner can record "skipped_uninstalled" instead of
      # "failed". The reporter buckets these out of the success rate.
      raise TaskAppNotInstalled(
          f"{type(self).__name__}: package {self.package_name!r} is not"
          " installed on the current device."
      )
    env.interaction_cache = ""
    self.initialize_device_time(env)
    if self.initialized:
      raise RuntimeError(f"{self.name}.initialize_task() is already called.")
    self.initialized = True

    seed = self.params.get("seed")
    if seed is not None:
      random.seed(seed)

    if self.isolate_category:
      # Heal any previous crash that left a sibling disabled, then keep only
      # the target package enabled in this category.
      isolate_package_category(
          env,
          self.package_name,
          task_name=type(self).__name__,
      )

    if self.clear_data_on_init:
      close_app(self.package_name, env)
    _grant_runtime_permissions(env, self.package_name)
    adb_utils.launch_app(self.package_name, env.controller)

  def tear_down(self, env: interface.AsyncEnv) -> None:
    try:
      adb_utils.close_recents(env.controller)
    except:  # pylint: disable=bare-except
      pass
    self.initialized = False
    if self.clear_data_on_teardown:
      close_app(self.package_name, env)
    if self.isolate_category:
      restore_package_category(env, self.package_name)
