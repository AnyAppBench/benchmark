#!/usr/bin/env python3
"""Verify that CATBench profiles only schedule supported task-app pairs.

This is a static harness audit. It does not launch emulators or models. The
checks are intentionally conservative: each runnable app must expose the
expected number of task templates, every scheduled task must be registered, and
the concrete task classes must not be scaffold placeholders.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
  sys.path.insert(0, str(BENCHMARK_ROOT))

from android_world import registry  # pylint: disable=wrong-import-position
from app_generalization_profiles import get_domain_profiles  # pylint: disable=wrong-import-position


DEFAULT_CATEGORIES = (
    "todo",
    "notes",
    "finance",
    "music",
    "calendar",
    "sms",
    "files",
    "maps",
    "contacts",
    "clock",
)

EXPECTED_TASKS_PER_APP = 10
PLACEHOLDER_PATTERNS = (
    "TODO_REPLACE_WITH_ANDROIDWORLD_APP_NAME",
    "TODO: port",
    "raise NotImplementedError",
)


def _class_source_bundle(cls: type[Any]) -> str:
  chunks: list[str] = []
  for candidate in inspect.getmro(cls)[:5]:
    try:
      chunks.append(inspect.getsource(candidate))
    except (OSError, TypeError):
      continue
  return "\n".join(chunks)


def _source_path(cls: type[Any]) -> str:
  path = inspect.getsourcefile(cls)
  if not path:
    return "<dynamic>"
  try:
    return str(Path(path).resolve().relative_to(BENCHMARK_ROOT.parent.resolve()))
  except ValueError:
    return path


def _app_names_are_package_names(app_names: Any) -> bool:
  return (
      isinstance(app_names, tuple)
      and bool(app_names)
      and all(isinstance(name, str) and "." in name for name in app_names)
  )


def _validator_mode(source: str) -> str:
  lowered = source.lower()
  if any(token in lowered for token in ("sqlite", "provider", "filesystem", "mediastore", "media store")):
    return "state/db/fs/provider"
  if any(token in lowered for token in ("ui", "screen", "accessibility", "screenshot")):
    return "ui/visual"
  return "unknown/static"


def _parse_categories(raw: str) -> tuple[str, ...]:
  return tuple(item.strip() for item in raw.split(",") if item.strip())


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--categories",
      default=",".join(DEFAULT_CATEGORIES),
      help="Comma-separated CATBench categories to audit.",
  )
  parser.add_argument(
      "--expected_tasks_per_app",
      type=int,
      default=EXPECTED_TASKS_PER_APP,
      help="Expected number of scheduled task templates for each runnable app.",
  )
  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Print every runnable app and validator-mode counts.",
  )
  args = parser.parse_args()

  profiles = get_domain_profiles()
  categories = _parse_categories(args.categories)
  unknown = [category for category in categories if category not in profiles]
  if unknown:
    print(f"Unknown categories: {unknown}. Valid: {sorted(profiles)}", file=sys.stderr)
    return 2

  task_registry = registry.TaskRegistry().get_registry(
      registry.TaskRegistry.ANDROID_WORLD_FAMILY
  )

  issues: list[tuple[str, str, str, str, str]] = []
  summary: list[tuple[str, str, str, int]] = []
  source_counts: Counter[str] = Counter()
  validator_modes: Counter[tuple[str, str, str]] = Counter()

  for category in categories:
    profile = profiles[category]
    for app in profile.apps:
      if app.optional and app.implemented_tasks:
        issues.append((
            "optional_app_scheduled",
            category,
            app.app_id,
            "",
            f"{len(app.implemented_tasks)} tasks",
        ))
      if not app.implemented_tasks:
        continue

      task_names = tuple(app.implemented_tasks)
      summary.append((category, app.app_id, app.display_name, len(task_names)))

      if len(task_names) != args.expected_tasks_per_app:
        issues.append((
            "wrong_task_count",
            category,
            app.app_id,
            "",
            f"{len(task_names)} != {args.expected_tasks_per_app}",
        ))

      duplicates = [name for name, count in Counter(task_names).items() if count > 1]
      for task_name in duplicates:
        issues.append(("duplicate_task", category, app.app_id, task_name, ""))

      for task_name in task_names:
        cls = task_registry.get(task_name)
        if cls is None:
          issues.append(("missing_registry", category, app.app_id, task_name, ""))
          continue

        source_path = _source_path(cls)
        source_counts[source_path] += 1
        source = _class_source_bundle(cls)
        validator_modes[(category, app.app_id, _validator_mode(source))] += 1

        cls_package = getattr(cls, "package_name", None)
        if cls_package and app.package_name and cls_package != app.package_name:
          issues.append((
              "package_mismatch",
              category,
              app.app_id,
              task_name,
              f"class={cls_package} profile={app.package_name}",
          ))

        app_names = getattr(cls, "app_names", None)
        if (
            app.package_name
            and _app_names_are_package_names(app_names)
            and app.package_name not in app_names
        ):
          issues.append((
              "app_names_mismatch",
              category,
              app.app_id,
              task_name,
              f"class={app_names} profile={app.package_name}",
          ))

        for pattern in PLACEHOLDER_PATTERNS:
          if pattern in source:
            issues.append((
                "placeholder_or_not_implemented",
                category,
                app.app_id,
                task_name,
                f"{pattern} in {source_path}",
            ))

  print("CATBench Task Support Audit")
  print(f"Categories: {', '.join(categories)}")
  print(f"Runnable apps: {len(summary)}")
  print(f"Scheduled task-app pairs: {sum(row[3] for row in summary)}")
  print(f"Issues: {len(issues)}")

  if args.verbose:
    print("\nRunnable Apps")
    for category, app_id, display_name, count in summary:
      print(f"{category:8s} {app_id:32s} {count:2d} {display_name}")

    print("\nValidator Modes")
    by_app: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for (category, app_id, mode), count in validator_modes.items():
      by_app[(category, app_id)][mode] = count
    for (category, app_id), modes in sorted(by_app.items()):
      rendered = ", ".join(f"{mode}={count}" for mode, count in sorted(modes.items()))
      print(f"{category:8s} {app_id:32s} {rendered}")

    print("\nTask Class Sources")
    for path, count in sorted(source_counts.items(), key=lambda item: (-item[1], item[0])):
      print(f"{count:4d} {path}")

  if issues:
    print("\nIssues")
    for issue_type, category, app_id, task_name, detail in issues:
      target = f"{category}/{app_id}"
      if task_name:
        target += f"/{task_name}"
      print(f"- {issue_type}: {target} {detail}".rstrip())
    return 1

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
