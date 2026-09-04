# App Generalization Extension

This extension helps evaluate whether a model can perform similar intents in
similar apps (cross-app generalization).

It follows AndroidWorld rules by reusing already-defined task names and
generating porting scaffolds for new apps in the same category.

## What Is Included

- Notes cohort profile with your requested apps:
  - Core: Joplin, Markor, NotallyX, QuickNotes, neutriNote, Notesnook
  - Optional harder variants: Orgzly Revived, My Brain
- To-Do cohort profile with your requested apps:
  - Core: Tasks.org, Cfait, Trudido, Todo List (PFA), ntodotxt, TaskMate
  - Optional harder variants: Super Productivity, Grit
- A runner that:
  - Uses canonical AndroidWorld task names per domain.
  - Executes implemented tasks for supported apps.
  - Emits per-app porting targets for unsupported apps.
  - Optionally generates Python task scaffold files for unsupported apps.

## Canonical Task Sets

Notes canonical task names:

- MarkorAddNoteHeader
- MarkorChangeNoteContent
- MarkorCreateFolder
- MarkorCreateNote
- MarkorCreateNoteAndSms
- MarkorCreateNoteFromClipboard
- MarkorDeleteAllNotes
- MarkorDeleteNewestNote
- MarkorDeleteNote
- MarkorEditNote
- MarkorMergeNotes
- MarkorMoveNote
- MarkorTranscribeReceipt
- MarkorTranscribeVideo
- NotesIsTodo
- NotesMeetingAttendeeCount
- NotesRecipeIngredientCount
- NotesTodoItemCount

To-Do canonical task names:

- TasksCompletedTasksForDate
- TasksDueNextWeek
- TasksDueOnDate
- TasksHighPriorityTasks
- TasksHighPriorityTasksDueOnDate
- TasksIncompleteTasksOnDate

## Files

- benchmark/app_generalization_profiles.py
- benchmark/run_app_generalization.py
- benchmark/app_generalization_apps.csv
- benchmark/download_app_generalization_apks.sh
- benchmark/install_app_generalization_apks.sh
- benchmark/grant_permissions_app_generalization.sh

## APK Setup For New Apps

The original AndroidWorld baseline install flow is still unchanged:

- benchmark/install_apps.sh
- benchmark/grant_permissions.sh
- benchmark/setup_apps.sh

For app-generalization apps, use the new catalog-driven setup scripts.

1. Fill package names and APK URLs in benchmark/app_generalization_apps.csv.
2. Download APKs:

```bash
bash benchmark/download_app_generalization_apks.sh
```

3. Install APKs:

```bash
bash benchmark/install_app_generalization_apks.sh
```

4. Grant runtime permissions:

```bash
bash benchmark/grant_permissions_app_generalization.sh
```

Optional harder variants are skipped by default. To include them:

```bash
INCLUDE_OPTIONAL=1 bash benchmark/download_app_generalization_apks.sh
INCLUDE_OPTIONAL=1 bash benchmark/install_app_generalization_apks.sh
INCLUDE_OPTIONAL=1 bash benchmark/grant_permissions_app_generalization.sh
```

## MAI-UI Prerequisites

MAI-UI uses an external UI-TARS adapter module that is **not bundled** in this
repository. You must provide an external repo root exposing:

- agents/uitars/adapters/android_world.py

Set one of these environment variables to that external repo root:

- CATBENCH_AGENT_ROOT
- CATBENCH_AGENT_ROOTS (multiple roots, separated by your OS path separator)
- PYTHONPATH (including the external repo root)

Validate environment + dependencies from repo root:

```bash
export CATBENCH_AGENT_ROOT=/path/to/adapter-repo
bash benchmark/prepare_maiui_benchmark.sh
```

If your MAI-UI environment is inconsistent, let the prep script repair pins:

```bash
AUTO_REPAIR=1 CONDA_ENV=catbench311 bash benchmark/prepare_maiui_benchmark.sh
```

Directory note:

- If you are in repo root (`CATBench`), use `python benchmark/run_app_generalization.py ...`
- If you are in `CATBench/benchmark`, use `python run_app_generalization.py ...`

## Run With MAI-UI-8B

```bash
python benchmark/run_app_generalization.py \
  --runner_script benchmark/run_maiui.py \
  --maiui_variant=8b \
  --domain notes \
  --write_scaffolds \
  --device=cuda:0
```

## Run With MAI-UI-2B

```bash
python benchmark/run_app_generalization.py \
  --runner_script benchmark/run_maiui.py \
  --maiui_variant=2b \
  --domain notes \
  --write_scaffolds \
  --device=cuda:0
```

## Include Optional Harder Variants

```bash
python benchmark/run_app_generalization.py \
  --runner_script benchmark/run_maiui.py \
  --maiui_variant=8b \
  --domain all \
  --include_optional \
  --write_scaffolds \
  --dry_run
```

## How To Extend Unsupported Apps

1. Start from scaffold files generated under
  benchmark/android_world/task_evals/single/app_generalization_generated
  (notes) and
  benchmark/android_world/task_evals/information_retrieval/app_generalization_generated
  (todo).
2. Implement app-specific setup and success validators for each scaffolded
  task.
3. Register finished task classes in android_world/registry.py.
4. Move each task from porting to implemented by updating
  benchmark/app_generalization_profiles.py.
5. Re-run benchmark/run_app_generalization.py.

The run manifest is written under the output root and includes every app,
status, canonical tasks, porting targets, commands, and return codes.

You can override scaffold output root with:

--scaffold_root benchmark/android_world/task_evals
