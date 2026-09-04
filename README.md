# AnyAppBench

A category-controlled live Android benchmark for **within-category cross-application
generalization**. The question it asks is narrow on purpose: if the user goal, the
parameters, the initial state and the success condition are all held fixed and *only the
application changes*, does a mobile GUI agent still succeed?

The harness is built on [AndroidWorld](https://github.com/google-research/android_world)
(vendored under `benchmark/`) and runs on a live Android emulator with deterministic,
programmatic verifiers. There are no model-graded success signals: every reported success
comes from an app-specific verifier that reads real device state.

> The project was previously called **CATBench**. The name survives in module names,
> environment variables (`CATBENCH_*`), directory names and result ledgers. Treat
> `CATBench` and `AnyAppBench` as the same thing.

---

## 1. Benchmark at a glance

| Quantity | Value |
| --- | --- |
| Functional categories | 10 (Tasks, Notes, Finance, Music, Calendar, SMS, File Manager, Maps, Contacts, Clock) |
| Task templates | 10 per category, 100 total |
| Runnable applications | 52 |
| Validated task–application pairs | 520 |
| Execution conditions | C1 (direct) and C2 (sub-goal augmented) |
| Success signal | deterministic per-pair verifier (`TaskEval.is_successful`) |

Verify these numbers against the tree rather than trusting this table:

```bash
cd benchmark
python scripts/verify_catbench_task_support.py
# Runnable apps: 52
# Scheduled task-app pairs: 520
```

A **task template** fixes the instruction, its parameters, the initialisation, the success
condition and the step budget. An **application** supplies the interface and the internal
state through which the agent must reach that condition. Every supported pair gets its own
verifier `v(template, app)` that translates the single success criterion into that app's
representation. The verifier never changes the task.

Each category has exactly one *original AndroidWorld application* (the app AndroidWorld
itself used for that category) plus several *new applications* offering the same function.
The cross-application gap is the difference between them.

---

## 2. Repository layout

```
benchmark/                              vendored AndroidWorld + all AnyAppBench code
├── android_world/
│   ├── suite_utils.py                  suite assembly, seeding, episode running,
│   │                                   condition labelling, result annotation
│   ├── episode_runner.py               single-episode loop
│   ├── registry.py                     task registry (all 520 pairs are wired here)
│   ├── agents/                         agent implementations and model adapters
│   ├── env/
│   │   ├── android_world_controller.py emulator control
│   │   └── runtime_health.py           per-step device health guard
│   └── task_evals/
│       ├── single/app_generalization_generated/
│       │   ├── _cross_app_base.py      PackageAppEval base, install + isolation
│       │   ├── _original_isolation.py  category isolation for AW-original apps
│       │   └── <category>_cross_app_tasks.py   the 10 category task modules
│       └── information_retrieval/app_generalization_generated/
├── app_generalization_profiles.py      the benchmark definition: category -> apps -> tasks
├── app_generalization_apps.csv         APK catalog (package, apk filename, url, optional)
├── task_breakdowns.py                  C2 plan lookup and prompt-goal composition
├── configs/
│   ├── catbench_5cat_primary_cohort.json   frozen cohort manifest
│   ├── app_versions_pinned.csv             pinned app versions
│   └── catbench.env                        LOCAL SECRETS, never committed
├── scripts/                            generation, preflight, launch, judging, reporting
├── docker_setup/                       headless emulator container
└── docs/                               methodology notes and runbooks
```

Everything outside `benchmark/` is either configuration or ignored experiment output.

---

## 3. Setup

### 3.1 Python environment

Python **3.11+**.

```bash
conda create -n catbench311 python=3.11 -y
conda activate catbench311
cd benchmark
pip install -r requirements.txt
pip install -e .
```

### 3.2 Emulator

A Pixel 6 AVD on **API level 33 (Tiramisu)**, launched from the command line with gRPC
enabled so the accessibility forwarder can attach:

```bash
$ANDROID_HOME/emulator/emulator -avd <AVD_NAME> -no-snapshot -grpc 8554
```

For multi-lane runs, `docker_setup/` builds a headless emulator image and
`scripts/manage_catbench_docker_pool.sh` / `manage_catbench_podman_58xx_pool.sh` manage a
pool of them. `CATBENCH_EMULATORS` maps console ports to gRPC ports:

```bash
export CATBENCH_EMULATORS=5800:8800,5802:8801,5804:8802,5806:8803,5808:8804
```

### 3.3 Applications

The 52 APKs are catalogued in `app_generalization_apps.csv`. Provision them with:

```bash
cd benchmark
./download_app_generalization_apks.sh     # fetch from the catalog urls
./install_app_generalization_apks.sh      # install onto the running emulator
./grant_permissions_app_generalization.sh # pre-grant runtime permissions
python scripts/provision_and_attest_catbench_apps.py   # record version + apk sha256
```

Version pinning matters. `app_versions_pinned.csv` and the per-episode `apk_sha256` field
are what make a result comparable across months, because live apps change under you.

### 3.4 Credentials

API keys live in `benchmark/configs/catbench.env`, which is git-ignored and should stay
mode `0600`. Nothing in this repository should ever contain a literal key. Load it with:

```bash
set -a && source benchmark/configs/catbench.env && set +a
```

---

## 4. Task model and conventions

Each generated task class carries the attributes the harness uses to pair episodes across
applications:

| Attribute | Meaning |
| --- | --- |
| `catbench_semantic_id` | template identity shared by every app implementing it (for example `SmsSendReceivedAddress`) |
| `catbench_app_display_name` | the app name as it appears in the rendered instruction |
| `package_name` / `app_names` | the Android package the task targets |
| `template` | instruction template, formatted with the app display name |
| `complexity` | drives the step budget via `_allocate_step_budget` |

Two derived quantities depend on `catbench_semantic_id` and must be understood before
running anything:

1. **Shared parameter seeding.** `suite_utils._get_instance_seed` derives the instance seed
   from `sha256(f'{seed}_{namespace}_{i}')` where the namespace is the semantic id. Apps
   that share a semantic id therefore draw *byte-identical parameters*, which is what makes
   "same task, different app" literally true rather than only distributionally true.
2. **Shared C2 plans.** The sub-goal generator is keyed by
   `semantic_task_id|instance|sha256(app-neutral goal)`, so every app of a template
   receives one and the same plan.

Without `catbench_semantic_id` both properties silently degrade to per-app behaviour.

The action space accepted from agents is defined by
`android_world/agents/mobile_action_schema.py::SUPPORTED_MOBILE_ACTIONS`:
`click`, `long_press`, `type`, `swipe`, `wait`, `answer`, `open`, `open_app`,
`system_button`, `terminate`.

---

## 5. The two execution conditions

`CATBENCH_CONDITION` labels every episode and is validated against what actually happened.
Valid values are `c1`, `c2_g` (Gemini-generated sub-goals) and `c2_o` (OpenAI-generated
sub-goals). A declared condition that contradicts the runtime state sets
`catbench_condition_config_valid=False` on the episode rather than being silently accepted.

### 5.1 C1 — direct execution

The agent receives the user instruction and nothing else.

```bash
export CATBENCH_CONDITION=c1
python scripts/run_catbench_5cat_matrix.py \
  --models UI-Venus-7B,GUI-Owl-7B,MAI-UI-8B \
  --n_task_combinations 3 \
  --task_random_seed 30 \
  --run_id c1_k3
```

### 5.2 C2 — sub-goal augmented execution

An external text-only generator reads the instruction with the app name replaced by
`[TARGET_APP]` and emits a short application-independent list of sub-goals. That list is
prepended to the agent's prompt for the whole episode. The generator never sees a
screenshot, an accessibility tree, a trajectory, an action, or any feedback, and the
validator rejects plans containing coordinates, accessibility identifiers, package names,
app names or low-level UI verbs.

C1 and C2 share the same initialisation, the same parameters and the same verifier. The
original instruction stays the evaluation goal; only the *prompt* goal is augmented.

**Step 1 — generate the plans** (one model call per semantic instance, reused across apps):

```bash
python scripts/generate_task_breakdowns.py \
  --output  /path/to/plans/c2_gemini_plans.json \
  --audit_log /path/to/plans/c2_gemini_attempts.jsonl \
  --provider gemini --model gemini-3.1-pro-preview \
  --cohort_manifest configs/catbench_5cat_primary_cohort.json \
  --n_task_combinations 3 --task_random_seed 30 \
  --strict_forbidden_check
```

Add `--dry_run` to print the schedule and the plan count without making a single API call.
Frozen-cohort generation refuses to run without an audit log, refuses `--overwrite`, and
refuses non-strict validation.

**Step 2 — preflight** (mandatory; the single best guard against a mid-run missing plan):

```bash
python scripts/preflight_task_breakdowns.py \
  --breakdown_file /path/to/plans/c2_gemini_plans.json \
  --categories sms,files,maps,contacts,clock \
  --n_task_combinations 3 --task_random_seed 30 \
  --report_json /path/to/plans/c2_gemini_plans.preflight.json
```

It exits non-zero on any scheduled instance missing from the file, any duplicate or legacy
entry lacking an exact instance-aware key, and any runner/file metadata mismatch (seed,
suite family, `n_task_combinations`, `fixed_task_seed`).

**Step 3 — run**:

```bash
export CATBENCH_CONDITION=c2_g
export CATBENCH_TASK_BREAKDOWN_FILE=/path/to/plans/c2_gemini_plans.json
export CATBENCH_TASK_BREAKDOWN_MODE=prepend
export CATBENCH_TASK_BREAKDOWN_REQUIRED=1     # missing plan aborts the episode
python scripts/run_catbench_5cat_matrix.py --models ... --run_id c2g_k2
```

With `CATBENCH_TASK_BREAKDOWN_REQUIRED=1` a missing or empty plan produces an
`invalid_infrastructure` episode instead of quietly degrading into a C1 rollout. This is
the property that makes matched C1/C2 pairing trustworthy.

Every C2 episode records `prompt_goal`, `task_breakdown_text`, `plan_file_sha256`,
`semantic_goal`, `semantic_goal_sha256` and `semantic_parameter_sha256`, so the condition
can be re-audited from the results alone without the plan file.

---

## 6. Running evaluations

| Entry point | Use |
| --- | --- |
| `scripts/run_catbench_5cat_matrix.py` | the main matrix launcher: models x categories x lanes |
| `scripts/run_catbench_target_cells.py` | re-run a named list of cells (repairs, top-ups) |
| `scripts/build_catbench_frozen_schedule.py` | freeze an immutable schedule manifest |
| `scripts/consume_catbench_frozen_schedule.py` | execute a frozen schedule |
| `run_<agent>.py` | single-agent entry points (`run_qwen3vl.py`, `run_gui_owl.py`, `run_maiui.py`, `run_ui_voyager.py`, `run_openai_python_action.py`, ...) |
| `scripts/preflight_catbench_aw_env.py` | validate the emulator/app environment before launch |

Useful runtime switches:

| Variable | Default | Effect |
| --- | --- | --- |
| `CATBENCH_EARLY_STOP_ON_SUCCESS` | `1` | end the episode the first step the verifier passes, so post-completion drift cannot turn a real success into a failure. Set `0` for strict AndroidWorld final-state-only scoring. |
| `CATBENCH_TASK_TIMEOUT_SECONDS` | — | per-task wall clock cap |
| `CATBENCH_INSTANCE_ID` | — | pin one scheduled instance |
| `CATBENCH_SKIP_AW_ENV_PREFLIGHT` | unset | skip the environment preflight (debugging only) |

Episodes are written as pickles under the run directory and carry a large provenance
block: `code_revision`, `model_revision`, `apk_sha256`, `app_version`, `cohort_sha256`,
`runner_config_sha256`, `model_config_sha256`, `plan_file_sha256`, `seed`, `instance_id`,
`catbench_episode_status` and the exception attribution fields. `catbench_episode_status`
is one of `valid_success`, `valid_failure`, `invalid_infrastructure`.

---

## 7. Failure analysis

Success is always the verifier's. The judging pipeline only explains *why* a
verifier-failed rollout failed, using a six-class taxonomy (planning, grounding, mixed,
execution/tooling, environment/evaluator, unknown).

```bash
# 1. Two-stage Gemini judge over verifier-failed trajectories.
python scripts/classify_catbench_failures.py --manifest <run_manifest.json>

# 2. Independent Qwen re-judge of the same cases (judge-model sensitivity).
python scripts/rejudge_c1_failures_qwen.py \
  --gemini_csv <gemini_results.csv> --out_dir <out> --workers 4

# 3. Cross-judge agreement.
python scripts/compare_failure_judges.py \
  --gemini_csv <gemini.csv> --qwen_jsonl <qwen.jsonl> --out_dir <out>

# 4. Human validation subset and blind annotation UI.
python scripts/prepare_proportional_human_validation.py ...
python scripts/serve_annotation_app.py \
  --sample_manifest <manifest.json> --out_dir <out> --annotator <name> --blind

# 5. Judge-versus-human agreement (accuracy and Cohen's kappa).
python scripts/validate_judge_vs_human.py \
  --annotations <annotations.jsonl> --judge_jsonl <judge.jsonl> --out_dir <out>
```

`--blind` hides every automated judge output from the annotator, which is what makes the
human labels usable as an independent reference rather than a confirmation of the judge.

---

## 8. Reporting

| Script | Output |
| --- | --- |
| `scripts/report_catbench_5cat_results.py` | headline per-model results |
| `scripts/report_c1_app_level_frozen.py` | per-application C1 tables from a frozen schedule |
| `scripts/report_c2_per_app_delta.py` | per-application C2 minus C1 deltas |
| `scripts/report_catbench_hierarchical_metrics.py` | equal-app-weighted metrics |
| `scripts/report_catbench_paired_hierarchical_contrasts.py` | matched paired contrasts |
| `scripts/report_catbench_attrition_bounds.py` | bounds under episode attrition |
| `scripts/write_catbench_paper_tables_md.py` | markdown tables |
| `scripts/build_catbench_failure_report.py` | failure-mode report |
| `scripts/report_app_generalization_delta.py` | cross-app gap summary |

Reports refuse to consume manifests that are not marked analysis-eligible, so a
development matrix cannot silently become a headline number.

---

## 9. Testing

```bash
cd benchmark
python -m pytest -q task_breakdowns_test.py \
                   android_world/suite_utils_test.py \
                   android_world/episode_runner_test.py \
                   android_world/task_evals/single/app_generalization_generated/
python -m pytest -q scripts/
```

The tests that matter most for experimental validity:

- `task_breakdowns_test.py` — plan lookup keys, fail-closed behaviour, prompt-goal
  composition and its exact inverse.
- `android_world/suite_utils_test.py` — that the agent receives the augmented goal, that a
  missing plan under `REQUIRED=1` yields `invalid_infrastructure`, and that condition
  labelling matches runtime state.
- `scripts/verify_catbench_task_support.py` — that the profiles only schedule registered,
  non-placeholder task classes.
