# CATBench Task-Breakdown Condition

> **SUPERSEDED FOR PAPER RERUNS (2026-07-10).** The commands below describe
> the legacy one-instance, per-app-plan pipeline and must not be used to replace
> Table 2. Use `acl_revision_experiment_protocol.md`, generate three shared
> semantic instances per template, and require the revised semantic-plan
> preflight. Legacy GPT-5.4/Gemini files are intentionally rejected.

This condition gives the evaluated agent the same user instruction plus a
precomputed, application-independent task breakdown for the whole episode.
The breakdown model is text-only: it receives no screen, no accessibility
tree, no app-specific UI layout, and no action history.

We generate breakdowns with **two independent planners** — Gemini 3.1 Pro
Preview and GPT-5.1 — so the published result is not confounded by a single
planner's bias. Each generator yields a separate JSON file; the matrix is
run once per file plus once for the baseline.

## 0. Senior-review fixes baked in

Compared to the first draft of this pipeline, the following are now enforced:

- **H1** Preflight script (`preflight_task_breakdowns.py`) validates that
  every scheduled task has an entry **before** launch.
- **H2** Runner config (`task_random_seed`, `n_task_combinations`,
  `fixed_task_seed`, `suite_family`) is recorded in the file's
  `metadata` and validated at preflight time.
- **H4** `--strict_forbidden_check` is on by default — generations that
  trip a forbidden-pattern regex fail loudly.
- **H5** Forbidden-pattern regex expanded to cover app names from your
  matrix, FAB/hamburger/notification-shade phrasing, and more.
- **L3** Prompt template is SHA-hashed into metadata; resuming after a
  prompt change is blocked unless explicitly allowed.
- **M5** Empty breakdowns are tagged `empty_breakdown: True` instead of
  silently degrading to baseline.
- **M6** Every episode now carries `catbench_condition ∈
  {baseline, breakdown, breakdown_missing_or_empty}` regardless of mode,
  so downstream analysis can split runs by condition from the pkl alone.

## 1. Generate the breakdowns (one file per generator)

### 1a. Gemini 3.1 Pro Preview

```bash
export GEMINI_API_KEY=...

python benchmark/scripts/generate_task_breakdowns.py \
  --output $HOME/anyappbench_plans/gemini31_seed30_5cat.json \
  --provider gemini \
  --model gemini-3.1-pro-preview \
  --categories sms,files,maps,contacts,clock \
  --n_task_combinations 3 \
  --task_random_seed 30 \
  --resume
```

### 1b. GPT-5.1

```bash
export OPENAI_API_KEY=...

python benchmark/scripts/generate_task_breakdowns.py \
  --output $HOME/anyappbench_plans/gpt51_seed30_5cat.json \
  --provider openai \
  --model gpt-5.1 \
  --categories sms,files,maps,contacts,clock \
  --n_task_combinations 3 \
  --task_random_seed 30 \
  --resume
```

The generator writes one entry per app/task/instance but makes only one model
call per shared semantic instance. The preflight reports both exact counts.

### Dry-run (no API calls)

```bash
python benchmark/scripts/generate_task_breakdowns.py \
  --dry_run --limit 10 \
  --output /tmp/preview.json
```

### Generator output shape

```json
{
  "metadata": {
    "generator_provider": "gemini | openai",
    "generator_model": "gemini-3.1-pro-preview | gpt-5.1",
    "prompt_sha256": "...",
    "suite_family": "android_world",
    "categories": ["sms", ...],
    "n_task_combinations": 3,
    "task_random_seed": 30,
    "fixed_task_seed": false,
    "condition": "application_independent_breakdown_prepend",
    "forbidden_patterns": ["app_name_mention", "coordinate_pair", ...]
  },
  "breakdowns": [
    {
      "key": "TaskName|instance=0|<goal_sha256>",
      "task_template": "TaskName",
      "instance_id": 0,
      "goal": "Original user instruction",
      "goal_sha256": "...",
      "generator_provider": "gemini",
      "generator_model": "gemini-3.1-pro-preview",
      "breakdown": {"steps": ["..."], "notes": []},
      "breakdown_text": "1. ...",
      "validation_warnings": []
    }
  ]
}
```

## 2. Preflight (mandatory before each run)

This is the single most important guard against the "missing entry mid-run"
class of bugs.

```bash
python benchmark/scripts/preflight_task_breakdowns.py \
  --breakdown_file $HOME/anyappbench_plans/gemini31_seed30_5cat.json \
  --categories sms,files,maps,contacts,clock \
  --n_task_combinations 3 \
  --task_random_seed 30 \
  --report_json $HOME/anyappbench_plans/gemini31_seed30_5cat.preflight.json
```

Exits non-zero on:
- any scheduled (template, instance_id, goal_sha256) missing from the file,
- any duplicate or legacy entry without an exact instance-aware key,
- any runner/file metadata mismatch (seed, family, ...),
- (with `--fail_on_warnings`) any entry with a non-empty
  `validation_warnings` list.

Run the preflight for **each generator's file** and for the exact seed /
categories the matrix runner will use.

## 3. Run the condition (once per generator) over ALL 11 models

The condition propagates through `suite_utils.run()` to every agent that uses
the suite runner — i.e. all the agents in the table. Independent code review
verified that **AutoDev, Mobile-Agent-v3, AgentProg, UI-TARS (MAIUI/Qwen3VL),
GUI-Owl, M3A-Venus, Mobile-RL, UI-Voyager, MaiUI** all consume the swapped
goal correctly. **OSAtlas** runs through an external adapter — verify
independently before publishing OSAtlas numbers with this condition.

### 3a. Gemini-31 condition

```bash
export ANDROID_ADB_SERVER_PORT=5039
export CATBENCH_EMULATORS=5800:8800,5802:8801,5804:8802,5806:8803,5808:8804,5810:8805,5812:8806,5814:8807,5816:8808,5818:8809

export CATBENCH_TASK_BREAKDOWN_FILE=$HOME/anyappbench_plans/gemini31_seed30_5cat.json
export CATBENCH_TASK_BREAKDOWN_MODE=prepend
export CATBENCH_TASK_BREAKDOWN_REQUIRED=1

bash benchmark/scripts/launch_5cat_all11_models_experiments.sh \
  --run_id 5cat_breakdown_gemini31_seed30
```

### 3b. GPT-51 condition

```bash
export CATBENCH_TASK_BREAKDOWN_FILE=$HOME/anyappbench_plans/gpt51_seed30_5cat.json
# CATBENCH_TASK_BREAKDOWN_MODE / REQUIRED stay set

bash benchmark/scripts/launch_5cat_all11_models_experiments.sh \
  --run_id 5cat_breakdown_gpt51_seed30
```

The launcher fans out 11 model rows × 5 categories × 25 apps × 10 templates
in parallel against the emulator pool. The env vars propagate to every
subprocess via `os.environ.copy()` (matrix runner line 290), so all agents
see the same breakdown source.

## 4. Run the baseline

Same seed / categories / launcher, but with the breakdown disabled:

```bash
unset CATBENCH_TASK_BREAKDOWN_FILE
unset CATBENCH_TASK_BREAKDOWN_REQUIRED
export CATBENCH_TASK_BREAKDOWN_MODE=off

bash benchmark/scripts/launch_5cat_all11_models_experiments.sh \
  --run_id 5cat_baseline_seed30
```

Baseline episodes now carry `catbench_condition: "baseline"` and the
original `goal` in the pkl, so a single downstream script can split
conditions from one mixed analysis dir.

## 5. Compare baseline vs. both generators

```bash
python benchmark/scripts/pair_baseline_and_breakdown.py \
  --baseline_manifest $HOME/anyappbench_runs/5cat_baseline_seed30/catbench_5cat_manifest.json \
  --breakdown_manifest $HOME/anyappbench_runs/5cat_breakdown_gemini31_seed30/catbench_5cat_manifest.json \
  --breakdown_manifest $HOME/anyappbench_runs/5cat_breakdown_gpt51_seed30/catbench_5cat_manifest.json \
  --label gemini31 --label gpt51 \
  --out_dir $HOME/catbench_paired/5cat_seed30
```

Outputs:
- `paired_summary.md` — per-model SR / Δ / wins / losses / ties for each
  generator
- `paired_summary.json` — same metrics machine-readable
- `paired_per_task.jsonl` — one row per (model, task) with the baseline
  success bool and the per-generator treatment bools, ready for paired
  significance tests (McNemar, sign test, etc.)

## 6. Diversity check across generators

Compare **per-model lift** for Gemini-31 vs. GPT-51:
- If both generators give large positive Δ → the breakdown structure
  generalises and the result is robust to planner choice.
- If only one gives Δ > 0 and the other not → the apparent lift is at
  least partly a "planner identity" effect (the strong Gemini agent
  benefits more from Gemini-authored breakdowns than from GPT-authored
  ones, etc.). Report both and discuss.
- If both give Δ ≈ 0 → the upfront breakdown does not help; report the
  null result honestly.

We deliberately did **not** average across the two generators — that hides
the diversity signal. Present them side by side.

## 7. Cost notes

For one full condition run (Gemini-31 OR GPT-51) over all 11 models the
extra cost at runtime is roughly:
- Breakdown adds ~300-600 tokens to every step's prompt.
- Average episode: ~15 steps → ~6-9k extra input tokens per episode.
- 11 models × 250 tasks × 15 steps ≈ 41k steps. For Pro-tier remote APIs
  the additional input-token cost is non-trivial; local VLMs eat the
  extra context inside their existing window.

Watch context-window pressure on the 8B/7B models (MaiUI-8B, GUI-Owl-7B,
Qwen3VL-8B, UI-Voyager-4B). A pre-condition diagnostic with
`--limit 5 --categories sms` confirms truncation behaviour without burning
compute.

## 8. Reporting checklist

For each table reporting the condition lift:
1. **Cite both generators**: never publish a "with breakdown" column
   averaged across them.
2. **Report `n_paired`, `wins`, `losses`, `ties`** from
   `pair_baseline_and_breakdown.py`, not just SR deltas.
3. **State that the evaluator uses the original `task.goal`**, not the
   augmented `prompt_goal` — this is the experimental-validity property
   that lets the lift be interpreted as causal.
4. **Note OSAtlas separately** if you keep it in the table — it's the
   only agent whose goal-consumption path is not in-tree-verified.
