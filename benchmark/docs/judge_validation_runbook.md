# CATBench Judge Validation Runbook

This runbook covers the VLM-based failure-mode classifier and the
human-annotation workflow that measures its accuracy.

> **Scope (corrected after advisor feedback).** The VLM does NOT produce a
> pass/fail verdict. The programmatic AW/CATBench validators own the
> verdict and define the SR we report. The VLM's only job is to classify
> *why* an already-failed episode failed (planning vs. grounding vs.
> execution vs. environment vs. mixed vs. unknown). Humans label the
> same failure-mode field — they do NOT override the validator's
> pass/fail either.
>
> Consequence: the `--progressive` and `--triage_*` flags on
> `classify_catbench_failures.py` are now **redundant for the default
> pipeline**. They are kept in the code only for the optional
> *validator-audit* pass described at the bottom of this runbook.

## What changed in the judge

[classify_catbench_failures.py](../scripts/classify_catbench_failures.py) now
supports:

| Flag | Effect |
|---|---|
| `--with_screenshots` | Stops pruning images from the pkl.gz and sends key-frame JPEGs (base64) to the multimodal judge. Requires Pillow. |
| `--screenshot_max_dim`, `--screenshot_max_frames`, `--triage_screenshot_max_frames` | Frame-size and budget controls. |
| `--smart_steps` | Replaces head+tail step sampling with error-adjacent / repeated-action / action-type-change selection. |
| `--progressive` | Two-stage scrutiny — cheap triage call resolves obvious cases; only uncertain/low-confidence cases escalate to the deep VLM. |
| `--triage_backend`, `--triage_model`, `--triage_base_url`, `--triage_api_key` | Configure the cheap first stage independently (e.g. `gemini-flash` then `gemini-3-pro`). |

The judgment JSONL now also includes:
- `verdict` (`success` / `failure` / `uncertain`) when the model returns one
- `agree_with_recorded_label` (flags suspected evaluator false-negatives)
- `usage` (token counts) and `stages` (triage + deep traces) when applicable

> **Important:** Screenshots are already inside your existing `pkl.gz`
> files (`raw_screenshot` is `(2400, 1080, 3) uint8`). You do **not** need to
> re-run any agent to enable visual grounding — only re-run the judge with
> `--with_screenshots`.

## Re-run recipes

### Default failure-mode classification (single Gemini 3.1 Pro call per failed episode)
```bash
python3 benchmark/scripts/classify_catbench_failures.py \
  --manifest /path/to/matrix_manifest.json \
  --out_dir /path/to/failure_mode_analysis \
  --env_file benchmark/configs/catbench.env \
  --judge_backend gemini \
  --judge_model gemini-3.1-pro-preview \
  --with_screenshots \
  --smart_steps \
  --screenshot_max_frames 6 \
  --resume \
  --continue_on_judge_error
```

The classifier runs only on validator-failed episodes (no `--include_successes`).
The `--progressive` / `--triage_*` flags are intentionally **not used** — they
add a redundant pass/fail call that the validator already provides.

The model id is not normalized or aliased in code. If
`gemini-3.1-pro-preview` is unavailable to the configured API key, the judge
fails loudly; do not silently substitute another model for paper numbers.

Compare the two output `failure_mode_judgments.jsonl` files with
`validate_judge_vs_human.py` (below) on the same annotated sample to see the
lift screenshots give you.

## Human annotation workflow

### Step A — Run the judge first (or use existing output)
You need a `failure_mode_judgments.jsonl` so the sampler can stratify by
the judge's prior label/confidence. If you don't have one yet, the sampler
falls back to model+category+verdict stratification.

### Step B — Sample episodes
```bash
python3 benchmark/scripts/sample_episodes_for_annotation.py \
  --manifest /path/to/matrix_manifest.json \
  --judge_jsonl /path/to/failure_mode_analysis/failure_mode_judgments.jsonl \
  --out_dir /path/to/annotation_run \
  --total 200 \
  --min_per_model 8 \
  --failure_share 1.0 \
  --double_annotation_share 0.10
```
Outputs `sample_manifest.jsonl` + `sample_summary.md`.

### Step C — Annotate
```bash
python3 benchmark/scripts/annotate_judge_cases.py \
  --sample_manifest /path/to/annotation_run/sample_manifest.jsonl \
  --out_dir /path/to/annotation_run \
  --annotator ttran \
  --pool primary
```
For each episode the script:
1. Saves key-frame JPEGs to `<out_dir>/frames/<episode_id>/step_*.jpg`.
2. Writes a markdown summary at `<out_dir>/frames/<episode_id>/summary.md`.
3. Prompts you on stdin for failure mode + planning/grounding scores +
   confidence + rationale. The validator verdict is read-only.

Resumable: re-running skips episodes already labelled by the same `--annotator`.

If you want a second annotator (for the 10% double-annotation pool):
```bash
python3 benchmark/scripts/annotate_judge_cases.py \
  --sample_manifest /path/to/annotation_run/sample_manifest.jsonl \
  --out_dir /path/to/annotation_run \
  --annotator second_rater \
  --pool double_annotation
```

### Step D — Measure agreement
```bash
python3 benchmark/scripts/validate_judge_vs_human.py \
  --annotations /path/to/annotation_run/annotations.jsonl \
  --judge_jsonl /path/to/failure_mode_analysis/failure_mode_judgments.jsonl \
  --out_dir /path/to/annotation_run/validation
```
Produces:
- `report.md` — macro-F1, per-mode precision/recall/F1, confusion matrix
- `report.json` — same metrics machine-readable
- `mismatches.jsonl` — every disagreement (review by opening the frames dir)
- `agreement.json` — Cohen's κ on the double-annotation pool

You can then re-run the judge with a different config (text-only vs.
screenshots, GPT-5.1 vs. Gemini-3-Pro, etc.) and re-run validation
against the **same** `annotations.jsonl`. That's how MemGUI compares its
M1/M2/M3 configurations.

## Sampling strategy — answering "do we annotate every model?"

**Short answer: no — stratify, don't enumerate.**

The MemGUI authors human-annotated **256 trajectories** for their full
benchmark and **78** for SPA-Bench. Your 11 models × 5 categories × 10
templates = 550 unique tasks per attempt, so ~150-300 annotations is a
reasonable target.

The defaults in `sample_episodes_for_annotation.py` implement the
following recommendations:

1. **Cover every model with a floor (`--min_per_model 8`).** Without a floor,
   models that fail more would dominate the sample and you'd never measure
   judge accuracy on models that mostly succeed.
2. **Use failures only for the classifier validation (`--failure_share 1.0`).**
   The classifier no longer produces a pass/fail verdict; it labels the
   failure mode for validator-failed episodes only. Validator-successes belong
   only in the separate validator audit.
3. **Stratify within a model across (category × judge_label).**
   This guarantees that every failure mode the judge produces has some
   ground-truth examples to compute per-mode F1 against. Otherwise rare
   modes (e.g. `execution_tooling`) end up with support=0 and the macro-F1
   silently ignores them.
4. **Boost low-confidence judge cases.** A `low` confidence judgment counts
   3× as much as `high` in the within-bucket weighting — those are the
   informative disagreements.
5. **Double-annotate ~10% (`--double_annotation_share 0.1`).** That gives
   you a Cohen's κ for inter-rater reliability. If κ < 0.6 your taxonomy
   is too ambiguous and you should reconcile before trusting the F1.

### Concrete budget for your 11-model × 5-category run

| Target | Per-model floor | Total | Time @ ~2 min/case |
|---|---:|---:|---|
| Quick smoke (one annotator, single sitting) | 5 | 60 | ~2 hrs |
| Publishable validation set | 8 | 150-200 | ~6-7 hrs |
| MemGUI-grade rigor | 12 | 250-300 | ~10 hrs |

Don't try to annotate every episode in every model — by the time you finish,
your judge config has changed and the labels are stale. Pick **one good
sample**, freeze it, and re-use it to compare every new judge variant.

### What to *not* do
- **Don't sample purely at random** — you'll over-sample failures from the
  worst model.
- **Don't mix validator-successes into the failure-mode validation set** —
  use the separate validator audit for pass/fail sanity checks.
- **Don't annotate without seeing screenshots** — a grounding failure looks
  like success if you only read the agent's self-narration.

## pass@k / FRR (multi-attempt evaluation)

pass@k is a **runner** change — your existing matrix already accepts a
`--task_random_seed`, so the new wrapper just invokes it `k` times with
different seeds and output dirs, and a separate script aggregates.

### Run k attempts
```bash
python3 benchmark/scripts/run_passk_attempts.py \
  --output_root $HOME/anyappbench_runs/passk \
  --k 3 \
  --base_seed 30 \
  -- \
  --run_id smoke --models gpt-5.1 --categories sms,clock \
  # ... whatever args your matrix runner normally takes
```
After all k attempts complete, this writes `passk_index.json` linking the
manifests.

### Compute pass@1 / pass@k / FRR
```bash
python3 benchmark/scripts/compute_passk_metrics.py \
  --passk_index $HOME/anyappbench_runs/passk/passk_index.json \
  --out_dir $HOME/anyappbench_runs/passk/metrics
# optional: use the judge's verdict instead of recorded is_successful
#   --judge_jsonl /path/to/failure_mode_judgments.jsonl
```
Produces `pass_k_summary.{json,md}` with per-model / per-category breakdowns
and the harmonic-decay FRR matching MemGUI's definition.

### Key caveats for pass@k on CATBench
1. **Compute cost.** k=3 with 11 models × 250 tasks = 8250 attempts. Don't
   run pass@k on the full matrix until you've sanity-checked it on one
   model × one category first.
2. **Agent memory does not persist by default.** Most agents in your matrix
   (UI-TARS, GUI-Owl, Voyager, MaiUI, Qwen3VL) are stateless across runs,
   so their pass@k is purely "did a different seed land on a successful
   trajectory?" — that is **not** the MemGUI definition of FRR. Only
   agents with explicit cross-session memory (Agent-S2-style, AgentProg
   with persistent plan cache) can be meaningfully scored on FRR.
3. **Emulator state.** The base matrix runner does the per-task reset; this
   wrapper does not snapshot-restore between attempts. If your tasks
   depend on cross-task state (e.g. "the contact you added in task 1
   should exist in task 2"), pass@k re-runs may not be reproducible.
4. **Recommended pilot:** run k=3 on one model and one category (~30 tasks
   × 3 = 90 attempts), compute pass@k vs. pass@1, then decide whether the
   delta is worth scaling up.

## Known remaining gaps (future work)

- **Cross-app memory taxonomy.** MemGUI's PMH/ProcMH/OMH modes apply to your
  `_cross_app_base` tasks but are not yet in `DEFAULT_FAILURE_MODES`. Add
  them after you have a baseline F1 on the current 6 modes.
- **pass@k support.** You currently run pass@1. MemGUI's FRR metric requires
  the runner to reset via snapshot and retry — a runner change, not a judge
  change.
- **Step Descriptor stage.** MemGUI generates third-party descriptions of
  each (before, after) pair before the semantic judge sees them. Currently
  the deep judge relies on the agent's own narration + screenshots, which is
  fine if you trust the screenshots; add this stage if you want to catch
  agents that lie about what they did.
- **Strict JSON schema.** OpenAI now supports `json_schema`; Gemini supports
  `responseSchema`. The brittle JSON-extraction fallbacks can shrink once
  every model in your config supports structured output.
