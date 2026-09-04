# CATBench Failure Mode Analysis

Use `benchmark/scripts/classify_catbench_failures.py` to classify failed
episodes as planning, grounding, mixed, execution/tooling,
environment/evaluator, or unknown.

Current run example:

```bash
MANIFEST=$HOME/anyappbench_results/20260511_045231_5cat_mobile_agentprog_autodev/matrix/20260511_045231_5cat_mobile_agentprog_autodev/catbench_5cat_manifest.json
OUT=$HOME/anyappbench_results/20260511_045231_5cat_mobile_agentprog_autodev/matrix/20260511_045231_5cat_mobile_agentprog_autodev/failure_mode_analysis
```

Build judge inputs without calling an LLM:

```bash
python benchmark/scripts/classify_catbench_failures.py \
  --manifest "$MANIFEST" \
  --judge_backend dry_run \
  --limit 20 \
  --out_dir "$OUT"
```

Run quick no-network heuristic triage:

```bash
python benchmark/scripts/classify_catbench_failures.py \
  --manifest "$MANIFEST" \
  --judge_backend heuristic \
  --limit 50 \
  --out_dir "$OUT"
```

Run the LLM judge with an OpenAI-compatible chat-completions endpoint:

```bash
python benchmark/scripts/classify_catbench_failures.py \
  --manifest "$MANIFEST" \
  --judge_backend llm \
  --env_file benchmark/configs/catbench.env \
  --judge_model "${FAILURE_JUDGE_MODEL:-${OPENAI_MODEL:-gpt-5.1}}" \
  --judge_base_url "${FAILURE_JUDGE_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1/chat/completions}}" \
  --limit 50 \
  --resume \
  --out_dir "$OUT"
```

Run the Gemini REST judge directly:

```bash
python benchmark/scripts/classify_catbench_failures.py \
  --manifest "$MANIFEST" \
  --judge_backend gemini \
  --env_file benchmark/configs/catbench.env \
  --judge_model "${FAILURE_JUDGE_MODEL:-gemini-3.1-pro-preview}" \
  --limit 50 \
  --resume \
  --out_dir "$OUT"
```

UI Voyager all-model run example:

```bash
MANIFEST=$HOME/anyappbench_results/20260502_120228_5cat_all11/matrix/20260502_120228_5cat_all11/catbench_5cat_manifest.json
OUT=$HOME/anyappbench_results/20260502_120228_5cat_all11/matrix/20260502_120228_5cat_all11/failure_mode_analysis/ui_voyager_gemini3

python benchmark/scripts/classify_catbench_failures.py \
  --manifest "$MANIFEST" \
  --judge_backend gemini \
  --env_file benchmark/configs/catbench.env \
  --judge_model gemini-3.1-pro-preview \
  --model 'UI Voyager-4B' \
  --resume \
  --out_dir "$OUT"
```

Useful filters:

```bash
# Focus one model/category.
python benchmark/scripts/classify_catbench_failures.py \
  --manifest "$MANIFEST" \
  --judge_backend llm \
  --env_file benchmark/configs/catbench.env \
  --model AutoDev \
  --category contacts \
  --limit 30 \
  --resume \
  --out_dir "$OUT/autodev_contacts"

# Focus tasks matching a regex.
python benchmark/scripts/classify_catbench_failures.py \
  --manifest "$MANIFEST" \
  --judge_backend llm \
  --env_file benchmark/configs/catbench.env \
  --task_regex 'Contacts(Call|Message|AddFavorite)' \
  --limit 30 \
  --resume \
  --out_dir "$OUT/contact_action_tasks"
```

Outputs:

- `failure_judge_inputs.jsonl`: compacted trace/log inputs sent to the judge.
- `failure_mode_judgments.jsonl`: per-episode judgments and evidence.
- `failure_mode_summary.json`: aggregate counts.
- `failure_mode_summary.md`: human-readable summary tables.
- `cache/*.json`: per-episode cached LLM judgments for `--resume`.
