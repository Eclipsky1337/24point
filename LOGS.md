# Work Log

## 2026-06-28 15:57 +08:00 - Base Model Hard-Split Evaluation On HF Data

- User requested baseline testing on HF-backed data and corrected scope to the 100 hard evaluation samples only.
- I mistakenly started an evaluation on `data/processed/train_nlile_solvable.jsonl`; stopped it before any record was appended. No training-data baseline result was kept.
- Ran the correct hard split evaluation on the `nlp` server:
  - model: `/home/ma-user/work/models/Qwen/Qwen2___5-1___5B-Instruct`
  - eval file: `data/processed/eval_game_of_24_hard.jsonl`
  - command settings: `--limit 100 --max_new_tokens 128 --num_samples 1 --temperature 0.8 --dtype float32 --device cuda`
  - record file: `outputs/baseline_hard_hfdata_20260628.jsonl`
- Metrics on 100 hard samples from `test-time-compute/game-of-24` indices 900-1000:
  - `success_rate=0.060`
  - `valid_rate=0.950`
  - `think_rate=1.000`
- No GRPO training or reward iteration was started.

## 2026-06-28 15:43 +08:00 - Rebuilt Hugging Face Dataset Splits With Mirror

- Re-read `/home/wuyan/study/nlp/final/AGENTS.md` and `24point/LOGS.md` before acting.
- On the `nlp` server, removed stale fallback/proxy artifacts if present:
  - `scripts/build_offline_eval_data.py`
  - `data/processed/eval_game_of_24_hard_fallback.jsonl`
  - `data/processed/eval_generated_solvable_proxy.jsonl`
  - `data/processed/eval_generated_unsolvable.jsonl`
  - `outputs/base_eval_records.jsonl`
- Ran `HF_ENDPOINT=https://hf-mirror.com python scripts/prepare_data.py --out-dir data/processed --force-download` in `/home/ma-user/work/24point`.
- Hugging Face mirror download succeeded and rebuilt:
  - `data/processed/train_nlile_solvable.jsonl`: 1362 rows
  - `data/processed/eval_game_of_24_hard.jsonl`: 100 rows
  - `data/processed/eval_nlile_unsolvable.jsonl`: 458 rows from `nlile/24-game`
- Verified no proxy/fallback/generated-unsolvable artifact filenames remain on the server.
- Server git status after this phase: intended deletion of `scripts/build_offline_eval_data.py`; unrelated untracked `src/twentyfour_grpo.egg-info/` remains from server installation.

## 2026-06-28 16:02 +08:00 - Retrying Hugging Face Dataset Download With Mirror

- User requested retrying Hugging Face dataset download with `HF_ENDPOINT=https://hf-mirror.com`.
- Read `/home/wuyan/study/nlp/final/AGENTS.md` and this log before acting.
- Local `24point/` working tree is clean.
- Server `24point/` still has generated fallback artifacts from the earlier HF outage:
  - `scripts/build_offline_eval_data.py`
  - `data/processed/eval_game_of_24_hard_fallback.jsonl`
  - `data/processed/eval_generated_solvable_proxy.jsonl`
  - `data/processed/eval_generated_unsolvable.jsonl`
  - `outputs/base_eval_records.jsonl`
- Next action: remove those fallback/proxy artifacts on the server and run `scripts/prepare_data.py` with the HF mirror endpoint to rebuild official dataset-backed splits.
- Removed stale server fallback artifacts:
  - `scripts/build_offline_eval_data.py`
  - `data/processed/eval_game_of_24_hard_fallback.jsonl`
  - `data/processed/eval_generated_solvable_proxy.jsonl`
  - `data/processed/eval_generated_unsolvable.jsonl`
  - `outputs/base_eval_records.jsonl`
- Ran `HF_ENDPOINT=https://hf-mirror.com python scripts/prepare_data.py --out-dir data/processed --force-download` on the server.
- Hugging Face mirror download succeeded.
- Rebuilt official dataset files on the server:
  - `data/processed/train_nlile_solvable.jsonl`: 1362 rows from `nlile/24-game`
  - `data/processed/eval_game_of_24_hard.jsonl`: 100 rows from `test-time-compute/game-of-24` hard slice
  - `data/processed/eval_nlile_unsolvable.jsonl`: 458 unsolvable rows from `nlile/24-game`
- Confirmed old proxy filenames are no longer present on the server.

## 2026-06-28 14:37 +08:00 - Initial Inspection Before Base Evaluation

- Read the project guidance from `/home/wuyan/study/nlp/final/AGENTS.md` supplied in the conversation.
- Confirmed `24point/` is the target repository; `24-game-grpo/` is reference-only; `24point-referer/` is prior-work reference-only.
- Inspected the current `24point/` structure, including `README.md`, `scripts/evaluate.py`, `scripts/prepare_data.py`, `src/twentyfour/data.py`, `src/twentyfour/prompts.py`, `src/twentyfour/rewards.py`, `src/twentyfour/verifier.py`, and `tests/test_verifier.py`.
- Observed the repository already has many modified files before my edits. I will treat those as user/prior-agent work and avoid reverting them.
- Current evaluator can run model evaluation and record per-example completions to JSONL with `--record_file`.
- Important caveat found: the evaluator's `valid_rate` currently counts any expression that parses/evaluates to a value, not strictly expressions that use exactly the given numbers once. This matters for the required "valid-expression rate" reporting.
- Attempted local verification with `python -m pytest -q`; it failed because the active Python environment does not have `pytest` installed.
- No GRPO training or reward iteration has been started. Per project guidance, the next required milestone remains a one-time full base model evaluation, then reporting results and waiting for explicit confirmation before GRPO work.

## 2026-06-28 14:45 +08:00 - Switched To Server-Only Execution

- User explicitly directed: do not test locally; use the `nlp` server.
- I will not run further local tests or local model evaluation.
- Confirmed `ssh nlp` works and lands in `/home/ma-user` on host `notebook-936bea3e-c862-4fca-8b53-e4d7f69b841d`.
- Confirmed local git remote is `https://github.com/wuyan1345/24point` on branch `main`.
- To follow the required local/server synchronization path, the current target repository state must be committed and pushed before server-side clone/pull and execution.

## 2026-06-28 14:51 +08:00 - Server Environment Ready

- Committed and pushed checkpoint `c552191` so the server can run the same repository state via git.
- Cloned the repository to `/home/ma-user/work/24point` on `nlp`.
- Server runtime: Python 3.10.6, Tesla V100-PCIE-32GB, NVIDIA driver 470.57.02, CUDA 11.4 reported by `nvidia-smi`.
- Preserved the server's existing `torch==1.12.1+cu113`; an initial dependency install attempted to resolve a newer Torch, so I switched to no-dependency installation for the evaluation-critical packages.
- Installed/import-checked `datasets==2.21.0`, `transformers==4.46.3`, `accelerate==1.2.1`, `pytest==9.1.1`, `safetensors==0.8.0`, and `sentencepiece==0.2.1` on the server.
- Server-side verification passed with `python -m pytest -q`: 11 tests passed.
- No local tests or local evaluations were run after the user directed server-only execution.

## 2026-06-28 15:00 +08:00 - Dataset And Model Access Decisions

- Server-side Hugging Face dataset loading failed for `nlile/24-game` with `ProxyError`.
- A direct Hugging Face rows API probe for `test-time-compute/game-of-24` also timed out through the server proxy.
- No usable Hugging Face dataset cache was found on the server.
- Added `scripts/build_offline_eval_data.py` as a server-runnable fallback:
  - hard split uses the 100-number processed reference list from `24-game-grpo/data/processed/eval.jsonl`;
  - unsolvable split is generated on the server from all 1-13 four-number combinations using the target solver;
  - in-distribution is only a generated solvable proxy while HF access is blocked, not a true `nlile/24-game` sample.
- Direct Hugging Face model access for `Qwen/Qwen2.5-1.5B-Instruct` timed out on a small `config.json` probe.
- Installed `modelscope` on the server and started downloading `Qwen/Qwen2.5-1.5B-Instruct` from ModelScope to `/home/ma-user/work/models`.

## 2026-06-28 15:07 +08:00 - Evaluation Runtime Fix

- ModelScope download completed at `/home/ma-user/work/models/Qwen/Qwen2___5-1___5B-Instruct`.
- First hard evaluation attempt failed before generation because `transformers` imported an incompatible server `torchvision==0.13.1+cu113`; removed `torchvision` on the server to unblock text-only Qwen loading.
- Second hard evaluation attempt loaded the model but ran CPU-bound with 0 MiB GPU memory, making the full evaluation impractical.
- Stopped that CPU-bound process before completion.
- Updated `scripts/evaluate.py` to resolve `--device auto` to CUDA when available, explicitly move the model to that device, record the resolved device, and flush per-example output.
- A subsequent GPU run produced repeated `!` tokens for the first 15 examples. Comparing against `24-game-grpo` showed the target evaluator was not applying the tokenizer chat template.
- Stopped that partial run before it completed or appended an experiment record.
- Updated `scripts/evaluate.py` to apply `tokenizer.apply_chat_template(..., add_generation_prompt=True)` when available and to set `pad_token=eos_token` when needed.

## 2026-06-28 15:19 +08:00 - fp32 Evaluation Smoke Check

- User suggested using fp32 to avoid repeated `!` outputs.
- Synced the chat-template evaluator fix to the server.
- Ran a 5-example hard-split smoke evaluation with `--dtype float32 --device cuda --max_new_tokens 128`.
- Result: no repeated `!` outputs; all five completions had `<think>`/`<answer>` structure and valid arithmetic expressions, but none solved the examples.
- Smoke metrics: `valid_rate=1.000`, `success_rate=0.000`, `think_rate=1.000`.
- Next step is to run the 100-example base evaluations with fp32.

## 2026-06-28 15:27 +08:00 - Base Model Evaluation Results

- Ran base `Qwen/Qwen2.5-1.5B-Instruct` from `/home/ma-user/work/models/Qwen/Qwen2___5-1___5B-Instruct` on the server V100.
- Generation settings for all full runs: greedy single sample (`num_samples=1`), `temperature=0.8` recorded but sampling disabled, `max_new_tokens=128`, `dtype=float32`, `device=cuda`.
- Detailed per-example records were appended to `/home/ma-user/work/24point/outputs/base_eval_records.jsonl`.
- Hard fallback split (`eval_game_of_24_hard_fallback.jsonl`, 100 rows from the reference processed hard list): `success_rate=0.060`, `valid_rate=0.950`, `think_rate=1.000`.
- Generated solvable proxy split (`eval_generated_solvable_proxy.jsonl`, 100 rows): `success_rate=0.040`, `valid_rate=0.920`, `think_rate=1.000`.
- Generated unsolvable split (`eval_generated_unsolvable.jsonl`, 100 rows): `success_rate=0.000`, `valid_rate=0.790`, `think_rate=1.000`, `unsolvable_valid_rate=0.790`, `unsolvable_success_rate=0.000`.
- Because Hugging Face dataset access is blocked on the server, the in-distribution value is a generated solvable proxy rather than a true `nlile/24-game` evaluation.
- No GRPO training or reward iteration has been started. Per project guidance, wait for explicit user confirmation before GRPO work.
