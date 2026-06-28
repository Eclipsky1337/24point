# Work Log

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
