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
