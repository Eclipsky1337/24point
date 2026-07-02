from __future__ import annotations

import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path

from twentyfour.data import load_game_of_24, load_nlile_24game
from twentyfour.prompts import build_prompt
from twentyfour.solver import solve_24


def write_generated_unsolvable(path: Path, limit: int | None) -> int:
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for numbers_tuple in combinations_with_replacement(range(1, 14), 4):
            numbers = list(numbers_tuple)
            if solve_24(numbers) is not None:
                continue
            record = {
                "numbers": numbers,
                "prompt": build_prompt(numbers),
                "solvable": False,
                "source": "generated/full-24-combinations",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if limit and written >= limit:
                break
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--unsolvable-limit", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--force-download", action="store_true", help="Redownload datasets instead of reusing HF cache.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    download_mode = "force_redownload" if args.force_download else None
    train = load_nlile_24game(solvable_only=False, limit=args.train_limit, download_mode=download_mode)
    eval_set = load_game_of_24(
        mode="nonhard",
        limit=args.eval_limit,
        download_mode=download_mode,
        sample_seed=args.eval_seed,
    )
    unsolvable = load_nlile_24game(
        solvable_only=False,
        limit=None,
        download_mode=download_mode,
    ).filter(lambda row: not row["solvable"])
    if args.unsolvable_limit:
        unsolvable = unsolvable.select(range(min(args.unsolvable_limit, len(unsolvable))))
    train.to_json(out_dir / "train_nlile_all.jsonl", orient="records", lines=True, force_ascii=False)
    eval_set.to_json(out_dir / "eval_game_of_24_nonhard_100.jsonl", orient="records", lines=True, force_ascii=False)
    unsolvable_path = out_dir / "eval_nlile_unsolvable.jsonl"
    if len(unsolvable):
        unsolvable.to_json(unsolvable_path, orient="records", lines=True, force_ascii=False)
        unsolvable_count = len(unsolvable)
    else:
        unsolvable_count = write_generated_unsolvable(unsolvable_path, args.unsolvable_limit)

    print(f"train: {len(train)} rows -> {out_dir / 'train_nlile_all.jsonl'}")
    print(f"eval: {len(eval_set)} rows -> {out_dir / 'eval_game_of_24_nonhard_100.jsonl'}")
    print(f"unsolvable eval: {unsolvable_count} rows -> {unsolvable_path}")
    if len(train):
        print(train[0]["prompt"])


if __name__ == "__main__":
    main()
