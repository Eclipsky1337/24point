from __future__ import annotations

import argparse
from pathlib import Path

from twentyfour.data import load_game_of_24, load_nlile_24game


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_nlile_24game(limit=args.train_limit)
    hard = load_game_of_24(mode="hard", limit=args.eval_limit)
    train.to_json(out_dir / "train_nlile_solvable.jsonl", orient="records", lines=True, force_ascii=False)
    hard.to_json(out_dir / "eval_game_of_24_hard.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"train: {len(train)} rows -> {out_dir / 'train_nlile_solvable.jsonl'}")
    print(f"hard eval: {len(hard)} rows -> {out_dir / 'eval_game_of_24_hard.jsonl'}")
    if len(train):
        print(train[0]["prompt"])


if __name__ == "__main__":
    main()

