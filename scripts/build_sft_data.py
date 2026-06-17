from __future__ import annotations

import argparse
import json
from pathlib import Path

from twentyfour.data import load_nlile_24game
from twentyfour.solver import format_sft_completion, solve_24


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", default="data/processed/sft_train.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    download_mode = "force_redownload" if args.force_download else None
    dataset = load_nlile_24game(limit=args.limit, download_mode=download_mode)

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_file.open("w", encoding="utf-8") as f:
        for row in dataset:
            expression = solve_24(row["numbers"])
            if not expression:
                continue
            record = {
                "numbers": row["numbers"],
                "prompt": row["prompt"],
                "completion": format_sft_completion(expression),
                "expression": expression,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"wrote {kept} SFT rows -> {out_file}")


if __name__ == "__main__":
    main()

