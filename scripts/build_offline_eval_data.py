from __future__ import annotations

import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path

from twentyfour.prompts import build_prompt
from twentyfour.solver import solve_24


HARD_GAME_OF_24_NUMBERS = [
    [4, 5, 6, 10],
    [1, 2, 4, 7],
    [2, 5, 8, 11],
    [3, 4, 4, 13],
    [6, 7, 8, 9],
    [1, 11, 11, 13],
    [1, 8, 10, 11],
    [2, 3, 6, 9],
    [1, 3, 5, 9],
    [3, 3, 7, 12],
    [4, 5, 7, 9],
    [1, 2, 8, 13],
    [4, 6, 6, 9],
    [1, 4, 4, 8],
    [1, 5, 10, 11],
    [3, 4, 6, 11],
    [2, 4, 8, 9],
    [1, 4, 5, 13],
    [2, 2, 7, 12],
    [3, 3, 6, 7],
    [1, 5, 9, 13],
    [5, 6, 7, 13],
    [5, 5, 8, 10],
    [2, 4, 6, 12],
    [6, 7, 8, 11],
    [7, 9, 9, 13],
    [3, 6, 9, 12],
    [6, 9, 12, 13],
    [4, 7, 9, 13],
    [5, 6, 8, 12],
    [2, 4, 6, 7],
    [2, 5, 10, 10],
    [6, 6, 7, 12],
    [6, 9, 9, 11],
    [5, 8, 11, 12],
    [5, 6, 8, 10],
    [6, 11, 12, 13],
    [2, 2, 8, 8],
    [2, 7, 12, 13],
    [2, 6, 8, 12],
    [3, 4, 9, 13],
    [4, 5, 10, 12],
    [1, 2, 7, 11],
    [4, 5, 6, 8],
    [6, 10, 12, 13],
    [1, 3, 9, 9],
    [1, 4, 4, 11],
    [2, 3, 9, 10],
    [1, 2, 3, 13],
    [1, 6, 6, 6],
    [1, 2, 2, 9],
    [1, 3, 6, 11],
    [5, 10, 12, 13],
    [2, 3, 6, 6],
    [6, 7, 10, 12],
    [7, 8, 8, 12],
    [3, 4, 6, 8],
    [1, 7, 9, 11],
    [2, 3, 6, 13],
    [2, 2, 5, 12],
    [2, 6, 8, 13],
    [8, 8, 10, 12],
    [1, 3, 8, 13],
    [4, 4, 7, 10],
    [1, 7, 10, 13],
    [1, 9, 10, 13],
    [3, 3, 4, 11],
    [2, 5, 7, 7],
    [3, 9, 10, 13],
    [2, 3, 4, 7],
    [4, 4, 8, 12],
    [1, 2, 6, 10],
    [1, 5, 12, 12],
    [5, 6, 6, 8],
    [7, 7, 8, 11],
    [1, 3, 7, 10],
    [3, 3, 9, 12],
    [3, 5, 7, 10],
    [4, 10, 12, 13],
    [2, 3, 10, 12],
    [3, 4, 6, 6],
    [5, 8, 8, 8],
    [6, 8, 8, 12],
    [2, 3, 4, 9],
    [2, 6, 7, 11],
    [5, 9, 12, 12],
    [1, 2, 7, 12],
    [2, 4, 5, 6],
    [5, 5, 8, 13],
    [2, 3, 3, 10],
    [3, 4, 8, 12],
    [2, 4, 6, 11],
    [2, 2, 8, 9],
    [1, 5, 6, 7],
    [5, 8, 10, 11],
    [4, 4, 9, 12],
    [2, 5, 6, 6],
    [2, 4, 9, 12],
    [4, 8, 11, 13],
    [4, 9, 10, 13],
]


def write_records(path: Path, numbers_list: list[list[int]], solvable: bool, source: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for numbers in numbers_list:
            record = {
                "numbers": numbers,
                "prompt": build_prompt(numbers),
                "solvable": solvable,
                "source": source,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generated_splits(limit: int) -> tuple[list[list[int]], list[list[int]]]:
    solvable: list[list[int]] = []
    unsolvable: list[list[int]] = []
    for numbers_tuple in combinations_with_replacement(range(1, 14), 4):
        numbers = list(numbers_tuple)
        if solve_24(numbers) is None:
            if len(unsolvable) < limit:
                unsolvable.append(numbers)
        elif len(solvable) < limit:
            solvable.append(numbers)
        if len(solvable) >= limit and len(unsolvable) >= limit:
            break
    return solvable, unsolvable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    solvable, unsolvable = generated_splits(args.limit)
    write_records(
        out_dir / "eval_generated_solvable_proxy.jsonl",
        solvable,
        True,
        "generated/full-24-combinations",
    )
    write_records(
        out_dir / "eval_generated_unsolvable.jsonl",
        unsolvable,
        False,
        "generated/full-24-combinations",
    )
    write_records(
        out_dir / "eval_game_of_24_hard_fallback.jsonl",
        HARD_GAME_OF_24_NUMBERS[: args.limit],
        True,
        "test-time-compute/game-of-24 hard slice from reference processed eval",
    )
    print(f"generated solvable proxy: {len(solvable)}")
    print(f"generated unsolvable: {len(unsolvable)}")
    print(f"hard fallback: {min(args.limit, len(HARD_GAME_OF_24_NUMBERS))}")


if __name__ == "__main__":
    main()
