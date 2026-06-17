from __future__ import annotations

from functools import lru_cache
from fractions import Fraction


def _combine(left: tuple[Fraction, str], right: tuple[Fraction, str]) -> list[tuple[Fraction, str]]:
    a, ae = left
    b, be = right
    out = [
        (a + b, f"({ae}+{be})"),
        (a - b, f"({ae}-{be})"),
        (b - a, f"({be}-{ae})"),
        (a * b, f"({ae}*{be})"),
    ]
    if b != 0:
        out.append((a / b, f"({ae}/{be})"))
    if a != 0:
        out.append((b / a, f"({be}/{ae})"))
    return out


@lru_cache(maxsize=None)
def _search(items: tuple[tuple[Fraction, str], ...], target: Fraction) -> str | None:
    if len(items) == 1:
        return items[0][1] if items[0][0] == target else None

    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            rest = tuple(items[k] for k in range(n) if k not in (i, j))
            for combined in _combine(items[i], items[j]):
                result = _search(rest + (combined,), target)
                if result:
                    return result
    return None


def solve_24(numbers: list[int], target: int = 24) -> str | None:
    items = tuple((Fraction(n), str(n)) for n in numbers)
    return _search(items, Fraction(target))


def format_sft_completion(expression: str) -> str:
    return (
        "<think>I will combine the four numbers with arithmetic operations and check that the final "
        "expression equals 24.</think>"
        f"<answer>{expression}</answer>"
    )

