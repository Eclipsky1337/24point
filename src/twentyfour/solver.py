from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction


@dataclass(frozen=True)
class SolutionTrace:
    expression: str
    steps: tuple[str, ...]


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


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


def _combine_with_steps(
    left: tuple[Fraction, str],
    right: tuple[Fraction, str],
) -> list[tuple[Fraction, str, str]]:
    a, ae = left
    b, be = right
    candidates = [
        (a + b, f"({ae}+{be})", f"{ae}+{be}={_format_fraction(a + b)}"),
        (a - b, f"({ae}-{be})", f"{ae}-{be}={_format_fraction(a - b)}"),
        (b - a, f"({be}-{ae})", f"{be}-{ae}={_format_fraction(b - a)}"),
        (a * b, f"({ae}*{be})", f"{ae}*{be}={_format_fraction(a * b)}"),
    ]
    if b != 0:
        candidates.append((a / b, f"({ae}/{be})", f"{ae}/{be}={_format_fraction(a / b)}"))
    if a != 0:
        candidates.append((b / a, f"({be}/{ae})", f"{be}/{ae}={_format_fraction(b / a)}"))
    return candidates


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


def _search_trace(
    items: tuple[tuple[Fraction, str], ...],
    target: Fraction,
    steps: tuple[str, ...] = (),
) -> SolutionTrace | None:
    if len(items) == 1:
        if items[0][0] == target:
            return SolutionTrace(items[0][1], steps)
        return None

    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            rest = tuple(items[k] for k in range(n) if k not in (i, j))
            for value, expression, step in _combine_with_steps(items[i], items[j]):
                result = _search_trace(rest + ((value, expression),), target, steps + (step,))
                if result:
                    return result
    return None


def solve_24(numbers: list[int], target: int = 24) -> str | None:
    items = tuple((Fraction(n), str(n)) for n in numbers)
    return _search(items, Fraction(target))


def solve_24_trace(numbers: list[int], target: int = 24) -> SolutionTrace | None:
    items = tuple((Fraction(n), str(n)) for n in numbers)
    return _search_trace(items, Fraction(target))


def format_sft_completion(expression: str, numbers: list[int] | None = None, target: int = 24) -> str:
    trace = solve_24_trace(numbers, target) if numbers else None
    if trace and trace.expression == expression:
        step_text = "; ".join(trace.steps)
        think = f"Combine step by step: {step_text}. Final value is {target}."
    else:
        number_text = ", ".join(str(number) for number in numbers) if numbers else "the given numbers"
        think = f"Check the expression: {expression} = {target}. It uses {number_text} exactly once."
    return (
        f"<think>{think}</think>"
        f"<answer>{expression}</answer>"
    )
