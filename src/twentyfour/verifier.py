from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"\d+")
ALLOWED_CHARS_RE = re.compile(r"^[\d\s+\-*/().]+$")


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    value: Fraction | None = None
    reason: str = ""
    expression: str = ""


def extract_answer(text: str) -> str:
    match = ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def normalize_answer(expression: str) -> str:
    expr = (expression or "").strip()
    if expr.count("=") == 1:
        expr = expr.split("=", maxsplit=1)[0].strip()
    return expr


def _eval_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return Fraction(node.value, 1)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
    raise ValueError(f"unsupported syntax: {ast.dump(node, include_attributes=False)}")


def verify_expression(expression: str, numbers: list[int], target: int = 24) -> VerificationResult:
    expr = normalize_answer(expression)
    if not expr:
        return VerificationResult(False, reason="empty expression", expression=expr)
    if not ALLOWED_CHARS_RE.fullmatch(expr):
        return VerificationResult(False, reason="contains illegal characters", expression=expr)

    used_numbers = [int(x) for x in NUMBER_RE.findall(expr)]
    expected = Counter(numbers)
    actual = Counter(used_numbers)
    if actual != expected:
        return VerificationResult(
            False,
            reason=f"numbers mismatch: expected {dict(expected)}, got {dict(actual)}",
            expression=expr,
        )

    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_node(tree)
    except Exception as exc:
        return VerificationResult(False, reason=str(exc), expression=expr)

    ok = math.isclose(float(value), float(target), rel_tol=0.0, abs_tol=1e-6)
    reason = "ok" if ok else f"value {float(value)} != {target}"
    return VerificationResult(ok, value=value, reason=reason, expression=expr)


def verify_completion(completion: str, numbers: list[int], target: int = 24) -> VerificationResult:
    return verify_expression(extract_answer(completion), numbers, target=target)


def has_r1_format(text: str) -> bool:
    text = text or ""
    return bool(re.search(r"<think>.*?</think>.*?<answer>.*?</answer>", text, re.I | re.S))
