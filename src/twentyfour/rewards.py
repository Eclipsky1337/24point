from __future__ import annotations

import re

from .verifier import extract_answer, verify_expression


STRICT_R1_RE = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.I | re.S)


def _normalize_numbers(value) -> list[int]:
    if isinstance(value, str):
        return [int(x) for x in value.replace(",", " ").split()]
    return [int(x) for x in value]


def _completion_text(completion) -> str:
    if isinstance(completion, list):
        return "".join(str(message.get("content", "")) for message in completion if isinstance(message, dict))
    return completion or ""


def answer_format_reward(completions, **kwargs) -> list[float]:
    return [0.1 if STRICT_R1_RE.fullmatch(_completion_text(text)) else 0.0 for text in completions]


def correctness_reward(completions, numbers=None, nums=None, **kwargs) -> list[float]:
    batch_numbers = numbers if numbers is not None else nums
    rewards = []
    for text, item_numbers in zip(completions, batch_numbers):
        result = verify_expression(extract_answer(_completion_text(text)), _normalize_numbers(item_numbers))
        rewards.append(1.0 if result.ok else 0.0)
    return rewards
