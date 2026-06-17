SYSTEM_PROMPT = """You are solving the 24-point game.
Use exactly the four given numbers once each.
You may use only +, -, *, /, and parentheses.
Return reasoning in <think>...</think> and the final expression in <answer>...</answer>."""


def build_prompt(numbers: list[int]) -> str:
    nums = " ".join(str(n) for n in numbers)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Numbers: {nums}\n"
        "Find an expression that equals 24."
    )

