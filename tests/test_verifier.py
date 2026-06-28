from twentyfour.prompts import build_prompt
from twentyfour.solver import format_sft_completion, solve_24_trace
from twentyfour.verifier import extract_answer, has_r1_format, verify_completion, verify_expression
from twentyfour.rewards import answer_format_reward, correctness_reward, proximity_reward, valid_expression_reward


def test_valid_expression() -> None:
    result = verify_expression("(8/(3-8/3))", [8, 3, 8, 3])
    assert result.ok


def test_rejects_reused_number() -> None:
    result = verify_expression("6*4", [6, 6, 6, 6])
    assert not result.ok
    assert "numbers mismatch" in result.reason


def test_rejects_illegal_function_call() -> None:
    result = verify_expression("__import__('os').system('echo bad')", [1, 2, 3, 4])
    assert not result.ok


def test_extract_answer_and_format() -> None:
    text = "<think>try factors</think><answer>(6-2)*(3+3)</answer>"
    assert has_r1_format(text)
    assert extract_answer(text) == "(6-2)*(3+3)"
    assert verify_completion(text, [6, 2, 3, 3]).ok


def test_verifier_normalizes_single_equals_sign() -> None:
    result = verify_expression("(6-2)*(3+3) = 24", [6, 2, 3, 3])
    assert result.ok
    assert result.expression == "(6-2)*(3+3)"


def test_prompt_forbids_equals_sign_in_answer() -> None:
    prompt = build_prompt([1, 2, 3, 4])
    assert "Do not include words, explanations, or an equals sign in <answer>." in prompt


def test_sft_completion_includes_verification_think() -> None:
    completion = format_sft_completion("(8*(1+(1+1)))", [1, 1, 1, 8])
    assert "Combine step by step:" in completion
    assert "Final value is 24." in completion
    assert "<answer>(8*(1+(1+1)))</answer>" in completion


def test_solve_24_trace_records_three_combine_steps() -> None:
    trace = solve_24_trace([1, 1, 1, 8])
    assert trace is not None
    assert trace.expression == "(8*(1+(1+1)))"
    assert len(trace.steps) == 3


def test_proximity_reward_prefers_closer_value() -> None:
    completions = ["<answer>(8*(8-(3+3)))</answer>", "<answer>((8-3)*3+8)</answer>"]
    rewards = proximity_reward(completions, numbers=[[3, 3, 8, 8], [3, 3, 8, 8]])
    assert rewards[1] > rewards[0]


def test_strict_format_reward_rejects_extra_text() -> None:
    completions = [
        "<think>check</think><answer>(6-2)*(3+3)</answer>",
        "<think>check</think>extra<answer>(6-2)*(3+3)</answer> trailing",
    ]
    assert answer_format_reward(completions) == [0.1, -0.1]


def test_correct_reward_dominates_valid_wrong_expression() -> None:
    completions = [
        "<think>check</think><answer>(6-2)*(3+3)</answer>",
        "<think>check</think><answer>(6+2)*(3+3)</answer>",
    ]
    numbers = [[6, 2, 3, 3], [6, 2, 3, 3]]
    valid_rewards = valid_expression_reward(completions, numbers=numbers)
    correct_rewards = correctness_reward(completions, numbers=numbers)
    assert valid_rewards == [0.2, 0.2]
    assert correct_rewards == [2.0, 0.0]


def test_rewards_accept_conversational_completions() -> None:
    completions = [[{"role": "assistant", "content": "<think>check</think><answer>(6-2)*(3+3)</answer>"}]]
    numbers = [[6, 2, 3, 3]]
    assert answer_format_reward(completions) == [0.1]
    assert valid_expression_reward(completions, numbers=numbers) == [0.2]
    assert correctness_reward(completions, numbers=numbers) == [2.0]
