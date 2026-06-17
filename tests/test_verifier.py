from twentyfour.verifier import extract_answer, has_r1_format, verify_completion, verify_expression


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

