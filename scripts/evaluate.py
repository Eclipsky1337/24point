from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from twentyfour.data import load_game_of_24, load_nlile_24game
from twentyfour.verifier import extract_answer, verify_expression


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split", choices=["id", "hard", "unsolvable"], default="hard")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    if args.split == "id":
        dataset = load_nlile_24game(limit=args.limit)
    elif args.split == "unsolvable":
        dataset = load_nlile_24game(solvable_only=False).filter(lambda row: not row["solvable"])
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    else:
        dataset = load_game_of_24(mode="hard", limit=args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    correct = 0
    valid = 0
    for idx, row in enumerate(dataset):
        completion = generate(model, tokenizer, row["prompt"], args.max_new_tokens)
        answer = extract_answer(completion)
        result = verify_expression(answer, row["numbers"])
        valid += int(result.value is not None)
        correct += int(result.ok)
        print(f"[{idx}] nums={row['numbers']} answer={answer!r} ok={result.ok} reason={result.reason}")

    total = max(len(dataset), 1)
    print(f"valid_rate={valid / total:.3f}")
    print(f"success_rate={correct / total:.3f}")


if __name__ == "__main__":
    main()

