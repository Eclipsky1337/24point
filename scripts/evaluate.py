from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from twentyfour.data import load_game_of_24, load_nlile_24game
from twentyfour.verifier import extract_answer, verify_expression


def generate(model, tokenizer, prompt: str, max_new_tokens: int, num_samples: int, temperature: float) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": num_samples > 1,
        "num_return_sequences": num_samples,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if num_samples > 1:
        generation_kwargs.update(temperature=temperature, top_p=0.9)
    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)
    return [
        tokenizer.decode(sequence[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        for sequence in output
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_file", default=None, help="Optional local JSONL evaluation file.")
    parser.add_argument("--split", choices=["id", "hard", "unsolvable"], default="hard")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.eval_file:
        dataset = load_dataset("json", data_files=args.eval_file, split="train")
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    elif args.split == "id":
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
        completions = generate(
            model,
            tokenizer,
            row["prompt"],
            args.max_new_tokens,
            args.num_samples,
            args.temperature,
        )
        results = [verify_expression(extract_answer(text), row["numbers"]) for text in completions]
        best = next((result for result in results if result.ok), results[0])
        valid += int(any(result.value is not None for result in results))
        correct += int(any(result.ok for result in results))
        print(
            f"[{idx}] nums={row['numbers']} answer={best.expression!r} "
            f"ok={best.ok} reason={best.reason} samples={args.num_samples}"
        )

    total = max(len(dataset), 1)
    print(f"valid_rate={valid / total:.3f}")
    print(f"success_rate={correct / total:.3f}")


if __name__ == "__main__":
    main()
