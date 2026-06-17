from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from twentyfour.prompts import build_prompt
from twentyfour.verifier import extract_answer, verify_expression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("numbers", nargs=4, type=int)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    prompt = build_prompt(args.numbers)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    answer = extract_answer(completion)
    result = verify_expression(answer, args.numbers)
    print(completion)
    print(f"\nanswer={answer!r} ok={result.ok} reason={result.reason}")


if __name__ == "__main__":
    main()

