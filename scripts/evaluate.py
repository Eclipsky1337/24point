from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from twentyfour.data import load_game_of_24, load_nlile_24game
from twentyfour.verifier import extract_answer, verify_expression


def extract_think(text: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def resolve_dtype(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "auto":
        return "auto"
    return None


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return device


def build_generation_prompt(prompt: str, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def generate(model, tokenizer, prompt: str, max_new_tokens: int, num_samples: int, temperature: float) -> list[str]:
    rendered_prompt = build_generation_prompt(prompt, tokenizer)
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
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
    parser.add_argument("--split", choices=["id", "nonhard", "hard", "unsolvable"], default="nonhard")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "auto"], default="float32")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--record_file", default=None, help="Append a JSONL experiment record to this path.")
    parser.add_argument("--experiment", default=None, help="Experiment name stored in --record_file.")
    args = parser.parse_args()

    if args.eval_file:
        dataset = load_dataset("json", data_files=args.eval_file, split="train")
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    elif args.split == "id":
        dataset = load_nlile_24game(limit=args.limit)
    elif args.split == "unsolvable":
        dataset = load_nlile_24game(solvable_only=False).filter(lambda row: not row["solvable"])
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    elif args.split == "nonhard":
        dataset = load_game_of_24(mode="nonhard", limit=args.limit)
    else:
        dataset = load_game_of_24(mode="hard", limit=args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = resolve_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    correct = 0
    valid = 0
    think_count = 0
    per_item_records = []
    for idx, row in enumerate(dataset):
        completions = generate(
            model,
            tokenizer,
            row["prompt"],
            args.max_new_tokens,
            args.num_samples,
            args.temperature,
        )
        sample_records = []
        for sample_idx, text in enumerate(completions):
            answer = extract_answer(text)
            result = verify_expression(answer, row["numbers"])
            think = extract_think(text)
            sample_records.append(
                {
                    "sample_idx": sample_idx,
                    "completion": text,
                    "think": think,
                    "answer": answer,
                    "expression": result.expression,
                    "ok": result.ok,
                    "reason": result.reason,
                    "value": float(result.value) if result.value is not None else None,
                    "value_exact": str(result.value) if result.value is not None else None,
                }
            )

        best_record = next((record for record in sample_records if record["ok"]), sample_records[0])
        valid += int(any(record["value"] is not None for record in sample_records))
        correct += int(any(record["ok"] for record in sample_records))
        think_count += int(any(record["think"] is not None for record in sample_records))
        per_item_records.append(
            {
                "idx": idx,
                "numbers": row["numbers"],
                "solvable": row.get("solvable", True),
                "best_sample_idx": best_record["sample_idx"],
                "best_answer": best_record["answer"],
                "best_expression": best_record["expression"],
                "best_ok": best_record["ok"],
                "best_reason": best_record["reason"],
                "samples": sample_records,
            }
        )
        print(
            f"[{idx}] nums={row['numbers']} answer={best_record['answer']!r} "
            f"ok={best_record['ok']} reason={best_record['reason']} samples={args.num_samples}",
            flush=True,
        )

    total = max(len(dataset), 1)
    success_metric_name = f"pass@{args.num_samples}"
    metrics = {
        "valid_rate": valid / total,
        success_metric_name: correct / total,
        "think_rate": think_count / total,
    }
    unsolvable_total = sum(1 for record in per_item_records if not record["solvable"])
    if unsolvable_total:
        false_positive = sum(
            1
            for record in per_item_records
            if not record["solvable"] and any(sample["value"] is not None for sample in record["samples"])
        )
        false_success = sum(1 for record in per_item_records if not record["solvable"] and record["best_ok"])
        metrics["unsolvable_valid_rate"] = false_positive / unsolvable_total
        metrics[f"unsolvable_{success_metric_name}"] = false_success / unsolvable_total
    print(f"valid_rate={metrics['valid_rate']:.3f}")
    print(f"{success_metric_name}={metrics[success_metric_name]:.3f}")
    print(f"think_rate={metrics['think_rate']:.3f}")

    if args.record_file:
        record_path = Path(args.record_file)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "experiment": args.experiment or Path(args.model_path).name,
            "model_path": args.model_path,
            "eval_file": args.eval_file,
            "split": args.split,
            "command": " ".join(sys.argv),
            "settings": {
                "limit": args.limit,
                "num_samples": args.num_samples,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "dtype": args.dtype,
                "device": device,
            },
            "environment": {
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
            },
            "results": per_item_records,
            "metrics": metrics,
        }
        with record_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"record_file={record_path}")


if __name__ == "__main__":
    main()
