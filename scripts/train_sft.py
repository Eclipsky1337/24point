from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_file", default="data/processed/sft_train.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen2.5-1.5b-24point-sft")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--report_to", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.fp16 else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.use_peft:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=args.train_file, split="train")

    def tokenize(row):
        prompt = row["prompt"].rstrip() + "\n"
        completion = row["completion"]
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full = tokenizer(
            prompt + completion,
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )
        labels = list(full["input_ids"])
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        labels = [label if mask else -100 for label, mask in zip(labels, full["attention_mask"])]
        full["labels"] = labels
        return full

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    first = tokenized[0]
    supervised_tokens = sum(1 for label in first["labels"] if label != -100)
    print(f"first sample supervised tokens: {supervised_tokens}")
    if supervised_tokens == 0:
        raise RuntimeError("No supervised tokens found. Increase --max_length or check SFT data.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dataloader = DataLoader(
        tokenized,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    global_step = 0
    running_loss = 0.0
    total_steps = int((len(dataloader) * args.num_train_epochs) / max(args.gradient_accumulation_steps, 1))
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(int(args.num_train_epochs)):
        for micro_step, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.detach().item()}")

            scaler.scale(loss).backward()
            running_loss += loss.detach().float().item()

            if micro_step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0 or global_step == 1:
                    avg_loss = running_loss / args.logging_steps if global_step % args.logging_steps == 0 else running_loss
                    print(
                        {
                            "step": global_step,
                            "total_steps": total_steps,
                            "loss": round(avg_loss, 6),
                            "grad_norm": float(grad_norm),
                            "epoch": epoch + micro_step / len(dataloader),
                        },
                        flush=True,
                    )
                    running_loss = 0.0

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
