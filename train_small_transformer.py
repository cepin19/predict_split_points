#!/usr/bin/env python3
"""
Train a small Hugging Face–compatible causal LM (GPT-2 architecture) on dummy
tasks or your own text (one training example per line).

Saved checkpoints work with run_inference.py, e.g.:
  python run_inference.py --model ./small_lm_out --prompt "5 4 3 "

Default dummy task: countdown — random start n in [1, max_num], target string
"n n-1 ... 1" (space-separated integers), then tokenizer EOS so generation can stop
instead of repeating the last digit after "1".
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a small GPT-2–compatible causal LM on countdown or text lines"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./small_lm_out",
        help="Directory to save model + tokenizer",
    )
    p.add_argument(
        "--task",
        type=str,
        choices=["countdown", "countup", "text"],
        default="countdown",
        help="countdown: synthetic n..1 strings; countup: synthetic 1..n strings; text: lines from --data-path",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="For task=text: UTF-8 text file with one example per line",
    )
    p.add_argument(
        "--max-num",
        type=int,
        default=20,
        help="For task=countdown: random start n is uniform in 1..max_num (inclusive)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=10_000,
        help="For task=countdown: number of random countdown strings in the dataset",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cuda/cpu (default: auto)")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="HF tokenizer name (vocab must match n_positions / standard GPT-2 BPE)",
    )
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--no-append-eos",
        action="store_true",
        help="Do not append tokenizer EOS to each line (default: append so the model learns to stop)",
    )
    return p


def get_device(device_str: str | None) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def countdown_string(n: int) -> str:
    return " ".join(str(i) for i in range(n, 0, -1))


def build_countdown_lines(num_samples: int, max_num: int, rng: random.Random) -> list[str]:
    if max_num < 1:
        raise ValueError("--max-num must be >= 1")
    return [countdown_string(rng.randint(1, max_num)) for _ in range(num_samples)]


def countup_string(start: int, max_num: int) -> str:
    return " ".join(str(i) for i in range(start, max_num + 1))


def build_countup_lines(num_samples: int, max_num: int, rng: random.Random) -> list[str]:
    if max_num < 2:
        raise ValueError("--max-num must be >= 2 for countup")
    return [countup_string(rng.randint(1, max_num), max_num) for _ in range(num_samples)]


def load_text_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"--data-path must be an existing file: {path}")
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    if not lines:
        raise ValueError(f"No non-empty lines in {path}")
    return lines


class LineLmDataset(Dataset):
    """Tokenize fixed strings; causal LM labels with padding ignored (-100)."""

    def __init__(
        self,
        lines: list[str],
        tokenizer: GPT2TokenizerFast,
        max_length: int,
        append_eos: bool = True,
    ) -> None:
        self.encodings: list[dict[str, torch.Tensor]] = []
        for text in lines:
            if append_eos and tokenizer.eos_token:
                text = text + tokenizer.eos_token
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attn = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attn == 0] = -100
            self.encodings.append(
                {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
            )

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.encodings[idx]


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
    }


def causal_lm_next_token_accuracy(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[float, int, int]:
    """Fraction of correct next-token predictions (padding / ignore_index=-100 excluded)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    preds = shift_logits.argmax(dim=-1)
    mask = shift_labels.ne(-100)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return 0.0, 0, 0
    n_correct = int(((preds == shift_labels) & mask).sum().item())
    return n_correct / n_valid, n_correct, n_valid


def linear_warmup_lr(
    step: int, warmup_steps: int, base_lr: float, total_steps: int
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    # cosine decay after warmup
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = get_parser().parse_args()
    device = get_device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.task == "countdown":
        lines = build_countdown_lines(args.num_samples, args.max_num, rng)
    elif args.task == "countup":
        lines = build_countup_lines(args.num_samples, args.max_num, rng)
    else:
        if not args.data_path:
            raise SystemExit("task=text requires --data-path to a UTF-8 text file")
        lines = load_text_lines(Path(args.data_path))

    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=min(1024, max(args.max_length, 128)),
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=bos,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model = model.to(device)

    dataset = LineLmDataset(
        lines, tokenizer, args.max_length, append_eos=not args.no_append_eos
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=False,
    )

    steps_per_epoch = max(len(loader), 1)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        epoch_acc_correct = 0
        epoch_acc_valid = 0
        for batch in loader:
            lr = linear_warmup_lr(
                global_step, args.warmup_steps, args.lr, total_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            with torch.no_grad():
                batch_acc, n_corr, n_val = causal_lm_next_token_accuracy(
                    out.logits, labels
                )
            epoch_acc_correct += n_corr
            epoch_acc_valid += n_val

            if global_step % args.log_every == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {global_step} "
                    f"loss {loss.item():.4f} acc {batch_acc:.4f} lr {lr:.2e}"
                )

        avg = epoch_loss / max(n_batches, 1)
        epoch_acc = (
            epoch_acc_correct / epoch_acc_valid if epoch_acc_valid > 0 else 0.0
        )
        print(
            f"epoch {epoch + 1}/{args.epochs} mean loss {avg:.4f} "
            f"pred acc {epoch_acc:.4f} ({epoch_acc_correct}/{epoch_acc_valid} tokens)"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved model and tokenizer to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
