#!/usr/bin/env python3
"""Fine-tune a pretrained Hugging Face causal LM on one-document-per-line text, with EOS
appended to every document, so that it learns to end its generations.

Used to make web-text models (gpt2, opt-125m, pythia-160m) stop after a TinyStories story.
The saved checkpoint works with run_inference.py --model <output-dir>.

  python finetune_lm.py --model gpt2 --data-path stories.txt --output-dir ./gpt2_tinystories
"""

import argparse
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Pretrained model name or path")
    p.add_argument("--data-path", required=True, help="Text file, one document per line")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-docs", type=int, default=None, help="Use only the first N documents")
    p.add_argument("--max-length", type=int, default=512, help="Documents longer than this are skipped")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = get_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model)
    # float32: transformers 5 loads checkpoints in their stored dtype, and fp16 (OPT, Pythia) gives NaN losses here
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    lines = [l.strip() for l in open(args.data_path, encoding="utf-8") if l.strip()]
    if args.max_docs:
        lines = lines[: args.max_docs]
    # tokenizer(...) adds BOS for models that use one (OPT); EOS is appended explicitly
    docs = [ids + [tok.eos_token_id] for ids in tok(lines)["input_ids"]]
    n_all = len(docs)
    docs = [d for d in docs if len(d) <= args.max_length]
    print(f"{len(docs)} of {n_all} documents fit in {args.max_length} tokens")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * math.ceil(len(docs) / args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, args.warmup_steps, steps)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(docs)
        for start in range(0, len(docs), args.batch_size):
            batch = docs[start : start + args.batch_size]
            L = max(len(d) for d in batch)
            ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for i, d in enumerate(batch):
                ids[i, : len(d)] = torch.tensor(d)
                mask[i, : len(d)] = 1
            labels = ids.masked_fill(mask == 0, -100)  # no loss on padding; EOS itself is a target
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch + 1} step {step}/{steps} loss {out.loss.item():.4f}", flush=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
