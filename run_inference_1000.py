#!/usr/bin/env python3
"""
Run inference N times (default 1000) with random max-new-tokens in [1, 100]
and temperature 1.0. Loads the model once, then runs generation in a loop.
"""

import argparse
import random
from pathlib import Path

from run_inference import (
    extract_activations,
    generate_one,
    get_device,
    get_final_softmax,
    load_model,
    save_activations,
)


def get_parser():
    p = argparse.ArgumentParser(
        description="Run many inferences with one model load (random lengths, fixed temperature)"
    )
    p.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name or path")
    p.add_argument("--prompt", type=str, default="A ", help="Input prompt")
    p.add_argument("--n-runs", type=int, default=1000, help="Number of inference runs")
    p.add_argument("--min-tokens", type=int, default=1, help="Min max-new-tokens (inclusive)")
    p.add_argument("--max-tokens", type=int, default=100, help="Max max-new-tokens (inclusive)")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--output-dir", type=str, default="./activations_out", help="Directory to save runs")
    p.add_argument(
        "--save-format",
        choices=["pt", "npz"],
        default="pt",
        help="Single activations.{pt,npz} blob per run (plus meta.json)",
    )
    p.add_argument(
        "--save-scores-softmax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include logits and softmax inside the activations blob; --no-save-scores-softmax omits them and skips score tensors in generation",
    )
    p.add_argument("--device", type=str, default=None, help="Device (cuda/cpu or auto)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for run lengths (and initial model load)")
    return p


def main():
    args = get_parser().parse_args()
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Loading model {args.model} on {device}...")
    model, tokenizer = load_model(args.model, device=device, seed=args.seed)
    print(f"Running {args.n_runs} inferences (lengths {args.min_tokens}-{args.max_tokens}, temperature={args.temperature})")
    print(f"Output: {output_dir}")
    print()

    for i in range(args.n_runs):
        max_new_tokens = random.randint(args.min_tokens, args.max_tokens)
        print(f"[{i + 1}/{args.n_runs}] max_new_tokens={max_new_tokens}")

        out = generate_one(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=max_new_tokens,
            device=device,
            temperature=args.temperature,
            output_scores=args.save_scores_softmax,
        )
        activations = extract_activations(out)
        activations["softmax_per_step"] = get_final_softmax(activations["scores"])

        run_dir = output_dir / f"run{i}"
        save_activations(
            activations, run_dir, args.save_format, save_scores_softmax=args.save_scores_softmax
        )

    print()
    print(f"Done. {args.n_runs} runs saved under {output_dir} (run0 .. run{args.n_runs - 1}).")


if __name__ == "__main__":
    main()
