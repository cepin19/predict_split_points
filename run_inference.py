#!/usr/bin/env python3
"""
Run inference on any Hugging Face causal language model and extract/save
inner activations of all layers at each generation step (one activations.{pt,npz}
blob per run; variable seq_len per step/layer), optionally logits/softmax
(--save-scores-softmax / --no-save-scores-softmax). A small meta.json is written
for fast metadata reads; legacy per-step files are still loadable.

Batch modes that create one save directory (run0, run1, …) per prompt:
  --input-file + --prompt-template (e.g. translation)
  --countdown + --countdown-max-num / --countdown-starts (countdown sweep)
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# One file per run: variable seq_len preserved as separate tensors per (step, layer).
ACTIVATIONS_BLOB_PT = "activations.pt"
ACTIVATIONS_BLOB_NPZ = "activations.npz"
ACTIVATIONS_FORMAT_VERSION = 2


def get_parser():
    p = argparse.ArgumentParser(description="Run LM inference and capture layer activations")
    p.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name or path")
    p.add_argument("--prompt", type=str, default="A ", help="Input prompt (ignored if --input-file is set)")
    p.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Text file with one sentence per line; prompt per line is built from --prompt-template with {line}",
    )
    p.add_argument(
        "--prompt-template",
        type=str,
        default="Translate the following line into English:\n{line}\n",
        help="Template for each line when using --input-file (must contain {line})",
    )
    p.add_argument(
        "--countdown",
        action="store_true",
        help="Run many prompts: one activation save per countdown start (see --countdown-*). Incompatible with --input-file.",
    )
    p.add_argument(
        "--countdown-max-num",
        type=int,
        default=20,
        metavar="N",
        help="With --countdown: use each start value 1..N (ignored if --countdown-starts is set)",
    )
    p.add_argument(
        "--countdown-starts",
        type=str,
        default=None,
        metavar="LIST",
        help='With --countdown: comma-separated starts, e.g. "3,7,12" (overrides --countdown-max-num range)',
    )
    p.add_argument(
        "--countdown-repetitions",
        type=int,
        default=1,
        metavar="R",
        help="With --countdown: repeat the full set of starts R times (more runN dirs); order is rep then start",
    )
    p.add_argument(
        "--countdown-prompt-template",
        type=str,
        default="{start} ",
        help="With --countdown: prompt per start; must contain {start} (integer as in training, e.g. '10 ')",
    )
    p.add_argument(
        "--countdown-shuffle-starts",
        action="store_true",
        help="With --countdown: shuffle order of starts each repetition (seed from --seed if set)",
    )
    p.add_argument("--max-new-tokens", type=int, default=100, help="Max new tokens to generate")
    p.add_argument("--output-dir", type=str, default="./activations_out", help="Directory to save activations")
    p.add_argument(
        "--save-format",
        choices=["pt", "npz"],
        default="pt",
        help="Single output blob per run: activations.pt or activations.npz (plus meta.json)",
    )
    p.add_argument("--print-shapes", action="store_true", help="Print activation shapes to stdout")
    p.add_argument(
        "--print-predictions",
        action="store_true",
        help="Print decoded model predictions (generated text and token ids)",
    )
    p.add_argument("--no-save", action="store_true", help="Do not save to disk (only print if --print-shapes)")
    p.add_argument(
        "--save-scores-softmax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include logits and softmax in the activations blob; --no-save-scores-softmax omits them and skips score tensors during generation",
    )
    p.add_argument("--device", type=str, default=None, help="Device (cuda/cpu or auto)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (enables sampling when set)")
    p.add_argument("--encoding", type=str, default="utf-8", help="Encoding for --input-file")
    p.add_argument(
        "--save-layers",
        type=str,
        default=None,
        help="Comma-separated hidden-state indices to save (0 = embeddings), keeping their original indices; default: all",
    )
    return p


def parse_countdown_starts(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("No integers parsed from --countdown-starts")
    return out


def build_countdown_runs(args: argparse.Namespace) -> list[tuple[str, dict]]:
    """Return (prompt, meta_extra) for each saved run."""
    tpl = args.countdown_prompt_template
    if "{start}" not in tpl:
        raise ValueError("--countdown-prompt-template must contain {start}")

    if args.countdown_starts:
        starts = parse_countdown_starts(args.countdown_starts)
    else:
        if args.countdown_max_num < 1:
            raise ValueError("--countdown-max-num must be >= 1 when not using --countdown-starts")
        starts = list(range(1, args.countdown_max_num + 1))

    reps = args.countdown_repetitions
    if reps < 1:
        raise ValueError("--countdown-repetitions must be >= 1")

    rng = random.Random(args.seed if args.seed is not None else 42)
    runs: list[tuple[str, dict]] = []
    for rep in range(reps):
        order = starts[:]
        if args.countdown_shuffle_starts:
            rng.shuffle(order)
        for idx, n in enumerate(order):
            prompt = tpl.format(start=n)
            runs.append(
                (
                    prompt,
                    {
                        "task": "countdown",
                        "countdown_start": n,
                        "countdown_repetition": rep,
                        "countdown_index_in_rep": idx,
                        "prompt": prompt,
                    },
                )
            )
    return runs


def next_run_dir(base_dir: str | Path) -> Path:
    """Return path to a new run subdir (run0, run1, ...) under base_dir."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_pattern = re.compile(r"^run(\d+)$")
    existing = []
    for p in base.iterdir():
        if p.is_dir():
            m = run_pattern.match(p.name)
            if m is not None:
                existing.append(int(m.group(1)))
    next_n = max(existing, default=-1) + 1
    return base / f"run{next_n}"


def get_device(device_str: str | None) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(
    model_name: str,
    device: torch.device | None = None,
    seed: int | None = 42,
):
    """Load model and tokenizer once. Returns (model, tokenizer)."""
    device = device or get_device(None)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float | None = None,
    output_scores: bool = True,
):
    """Run one generation with already-loaded model and tokenizer. Returns generation output."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = getattr(model.config, "eos_token_id", None)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_scores=output_scores,
        pad_token_id=pad_id,
    )
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id
    if temperature is not None:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    return out


def run_inference(
    model_name: str,
    prompt: str,
    max_new_tokens: int = 10,
    device: torch.device | None = None,
    seed: int = 42,
    temperature: float | None = None,
):
    """Load model and tokenizer, run generation with hidden states and scores."""
    device = device or get_device(None)
    model, tokenizer = load_model(model_name, device=device, seed=seed)
    out = generate_one(model, tokenizer, prompt, max_new_tokens, device, temperature=temperature)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    return model, tokenizer, out, inputs


def extract_activations(generation_output) -> dict:
    """
    Extract activations from generation output.
    Returns dict with:
      - step_activations: list of dicts, one per generation step; each dict has
        layer index -> hidden state tensor (batch, seq_len, hidden_size)
      - scores: list of tensors (logits) per step
      - sequences: generated token ids
    """
    step_activations = []
    if generation_output.hidden_states is not None:
        # hidden_states: tuple of (step0_layers, step1_layers, ...)
        # each step: tuple of (embedding, layer_0_out, layer_1_out, ...)
        for step_idx, layer_outputs in enumerate(generation_output.hidden_states):
            step_dict = {}
            for layer_idx, h in enumerate(layer_outputs):
                step_dict[layer_idx] = h.detach().cpu()
            step_activations.append(step_dict)

    scores = None
    if generation_output.scores is not None:
        scores = [s.detach().cpu() for s in generation_output.scores]

    sequences = generation_output.sequences.cpu() if generation_output.sequences is not None else None

    return {
        "step_activations": step_activations,
        "scores": scores,
        "sequences": sequences,
    }


def _normalize_step_layer_keys(step_dict: dict) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for k, v in step_dict.items():
        out[int(k)] = v
    return out


def _scores_softmax_for_save(
    scores: list[torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[torch.Tensor] | None, list[torch.Tensor] | None]:
    """Return stacked (scores, softmax) if shapes match; else list forms."""
    if not scores:
        return None, None, None, None
    sm = get_final_softmax(scores)
    try:
        return torch.stack(scores, dim=0), torch.stack(sm, dim=0), None, None
    except RuntimeError:
        return None, None, scores, sm


def _unpack_scores_from_blob(data: dict) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
    scores_out: list[torch.Tensor] | None = None
    sm_out: list[torch.Tensor] | None = None
    if isinstance(data.get("scores"), torch.Tensor):
        t = data["scores"]
        scores_out = [t[i].contiguous() for i in range(t.shape[0])]
    elif data.get("scores_list") is not None:
        scores_out = list(data["scores_list"])
    if isinstance(data.get("softmax"), torch.Tensor):
        t = data["softmax"]
        sm_out = [t[i].contiguous() for i in range(t.shape[0])]
    elif data.get("softmax_list") is not None:
        sm_out = list(data["softmax_list"])
    return scores_out, sm_out


def get_final_softmax(scores: list[torch.Tensor], temperature: float = 1.0) -> list[torch.Tensor]:
    """
    Compute softmax over the last dimension of each step's logits (scores).
    Returns list of tensors of shape (batch_size, vocab_size) or (batch_size, 1, vocab_size).
    """
    if scores is None:
        return []
    return [torch.softmax(s / temperature, dim=-1) for s in scores]


def load_activations(output_dir: str | Path) -> dict:
    """
    Load previously saved activations from output_dir.
    Returns dict with step_activations (list of layer dicts), scores, softmax_per_step, sequences,
    and meta (from blob or meta.json).

    Prefers a single-file layout (activations.pt / activations.npz, format v2); falls back to
    legacy per-step files (step_0.*, …), scores_and_softmax.*, sequences.pt.
    """
    output_dir = Path(output_dir)
    blob_pt = output_dir / ACTIVATIONS_BLOB_PT
    blob_npz = output_dir / ACTIVATIONS_BLOB_NPZ

    if blob_pt.exists():
        data = torch.load(blob_pt, map_location="cpu", weights_only=False)
        if data.get("format_version") != ACTIVATIONS_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported activations.pt format_version {data.get('format_version')!r} in {blob_pt}"
            )
        step_activations = [_normalize_step_layer_keys(d) for d in data["step_activations"]]
        scores, softmax_per_step = _unpack_scores_from_blob(data)
        sequences = data.get("sequences")
        meta = data.get("meta")
        result = {
            "step_activations": step_activations,
            "scores": scores,
            "sequences": sequences,
            "softmax_per_step": softmax_per_step,
        }
        if data.get("generated_text") is not None:
            result["generated_text"] = data["generated_text"]
        if meta is not None:
            result["meta"] = meta
        else:
            meta_path = output_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    result["meta"] = json.load(f)
        return result

    if blob_npz.exists():
        import numpy as np

        raw = np.load(blob_npz)
        if int(raw["format_version"]) != ACTIVATIONS_FORMAT_VERSION:
            raise ValueError(f"Unsupported {ACTIVATIONS_BLOB_NPZ} format_version in {blob_npz}")

        meta = json.loads(raw["meta_json"].tobytes().decode("utf-8"))
        by_step: dict[int, dict[int, torch.Tensor]] = {}
        for name in raw.files:
            if name in ("format_version", "meta_json", "sequences", "scores", "softmax", "generated_text"):
                continue
            if name.startswith("scores_") or name.startswith("softmax_"):
                continue
            if not name.startswith("s") or "_l" not in name:
                continue
            _, rest = name.split("s", 1)
            si_s, li_s = rest.split("_l", 1)
            si, li = int(si_s), int(li_s)
            by_step.setdefault(si, {})[li] = torch.from_numpy(raw[name])

        step_activations = [by_step[k] for k in sorted(by_step.keys())]

        scores: list[torch.Tensor] | None = None
        softmax_per_step: list[torch.Tensor] | None = None
        if "scores" in raw.files:
            s = torch.from_numpy(raw["scores"])
            scores = [s[i].contiguous() for i in range(s.shape[0])]
        else:
            score_keys = sorted(
                (int(k.split("_", 1)[1]) for k in raw.files if k.startswith("scores_")),
                key=int,
            )
            if score_keys:
                scores = [torch.from_numpy(raw[f"scores_{i}"]) for i in score_keys]
        if "softmax" in raw.files:
            s = torch.from_numpy(raw["softmax"])
            softmax_per_step = [s[i].contiguous() for i in range(s.shape[0])]
        else:
            sm_keys = sorted(
                (int(k.split("_", 1)[1]) for k in raw.files if k.startswith("softmax_")),
                key=int,
            )
            if sm_keys:
                softmax_per_step = [torch.from_numpy(raw[f"softmax_{i}"]) for i in sm_keys]

        sequences = torch.from_numpy(raw["sequences"]) if "sequences" in raw.files else None
        result = {
            "step_activations": step_activations,
            "scores": scores,
            "sequences": sequences,
            "softmax_per_step": softmax_per_step,
            "meta": meta,
        }
        if "generated_text" in raw.files:
            gt = raw["generated_text"]
            result["generated_text"] = str(gt.item()) if gt.shape == () else str(gt)
        return result

    # Legacy: step_0.pt, step_1.pt, … (variable seq_len per file)
    step_activations = []
    step_idx = 0
    while True:
        for ext in ("pt", "npz"):
            path = output_dir / f"step_{step_idx}.{ext}"
            if path.exists():
                if ext == "pt":
                    step_activations.append(
                        _normalize_step_layer_keys(torch.load(path, map_location="cpu", weights_only=True))
                    )
                else:
                    import numpy as np

                    npdata = np.load(path)
                    step_activations.append(
                        {int(k): torch.from_numpy(npdata[k]) for k in npdata.files}
                    )
                step_idx += 1
                break
        else:
            break

    result = {"step_activations": step_activations, "scores": None, "sequences": None, "softmax_per_step": None}

    scores_path = output_dir / "scores_and_softmax.pt"
    if not scores_path.exists():
        scores_path = output_dir / "scores_and_softmax.npz"
    if scores_path.exists():
        if scores_path.suffix == ".pt":
            sdata = torch.load(scores_path, map_location="cpu", weights_only=True)
            result["scores"] = [sdata["scores"][i] for i in range(sdata["scores"].shape[0])]
            result["softmax_per_step"] = [sdata["softmax"][i] for i in range(sdata["softmax"].shape[0])]
        else:
            import numpy as np

            sdata = np.load(scores_path)
            result["scores"] = [torch.from_numpy(sdata["scores"][i]) for i in range(len(sdata["scores"]))]
            result["softmax_per_step"] = [
                torch.from_numpy(sdata["softmax"][i]) for i in range(len(sdata["softmax"]))
            ]

    seq_path = output_dir / "sequences.pt"
    if seq_path.exists():
        result["sequences"] = torch.load(seq_path, map_location="cpu", weights_only=True)

    meta_path = output_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            result["meta"] = json.load(f)
    return result


def save_activations(
    activations: dict, output_dir: Path, save_format: str, save_scores_softmax: bool = True
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_activations = activations["step_activations"]

    meta = {
        "num_steps": len(step_activations),
        "num_layers": len(step_activations[0]) if step_activations else 0,
        "shapes_per_step": {
            str(k): list(v.shape) for k, v in step_activations[0].items()
        }
        if step_activations
        else {},
        "activations_format_version": ACTIVATIONS_FORMAT_VERSION,
    }
    if activations.get("meta_extra"):
        meta.update(activations["meta_extra"])

    if save_format == "pt":
        payload: dict = {
            "format_version": ACTIVATIONS_FORMAT_VERSION,
            "meta": meta,
            "step_activations": [
                {int(li): t for li, t in step_dict.items()} for step_dict in step_activations
            ],
        }
        if activations.get("generated_text") is not None:
            payload["generated_text"] = activations["generated_text"]
        if activations["sequences"] is not None:
            payload["sequences"] = activations["sequences"]
        scores = activations.get("scores")
        if save_scores_softmax and scores:
            sc_st, sm_st, sc_li, sm_li = _scores_softmax_for_save(scores)
            if sc_st is not None:
                payload["scores"] = sc_st
                payload["softmax"] = sm_st
            else:
                payload["scores_list"] = sc_li
                payload["softmax_list"] = sm_li
        torch.save(payload, output_dir / ACTIVATIONS_BLOB_PT)
    else:
        import numpy as np

        arrs: dict[str, object] = {
            "format_version": np.array(ACTIVATIONS_FORMAT_VERSION, dtype=np.int64),
            "meta_json": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        }
        for si, step_dict in enumerate(step_activations):
            for li, t in step_dict.items():
                arrs[f"s{si}_l{li}"] = t.numpy()
        if activations.get("generated_text") is not None:
            arrs["generated_text"] = np.array(activations["generated_text"], dtype=object)
        if activations["sequences"] is not None:
            arrs["sequences"] = activations["sequences"].numpy()
        scores = activations.get("scores")
        if save_scores_softmax and scores:
            sc_st, sm_st, sc_li, sm_li = _scores_softmax_for_save(scores)
            if sc_st is not None:
                arrs["scores"] = sc_st.numpy()
                arrs["softmax"] = sm_st.numpy()
            else:
                for i, (a, b) in enumerate(zip(sc_li, sm_li)):
                    arrs[f"scores_{i}"] = a.numpy()
                    arrs[f"softmax_{i}"] = b.numpy()
        np.savez_compressed(output_dir / ACTIVATIONS_BLOB_NPZ, **arrs)

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def print_activation_shapes(activations: dict, tokenizer, sequences) -> None:
    print("=== Activation shapes ===")
    step_activations = activations["step_activations"]
    if not step_activations:
        print("No step activations.")
        return
    print(f"Number of generation steps: {len(step_activations)}")
    print(f"Number of layers (incl. embedding): {len(step_activations[0])}")
    print("\nShapes at step 0 (example):")
    for layer_idx, t in step_activations[0].items():
        print(f"  layer_{layer_idx}: {tuple(t.shape)}")
    sc = activations.get("scores")
    if sc:
        print(f"\nScores per step: {tuple(sc[0].shape)} (batch, seq, vocab_size)")
    if sequences is not None:
        decoded = tokenizer.decode(sequences[0], skip_special_tokens=True)
        print(f"\nGenerated sequence (decoded): {repr(decoded)}")
        print(f"Generated token ids: {sequences[0].tolist()}")


def main():
    args = get_parser().parse_args()
    device = get_device(args.device)

    run_meta_extra: list[dict] | None = None

    if args.countdown and args.input_file is not None:
        raise SystemExit("Use either --countdown or --input-file, not both.")

    if args.countdown:
        countdown_runs = build_countdown_runs(args)
        prompts = [p for p, _ in countdown_runs]
        run_meta_extra = [m for _, m in countdown_runs]
    elif args.input_file is not None:
        # File mode: one run per line, same saving as single-prompt mode
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        with open(input_path, encoding=args.encoding) as f:
            lines = [ln.rstrip("\n") for ln in f.readlines()]
        prompts = [args.prompt_template.format(line=ln.strip()) for ln in lines]
        if not prompts:
            print("No lines in input file.")
            return None
        run_meta_extra = [
            {"task": "prompt_file", "prompt": p, "line_index": i}
            for i, p in enumerate(prompts)
        ]
    else:
        prompts = [args.prompt]
        run_meta_extra = [{"task": "single_prompt", "prompt": prompts[0]}]

    print(f"Loading model {args.model} on {device}...")
    model, tokenizer = load_model(args.model, device=device, seed=args.seed)

    for i, prompt in enumerate(prompts):
        if len(prompts) > 1:
            print(f"Run {i + 1}/{len(prompts)}: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        generation_output = generate_one(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            device,
            temperature=args.temperature,
            output_scores=args.save_scores_softmax,
        )
        activations = extract_activations(generation_output)
        if args.save_layers:
            keep = {int(x) for x in args.save_layers.split(",")}
            activations["step_activations"] = [
                {k: v for k, v in s.items() if k in keep} for s in activations["step_activations"]
            ]
        activations["softmax_per_step"] = get_final_softmax(activations["scores"])
        sequences = activations["sequences"]
        # Decode generated text only (strip prompt) for saving
        prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.size(1)
        generated_text = tokenizer.decode(
            sequences[0, prompt_len:].cpu(), skip_special_tokens=True
        ).strip()
        activations["generated_text"] = generated_text
        if run_meta_extra is not None and i < len(run_meta_extra):
            extra = dict(run_meta_extra[i])
            extra["generated_text"] = generated_text
            activations["meta_extra"] = extra

        if args.print_predictions:
            full_decoded = tokenizer.decode(sequences[0], skip_special_tokens=True)
            print("=== Prediction ===")
            print(f"Prompt: {repr(prompt)}")
            print(f"Full decoded: {repr(full_decoded)}")
            print(f"Generated (without prompt): {repr(generated_text)}")
            print(f"Token ids: {sequences[0].tolist()}")

        if args.print_shapes and (len(prompts) == 1 or i == 0):
            print_activation_shapes(activations, tokenizer, sequences)

        if not args.no_save:
            run_dir = next_run_dir(args.output_dir)
            save_activations(
                activations, run_dir, args.save_format, save_scores_softmax=args.save_scores_softmax
            )
            print(f"Activations saved to {run_dir}")

    return model, tokenizer


if __name__ == "__main__":
    main()
