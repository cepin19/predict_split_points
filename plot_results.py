"""Plot ELRC-small probe results (held-out test set) into figures/.

Needs only numpy + matplotlib (the project env has neither matplotlib; run with e.g.
/home/jon/miniconda3/bin/python3 plot_results.py).

Figures:
  tokens_left_error_by_distance  - probe vs step-only baseline, error vs true tokens_left
  pred_vs_true                   - 2D histograms of predicted vs true, timestep and tokens_left
  tokens_left_trajectories       - predicted vs true countdown for example test runs
  run_lengths                    - generated length distribution, probe-train vs test runs
"""
import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# --- Palette (reference instance of the dataviz skill, light mode) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PROBE = "#2a78d6"  # categorical slot 1 (blue)
BASELINE = "#eb6834"  # categorical slot 2 (orange)
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", default="./elrc_small_activations_tokens_left_train",
                   help="Probe-train runs (used for the step-only baseline and length distribution)")
    p.add_argument("--test-dir", default="./elrc_small_activations_heldout_test")
    p.add_argument("--tokens-left-csv",
                   default="./saved_predictor_elrc_small_tokens_left/predictions_heldout_tokens_left_mlp_256_128.csv")
    p.add_argument("--timestep-csv",
                   default="./saved_predictor_elrc_small_timestep/predictions_heldout_timestep_mlp_256_128.csv")
    p.add_argument("--out-dir", default="./figures")
    p.add_argument("--bin", type=int, default=5, help="Bin width (tokens) for the error-by-distance plot")
    p.add_argument("--min-bin-count", type=int, default=200, help="Drop bins with fewer test steps than this")
    p.add_argument("--seed", type=int, default=0, help="Seed for picking example runs")
    return p


def run_lengths(data_dir):
    return np.array([json.load(open(f))["num_steps"] for f in glob.glob(f"{data_dir}/run*/meta.json")])


def load_predictions(path):
    rows = list(csv.DictReader(open(path)))
    run = np.array([r["run"] for r in rows])
    step = np.array([int(r["step"]) for r in rows])
    pred = np.array([int(r["predicted"]) for r in rows])
    true = np.array([int(r["true"]) for r in rows])
    return run, step, pred, true


def step_only_baseline(train_lengths, steps):
    """Median tokens_left at each step over train runs that reach that step."""
    by_step = defaultdict(list)
    for L in train_lengths:
        for s in range(L):
            by_step[s].append(L - 1 - s)
    med = {s: np.median(v) for s, v in by_step.items()}
    return np.round(np.array([med.get(s, 0) for s in steps]))


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / name}.{{pdf,png}}")


def plot_error_by_distance(true, pred, base, out_dir, bin_width, min_count):
    edges = np.arange(0, true.max() + bin_width + 1, bin_width)
    idx = np.digitize(true, edges) - 1
    centers, mae_p, mae_b, w5_p, w5_b = [], [], [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < min_count:
            continue
        ep, eb = np.abs(pred[m] - true[m]), np.abs(base[m] - true[m])
        centers.append(edges[b] + bin_width / 2)
        mae_p.append(ep.mean())
        mae_b.append(eb.mean())
        w5_p.append(np.mean(ep <= 5) * 100)
        w5_b.append(np.mean(eb <= 5) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    panels = [
        (axes[0], mae_p, mae_b, "Mean absolute error (tokens)", "Error by distance to end"),
        (axes[1], w5_p, w5_b, "Predictions within ±5 tokens (%)", "Share of near-correct predictions"),
    ]
    for ax, yp, yb, ylabel, title in panels:
        ax.plot(centers, yp, color=PROBE, label="Probe (hidden state)")
        ax.plot(centers, yb, color=BASELINE, label="Step-only baseline")
        # selective direct labels at the left end, where the gap is largest
        ax.annotate("Probe", (centers[0], yp[0]), xytext=(6, 0), textcoords="offset points",
                    color=INK_2, va="center", fontsize=9)
        ax.annotate("Step only", (centers[0], yb[0]), xytext=(6, 8), textcoords="offset points",
                    color=INK_2, va="center", fontsize=9)
        ax.set_xlabel("True tokens left")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, centers[-1] + bin_width / 2)
        ax.set_ylim(bottom=0)
    axes[1].set_ylim(0, 100)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2, fontsize=9)
    save(fig, out_dir, "tokens_left_error_by_distance")


def plot_pred_vs_true(ts, tl, out_dir):
    hi = max(ts[1].max(), ts[0].max(), tl[1].max(), tl[0].max()) + 1
    bins = np.arange(0, hi + 5, 5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    counts = [np.histogram2d(t, p, bins=[bins, bins])[0] for p, t in (ts, tl)]
    vmax = max(c.max() for c in counts)
    for ax, c, title in zip(axes, counts, ["Timestep (position from start)", "Tokens left (distance to end)"]):
        c = np.where(c > 0, c, np.nan)
        im = ax.pcolormesh(bins, bins, c.T, cmap=SEQ_BLUE, norm=LogNorm(vmin=1, vmax=vmax), shading="flat")
        ax.plot([0, hi], [0, hi], color=MUTED, linewidth=1, linestyle="--")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("True value")
        ax.set_ylabel("Predicted value")
        ax.set_title(title, loc="left")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Test steps per 5×5 cell (log)", color=INK_2)
    cb.outline.set_visible(False)
    save(fig, out_dir, "pred_vs_true")


def plot_trajectories(run, step, pred, true, test_lengths_by_run, out_dir, seed):
    # pick runs spread across the length distribution: ~10th..90th percentile
    runs = np.array(sorted(test_lengths_by_run))
    lens = np.array([test_lengths_by_run[r] for r in runs])
    rng = np.random.default_rng(seed)
    picks = []
    for q in (10, 26, 42, 58, 74, 90):
        target = np.percentile(lens, q)
        cands = runs[np.abs(lens - target) <= max(2, 0.05 * target)]
        picks.append(rng.choice(cands))

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), constrained_layout=True)
    for ax, r in zip(axes.flat, picks):
        m = run == r
        order = np.argsort(step[m])
        s, p, t = step[m][order], pred[m][order], true[m][order]
        ax.plot(s, t, color=MUTED, linewidth=2, label="True tokens left")
        ax.plot(s, p, color=PROBE, linewidth=1, alpha=0.5)
        ax.scatter(s, p, color=PROBE, s=10, linewidths=0, label="Probe prediction", zorder=3)
        ax.set_title(f"{Path(r).name}, {len(s)} steps", loc="left", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, len(s))
    for ax in axes[1]:
        ax.set_xlabel("Generation step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Tokens left")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    save(fig, out_dir, "tokens_left_trajectories")


def plot_run_lengths(train_lengths, test_lengths, out_dir):
    bins = np.arange(0, max(train_lengths.max(), test_lengths.max()) + 10, 10)
    fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    for lens, color, name in [(train_lengths, PROBE, "Probe train"), (test_lengths, BASELINE, "Held-out test")]:
        ax.hist(lens, bins=bins, density=True, histtype="step", linewidth=2, color=color,
                label=f"{name} ({len(lens)} runs, median {np.median(lens):.0f})")
    ax.set_xlabel("Generated tokens per run (cap = 250)")
    ax.set_ylabel("Density")
    ax.set_title("Run-length distribution", loc="left")
    ax.legend(fontsize=9)
    save(fig, out_dir, "run_lengths")


def main():
    args = get_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_lengths = run_lengths(args.train_dir)
    test_lengths = run_lengths(args.test_dir)
    tl_run, tl_step, tl_pred, tl_true = load_predictions(args.tokens_left_csv)
    _, _, ts_pred, ts_true = load_predictions(args.timestep_csv)
    base = step_only_baseline(train_lengths, tl_step)
    test_lengths_by_run = defaultdict(int)
    for r in tl_run:
        test_lengths_by_run[r] += 1

    plot_error_by_distance(tl_true, tl_pred, base, out_dir, args.bin, args.min_bin_count)
    plot_pred_vs_true((ts_pred, ts_true), (tl_pred, tl_true), out_dir)
    plot_trajectories(tl_run, tl_step, tl_pred, tl_true, test_lengths_by_run, out_dir, args.seed)
    plot_run_lengths(train_lengths, test_lengths, out_dir)


if __name__ == "__main__":
    main()
