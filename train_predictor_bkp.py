#!/usr/bin/env python3
"""
Load activations from all run subdirs in a given directory, build train/test sets,
and train a simple model to predict a target from the activations.
Default target: timestep (generation step index).
Save/load the trained model and run prediction on new data.

Use --batch-runs N to train with limited memory: load at most N runs at a time
and use incremental training (SGDClassifier + incremental scaler).
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import torch

from run_inference import ACTIVATIONS_BLOB_NPZ, ACTIVATIONS_BLOB_PT, load_activations


RUN_PATTERN = re.compile(r"^run(\d+)$")
MODEL_FILENAME = "predictor.joblib"


def parse_hidden_layers(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def get_parser():
    p = argparse.ArgumentParser(description="Train a predictor on saved activations or run it on new data")
    p.add_argument("mode", choices=["train", "predict"], help="train: fit and save model; predict: load model and predict on new data")
    p.add_argument("--data-dir", type=str, default="./activations_out", help="Directory containing run0, run1, ... (train or predict)")
    p.add_argument("--model-dir", type=str, default="./saved_predictor", help="Directory to save/load the trained model (and scaler)")
    p.add_argument("--test-frac", type=float, default=0.2, help="Fraction of data for testing (0–1), train only")
    p.add_argument("--seed", type=int, default=42, help="Random seed for train/test split and training")
    p.add_argument(
        "--target",
        type=str,
        default="timestep",
        choices=["timestep", "tokens_left"],
        help="Target: timestep (position from start), tokens_left (how many tokens before end)",
    )
    p.add_argument("--layer", type=int, default=None, help="Use only this layer index (default: concat all layers)")
    p.add_argument("--last-position-only", action="store_true", default=True, help="Use only last sequence position (default: True)")
    p.add_argument("--no-last-position-only", action="store_false", dest="last_position_only")
    p.add_argument("--predictions-out", type=str, default=None, help="Save predictions to this path (predict mode; default: print only)")
    p.add_argument(
        "--batch-runs",
        type=int,
        default=None,
        metavar="N",
        help="Low-memory mode: load at most N runs at a time, use incremental training (default: load all)",
    )
    p.add_argument("--debug", action="store_true", help="Print progress and extra info during training/predict")
    # Classifier architecture
    p.add_argument(
        "--classifier",
        type=str,
        default="mlp",
        choices=["mlp", "sgd"],
        help="Classifier type for full-memory path; --batch-runs always uses sgd regardless",
    )
    p.add_argument(
        "--hidden-layers",
        type=str,
        default="256,128",
        help="Comma-separated MLP hidden layer sizes, e.g. '256,128' or '512,256,128' (mlp only)",
    )
    p.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "logistic"],
        help="Activation function for MLP hidden layers",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.0001,
        help="L2 regularization strength (MLP and SGD)",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Max training iterations (MLP: epochs with early stopping; SGD: passes over data)",
    )
    p.add_argument(
        "--learning-rate-init",
        type=float,
        default=0.001,
        help="Initial learning rate for MLP",
    )
    p.add_argument(
        "--sgd-loss",
        type=str,
        default="log_loss",
        choices=["hinge", "log_loss", "modified_huber"],
        help="Loss function for SGD classifier (batch path and --classifier sgd)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of full passes over training data in batch mode (--batch-runs); for full-memory path use --max-iter",
    )
    return p


def discover_run_dirs(data_dir: Path) -> list[Path]:
    """Return sorted list of run subdirs (run0, run1, ...). If data_dir is itself one run (step_0.* or activations blob), return [data_dir]."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    if (
        (data_dir / "step_0.pt").exists()
        or (data_dir / "step_0.npz").exists()
        or (data_dir / ACTIVATIONS_BLOB_PT).exists()
        or (data_dir / ACTIVATIONS_BLOB_NPZ).exists()
    ):
        return [data_dir]
    runs = []
    for p in data_dir.iterdir():
        if p.is_dir() and RUN_PATTERN.match(p.name):
            runs.append((int(RUN_PATTERN.match(p.name).group(1)), p))
    return [p for _, p in sorted(runs)]


def get_num_steps_per_run(run_dirs: list[Path]) -> list[int]:
    """Read meta.json from each run dir; return list of num_steps (no activation loading)."""
    out = []
    for run_path in run_dirs:
        meta_path = run_path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {run_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        out.append(meta["num_steps"])
    return out


def get_n_features_from_one_run(
    run_path: Path, last_position_only: bool, layer_idx: int | None
) -> int:
    """Load first step from one run to get feature vector length (minimal load)."""
    data = load_activations(run_path)
    step_activations = data.get("step_activations") or []
    if not step_activations or not step_activations[0]:
        raise ValueError(f"No step activations in {run_path}")
    x = activations_to_feature_vector(step_activations[0], last_position_only, layer_idx)
    return x.size


def load_runs_chunk(
    run_dirs: list[Path],
    last_position_only: bool,
    layer_idx: int | None,
    target: str,
    num_steps_per_run: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load a subset of runs and return (X, y). Uses num_steps_per_run for labels (same length as run_dirs)."""
    X_list = []
    y_list = []
    for run_path, num_steps in zip(run_dirs, num_steps_per_run):
        data = load_activations(run_path)
        step_activations = data.get("step_activations") or []
        for step_idx, step_layers in enumerate(step_activations):
            if not step_layers:
                continue
            x = activations_to_feature_vector(step_layers, last_position_only, layer_idx)
            X_list.append(x)
            if target == "timestep":
                y_list.append(step_idx)
            elif target == "tokens_left":
                y_list.append(num_steps - 1 - step_idx)
            else:
                raise ValueError(f"Unknown target: {target}")
    if not X_list:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int64)
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def activations_to_feature_vector(step_layers: dict[int, torch.Tensor], last_position_only: bool, layer_idx: int | None) -> np.ndarray:
    """
    Convert one step's layer activations to a single feature vector.
    step_layers: dict layer_index -> (batch, seq_len, hidden_size)
    If layer_idx is set, use only that layer; else concatenate all layers (in order).
    If last_position_only, take only the last position along seq_len; else flatten.
    """
    if layer_idx is not None:
        h = step_layers[layer_idx]  # (batch, seq_len, hidden_size)
        if last_position_only:
            h = h[:, -1:, :]  # (batch, 1, hidden_size)
        x = h.float().reshape(-1).numpy()
        return x

    # Concat all layers (convert to float32 so bfloat16/float16 work with numpy)
    parts = []
    for k in sorted(step_layers.keys()):
        h = step_layers[k]
        if last_position_only:
            h = h[:, -1, :]  # (batch, hidden_size)
        else:
            h = h.reshape(h.size(0), -1)
        parts.append(h.float().numpy())
    return np.concatenate(parts, axis=1).flatten()


def load_all_runs(
    data_dir: Path,
    last_position_only: bool,
    layer_idx: int | None,
    target: str = "timestep",
):
    """Load all runs and return (X, y) arrays.
    target: timestep = step index (position from start); tokens_left = num steps remaining before end.
    """
    run_dirs = discover_run_dirs(data_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run subdirs (run0, run1, ...) found in {data_dir}")

    X_list = []
    y_list = []
    for run_path in run_dirs:
        data = load_activations(run_path)
        step_activations = data.get("step_activations") or []
        num_steps = len(step_activations)
        for step_idx, step_layers in enumerate(step_activations):
            if not step_layers:
                continue
            x = activations_to_feature_vector(step_layers, last_position_only, layer_idx)
            X_list.append(x)
            if target == "timestep":
                y_list.append(step_idx)
            elif target == "tokens_left":
                y_list.append(num_steps - 1 - step_idx)  # tokens remaining before end
            else:
                raise ValueError(f"Unknown target: {target}")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, run_dirs


def train_test_split(X: np.ndarray, y: np.ndarray, test_frac: float, seed: int):
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    n_train = n - n_test
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def welford_update(count: int, mean: np.ndarray, M2: np.ndarray, X_batch: np.ndarray):
    """Online update for mean and M2 (sum of squared differences from current mean). Returns (new_count, new_mean, new_M2)."""
    n_batch = X_batch.shape[0]
    if n_batch == 0:
        return count, mean, M2
    if count == 0:
        new_mean = X_batch.mean(axis=0).astype(np.float64)
        new_M2 = ((X_batch - new_mean) ** 2).sum(axis=0).astype(np.float64)
        return n_batch, new_mean, new_M2
    total = count + n_batch
    delta = X_batch - mean
    new_mean = mean + delta.sum(axis=0) / total
    delta2 = X_batch - new_mean
    new_M2 = M2 + (delta * delta2).sum(axis=0)
    return total, new_mean, new_M2


class IncrementalScaler:
    """Scaler with mean_ and scale_ (std) set manually; used for low-memory path."""

    def __init__(self, mean_: np.ndarray, scale_: np.ndarray):
        self.mean_ = mean_
        self.scale_ = scale_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X.astype(np.float64) - self.mean_) / np.where(self.scale_ != 0, self.scale_, 1.0)


def incremental_fit_scaler(
    run_dirs: list[Path],
    num_steps_per_run: list[int],
    batch_runs: int,
    last_position_only: bool,
    layer_idx: int | None,
    n_features: int,
    target: str,
) -> IncrementalScaler:
    """One pass over run_dirs in chunks; return scaler (mean, scale) for transform."""
    count = 0
    mean = np.zeros(n_features, dtype=np.float64)
    M2 = np.zeros(n_features, dtype=np.float64)
    for start in range(0, len(run_dirs), batch_runs):
        chunk = run_dirs[start : start + batch_runs]
        num_steps_chunk = num_steps_per_run[start : start + batch_runs]
        X, _ = load_runs_chunk(chunk, last_position_only, layer_idx, target, num_steps_chunk)
        if X.shape[0] == 0:
            continue
        count, mean, M2 = welford_update(count, mean, M2, X)
    scale = np.sqrt(M2 / count)
    scale[scale == 0] = 1.0
    return IncrementalScaler(mean.astype(np.float32), scale.astype(np.float32))


def build_model_artifact(clf, scaler, config: dict) -> dict:
    """Bundle classifier, scaler, and feature config for saving."""
    return {"clf": clf, "scaler": scaler, "config": config}


def save_model(artifact: dict, model_dir: str | Path) -> Path:
    """Save trained model, scaler, and config to model_dir."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / MODEL_FILENAME
    joblib.dump(artifact, path)
    return path


def load_model(model_dir: str | Path) -> dict:
    """Load model artifact (clf, scaler, config) from model_dir."""
    path = Path(model_dir) / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {path}")
    return joblib.load(path)


def run_train(args) -> dict:
    """Train and save model. Returns the model artifact."""
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    batch_runs = args.batch_runs

    run_dirs = discover_run_dirs(data_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run subdirs (run0, run1, ...) found in {data_dir}")

    if batch_runs is not None:
        # Low-memory path: load in chunks, incremental scaler + SGDClassifier
        import time
        from sklearn.linear_model import SGDClassifier

        num_steps_per_run = get_num_steps_per_run(run_dirs)
        n_features = get_n_features_from_one_run(run_dirs[0], args.last_position_only, args.layer)
        n_classes = max(num_steps_per_run)  # max class value + 1 for 0..n_classes-1
        classes = np.arange(n_classes, dtype=np.int64)
        total_samples = sum(num_steps_per_run)
        target_desc = "timestep 0..n-1" if args.target == "timestep" else "tokens_left 0..n-1"
        print(f"Found {len(run_dirs)} runs, ~{total_samples} samples, {n_features} features, {n_classes} classes ({target_desc})")
        print(f"Low-memory mode: batch_runs={batch_runs}")

        # Run-level train/test split
        rng = np.random.default_rng(args.seed)
        n_runs = len(run_dirs)
        perm = rng.permutation(n_runs)
        n_test_runs = max(1, int(n_runs * args.test_frac))
        n_train_runs = n_runs - n_test_runs
        train_idx = perm[:n_train_runs]
        test_idx = perm[n_train_runs:]
        train_run_dirs = [run_dirs[i] for i in train_idx]
        test_run_dirs = [run_dirs[i] for i in test_idx]
        train_num_steps = [num_steps_per_run[i] for i in train_idx]
        test_num_steps = [num_steps_per_run[i] for i in test_idx]
        n_train_samples = sum(train_num_steps)
        n_test_samples = sum(test_num_steps)
        print(f"Train: {len(train_run_dirs)} runs (~{n_train_samples} samples), Test: {len(test_run_dirs)} runs (~{n_test_samples} samples)")

        # Pass 1: incremental scaler over train runs
        print("Pass 1: fitting scaler on train runs...")
        scaler = incremental_fit_scaler(
            train_run_dirs, train_num_steps, batch_runs,
            args.last_position_only, args.layer, n_features, args.target,
        )

        # Pass 2: SGDClassifier partial_fit over train runs (args.epochs full passes)
        print(f"Pass 2: training classifier ({args.epochs} epoch(s))...")
        clf = SGDClassifier(
            loss=args.sgd_loss,
            alpha=args.alpha,
            max_iter=args.max_iter,
            tol=1e-3,
            random_state=args.seed,
        )
        for epoch in range(args.epochs):
            if args.epochs > 1:
                print(f"  Epoch {epoch + 1}/{args.epochs}")
            for start in range(0, len(train_run_dirs), batch_runs):
                chunk = train_run_dirs[start : start + batch_runs]
                num_steps_chunk = train_num_steps[start : start + batch_runs]
                X_batch, y_batch = load_runs_chunk(chunk, args.last_position_only, args.layer, args.target, num_steps_chunk)
                if X_batch.shape[0] == 0:
                    continue
                X_batch_s = scaler.transform(X_batch)
                clf.partial_fit(X_batch_s, y_batch, classes=classes)

        # Evaluate on test runs (in chunks)
        if args.debug:
            print("Evaluating on test runs...")
        train_acc = None
        test_preds, test_ys = [], []
        n_test_chunks = (len(test_run_dirs) + batch_runs - 1) // batch_runs
        for chunk_idx, start in enumerate(range(0, len(test_run_dirs), batch_runs)):
            t0 = time.perf_counter() if args.debug else None
            chunk = test_run_dirs[start : start + batch_runs]
            num_steps_chunk = test_num_steps[start : start + batch_runs]
            X_batch, y_batch = load_runs_chunk(chunk, args.last_position_only, args.layer, args.target, num_steps_chunk)
            if X_batch.shape[0] == 0:
                continue
            X_batch_s = scaler.transform(X_batch)
            test_preds.append(clf.predict(X_batch_s))
            test_ys.append(y_batch)
            if args.debug:
                elapsed = time.perf_counter() - t0
                print(f"  [debug] Test chunk {chunk_idx + 1}/{n_test_chunks}: {X_batch.shape[0]} samples, {elapsed:.2f}s")
        if test_preds:
            y_test = np.concatenate(test_ys)
            y_pred_test = np.concatenate(test_preds)
            test_acc = (y_pred_test == y_test).mean()
            zero_acc = (y_test == 0).mean()
            random_acc = 1.0 / n_classes
            print(f"Test accuracy:  {test_acc:.4f}")
            print(f"Random baseline (1/{n_classes}): {random_acc:.4f}")
            print(f"Always-predict-0 baseline: {zero_acc:.4f}")

        else:
            test_acc = None

        config = {
            "last_position_only": args.last_position_only,
            "layer": args.layer,
            "target": args.target,
            "classifier": "sgd",
            "sgd_loss": args.sgd_loss,
            "alpha": args.alpha,
            "max_iter": args.max_iter,
            "epochs": args.epochs,
        }
        artifact = build_model_artifact(clf, scaler, config)
        save_path = save_model(artifact, model_dir)
        print(f"Model saved to {save_path}")
        return artifact

    # Default path: load all into memory
    print(f"Loading runs from {data_dir}...")
    X, y, run_dirs = load_all_runs(data_dir, args.last_position_only, args.layer, target=args.target)
    n_samples, n_features = X.shape
    n_classes = int(y.max()) + 1
    target_desc = "timestep 0..n-1" if args.target == "timestep" else "tokens_left 0..n-1"
    print(f"Found {len(run_dirs)} runs, {n_samples} samples, {n_features} features, {n_classes} classes ({target_desc})")
    print(f"Random baseline (1/{n_classes}): {1.0 / n_classes:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, args.test_frac, args.seed)
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if args.classifier == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=parse_hidden_layers(args.hidden_layers),
            activation=args.activation,
            alpha=args.alpha,
            max_iter=args.max_iter,
            learning_rate_init=args.learning_rate_init,
            random_state=args.seed,
            early_stopping=True,
        )
    else:
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(
            loss=args.sgd_loss,
            alpha=args.alpha,
            max_iter=args.max_iter,
            random_state=args.seed,
        )
    clf.fit(X_train_s, y_train)

    train_acc = (clf.predict(X_train_s) == y_train).mean()
    test_acc = (clf.predict(X_test_s) == y_test).mean()
    zero_acc = (y_test == 0).mean()
    random_acc = 1.0 / n_classes
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Random baseline (1/{n_classes}): {random_acc:.4f}")
    print(f"Always-predict-0 baseline: {zero_acc:.4f}")

    config = {
        "last_position_only": args.last_position_only,
        "layer": args.layer,
        "target": args.target,
        "classifier": args.classifier,
        "hidden_layers": args.hidden_layers if args.classifier == "mlp" else None,
        "activation": args.activation if args.classifier == "mlp" else None,
        "alpha": args.alpha,
        "max_iter": args.max_iter,
        "sgd_loss": args.sgd_loss if args.classifier == "sgd" else None,
    }
    artifact = build_model_artifact(clf, scaler, config)
    save_path = save_model(artifact, model_dir)
    print(f"Model saved to {save_path}")
    return artifact


def run_predict(args) -> None:
    """Load model and run prediction on new data."""
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    batch_runs = args.batch_runs

    print(f"Loading model from {model_dir}...")
    artifact = load_model(model_dir)
    clf = artifact["clf"]
    scaler = artifact["scaler"]
    config = artifact["config"]
    last_position_only = config["last_position_only"]
    layer_idx = config.get("layer")
    target = config.get("target", "timestep")

    run_dirs = discover_run_dirs(data_dir)
    if not run_dirs:
        raise ValueError(f"No run subdirs found in {data_dir}")
    num_steps_per_run = get_num_steps_per_run(run_dirs)

    if batch_runs is not None:
        # Low-memory predict: load and predict in chunks
        print(f"Predicting on {len(run_dirs)} runs (batch_runs={batch_runs})...")
        all_preds = []
        all_ys = []
        for start in range(0, len(run_dirs), batch_runs):
            chunk = run_dirs[start : start + batch_runs]
            num_steps_chunk = num_steps_per_run[start : start + batch_runs]
            X_batch, y_batch = load_runs_chunk(chunk, last_position_only, layer_idx, target, num_steps_chunk)
            if X_batch.shape[0] == 0:
                continue
            X_batch_s = scaler.transform(X_batch)
            all_preds.append(clf.predict(X_batch_s))
            all_ys.append(y_batch)
        y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
        y_true = np.concatenate(all_ys) if all_ys else None
    else:
        print(f"Loading data from {data_dir}...")
        X, y_true, run_dirs = load_all_runs(data_dir, last_position_only, layer_idx, target=target)
        if X.size == 0:
            raise ValueError(f"No activation data found in {data_dir}")
        X_s = scaler.transform(X)
        y_pred = clf.predict(X_s)

    if y_true is not None and len(y_true) == len(y_pred):
        acc = (y_pred == y_true).mean()
        print(f"Accuracy (if labels present): {acc:.4f}")

    # Build per-step lines from num_steps_per_run (no extra activation loading)
    rows = []
    idx = 0
    for run_path, n_steps in zip(run_dirs, num_steps_per_run):
        for step in range(n_steps):
            if idx >= len(y_pred):
                break
            true_val = int(y_true[idx]) if y_true is not None else None
            rows.append((str(run_path), step, int(y_pred[idx]), true_val))
            idx += 1

    if args.predictions_out:
        out_path = Path(args.predictions_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("run,step,predicted,true\n")
            for run_name, step, pred, true_val in rows:
                true_s = str(true_val) if true_val is not None else ""
                f.write(f"{run_name},{step},{pred},{true_s}\n")
        print(f"Predictions written to {out_path}")

    print(f"Predictions (run, step, predicted {target}):")
    for run_name, step, pred, _ in rows[:20]:
        print(f"  {run_name} step={step} -> {pred}")
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more")


def main():
    args = get_parser().parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_predict(args)


if __name__ == "__main__":
    main()
