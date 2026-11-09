"""Small PyTorch MLP regressor with an sklearn-like fit/predict interface.

Used by train_predictor.py --regression when --regression-loss is absolute or huber,
which sklearn's MLPRegressor (squared error only) does not support. Defaults mirror
MLPRegressor: Adam, batch size 200, early stopping on a 10% validation split.
Kept in its own module so pickled models load regardless of how train_predictor runs.
"""

import numpy as np
import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "logistic": nn.Sigmoid}
LOSSES = {"squared": nn.MSELoss, "absolute": nn.L1Loss, "huber": nn.HuberLoss}


class TorchMLPRegressor:
    def __init__(
        self,
        hidden_layer_sizes=(256, 128),
        activation="relu",
        loss="absolute",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=500,
        batch_size=200,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=None,
        device=None,
        verbose=True,
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.loss = loss
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.model_ = None

    def _build(self, n_features: int) -> nn.Module:
        layers, d = [], n_features
        for h in self.hidden_layer_sizes:
            layers += [nn.Linear(d, h), ACTIVATIONS[self.activation]()]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPRegressor":
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        perm = rng.permutation(len(X))
        n_val = max(1, int(len(X) * self.validation_fraction))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)

        model = self._build(X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        loss_fn = LOSSES[self.loss]()

        best_val, best_state, bad_epochs = float("inf"), None, 0
        for epoch in range(self.max_iter):
            model.train()
            order = torch.as_tensor(rng.permutation(tr_idx), device=device)
            for start in range(0, len(order), self.batch_size):
                b = order[start : start + self.batch_size]
                opt.zero_grad()
                loss_fn(model(X_t[b]).squeeze(1), y_t[b]).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_t[val_idx]).squeeze(1), y_t[val_idx]).item()
            if self.verbose:
                print(f"  epoch {epoch + 1}: val {self.loss} loss {val_loss:.4f}")
            if val_loss < best_val - self.tol:
                best_val, bad_epochs = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.n_iter_no_change:
                    break

        model.load_state_dict(best_state)
        self.model_ = model.cpu().eval()
        self.n_iter_ = epoch + 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for start in range(0, len(X), 65536):
                xb = torch.as_tensor(X[start : start + 65536], dtype=torch.float32)
                out.append(self.model_(xb).squeeze(1).numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)
