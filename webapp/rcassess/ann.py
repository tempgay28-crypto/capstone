"""NumPy forward pass of the exported ANN ensemble.

Reproduces the scikit-learn pipeline used in training (02_train_models.py) step by step:
  log1p of size-type inputs -> median imputation with "was missing" flags -> standardisation
  -> hidden layers (relu or tanh) -> linear output -> undo the target scaling.
The weights come from model.json, written and checked against scikit-learn by
ann_model/04_export_web_model.py.
"""
import numpy as np

ACTIVATIONS = {
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "identity": lambda z: z,
}


class Network:
    """One trained network of the ensemble."""

    def __init__(self, spec, features):
        index = {name: i for i, name in enumerate(features)}
        self.blocks = [{
            "cols": np.array([index[c] for c in b["cols"]]),
            "log": b["log"],
            "medians": np.array(b["medians"], dtype=float),
            "indicator": np.array(b["indicator"], dtype=int),
            "mean": np.array(b["mean"], dtype=float),
            "scale": np.array(b["scale"], dtype=float),
        } for b in spec["blocks"]]
        self.weights = [np.array(w, dtype=float) for w in spec["weights"]]
        self.biases = [np.array(b, dtype=float) for b in spec["biases"]]
        self.activation = ACTIVATIONS[spec["activation"]]
        self.y_mean, self.y_scale, self.log_target = spec["y_mean"], spec["y_scale"], spec["log_target"]

    def transform(self, X):
        parts = []
        for b in self.blocks:
            v = X[:, b["cols"]]
            if b["log"]:
                v = np.log1p(v)
            missing = np.isnan(v)
            v = np.where(missing, b["medians"], v)
            if b["indicator"].size:
                v = np.hstack([v, missing[:, b["indicator"]].astype(float)])
            parts.append((v - b["mean"]) / b["scale"])
        return np.hstack(parts)

    def predict(self, X):
        a = self.transform(X)
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            a = a @ w + b
            if i < last:
                a = self.activation(a)
        y = a[:, 0] * self.y_scale + self.y_mean
        return np.expm1(y) if self.log_target else y


class Ensemble:
    def __init__(self, spec):
        self.features = list(spec["features"])
        self.networks = [Network(m, self.features) for m in spec["members"]]

    def predict_members(self, X):
        """Predictions of every network, shape (n_networks, n_rows). X columns follow self.features."""
        X = np.asarray(X, dtype=float)
        return np.array([n.predict(X) for n in self.networks])

    def predict(self, X):
        return self.predict_members(X).mean(axis=0)
