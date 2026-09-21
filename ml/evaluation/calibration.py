import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin

    return float(ece)

class TemperatureScaler:
    """Post-hoc probability calibration via temperature scaling."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        t_logits = torch.tensor(logits, dtype=torch.float32)
        t_labels = torch.tensor(labels, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        def loss_fn(t_val):
            scaled = t_logits / max(1e-3, t_val[0])
            return criterion(scaled, t_labels).item()

        res = minimize(loss_fn, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
