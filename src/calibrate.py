"""Temperature scaling: make the model's probabilities honest.

A network trained with cross-entropy is often over-confident: it says 0.99 when
it is right only 95% of the time. Temperature scaling divides every logit by one
number T before the sigmoid:   p = sigmoid(z / T)
  T > 1 softens probabilities (towards 0.5), T < 1 sharpens them.
T is chosen to minimise the negative log-likelihood (NLL) on the VALIDATION set,
never the test set, so the test set stays an honest check.
Because T > 0 never changes the sign of z, predictions at threshold 0.5,
accuracy and ROC-AUC are unchanged; only the confidence changes.

ECE (expected calibration error): put predictions into 15 confidence bins; in
each bin compare average confidence with actual accuracy; ECE is the weighted
average gap. 0 = perfectly calibrated.

Run:  python -I -m src.calibrate --checkpoint checkpoints/smallcnn.pt [--device cpu]
Outputs reports/calibration_<run>.json and reports/figures/<run>_reliability.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from src.data import ImageDataset, build_transforms, get_splits
from src.evaluate import predict_logits, sigmoid
from src.models import load_checkpoint


def nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    """Mean binary negative log-likelihood (= binary cross-entropy) at temperature T.

    Written with logaddexp instead of log(sigmoid(.)): -log(sigmoid(x)) = log(1 + e^-x)
    = logaddexp(0, -x). For a very confident logit, sigmoid rounds to exactly 1.0
    and log(1 - 1.0) = -inf; the logaddexp form never rounds like that.
    """
    x = logits.astype(np.float64) / T
    return float(np.mean(labels * np.logaddexp(0, -x) + (1 - labels) * np.logaddexp(0, x)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find T by searching 2,000 values between 0.05 and 20 (log-spaced).

    A plain grid search is used instead of gradient descent because there is
    only one parameter: it is easy to explain and cannot get stuck.
    """
    grid = np.logspace(np.log10(0.05), np.log10(20), 2000)
    losses = [nll(logits, labels, T) for T in grid]
    return float(grid[int(np.argmin(losses))])


def reliability_points(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Group predictions into confidence bins.

    Confidence = probability of the PREDICTED class = max(p, 1-p), so it lies in
    [0.5, 1]; the bins split that range evenly. Returns, per non-empty bin,
    (mean confidence, accuracy, count).
    """
    preds = (probs >= 0.5).astype(int)
    conf = np.where(preds == 1, probs, 1 - probs)
    correct = (preds == labels).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    return [(float(conf[idx == b].mean()), float(correct[idx == b].mean()), int((idx == b).sum()))
            for b in range(n_bins) if (idx == b).any()]


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error = sum over bins of (bin share) * |accuracy - confidence|."""
    n = len(labels)
    return float(sum(cnt / n * abs(acc - conf) for conf, acc, cnt in reliability_points(probs, labels, n_bins)))


def plot_reliability(before, after, ece_b, ece_a, T, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0.5, 1], [0.5, 1], "--", color="grey", label="perfect calibration")
    for pts, lab, mk in ((before, f"before (ECE {ece_b:.4f})", "o-"),
                         (after, f"after T={T:.3f} (ECE {ece_a:.4f})", "s-")):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, label=lab)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, cfg = load_checkpoint(args.checkpoint, args.device)
    run, d = cfg["run_name"], cfg["data"]
    splits = get_splits(cfg)
    tf = build_transforms(d["img_size"], d["mean"], d["std"], train=False)

    def logits_for(split):
        loader = DataLoader(ImageDataset(Path(d["root"]), splits[split], tf), batch_size=256,
                            num_workers=d["num_workers"])
        return predict_logits(model, loader, args.device)

    val_z, val_y = logits_for("val")
    test_z, test_y = logits_for("test")
    T = fit_temperature(val_z, val_y)

    p_before, p_after = sigmoid(test_z), sigmoid(test_z / T)
    out = {
        "run": run, "temperature": T, "fitted_on": "validation split",
        "reported_on": "CIFAKE test set",
        "val_nll_before": nll(val_z, val_y, 1.0), "val_nll_after": nll(val_z, val_y, T),
        "test_nll_before": nll(test_z, test_y, 1.0), "test_nll_after": nll(test_z, test_y, T),
        "test_ece_before": ece(p_before, test_y), "test_ece_after": ece(p_after, test_y),
        "n_bins": 15,
    }
    plot_reliability(reliability_points(p_before, test_y), reliability_points(p_after, test_y),
                     out["test_ece_before"], out["test_ece_after"], T,
                     Path("reports/figures") / f"{run}_reliability.png",
                     f"{run}: reliability diagram (test)")
    with open(Path("reports") / f"calibration_{run}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
