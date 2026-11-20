"""Final evaluation on the untouched CIFAKE test set.

Run:  python -I -m src.evaluate --checkpoint checkpoints/smallcnn.pt

Outputs:
  reports/metrics_<run>.json                accuracy, precision, recall, F1, ROC-AUC,
                                            confusion matrix, parameters, CPU latency
  reports/figures/<run>_confusion.png
  reports/figures/<run>_roc.png

Cross-generator mode (COCOAI held-out test split, broken down per generator):
  python -I -m src.evaluate --checkpoint checkpoints/smallcnn.pt --cocoai data/cocoai/manifest.csv
  -> reports/cocoai_<run>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from torch.utils.data import DataLoader

from src.data import ImageDataset, build_transforms, get_splits, load_cocoai
from src.models import count_parameters, load_checkpoint


@torch.no_grad()
def predict_logits(model, loader, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader. Returns (logits, labels) as numpy arrays."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Standard metrics with FAKE (label 1) as the positive class.

    precision = of images we called FAKE, how many really were FAKE
    recall    = of the FAKE images, how many we caught
    ROC-AUC   = probability a random FAKE image gets a higher score than a random
                REAL one; threshold-free, 0.5 = guessing, 1.0 = perfect ranking
    """
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    out = {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    # AUC is undefined when only one class is present (e.g. a generator-only subset)
    out["roc_auc"] = float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else None
    return out


def plot_confusion(cm: dict, out_path: Path, title: str) -> None:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["REAL", "FAKE"])
    ax.set_yticks([0, 1], ["REAL", "FAKE"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc(labels: np.ndarray, probs: np.ndarray, auc: float, out_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random guess")
    ax.set_xlabel("False positive rate (real called fake)")
    ax.set_ylabel("True positive rate (fake caught)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def cpu_latency_ms(model, img_size: int, n: int = 200) -> float:
    """Average CPU time for ONE image (batch size 1), like a single web request.
    The first 20 runs are warm-up and not timed (memory allocation, caches)."""
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(20):
        model(x)
    start = time.perf_counter()
    for _ in range(n):
        model(x)
    return (time.perf_counter() - start) / n * 1000


def evaluate_cocoai(model, cfg: dict, manifest: str, device: str = "cpu", split: str = "test") -> dict:
    """Metrics on the COCOAI split: overall, and real-vs-each-generator.

    Per generator we compare the SAME 1000 real photos against that generator's
    1000 fakes, so every row is a balanced 50/50 problem.
    """
    d = cfg["data"]
    rows = load_cocoai(manifest, split)
    tf = build_transforms(d["img_size"], d["mean"], d["std"], train=False)
    loader = DataLoader(ImageDataset(Path(manifest).parent, [(p, y) for p, y, _ in rows], tf),
                        batch_size=128, num_workers=d["num_workers"])
    logits, labels = predict_logits(model, loader, device)
    probs = sigmoid(logits)
    gens = np.array([g for _, _, g in rows])
    real = gens == "real"
    out = {"overall": classification_metrics(labels, probs),
           "real_photos_correct": float((probs[real] < 0.5).mean()),
           "per_generator": {}}
    for g in sorted(set(gens) - {"real"}):
        mask = real | (gens == g)
        m = classification_metrics(labels[mask], probs[mask])
        m["fakes_caught"] = float((probs[gens == g] >= 0.5).mean())
        out["per_generator"][g] = m
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cocoai", help="path to COCOAI manifest.csv: run cross-generator evaluation")
    args = parser.parse_args()

    model, cfg = load_checkpoint(args.checkpoint, args.device)
    run, d = cfg["run_name"], cfg["data"]
    if args.cocoai:
        out = {"run": run, "dataset": "COCOAI subset, held-out test split", "threshold": 0.5,
               **evaluate_cocoai(model, cfg, args.cocoai, args.device)}
        with open(Path("reports") / f"cocoai_{run}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
        return
    test_items = get_splits(cfg)["test"]
    tf = build_transforms(d["img_size"], d["mean"], d["std"], train=False)
    loader = DataLoader(ImageDataset(Path(d["root"]), test_items, tf),
                        batch_size=256, num_workers=d["num_workers"])

    logits, labels = predict_logits(model, loader, args.device)
    probs = sigmoid(logits)
    metrics = classification_metrics(labels, probs)
    metrics.update({
        "run": run, "dataset": "CIFAKE official test set", "threshold": 0.5,
        "parameters": count_parameters(model),
        "cpu_ms_per_image": cpu_latency_ms(model, d["img_size"]),
        "cpu_threads": torch.get_num_threads(),
        "input_size": d["img_size"],
    })

    figs = Path("reports/figures")
    figs.mkdir(parents=True, exist_ok=True)
    plot_confusion(metrics["confusion_matrix"], figs / f"{run}_confusion.png", f"{run}: test confusion matrix")
    plot_roc(labels, probs, metrics["roc_auc"], figs / f"{run}_roc.png", f"{run}: test ROC")
    with open(Path("reports") / f"metrics_{run}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
