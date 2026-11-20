"""Training loop shared by every model; the YAML config decides which model and settings.

Run:  python -I -m src.train --config configs/smallcnn.yaml [--epochs N] [--device cpu|mps|cuda|auto]

Outputs:
  checkpoints/<run>.pt                 best weights (by validation ROC-AUC)
  reports/<run>_train_log.csv          one row per epoch
  reports/figures/<run>_curves.png     loss / accuracy / AUC curves
  reports/train_<run>.json             summary: device, epochs, time per epoch, best epoch
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from src.data import load_config, make_loaders
from src.models import build_model, count_parameters, get_device


def set_seed(seed: int) -> None:
    """Fix every random number generator so a run can be repeated.
    (GPU kernels may still differ in the last decimal places.)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    """One pass over the training set. Returns mean training loss."""
    model.train()  # enables dropout and makes BatchNorm use batch statistics
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad()           # gradients accumulate by default, so reset them
        loss = loss_fn(model(x), y)     # forward pass + binary cross-entropy
        loss.backward()                 # backpropagation: d(loss)/d(every weight)
        optimizer.step()                # Adam update of the weights
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def evaluate_loader(model, loader, loss_fn, device) -> dict:
    """Loss, accuracy and ROC-AUC on a validation/test loader."""
    model.eval()  # disables dropout; BatchNorm uses running statistics
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    logits, labels = torch.cat(logits), torch.cat(labels)
    probs = torch.sigmoid(logits).numpy()
    return {
        "loss": loss_fn(logits, labels.float()).item(),
        "acc": float(((probs >= 0.5) == labels.numpy()).mean()),
        "auc": float(roc_auc_score(labels.numpy(), probs)),
    }


def plot_curves(rows: list[dict], out_path: Path, title: str) -> None:
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(epochs, [r["train_loss"] for r in rows], "o-", label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in rows], "o-", label="val")
    axes[0].set_title("Binary cross-entropy loss")
    axes[1].plot(epochs, [r["val_acc"] for r in rows], "o-", color="tab:green")
    axes[1].set_title("Validation accuracy")
    axes[2].plot(epochs, [r["val_auc"] for r in rows], "o-", color="tab:purple")
    axes[2].set_title("Validation ROC-AUC")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, help="override max epochs from config")
    parser.add_argument("--device", help="override device from config (cpu/mps/cuda/auto)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t = cfg["train"]
    run = cfg["run_name"]
    epochs = args.epochs or t["epochs"]
    device = get_device(args.device or t.get("device", "cpu"))
    set_seed(cfg["seed"])

    loaders = make_loaders(cfg)
    model = build_model(cfg)
    if t.get("init_checkpoint"):
        # Fine-tuning run: start from weights we trained earlier instead of from scratch/ImageNet.
        ckpt = torch.load(t["init_checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print(f"initialised from {t['init_checkpoint']}", flush=True)
    model = model.to(device)
    # BCEWithLogitsLoss = sigmoid + binary cross-entropy in one numerically stable step
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    ckpt_path = Path("checkpoints") / f"{run}.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    log_path = Path("reports") / f"{run}_train_log.csv"
    log_path.parent.mkdir(exist_ok=True)

    print(f"run={run} device={device} params={count_parameters(model):,} "
          f"train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)}", flush=True)

    rows, best_auc, best_epoch, bad_epochs = [], -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
        val = evaluate_loader(model, loaders["val"], loss_fn, device)
        secs = time.time() - start
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
               "val_acc": val["acc"], "val_auc": val["auc"], "seconds": secs}
        rows.append(row)
        print(f"epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} "
              f"| val_acc {val['acc']:.4f} | val_auc {val['auc']:.4f} | {secs:.0f}s", flush=True)

        # Early stopping on validation ROC-AUC: keep the best model and stop once
        # it has not improved for `patience` epochs (it has started to overfit or plateau).
        if val["auc"] > best_auc:
            best_auc, best_epoch, bad_epochs = val["auc"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= t["patience"]:
                print(f"early stop: no val AUC improvement for {t['patience']} epochs", flush=True)
                break

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_curves(rows, Path("reports/figures") / f"{run}_curves.png", f"{run} training curves")
    summary = {
        "run": run, "device": str(device), "max_epochs": epochs, "epochs_run": len(rows),
        "best_epoch": best_epoch, "best_val_auc": best_auc,
        "mean_seconds_per_epoch": float(np.mean([r["seconds"] for r in rows])),
        "parameters": count_parameters(model), "torch_threads": torch.get_num_threads(),
    }
    with open(Path("reports") / f"train_{run}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
