"""Robustness: does the detector still work after everyday image processing?

Apps like WhatsApp and Instagram recompress (JPEG) and resize uploads. AI-image
artefacts are often tiny, high-frequency pixel patterns, which is exactly what
JPEG compression, downscaling and blur remove. So we re-evaluate the test set
after each degradation and see how far the metrics fall.

Degradations are applied to the ORIGINAL 32x32 image, before the model's own
resize/normalise, exactly as if someone had shared the image and we received
the processed copy.

Run:  python -I -m src.robustness --checkpoint checkpoints/smallcnn.pt [--device cpu]
Outputs reports/robustness_<run>.json and reports/figures/<run>_robustness.png
"""
from __future__ import annotations

import argparse
import io
import json
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from src.data import ImageDataset, build_transforms, get_splits
from src.evaluate import classification_metrics, predict_logits, sigmoid
from src.models import load_checkpoint


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode as JPEG at the given quality (100 = best, 30 = heavy compression)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def down_up(img: Image.Image, scale: float) -> Image.Image:
    """Shrink by `scale` then enlarge back: fine detail is lost but the size is unchanged."""
    w, h = img.size
    small = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


# partial(f, x=...) fixes one argument. Unlike a nested function it can be pickled,
# which DataLoader worker processes need (they receive the dataset by pickling).
# Ordered from mild to severe inside each family; "clean" is the reference.
DEGRADATIONS = {
    "clean": None,
    **{f"jpeg_q{q}": partial(jpeg, quality=q) for q in (90, 70, 50, 30)},
    "resize_0.75": partial(down_up, scale=0.75), "resize_0.5": partial(down_up, scale=0.5),
    "blur_r0.5": partial(blur, radius=0.5), "blur_r1.0": partial(blur, radius=1.0),
}


def plot_robustness(results: dict, out_path: Path, title: str) -> None:
    names = list(results)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = range(len(names))
    ax.plot(x, [results[n]["accuracy"] for n in names], "o-", label="accuracy")
    ax.plot(x, [results[n]["roc_auc"] for n in names], "s-", label="ROC-AUC")
    ax.plot(x, [results[n]["recall"] for n in names], "^-", label="recall (fakes caught)")
    ax.axhline(0.5, color="grey", ls=":", label="chance (0.5)")
    ax.set_xticks(list(x), names, rotation=30, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("metric on CIFAKE test")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
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
    test_items = get_splits(cfg)["test"]
    tf = build_transforms(d["img_size"], d["mean"], d["std"], train=False)

    results = {}
    for name, fn in DEGRADATIONS.items():
        loader = DataLoader(ImageDataset(Path(d["root"]), test_items, tf, degrade=fn),
                            batch_size=256, num_workers=d["num_workers"])
        z, y = predict_logits(model, loader, args.device)
        m = classification_metrics(y, sigmoid(z))
        results[name] = {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
        print(f"{name:12s} acc {m['accuracy']:.4f}  auc {m['roc_auc']:.4f}  recall {m['recall']:.4f}", flush=True)

    plot_robustness(results, Path("reports/figures") / f"{run}_robustness.png",
                    f"{run}: metrics under degradation (test set, n={len(test_items)})")
    with open(Path("reports") / f"robustness_{run}.json", "w") as f:
        json.dump({"run": run, "n": len(test_items), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
