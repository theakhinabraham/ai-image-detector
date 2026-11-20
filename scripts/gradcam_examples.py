"""Save a grid of Grad-CAM overlays on CIFAKE test images for each model.

Run:  python -I scripts/gradcam_examples.py   -> reports/figures/gradcam_examples.png
Images are picked at random (fixed seed); the title shows the true label and the
calibrated probability of FAKE.
"""
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import get_splits, load_config
from src.gradcam import overlay
from src.infer import load_image_safe, load_predictors

N_PER_CLASS = 4


def main() -> None:
    preds = load_predictors("configs/app.yaml")
    names = [n for n in ("smallcnn", "resnet18") if n in preds]
    cfg = load_config("configs/smallcnn.yaml")
    test = get_splits(cfg)["test"]
    rng = random.Random(7)
    chosen = [it for lab in (0, 1) for it in rng.sample([t for t in test if t[1] == lab], N_PER_CLASS)]

    fig, axes = plt.subplots(len(names), len(chosen), figsize=(2 * len(chosen), 2.4 * len(names)))
    for r, name in enumerate(names):
        p = preds[name]
        for c, (rel, label) in enumerate(chosen):
            data = (Path(cfg["data"]["root"]) / rel).read_bytes()
            img, _ = load_image_safe(data)
            cam, logit = p.gradcam(p.transform(img).unsqueeze(0))
            prob = 1 / (1 + math.exp(-logit / p.temperature))  # calibrated, as in the app
            ax = axes[r, c]
            ax.imshow(overlay(img.resize((128, 128), resample=0), cam, alpha=0.45))
            ax.set_title(f"{'REAL' if label == 0 else 'FAKE'} | p={prob:.2f}", fontsize=8)
            ax.axis("off")
        axes[r, 0].text(-0.15, 0.5, name, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=10)
    fig.suptitle("Grad-CAM on CIFAKE test images (true label | calibrated P(fake)); red = most influential")
    fig.tight_layout()
    fig.savefig("reports/figures/gradcam_examples.png", dpi=120)
    print("saved reports/figures/gradcam_examples.png")


if __name__ == "__main__":
    main()
