"""Phase 1 sanity check: class balance, split sizes, image sizes, and a sample grid.

Run:  python -I scripts/inspect_data.py --config configs/smallcnn.yaml
Writes reports/data_summary.json and reports/figures/sample_grid.png.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
from PIL import Image

from src.data import LABEL_TO_CLASS, get_splits, load_config


def image_size_counts(root: Path, items) -> dict[str, int]:
    """Count image sizes. Image.open reads only the header, so this is fast."""
    sizes: Counter = Counter()
    for rel, _ in items:
        with Image.open(root / rel) as img:
            sizes[f"{img.width}x{img.height} {img.mode}"] += 1
    return dict(sizes)


def save_sample_grid(root: Path, items, out_path: Path, per_class: int, seed: int) -> None:
    """Top row(s): REAL, bottom row(s): FAKE, drawn at random from the train split."""
    rng = random.Random(seed)
    cols = 8
    rows_per_class = per_class // cols
    fig, axes = plt.subplots(2 * rows_per_class, cols, figsize=(cols * 1.4, 2 * rows_per_class * 1.6))
    for label in (0, 1):
        chosen = rng.sample([it for it in items if it[1] == label], per_class)
        for i, (rel, _) in enumerate(chosen):
            ax = axes[label * rows_per_class + i // cols, i % cols]
            with Image.open(root / rel) as img:
                # nearest-neighbour keeps the real 32x32 pixels visible instead of blurring them
                ax.imshow(img.convert("RGB"), interpolation="nearest")
            ax.set_title(LABEL_TO_CLASS[label], fontsize=8, color="tab:blue" if label == 0 else "tab:red")
            ax.axis("off")
    fig.suptitle("CIFAKE train samples (32x32): REAL (top) vs FAKE (bottom)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["data"]["root"])
    splits = get_splits(cfg)

    summary = {"seed": cfg["seed"], "val_fraction": cfg["data"]["val_fraction"], "splits": {}}
    for name, items in splits.items():
        counts = Counter(LABEL_TO_CLASS[y] for _, y in items)
        summary["splits"][name] = {
            "total": len(items),
            "REAL": counts["REAL"],
            "FAKE": counts["FAKE"],
            "image_sizes": image_size_counts(root, items),
        }
    # Leakage check: no file may appear in more than one split.
    sets = {k: {p for p, _ in v} for k, v in splits.items()}
    summary["overlap"] = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }

    Path("reports").mkdir(exist_ok=True)
    with open("reports/data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    save_sample_grid(root, splits["train"], Path("reports/figures/sample_grid.png"), per_class=16, seed=cfg["seed"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
