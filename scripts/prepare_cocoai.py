"""Build a small, balanced COCOAI (Defactify) subset from the downloaded parquet files.

Run:  python -I scripts/prepare_cocoai.py --parquet-dir data/raw/cocoai --out data/cocoai

Output:
  data/cocoai/images/*.jpg      the selected images (original bytes, not re-encoded)
  data/cocoai/manifest.csv      path,label,generator,split,width,height
  reports/cocoai_subset.json    counts per split and generator (committed)

Splits (per class: real, and each of the 5 generators):
  test      1000 each  -> cross-generator evaluation (never trained on)
  ft_train   800 real + 160 per generator (800 fake) -> fine-tuning
  ft_val     200 real +  40 per generator (200 fake) -> early stopping while fine-tuning

WHY split by caption: each real COCO photo has 5 AI images generated from the
same caption, so they show the same scene. If the real photo were in ft_train and
its fakes in test, the test would be partly "seen" content. We therefore assign
whole caption groups to either the test side or the fine-tuning side.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

GENERATORS = {0: "real", 1: "SD2.1", 2: "SDXL", 3: "SD3", 4: "DALLE3", 5: "Midjourney6"}
QUOTA = {  # split -> (n real, n per generator)
    "test": (1000, 1000),
    "ft_train": (800, 160),
    "ft_val": (200, 40),
}
Image.MAX_IMAGE_PIXELS = 40_000_000  # refuse absurdly large images (decompression bombs)


def caption_side(caption: str) -> str:
    """Deterministically send a caption group to 'test' (55%) or 'ft' (45%).
    A hash (not random) means the same caption always lands on the same side."""
    key = " ".join(caption.lower().split())
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
    return "test" if h < 55 else "ft"


def is_valid_image(data: bytes) -> tuple[bool, int, int]:
    """Check the bytes really are a decodable image; never trust a downloaded file."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()  # checks file integrity without decoding all pixels
        with Image.open(io.BytesIO(data)) as img:
            return img.format in {"JPEG", "PNG", "WEBP"}, img.width, img.height
    except Exception:
        return False, 0, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Read only the light columns first and bucket row ids by (side, label_b).
    pools: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    files = sorted(Path(args.parquet_dir).glob("*.parquet"))
    for f in files:
        t = pq.read_table(f, columns=["Caption", "Label_B"])
        for i, (cap, lb) in enumerate(zip(t["Caption"].to_pylist(), t["Label_B"].to_pylist())):
            pools[(caption_side(cap), lb)].append((f.name, i))
    print({f"{s}/{GENERATORS[lb]}": len(v) for (s, lb), v in sorted(pools.items())})

    # 2. Randomly choose rows per split; test from the 'test' side, ft_* from the 'ft' side.
    rng = random.Random(args.seed)
    wanted: dict[tuple[str, int], tuple[str, str]] = {}  # (file, row) -> (split, generator)
    for lb, gen in GENERATORS.items():
        test_pool = pools[("test", lb)][:]
        ft_pool = pools[("ft", lb)][:]
        rng.shuffle(test_pool)
        rng.shuffle(ft_pool)
        n_test = QUOTA["test"][0 if lb == 0 else 1]
        n_tr = QUOTA["ft_train"][0 if lb == 0 else 1]
        n_va = QUOTA["ft_val"][0 if lb == 0 else 1]
        if len(test_pool) < n_test or len(ft_pool) < n_tr + n_va:
            raise SystemExit(f"Not enough {gen} images: test {len(test_pool)}, ft {len(ft_pool)}")
        for r in test_pool[:n_test]:
            wanted[r] = ("test", gen)
        for r in ft_pool[:n_tr]:
            wanted[r] = ("ft_train", gen)
        for r in ft_pool[n_tr:n_tr + n_va]:
            wanted[r] = ("ft_val", gen)

    # 3. Read image bytes only for chosen rows, validate, and save the ORIGINAL bytes
    #    (re-encoding would add our own JPEG artefacts on top of the dataset's).
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for f in files:
        table = pq.read_table(f, columns=["Image", "Label_B"])
        for i, (img, lb) in enumerate(zip(table["Image"].to_pylist(), table["Label_B"].to_pylist())):
            if (f.name, i) not in wanted:
                continue
            split, gen = wanted[(f.name, i)]
            ok, w, h = is_valid_image(img["bytes"])
            if not ok:
                skipped += 1
                continue
            rel = f"images/{f.stem}_{i:05d}.jpg"
            (out / rel).write_bytes(img["bytes"])
            rows.append({"path": rel, "label": int(lb != 0), "generator": gen, "split": split,
                         "width": w, "height": h})

    rows.sort(key=lambda r: r["path"])
    with open(out / "manifest.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"source_files": [f.name for f in files], "skipped_invalid": skipped,
               "counts": {s: dict(Counter(r["generator"] for r in rows if r["split"] == s))
                          for s in QUOTA},
               "square_fraction_by_generator": {
                   g: round(sum(r["width"] == r["height"] for r in rows if r["generator"] == g)
                            / max(1, sum(r["generator"] == g for r in rows)), 3)
                   for g in GENERATORS.values()},
               "common_sizes_by_generator": {
                   g: dict(Counter(f"{r['width']}x{r['height']}" for r in rows
                                   if r["generator"] == g).most_common(3))
                   for g in GENERATORS.values()}}
    Path("reports").mkdir(exist_ok=True)
    with open("reports/cocoai_subset.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
