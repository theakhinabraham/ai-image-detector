"""Estimate the JPEG quality of dataset files, per class / generator.

WHY: if real and fake images had different JPEG quality, a model could learn
"compression level" instead of "AI artefacts" (a shortcut). This checks for that.

Run:  python -I scripts/jpeg_quality_check.py   -> reports/jpeg_quality.json
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.infer import estimate_jpeg_quality

SAMPLE = 2000


def estimate(path: Path):
    with Image.open(path) as im:
        return estimate_jpeg_quality(im) if im.format == "JPEG" else im.format


def main() -> None:
    out = {"sample_per_dataset": SAMPLE, "cifake_test": {}, "cocoai_test": {}}
    items = json.load(open("data/splits.json"))["test"]
    random.Random(0).shuffle(items)
    counts = defaultdict(Counter)
    for rel, y in items[:SAMPLE]:
        counts["REAL" if y == 0 else "FAKE"][estimate(Path("data/raw/cifake") / rel)] += 1
    out["cifake_test"] = {k: dict(v.most_common(3)) for k, v in counts.items()}

    rows = [r for r in csv.DictReader(open("data/cocoai/manifest.csv")) if r["split"] == "test"]
    random.Random(0).shuffle(rows)
    counts = defaultdict(Counter)
    for r in rows[:SAMPLE]:
        counts[r["generator"]][estimate(Path("data/cocoai") / r["path"])] += 1
    out["cocoai_test"] = {k: dict(v.most_common(3)) for k, v in counts.items()}
    Path("reports/jpeg_quality.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
