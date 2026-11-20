"""Tests for src/data.py. Uses a tiny fake dataset built in a temp folder,
so the tests run without downloading CIFAKE."""
from pathlib import Path

import torch
from PIL import Image

from src.data import CifakeDataset, build_transforms, make_splits


def _make_fake_cifake(root: Path, n_per_class: int = 20) -> None:
    for split in ("train", "test"):
        for cls, colour in (("REAL", (200, 50, 50)), ("FAKE", (50, 50, 200))):
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(n_per_class):
                Image.new("RGB", (32, 32), colour).save(folder / f"{i}.jpg")


def test_split_sizes_stratified_and_disjoint(tmp_path):
    _make_fake_cifake(tmp_path)
    splits = make_splits(tmp_path, val_fraction=0.1, seed=42)
    assert len(splits["train"]) == 36 and len(splits["val"]) == 4 and len(splits["test"]) == 40
    # stratified: val keeps the 50/50 class ratio
    assert sum(y for _, y in splits["val"]) == 2
    # no leakage between splits
    assert not {p for p, _ in splits["train"]} & {p for p, _ in splits["val"]}


def test_split_is_reproducible(tmp_path):
    _make_fake_cifake(tmp_path)
    assert make_splits(tmp_path, 0.1, seed=42) == make_splits(tmp_path, 0.1, seed=42)


def test_transform_shape_and_dtype(tmp_path):
    _make_fake_cifake(tmp_path, n_per_class=2)
    items = [("train/REAL/0.jpg", 0)]
    for size in (32, 64):
        for train in (True, False):
            tf = build_transforms(size, [0.5] * 3, [0.5] * 3, train=train)
            x, y = CifakeDataset(tmp_path, items, tf)[0]
            assert x.shape == (3, size, size) and x.dtype == torch.float32 and y == 0
            assert x.min() >= -1.0 and x.max() <= 1.0  # normalised range
