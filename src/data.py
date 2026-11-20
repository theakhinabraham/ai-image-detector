"""Data loading: CIFAKE file listing, train/val split, COCOAI subset, transforms, DataLoaders.

Labels: REAL = 0, FAKE = 1. So the model's positive class (probability output)
means "AI-generated", which is what the app reports.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

CLASS_TO_LABEL = {"REAL": 0, "FAKE": 1}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

Item = tuple[str, int]  # (path relative to a dataset root, label)


def load_config(path: str | Path) -> dict:
    """Read a YAML experiment config. safe_load never executes code from the file."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- CIFAKE

def list_images(root: Path, split: str) -> list[Item]:
    """List every image under root/<split>/{REAL,FAKE}, sorted.

    Sorting matters: the file system returns files in arbitrary order, and the
    random split below is only reproducible if its input order is fixed.
    """
    items: list[Item] = []
    for class_name, label in CLASS_TO_LABEL.items():
        folder = root / split / class_name
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((str(p.relative_to(root)), label))
    return items


def make_splits(root: Path, val_fraction: float, seed: int) -> dict[str, list[Item]]:
    """Split the official train set into train/val; keep the official test set as-is.

    stratify=labels keeps the REAL/FAKE ratio identical in train and val, so
    validation metrics are not skewed by an accidental class imbalance.
    """
    train_all = list_images(root, "train")
    labels = [label for _, label in train_all]
    train, val = train_test_split(
        train_all, test_size=val_fraction, stratify=labels, random_state=seed
    )
    return {"train": train, "val": val, "test": list_images(root, "test")}


def get_splits(cfg: dict) -> dict[str, list[Item]]:
    """Load CIFAKE splits from disk if they exist, otherwise create and save them.

    Saving the split to a file means every later run (training, evaluation,
    calibration) uses exactly the same validation images.
    """
    splits_file = Path(cfg["data"]["splits_file"])
    if splits_file.exists():
        with open(splits_file) as f:
            raw = json.load(f)
        return {k: [(p, int(y)) for p, y in v] for k, v in raw.items()}
    splits = make_splits(Path(cfg["data"]["root"]), cfg["data"]["val_fraction"], cfg["seed"])
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w") as f:
        json.dump(splits, f)
    return splits


# ---------------------------------------------------------------- COCOAI

def load_cocoai(manifest: str | Path, split: str) -> list[tuple[str, int, str]]:
    """Rows of the COCOAI subset manifest (built by scripts/prepare_cocoai.py).

    Returns (path relative to the manifest's folder, label, generator name).
    split is one of: ft_train, ft_val, test.
    """
    with open(manifest, newline="") as f:
        return [(r["path"], int(r["label"]), r["generator"])
                for r in csv.DictReader(f) if r["split"] == split]


# ---------------------------------------------------------------- transforms

def build_transforms(img_size: int, mean: list[float], std: list[float], train: bool):
    """Image -> normalised tensor. Augmentation is applied to training data only.

    Resize (shorter side) + centre crop turns ANY image into img_size x img_size.
    For CIFAKE at 32 this does nothing; it matters for ResNet18 (upscale to 64)
    and for COCOAI / uploaded photos (large images shrunk down).

    WHY augmentation only on train: it creates slightly varied copies so the model
    learns features that survive small changes. Validation and test must stay
    fixed so that their metrics measure the model, not random augmentation.

    - Horizontal flip: a mirrored photo is still real (or still fake).
    - Light crop (pad 1/16 of the size, then crop back): small shifts in position.
      Kept light on purpose, since heavy crops/resizes could destroy the
      fine pixel-level artefacts that reveal AI generation.
    """
    steps = [
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
    ]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(img_size, padding=max(1, img_size // 16), padding_mode="reflect"),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


class ImageDataset(Dataset):
    """Reads images lazily from disk (only the current batch is in memory).

    `degrade` is an optional function PIL->PIL applied BEFORE the transforms;
    robustness.py uses it to simulate JPEG recompression, blur, etc.
    """

    def __init__(self, root: Path, items: list[Item], transform, degrade=None):
        self.root = Path(root)
        self.items = items
        self.transform = transform
        self.degrade = degrade

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rel_path, label = self.items[idx][:2]
        with Image.open(self.root / rel_path) as img:
            img = img.convert("RGB")  # guarantees 3 channels even for greyscale/RGBA files
        if self.degrade is not None:
            img = self.degrade(img)
        return self.transform(img), label


def _loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      persistent_workers=num_workers > 0)


def make_loaders(cfg: dict) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from a config.

    Normal mode: CIFAKE train/val/test.
    Mix mode (config has data.cocoai_manifest): train/val are a MIX of a random
    CIFAKE sample and the COCOAI fine-tuning split. Mixing in CIFAKE stops the
    model from forgetting what it learnt there ("catastrophic forgetting").
    """
    d = cfg["data"]
    root = Path(d["root"])
    splits = get_splits(cfg)
    tf_train = build_transforms(d["img_size"], d["mean"], d["std"], train=True)
    tf_eval = build_transforms(d["img_size"], d["mean"], d["std"], train=False)
    bs, nw = d["batch_size"], d["num_workers"]

    if "cocoai_manifest" not in d:
        return {
            # shuffle only training data; evaluation order does not matter
            "train": _loader(ImageDataset(root, splits["train"], tf_train), bs, True, nw),
            "val": _loader(ImageDataset(root, splits["val"], tf_eval), bs, False, nw),
            "test": _loader(ImageDataset(root, splits["test"], tf_eval), bs, False, nw),
        }

    rng = random.Random(cfg["seed"])
    coco_root = Path(d["cocoai_manifest"]).parent
    loaders = {}
    for name, cifake_n, coco_split, tf, shuffle in (
        ("train", d["cifake_mix_train"], "ft_train", tf_train, True),
        ("val", d["cifake_mix_val"], "ft_val", tf_eval, False),
    ):
        cifake_items = rng.sample(splits[name], cifake_n)
        coco_items = [(p, y) for p, y, _ in load_cocoai(d["cocoai_manifest"], coco_split)]
        mixed = ConcatDataset([ImageDataset(root, cifake_items, tf),
                               ImageDataset(coco_root, coco_items, tf)])
        loaders[name] = _loader(mixed, bs, shuffle, nw)
    return loaders

