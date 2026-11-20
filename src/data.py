"""Data loading for CIFAKE: file listing, train/val split, transforms, DataLoaders.

Labels: REAL = 0, FAKE = 1. So the model's positive class (probability output)
means "AI-generated", which is what the app reports.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_TO_LABEL = {"REAL": 0, "FAKE": 1}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

Item = tuple[str, int]  # (path relative to data root, label)


def load_config(path: str | Path) -> dict:
    """Read a YAML experiment config. safe_load never executes code from the file."""
    with open(path) as f:
        return yaml.safe_load(f)


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
    """Load splits from disk if they exist, otherwise create and save them.

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


def build_transforms(img_size: int, mean: list[float], std: list[float], train: bool):
    """Image -> normalised tensor. Augmentation is applied to training data only.

    WHY only on train: augmentation creates slightly varied copies so the model
    learns features that survive small changes. Validation and test must stay
    fixed so that their metrics measure the model, not random augmentation.

    - Horizontal flip: a mirrored photo is still real (or still fake).
    - Light crop (pad 1/16 of the size, then crop back): small shifts in position.
      Kept light on purpose, since heavy crops/resizes could destroy the
      fine pixel-level artefacts that reveal AI generation.
    """
    steps = []
    if img_size != 32:  # CIFAKE is 32x32; bigger models (ResNet18) need upscaling
        steps.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR))
    if train:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(img_size, padding=max(1, img_size // 16), padding_mode="reflect"),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


class CifakeDataset(Dataset):
    """Reads images lazily from disk (only the current batch is in memory)."""

    def __init__(self, root: Path, items: list[Item], transform):
        self.root = Path(root)
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rel_path, label = self.items[idx]
        with Image.open(self.root / rel_path) as img:
            img = img.convert("RGB")  # guarantees 3 channels even for greyscale files
        return self.transform(img), label


def make_loaders(cfg: dict) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from a config."""
    d = cfg["data"]
    root = Path(d["root"])
    splits = get_splits(cfg)
    loaders = {}
    for name, items in splits.items():
        is_train = name == "train"
        tf = build_transforms(d["img_size"], d["mean"], d["std"], train=is_train)
        loaders[name] = DataLoader(
            CifakeDataset(root, items, tf),
            batch_size=d["batch_size"],
            shuffle=is_train,  # shuffle only training data; eval order does not matter
            num_workers=d["num_workers"],
        )
    return loaders
