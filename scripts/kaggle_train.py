"""Train + evaluate ResNet18 inside a Kaggle notebook with the free GPU.

How to run (see README "Training on Kaggle" for the click-by-click version):
  1. Zip this repo WITHOUT data/, .venv/ and checkpoints/ and upload it to Kaggle
     as a private Dataset named `ai-image-detector-code`.
  2. New Notebook -> Settings -> Accelerator: GPU T4 x2 (or P100); Internet: On.
  3. "Add Input": the CIFAKE dataset (birdy654/cifake-real-and-ai-generated-synthetic-images)
     and your `ai-image-detector-code` dataset.
  4. In a cell:   !python /kaggle/input/ai-image-detector-code/scripts/kaggle_train.py
  5. Download /kaggle/working/kaggle_outputs.zip from the Output panel and unzip
     it into the repo root (it contains checkpoints/ and reports/).

Kaggle's own packages (torch, sklearn, matplotlib, yaml) are used as installed there.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]   # /kaggle/input/ai-image-detector-code
WORK = Path("/kaggle/working")


def find_cifake_root() -> Path:
    """Kaggle mounts datasets under /kaggle/input/<name>/... ; locate the folder holding train/REAL."""
    for p in Path("/kaggle/input").rglob("REAL"):
        if p.parent.name == "train":
            return p.parent.parent
    raise FileNotFoundError("CIFAKE not found under /kaggle/input. Did you add it as an input?")


def main() -> None:
    cfg = yaml.safe_load(open(REPO / "configs" / "resnet18.yaml"))
    cfg["data"]["root"] = str(find_cifake_root())
    cfg["data"]["splits_file"] = str(WORK / "data" / "splits.json")  # same seed -> same split as local
    cfg["data"]["num_workers"] = 4
    cfg["train"]["device"] = "cuda"
    cfg_path = WORK / "resnet18_kaggle.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"))

    # /kaggle/input is read-only, so run from /kaggle/working and put the repo on PYTHONPATH.
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    run = lambda *args: subprocess.run([sys.executable, *args], cwd=WORK, env=env, check=True)
    run("-m", "src.train", "--config", str(cfg_path))
    run("-m", "src.evaluate", "--checkpoint", "checkpoints/resnet18.pt", "--device", "cuda")

    shutil.make_archive(str(WORK / "kaggle_outputs"), "zip", WORK, ".")
    print("Done. Download /kaggle/working/kaggle_outputs.zip")


if __name__ == "__main__":
    main()
