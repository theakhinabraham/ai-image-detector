"""Model definitions: a small CNN trained from scratch and a fine-tuned ResNet18.

Both output ONE number per image: a logit z. sigmoid(z) = P(image is AI-generated).
One logit (instead of two softmax outputs) is enough for a yes/no question and
makes temperature scaling and Grad-CAM simpler to explain.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """conv -> batch norm -> ReLU -> 2x2 max pool (halves height and width).

    - 3x3 conv learns local patterns (edges, textures, generator artefacts).
    - BatchNorm keeps activations at a stable scale, which makes training faster
      and less sensitive to the learning rate.
    - ReLU adds non-linearity; without it stacked convs collapse into one linear map.
    - MaxPool shrinks the map so later layers see a wider area of the image.
    bias=False because BatchNorm's own shift (beta) makes a conv bias redundant.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """3 conv blocks + small classifier head, for 32x32 RGB input.

    Shapes: 3x32x32 -> 32x16x16 -> 64x8x8 -> 128x4x4 -> 2048 -> 128 -> 1
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # randomly zeroes units in training to reduce overfitting
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)  # shape (batch,)

    def gradcam_layer(self) -> nn.Module:
        """Last conv layer (8x8 output map in block 3, before pooling)."""
        return self.features[2][0]


class ResNet18Detector(nn.Module):
    """ImageNet-pretrained ResNet18 with the 1000-class layer replaced by 1 output.

    The max-pool after the first conv is removed. WHY: at 64x64 input the standard
    ResNet shrinks the image 32x, leaving a 2x2 map at the end, which throws away
    spatial detail and makes Grad-CAM useless. Without that pool the final map is
    4x4. The pretrained weights still load because max-pool has no weights.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.resnet18(weights=weights)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

    def gradcam_layer(self) -> nn.Module:
        """Last conv layer of the network (in the final residual block)."""
        return self.net.layer4[-1].conv2


def build_model(cfg: dict, pretrained: bool = True) -> nn.Module:
    """Create the model named in the config. pretrained=False when we will load our own weights."""
    name = cfg["model"]["name"]
    if name == "smallcnn":
        return SmallCNN(dropout=cfg["model"].get("dropout", 0.3))
    if name == "resnet18":
        return ResNet18Detector(pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_device(name: str = "cpu") -> torch.device:
    """'cpu', 'mps' (Apple GPU) or 'cuda' (NVIDIA GPU, e.g. on Kaggle). 'auto' picks the best available."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_checkpoint(path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load a checkpoint saved by train.py. Returns (model in eval mode, config).

    weights_only=True: torch.load will only unpickle tensors and plain Python
    types, so a tampered checkpoint file cannot run arbitrary code.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["config"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["config"]
