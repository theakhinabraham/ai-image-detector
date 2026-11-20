"""Grad-CAM, written by hand with PyTorch hooks.

Idea (Selvaraju et al., 2017): the last conv layer keeps a coarse spatial map of
what the network found. For each of its K channels (feature maps A_k):
  1. weight alpha_k = average over positions of d(score)/d(A_k)
     ("how much does this channel push the score up?")
  2. heatmap = ReLU( sum_k alpha_k * A_k )
     (ReLU keeps only regions that INCREASE the score)
  3. upsample the small map (8x8 or 4x4) to the image size and scale to [0, 1].

Score: our model outputs one logit z (high = FAKE). To explain a FAKE verdict we
use z; to explain a REAL verdict we use -z. So the heatmap always shows the
regions that most supported the model's own decision.
"""
from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        # Forward hook: runs every time `layer` produces output. We keep the output
        # and attach a tensor hook that will catch its gradient during backward().
        layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:  # false during ordinary no_grad predictions
            output.register_hook(self._save_grad)

    def _save_grad(self, grad: torch.Tensor):
        self.gradients = grad.detach()

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        """x: one preprocessed image, shape (1, 3, H, W).
        Returns (heatmap HxW in [0,1], raw logit)."""
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)  # ensures a graph exists even if weights are frozen
            logit = self.model(x)[0]
            score = logit if logit.item() >= 0 else -logit  # explain the predicted class
            score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # (1, K, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam  # all-zero map stays zero
        return cam.numpy(), float(logit.item())


def heatmap_rgba(cam: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Colour the [0,1] map with the 'jet' colormap (blue = low, red = high) and resize."""
    coloured = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(coloured).resize(size, Image.BILINEAR)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend the coloured heatmap over the image (alpha = heatmap opacity)."""
    heat = heatmap_rgba(cam, image.size)
    return Image.blend(image.convert("RGB"), heat, alpha)


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
