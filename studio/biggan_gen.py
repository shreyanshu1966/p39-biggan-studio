"""BigGAN-deep-128 wrapper: class-conditional generation with the truncation trick.

Truncation trick (U1): the latent z is drawn from a standard normal N(0, I) that is
*truncated* to [-2, 2] and then scaled by `truncation`. A small truncation squeezes
samples towards the mode of the distribution -> higher fidelity, lower diversity.
A value near 1.0 samples the full distribution -> more variety, more artefacts.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import truncnorm

from pytorch_pretrained_biggan import BigGAN, BigGANConfig

_S3 = "https://s3.amazonaws.com/models.huggingface.co/biggan/"
WEIGHTS_DIR = Path(os.environ.get("BIGGAN_DIR", Path(__file__).resolve().parent.parent / "weights" / "biggan-deep-128"))
Z_DIM = 128
NUM_CLASSES = 1000

_model: BigGAN | None = None


def _ensure_weights() -> None:
    """Download the official weights (published by the pytorch-pretrained-biggan authors)."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for local, remote in [("config.json", "biggan-deep-128-config.json"),
                          ("pytorch_model.bin", "biggan-deep-128-pytorch_model.bin")]:
        dst = WEIGHTS_DIR / local
        if not dst.exists():
            tmp = dst.with_suffix(".part")
            urllib.request.urlretrieve(_S3 + remote, tmp)
            tmp.rename(dst)


def get_model() -> BigGAN:
    global _model
    if _model is None:
        _ensure_weights()
        config = BigGANConfig.from_json_file(str(WEIGHTS_DIR / "config.json"))
        model = BigGAN(config)
        # weights_only=True: load tensors only, never execute pickled code.
        state = torch.load(WEIGHTS_DIR / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        _model = model.eval()
    return _model


def truncated_noise(n: int, truncation: float, seed: int | None = None) -> np.ndarray:
    """z ~ TruncNormal(-2, 2) * truncation, shape (n, 128)."""
    rng = np.random.RandomState(seed)
    z = truncnorm.rvs(-2, 2, size=(n, Z_DIM), random_state=rng)
    return (truncation * z).astype(np.float32)


def _to_pil(batch: torch.Tensor) -> list[Image.Image]:
    arr = ((batch.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).byte().cpu().numpy()
    return [Image.fromarray(a) for a in arr]


@torch.no_grad()
def generate(class_idx: int, truncation: float = 0.4, n: int = 4, seed: int | None = None) -> list[Image.Image]:
    """Generate `n` 128x128 images of ImageNet class `class_idx`."""
    if not 0 <= class_idx < NUM_CLASSES:
        raise ValueError(f"class_idx must be in [0, {NUM_CLASSES}), got {class_idx}")
    truncation = float(min(max(truncation, 0.02), 1.0))
    model = get_model()
    z = torch.from_numpy(truncated_noise(n, truncation, seed))
    y = torch.zeros(n, NUM_CLASSES)
    y[:, class_idx] = 1.0
    return _to_pil(model(z, y, truncation))
