"""CLIP ranking (U3): score each generated image against the user's prompt.

CLIP score = 100 * cosine(image_embedding, text_embedding), the usual "CLIPScore" scale.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

CLIP_MODEL = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")


@lru_cache(maxsize=1)
def _clip():
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return model, proc


@torch.no_grad()
def image_embeddings(images: list[Image.Image]) -> torch.Tensor:
    model, proc = _clip()
    inputs = proc(images=images, return_tensors="pt")
    feats = model.get_image_features(**inputs)
    if not isinstance(feats, torch.Tensor):  # newer transformers may return a model output
        feats = feats.pooler_output
    return feats / feats.norm(dim=-1, keepdim=True)


@torch.no_grad()
def text_embedding(text: str) -> torch.Tensor:
    model, proc = _clip()
    inputs = proc(text=[text], return_tensors="pt", padding=True, truncation=True)
    feats = model.get_text_features(**inputs)
    if not isinstance(feats, torch.Tensor):
        feats = feats.pooler_output
    return (feats / feats.norm(dim=-1, keepdim=True))[0]


def clip_scores(images: list[Image.Image], text: str) -> list[float]:
    if not images:
        return []
    sims = image_embeddings(images) @ text_embedding(text)
    return [float(100 * s) for s in sims]


def rank(images: list[Image.Image], text: str) -> list[tuple[Image.Image, float]]:
    scores = clip_scores(images, text)
    return sorted(zip(images, scores), key=lambda p: -p[1])


def diversity(images: list[Image.Image]) -> float:
    """Mean pairwise (1 - cosine) between CLIP image embeddings: higher = more varied."""
    if len(images) < 2:
        return 0.0
    e = image_embeddings(images)
    sim = (e @ e.T).numpy()
    n = len(images)
    off = sim[~np.eye(n, dtype=bool)]
    return float(1 - off.mean())
