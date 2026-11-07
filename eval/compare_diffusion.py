"""U3: compare BigGAN (class-conditional GAN) with a text-conditional diffusion model.

For each prompt: BigGAN best-of-4 (CLIP-ranked) vs sd-turbo (1-step diffusion, 512px),
both scored with CLIP against the *original free text*. Also reports wall-clock time.

Backends for sd-turbo:
  --backend local  : diffusers on CPU (downloads stabilityai/sd-turbo, ~2.5 GB, a few s/image)
  --backend api    : Hugging Face Inference API (needs HF_TOKEN)

Run:  python eval/compare_diffusion.py --backend local
Out:  results/diffusion_vs_biggan.md, results/diffusion_vs_biggan.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from studio import mapper  # noqa: E402
from studio.biggan_gen import generate  # noqa: E402
from studio.clip_rank import clip_scores, rank  # noqa: E402

PROMPTS = [
    "a friendly family dog with golden fur",
    "a mountain erupting with lava",
    "a slice of italian food with cheese and tomato",
    "an orange and black butterfly that migrates",
    "a cat riding a bicycle",  # out-of-vocabulary for ImageNet: diffusion should win clearly
]
SD_MODEL = "stabilityai/sd-turbo"

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["local", "api"], default="local")
ap.add_argument("--method", default="llm_fewshot")
args = ap.parse_args()
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

if args.backend == "local":
    import torch
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(SD_MODEL, torch_dtype=torch.float32, use_safetensors=True)
    pipe.set_progress_bar_config(disable=True)

    def diffuse(p: str, seed: int) -> Image.Image:
        g = torch.Generator().manual_seed(seed)
        return pipe(p, num_inference_steps=1, guidance_scale=0.0, generator=g, height=512, width=512).images[0]
else:
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=os.environ["HF_TOKEN"])

    def diffuse(p: str, seed: int) -> Image.Image:
        return client.text_to_image(p, model=os.environ.get("SD_API_MODEL", SD_MODEL))

generate(0, 0.4, 1, seed=0)  # warm-up: keep model loading out of the timing
rows, pairs = [], []
for k, p in enumerate(PROMPTS):
    res = mapper.map_text(p, args.method)
    t0 = time.perf_counter()
    g_img, g_score = rank(generate(res.idx, 0.4, 4, seed=300 + k), p)[0]
    g_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    d_img = diffuse(p, 300 + k)
    d_t = time.perf_counter() - t0
    d_score = clip_scores([d_img], p)[0]
    rows.append((p, res.label, g_score, d_score, g_t, d_t))
    pairs.append((p, res.label, g_img, d_img, g_score, d_score))
    print(f"{p}: BigGAN[{res.label}] {g_score:.1f} ({g_t:.1f}s) vs sd-turbo {d_score:.1f} ({d_t:.1f}s)")

S, LW = 200, 260
sheet = Image.new("RGB", (LW + 2 * S, 24 + S * len(pairs)), "white")
dr = ImageDraw.Draw(sheet)
dr.text((LW + 40, 6), "BigGAN-deep-128 (best of 4)", fill="black")
dr.text((LW + S + 60, 6), "sd-turbo (1 step)", fill="black")
for r, (p, lab, gi, di, gs, ds) in enumerate(pairs):
    y = 24 + r * S
    dr.text((6, y + 80), p[:38], fill="black")
    dr.text((6, y + 98), f"BigGAN class: {lab}", fill="gray")
    sheet.paste(gi.resize((S - 4, S - 4), Image.LANCZOS), (LW + 2, y + 2))
    sheet.paste(di.resize((S - 4, S - 4), Image.LANCZOS), (LW + S + 2, y + 2))
    dr.text((LW + 8, y + S - 18), f"CLIP {gs:.1f}", fill="white")
    dr.text((LW + S + 8, y + S - 18), f"CLIP {ds:.1f}", fill="white")
sheet.save(OUT / "diffusion_vs_biggan.png")

md = ["# BigGAN vs diffusion (sd-turbo), CLIP score against the free-text prompt\n",
      f"sd-turbo backend: {args.backend} · BigGAN truncation 0.4, best of 4 by CLIP\n",
      "| Prompt | BigGAN class | BigGAN CLIP | sd-turbo CLIP | BigGAN time | sd-turbo time |", "|---|---|---|---|---|---|"]
md += [f"| {p} | {lab} | {gs:.1f} | {ds:.1f} | {gt:.1f} s | {dt:.1f} s |" for p, lab, gs, ds, gt, dt in rows]
mg = sum(r[2] for r in rows) / len(rows)
md_ = sum(r[3] for r in rows) / len(rows)
md.append(f"| **Mean** | | **{mg:.1f}** | **{md_:.1f}** | | |")
md.append("\nBigGAN can only draw the 1000 ImageNet classes at 128 px, so attributes and compositions in the text "
          "(\"riding a bicycle\", \"orange and black\") are lost. Diffusion follows the full text but costs more compute.")
(OUT / "diffusion_vs_biggan.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("wrote results/diffusion_vs_biggan.md")
