"""U4 deliverable: 5 prompts x 3 truncation values grid + truncation trade-off table.

Each cell = the CLIP-best of 4 samples (fixed seed per prompt, so only truncation varies).
Table = mean CLIP score and CLIP-diversity over 8 samples per (prompt, truncation).

Run:  python eval/make_grid.py [--method llm_fewshot]
Out:  results/grid_5x3.png, results/truncation_table.md, results/grid_classes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from studio import mapper  # noqa: E402
from studio.biggan_gen import generate  # noqa: E402
from studio.clip_rank import clip_scores, diversity, rank  # noqa: E402

PROMPTS = [
    "a friendly family dog with golden fur",
    "a mountain erupting with lava",
    "a slice of italian food with cheese and tomato",
    "an old wooden ship with pirate flags",
    "an orange and black butterfly that migrates",
]
TRUNCS = [0.2, 0.5, 1.0]
CELL, LABEL_W, HEAD_H = 160, 230, 26

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="llm_fewshot")
args = ap.parse_args()
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

grid = Image.new("RGB", (LABEL_W + CELL * len(TRUNCS), HEAD_H + CELL * len(PROMPTS)), "white")
draw = ImageDraw.Draw(grid)
for c, t in enumerate(TRUNCS):
    draw.text((LABEL_W + c * CELL + 50, 7), f"truncation {t}", fill="black")

rows, classes = [], {}
for r, prompt in enumerate(PROMPTS):
    res = mapper.map_text(prompt, args.method)
    classes[prompt] = {"idx": res.idx, "label": res.label, "method": res.method}
    y = HEAD_H + r * CELL
    words, line, lines = prompt.split(), "", []
    for w in words:
        if len(line) + len(w) > 30:
            lines.append(line)
            line = ""
        line += w + " "
    lines.append(line)
    lines.append(f"-> {res.label} (#{res.idx})")
    for k, ln in enumerate(lines):
        draw.text((6, y + 40 + 16 * k), ln, fill="black")
    for c, t in enumerate(TRUNCS):
        best_img, best_score = rank(generate(res.idx, t, 4, seed=100 + r), prompt)[0]
        grid.paste(best_img.resize((CELL - 4, CELL - 4), Image.LANCZOS), (LABEL_W + c * CELL + 2, y + 2))
        draw.text((LABEL_W + c * CELL + 6, y + CELL - 18), f"CLIP {best_score:.1f}", fill="white")
        samples = generate(res.idx, t, 8, seed=200 + r)
        sc = clip_scores(samples, prompt)
        rows.append({"prompt": prompt, "class": res.label, "truncation": t,
                     "mean_clip": sum(sc) / len(sc), "diversity": diversity(samples)})
    print(f"done: {prompt} -> {res.label}")

grid.save(OUT / "grid_5x3.png")
(OUT / "grid_classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

md = ["# Truncation trade-off (8 samples per cell)\n",
      "| Prompt | Class | " + " | ".join(f"CLIP @ {t}" for t in TRUNCS) + " | " + " | ".join(f"Div @ {t}" for t in TRUNCS) + " |",
      "|---|---|" + "---|" * (2 * len(TRUNCS))]
for p in PROMPTS:
    rr = [x for x in rows if x["prompt"] == p]
    md.append(f"| {p} | {rr[0]['class']} | " + " | ".join(f"{x['mean_clip']:.1f}" for x in rr) + " | "
              + " | ".join(f"{x['diversity']:.3f}" for x in rr) + " |")
avg = lambda key, t: sum(x[key] for x in rows if x["truncation"] == t) / len(PROMPTS)  # noqa: E731
md.append("| **Mean** | | " + " | ".join(f"**{avg('mean_clip', t):.1f}**" for t in TRUNCS) + " | "
          + " | ".join(f"**{avg('diversity', t):.3f}**" for t in TRUNCS) + " |")
md.append("\nDiversity = mean pairwise (1 − cosine) of CLIP image embeddings. Higher truncation → more diverse samples.")
(OUT / "truncation_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("wrote results/grid_5x3.png and results/truncation_table.md")
