"""Draws docs/lifecycle.png: the Generative-AI lifecycle of this project (U1)."""
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

_FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
F = ImageFont.truetype(str(_FD / "DejaVuSans.ttf"), 14)
FB = ImageFont.truetype(str(_FD / "DejaVuSans-Bold.ttf"), 16)

STAGES = [
    ("1. Problem", "free text -> image;\nclass-conditional GAN"),
    ("2. Data", "ImageNet-1k (1.28M imgs,\n1000 classes) - BigGAN\npretrained on it"),
    ("3. Model", "BigGAN-deep-128 (G)\nLLM + MiniLM mapper\nCLIP ranker"),
    ("4. Adapt", "prompt-based learning:\nzero-shot vs few-shot\n+ retrieval (no fine-tune)"),
    ("5. Evaluate", "mapping accuracy,\nhallucination rate,\nCLIP score, diversity"),
    ("6. Deploy", "Gradio on HF Spaces\n(CPU), secrets for keys"),
    ("7. Monitor/Govern", "guardrail, misuse note,\nmodel card, licences"),
]
W, H, BW, BH, GAP = 1790, 360, 222, 160, 20
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
d.text((20, 15), "P39 Generative-AI lifecycle: text -> ImageNet class -> BigGAN -> CLIP rank", fill="black", font=FB)
x0, y0 = 20, 70
for i, (title, body) in enumerate(STAGES):
    x = x0 + i * (BW + GAP + 10)
    d.rounded_rectangle([x, y0, x + BW, y0 + BH], radius=12, outline="#2b5797", width=3, fill="#eef3fb")
    d.text((x + 10, y0 + 10), title, fill="#2b5797", font=FB)
    d.multiline_text((x + 10, y0 + 44), body, fill="black", spacing=6, font=F)
    if i < len(STAGES) - 1:
        ax = x + BW + 2
        d.line([ax, y0 + BH // 2, ax + GAP + 6, y0 + BH // 2], fill="black", width=3)
        d.polygon([(ax + GAP + 8, y0 + BH // 2), (ax + GAP, y0 + BH // 2 - 6), (ax + GAP, y0 + BH // 2 + 6)], fill="black")
d.line([x0 + 6 * (BW + GAP + 10) + BW // 2, y0 + BH, x0 + 6 * (BW + GAP + 10) + BW // 2, y0 + BH + 50,
        x0 + BW // 2, y0 + BH + 50, x0 + BW // 2, y0 + BH + 5], fill="gray", width=2)
d.text((W // 2 - 230, y0 + BH + 56), "feedback: failure cases -> new prompts / guardrail rules", fill="gray", font=F)
img.save(Path(__file__).parent / "lifecycle.png")
print("saved docs/lifecycle.png")
