"""P39 - Class-Conditional Image Studio (BigGAN).

free text -> guardrail -> LLM picks nearest ImageNet class -> BigGAN generates
-> CLIP ranks the samples against the prompt.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from studio import mapper
from studio.biggan_gen import generate, get_model
from studio.clip_rank import diversity, rank
from studio.guardrail import check_class, check_prompt

ROOT = Path(__file__).parent
METHOD_LABELS = {
    "LLM few-shot + query expansion (default)": "llm_fewshot",
    "LLM few-shot v1 (retrieval only)": "llm_fewshot_v1",
    "LLM zero-shot": "llm_zero",
    "Embedding only (no LLM)": "embedding",
}


def _seed(seed) -> int:
    s = int(seed or 0)
    return s if s > 0 else random.randint(1, 2**31 - 1)


def _class_choice_md(res: mapper.MapResult) -> str:
    lines = [f"### Chosen class: **{res.label}** (ImageNet #{res.idx})",
             f"- Method: `{res.method}`" + (f" (fallback: {res.fallback})" if res.fallback else ""),
             f"- Reason: {res.reason or '-'}",
             f"- Encoder similarity: {res.similarity:.2f} (closest label overall: {res.best_similarity:.2f})" + ("  ⚠️ *low: no close ImageNet class, output may not match your text*" if res.low_confidence else ""),
             f"- Mapping latency: {res.latency_ms:.0f} ms"]
    if res.hallucinated:
        lines.append("- ⚠️ LLM output was invalid (hallucinated class) and was corrected.")
    lines.append("\n**Top-5 encoder candidates:** " + ", ".join(f"{n} ({s:.2f})" for _, n, s in res.candidates))
    return "\n".join(lines)


def studio(prompt, method_label, truncation, n, seed):
    ok, msg = check_prompt(prompt)
    if not ok:
        return [], f"### 🚫 {msg}"
    res = mapper.map_text(prompt, METHOD_LABELS[method_label])
    ok, note = check_class(prompt, res.idx)
    if not ok:
        return [], f"### 🚫 {note}"
    s = _seed(seed)
    t0 = time.perf_counter()
    imgs = generate(res.idx, truncation, int(n), seed=s)
    gen_ms = (time.perf_counter() - t0) * 1000
    ranked = rank(imgs, prompt)
    gallery = [(im.resize((256, 256), Image.LANCZOS), f"#{k + 1} · CLIP {sc:.1f}") for k, (im, sc) in enumerate(ranked)]
    md = _class_choice_md(res)
    md += (f"\n\n**Generation:** {len(imgs)} images · truncation {truncation:.2f} · seed {s} · {gen_ms:.0f} ms"
           f"\n\n**CLIP scores:** best {ranked[0][1]:.1f}, mean {sum(x for _, x in ranked) / len(ranked):.1f} · "
           f"diversity {diversity(imgs):.3f}")
    if note:
        md += f"\n\n> {note}"
    return gallery, md


def _label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 18), "white")
    out.paste(img, (0, 18))
    ImageDraw.Draw(out).text((4, 3), text, fill="black")
    return out


def truncation_explorer(prompt, method_label, seed):
    ok, msg = check_prompt(prompt)
    if not ok:
        return None, f"### 🚫 {msg}"
    res = mapper.map_text(prompt, METHOD_LABELS[method_label])
    ok, note = check_class(prompt, res.idx)
    if not ok:
        return None, f"### 🚫 {note}"
    s = _seed(seed)
    values = [0.1, 0.3, 0.5, 0.7, 1.0]
    rows, table = [], ["| truncation | mean CLIP | diversity |", "|---|---|---|"]
    for t in values:
        imgs = generate(res.idx, t, 4, seed=s)
        ranked = rank(imgs, prompt)
        table.append(f"| {t} | {sum(x for _, x in ranked) / 4:.1f} | {diversity(imgs):.3f} |")
        rows.append([_label(im, f"t={t}") for im in imgs])
    w, h = rows[0][0].size
    grid = Image.new("RGB", (w * 4, h * len(rows)), "white")
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            grid.paste(im, (c * w, r * h))
    md = (f"**Class:** {res.label} (#{res.idx}) · same seed {s} in every row, so only the truncation changes.\n\n"
          + "\n".join(table)
          + "\n\nLow truncation → samples near the mode (clean but similar). High truncation → full distribution (varied, more artefacts).")
    return grid, md


ABOUT = (ROOT / "docs" / "about.md").read_text(encoding="utf-8") if (ROOT / "docs" / "about.md").exists() else ""
EXAMPLES = [
    ["a friendly family dog with golden fur", 0.4],
    ["a mountain erupting with lava", 0.5],
    ["an instrument with black and white keys", 0.4],
    ["a slice of italian food with cheese and tomato", 0.3],
    ["a pirate ship with big sails", 0.6],
]

with gr.Blocks(title="P39 · BigGAN Class-Conditional Studio") as demo:
    llm_status = (f"LLM: `{mapper.llm_config()['model']}`" if mapper.llm_available()
                  else "LLM: **not configured**, using the embedding mapper (set `GROQ_API_KEY` or `HF_TOKEN` in Space secrets)")
    gr.Markdown("# 🎨 Class-Conditional Image Studio (BigGAN)\n"
                "Type anything. An LLM maps it to the nearest of the 1000 ImageNet classes, "
                "**BigGAN-deep-128** generates samples, and **CLIP** ranks them against your text.\n\n" + llm_status)
    with gr.Tab("Studio"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Describe an image", placeholder="e.g. a slow reptile with a shell", max_lines=2)
                method = gr.Dropdown(list(METHOD_LABELS), value=list(METHOD_LABELS)[0], label="Text → class method")
                trunc = gr.Slider(0.02, 1.0, value=0.4, step=0.02, label="Truncation (low = faithful, high = diverse)")
                n = gr.Slider(1, 8, value=4, step=1, label="Number of samples")
                seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")
                go = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Ranked by CLIP score", columns=4, height=560)
                info = gr.Markdown()
        gr.Examples(EXAMPLES, [prompt, trunc])
        go.click(studio, [prompt, method, trunc, n, seed], [gallery, info])
        prompt.submit(studio, [prompt, method, trunc, n, seed], [gallery, info])
    with gr.Tab("Truncation explorer"):
        p2 = gr.Textbox(label="Describe an image", value="a friendly family dog with golden fur")
        with gr.Row():
            m2 = gr.Dropdown(list(METHOD_LABELS), value=list(METHOD_LABELS)[0], label="Text → class method")
            s2 = gr.Number(value=7, precision=0, label="Seed")
        go2 = gr.Button("Build truncation grid", variant="primary")
        grid = gr.Image(label="Rows: truncation 0.1 → 1.0", type="pil")
        info2 = gr.Markdown()
        go2.click(truncation_explorer, [p2, m2, s2], [grid, info2])
    with gr.Tab("About · Ethics · Results"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    get_model()  # warm up
    demo.queue(max_size=16).launch()
