"""P39 - Class-Conditional Image Studio (BigGAN), Streamlit version (deployed on Streamlit Community Cloud).

Same pipeline as app.py (Gradio): free text -> guardrail -> LLM picks nearest ImageNet class
-> BigGAN-deep-128 generates -> CLIP ranks the samples against the prompt.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import streamlit as st

# Streamlit secrets -> env vars (mapper reads GROQ_API_KEY / HF_TOKEN / LLM_* from the environment)
try:
    for _k in ("GROQ_API_KEY", "HF_TOKEN", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # no secrets file locally
    pass

from PIL import Image, ImageDraw  # noqa: E402

from studio import mapper  # noqa: E402
from studio.biggan_gen import generate, get_model  # noqa: E402
from studio.clip_rank import diversity, rank  # noqa: E402
from studio.guardrail import check_class, check_prompt  # noqa: E402

ROOT = Path(__file__).parent
METHOD_LABELS = {
    "LLM few-shot + query expansion (default)": "llm_fewshot",
    "LLM few-shot v1 (retrieval only)": "llm_fewshot_v1",
    "LLM zero-shot": "llm_zero",
    "Embedding only (no LLM)": "embedding",
}
EXAMPLES = [
    "the king of the jungle roaring",
    "a mountain erupting with lava",
    "an instrument with black and white keys",
    "a slice of italian food with cheese and tomato",
    "an orange and black butterfly that migrates",
    "a cat riding a bicycle",
]

st.set_page_config(page_title="P39 BigGAN Studio", page_icon="🎨", layout="wide")


@st.cache_resource(show_spinner="Loading BigGAN, CLIP and the text encoder (first start only)…")
def warm_up() -> bool:
    get_model()
    mapper.top_k("warm up", 1)
    rank(generate(0, 0.4, 1, seed=0), "warm up")
    return True


def class_info(res: mapper.MapResult) -> None:
    st.markdown(f"### Chosen class: **{res.label}** (ImageNet #{res.idx})")
    cols = st.columns(4)
    cols[0].metric("Method", res.method)
    cols[1].metric("Similarity to text", f"{res.similarity:.2f}")
    cols[2].metric("Closest label overall", f"{res.best_similarity:.2f}")
    cols[3].metric("Mapping latency", f"{res.latency_ms:.0f} ms")
    if res.reason:
        st.caption(f"Reason: {res.reason}")
    if res.fallback:
        st.caption(f"Fallback: {res.fallback}")
    if res.hallucinated:
        st.warning("The LLM's answer was not a valid ImageNet class (hallucination); it was corrected.")
    if res.low_confidence:
        st.warning("Low confidence: no ImageNet class is close to your text, so the output may not match it.")
    st.caption("Top candidates: " + ", ".join(f"{n} ({s:.2f})" for _, n, s in res.candidates))


def labelled(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 18), "white")
    out.paste(img, (0, 18))
    ImageDraw.Draw(out).text((4, 3), text, fill="black")
    return out


# ------------------------------------------------------------------ header + sidebar
st.title("🎨 Class-Conditional Image Studio (BigGAN)")
st.markdown("Type anything. An LLM maps it to the nearest of the 1000 ImageNet classes, "
            "**BigGAN-deep-128** generates samples, and **CLIP** ranks them against your text.")
if mapper.llm_available():
    st.caption(f"LLM: `{mapper.llm_config()['model']}` · encoder: `{mapper.EMBED_MODEL}` · ranker: `openai/clip-vit-base-patch32`")
else:
    st.info("LLM not configured, so the embedding mapper is used. Add `GROQ_API_KEY` in the app's secrets.")

warm_up()

with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Text → class method", list(METHOD_LABELS))
    trunc = st.slider("Truncation ψ (low = faithful, high = diverse)", 0.02, 1.0, 0.4, 0.02)
    n = st.slider("Number of samples", 1, 8, 4)
    seed = st.number_input("Seed (0 = random)", min_value=0, value=0, step=1)
    st.caption("P39 · Roll No. 39 · Generative AI mini-project")

tab_studio, tab_trunc, tab_about = st.tabs(["Studio", "Truncation explorer", "About · Ethics · Results"])

# ------------------------------------------------------------------ studio
with tab_studio:
    ex = st.selectbox("Example prompts", ["(type your own)"] + EXAMPLES)
    prompt = st.text_input("Describe an image", value="" if ex == "(type your own)" else ex,
                           placeholder="e.g. a slow reptile with a shell", max_chars=200)
    if st.button("Generate", type="primary") and prompt:
        ok, msg = check_prompt(prompt)
        if not ok:
            st.error(f"🚫 {msg}")
        else:
            with st.spinner("Mapping text to an ImageNet class…"):
                res = mapper.map_text(prompt, METHOD_LABELS[method])
            ok, note = check_class(prompt, res.idx)
            if not ok:
                st.error(f"🚫 {note}")
            else:
                class_info(res)
                if note:
                    st.info(note)
                s = int(seed) or random.randint(1, 2**31 - 1)
                with st.spinner("BigGAN is generating…"):
                    t0 = time.perf_counter()
                    imgs = generate(res.idx, trunc, int(n), seed=s)
                    gen_ms = (time.perf_counter() - t0) * 1000
                    ranked = rank(imgs, prompt)
                cols = st.columns(min(4, len(ranked)))
                for k, (im, sc) in enumerate(ranked):
                    cols[k % len(cols)].image(im.resize((256, 256), Image.LANCZOS), caption=f"#{k + 1} · CLIP {sc:.1f}", width=200)
                st.caption(f"{len(imgs)} images · truncation {trunc:.2f} · seed {s} · {gen_ms:.0f} ms · "
                           f"CLIP best {ranked[0][1]:.1f}, mean {sum(x for _, x in ranked) / len(ranked):.1f} · "
                           f"diversity {diversity(imgs):.3f}")

# ------------------------------------------------------------------ truncation explorer
with tab_trunc:
    st.markdown("Same seed in every row, so **only the truncation changes**. Low ψ samples near the mode "
                "(clean but similar); high ψ samples the full distribution (varied, more artefacts).")
    p2 = st.text_input("Prompt for the truncation grid", value="a friendly family dog with golden fur", max_chars=200)
    s2 = st.number_input("Seed", min_value=1, value=7, step=1)
    if st.button("Build truncation grid", type="primary"):
        ok, msg = check_prompt(p2)
        if not ok:
            st.error(f"🚫 {msg}")
        else:
            res = mapper.map_text(p2, METHOD_LABELS[method])
            ok, note = check_class(p2, res.idx)
            if not ok:
                st.error(f"🚫 {note}")
            else:
                st.markdown(f"**Class:** {res.label} (#{res.idx})")
                rows, table = [], []
                with st.spinner("Generating 5 truncation levels…"):
                    for t in [0.1, 0.3, 0.5, 0.7, 1.0]:
                        imgs = generate(res.idx, t, 4, seed=int(s2))
                        ranked = rank(imgs, p2)
                        table.append({"truncation": t, "mean CLIP": round(sum(x for _, x in ranked) / 4, 1),
                                      "diversity": round(diversity(imgs), 3)})
                        rows.append([labelled(im, f"t={t}") for im in imgs])
                w, h = rows[0][0].size
                grid = Image.new("RGB", (w * 4, h * len(rows)), "white")
                for r, row in enumerate(rows):
                    for c, im in enumerate(row):
                        grid.paste(im, (c * w, r * h))
                c1, c2 = st.columns([3, 2])
                c1.image(grid, caption="Rows: truncation 0.1 → 1.0")
                c2.dataframe(table, hide_index=True)

# ------------------------------------------------------------------ about
with tab_about:
    about = ROOT / "docs" / "about.md"
    st.markdown(about.read_text(encoding="utf-8") if about.exists() else "")
    for f, title in [("mapping_eval.md", None), ("truncation_table.md", None), ("diffusion_vs_biggan.md", None)]:
        p = ROOT / "results" / f
        if p.exists():
            st.markdown(p.read_text(encoding="utf-8").split("\n## Errors")[0])
    for img in ["grid_5x3.png", "diffusion_vs_biggan.png"]:
        if (ROOT / "results" / img).exists():
            st.image(str(ROOT / "results" / img))
