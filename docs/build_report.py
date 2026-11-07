"""Builds report/P39_Report.pdf from the results/ files (re-run after every eval run).

Usage: python docs/build_report.py --name "Your Name" [--url https://huggingface.co/spaces/...] [--github https://github.com/...]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from html import escape
import matplotlib

_FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(_FD / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_FD / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(_FD / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", str(_FD / "DejaVuSans-BoldOblique.ttf")))
addMapping("DejaVu", 0, 0, "DejaVu"); addMapping("DejaVu", 1, 0, "DejaVu-Bold")
addMapping("DejaVu", 0, 1, "DejaVu-Oblique"); addMapping("DejaVu", 1, 1, "DejaVu-BoldOblique")

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="Shreyanshu Ramesh Maske")
ap.add_argument("--klass", default="AIA5")
ap.add_argument("--roll", default="39")
ap.add_argument("--url", default="<live app URL>")
ap.add_argument("--github", default="<GitHub repo URL>")
args = ap.parse_args()

ss = getSampleStyleSheet()
for _st in ss.byName.values():
    _st.fontName = "DejaVu-Bold" if "Heading" in _st.name or _st.name == "Title" else "DejaVu"
for _k in ("name", "klass", "roll", "url", "github"):
    setattr(args, _k, escape(getattr(args, _k)))
H1, H2, B = ss["Heading1"], ss["Heading2"], ss["BodyText"]
SMALL = ParagraphStyle("small", parent=B, fontSize=8, leading=10)
CELL = ParagraphStyle("cell", parent=B, fontSize=7.5, leading=9)

def _cell(c):
    return escape(c)
story = []


def p(text, style=B):
    story.append(Paragraph(text, style))


def md_tables(path: Path, max_rows: int = 40) -> list[tuple[str, list[list[str]]]]:
    """Parse markdown tables (with the heading just above each) from a results file."""
    if not path.exists():
        return []
    out, heading, rows = [], "", []
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
        if line.startswith("|"):
            cells = [c.strip().replace("**", "") for c in line.strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
        elif rows:
            out.append((heading, rows[: max_rows + 1]))
            rows = []
    return out


def table(rows, widths=None):
    data = [[Paragraph(_cell(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f5")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t)
    story.append(Spacer(1, 8))


def image(path: Path, width_cm: float):
    if path.exists():
        from PIL import Image as PIL

        w, h = PIL.open(path).size
        story.append(Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w))
        story.append(Spacer(1, 6))


# ---------------------------------------------------------------- title
p("P39 · Class-Conditional Image Studio (BigGAN)", ss["Title"])
p(f"<b>Name:</b> {args.name} &nbsp;&nbsp; <b>Roll No.:</b> {args.roll} &nbsp;&nbsp; <b>Class:</b> {args.klass}<br/><b>Course:</b> Generative AI, CA mini-project")
p(f"<b>Live app:</b> {args.url}<br/><b>GitHub:</b> {args.github}")
p("<b>Goal:</b> generate images from free text via the nearest ImageNet class. Pipeline: guardrail → "
  "LLM (gpt-oss-20b) guesses class names → MiniLM retrieves candidate classes → LLM (few-shot) picks one → "
  "BigGAN-deep-128 generates with truncation ψ → CLIP ranks samples against the text.")
p("<b>Headline results:</b> text→class accuracy 47% (embedding) → 87% (few-shot v1) → <b>93–97%</b> (few-shot + query expansion, two runs), "
  "0% invalid answers vs 27% for zero-shot; truncation 0.2→1.0 raises sample diversity ~4×; CLIP score BigGAN 28.5 vs "
  "sd-turbo 32.2, while BigGAN is ~18× faster on CPU.")
image(ROOT / "docs" / "lifecycle.png", 17)

# ---------------------------------------------------------------- U1
p("U1 · Foundations: truncation as sampling from a narrower distribution", H1)
p("BigGAN's generator G(z, y) maps a latent z ∈ R<super>128</super> and a one-hot class y to an image. During training z ~ N(0, I). "
  "The <b>truncation trick</b> samples z from a normal truncated to [−2, 2] and scales it by ψ ≤ 1, so z lies in a narrower region "
  "around the mode of the prior. Images near the mode are the most \"typical\" for the class, which gives higher fidelity, while the "
  "lower spread means less diversity. This is a precision/recall trade-off controlled at sampling time, with no retraining.")
p("<b>Generative vs discriminative:</b> CLIP and the sentence encoder are used discriminatively (scoring p(match | image, text)); "
  "BigGAN is generative (sampling x ~ p(x | y)). <b>Prompt techniques compared:</b> zero-shot (\"name the ImageNet class\") vs "
  "few-shot with a retrieved candidate list (see U2 table).")
for heading, rows in md_tables(RES / "truncation_table.md")[:1]:
    p("Truncation trade-off: mean CLIP score and CLIP-embedding diversity (8 samples per cell)", H2)
    table(rows)
p("As ψ grows from 0.2 to 1.0, diversity rises roughly fourfold while the CLIP score changes little or falls slightly. "
  "This is the latent-space view of truncation.")

# ---------------------------------------------------------------- U2
story.append(PageBreak())
p("U2 · LLM maps text → class; pretrained BigGAN generates", H1)
p("The LLM is used with <b>prompt-based learning</b> (no fine-tuning). Zero-shot asks for a class name, which may not exist "
  "(a hallucination). Few-shot + retrieval gives three worked examples and the top-20 classes from all-MiniLM-L6-v2. The answer "
  "must be an index from that list, so it is validated. The embedding-only nearest label is the no-LLM baseline. "
  "<b>Iteration:</b> every v1 error happened because the right class was not among the encoder's top 20 (\"king of the jungle\" "
  "is not near \"lion\" in embedding space). v2 adds <b>query expansion</b>: the LLM first guesses 3 class names and candidates "
  "are retrieved for the text and for each guess. Accuracy rose from 87% to 93–97% over two runs (28/30 and 29/30; the LLM is not fully deterministic at temperature 0), at the cost of a second LLM call. "
  "BigGAN-deep injects the class through conditional batch-norm in every residual block and uses a <b>self-attention</b> block at "
  "64×64 so distant regions of the image stay consistent.")
tabs = md_tables(RES / "mapping_eval.md")
for heading, rows in tabs:
    if heading.startswith("Text"):
        p("Mapping accuracy on 30 labelled prompts", H2)
        table(rows)
for heading, rows in tabs:
    if heading.startswith("Errors"):
        p("Mapping errors", H2)
        table(rows)
if (RES / "grid_classes.json").exists():
    cls = json.loads((RES / "grid_classes.json").read_text())
    p("Classes chosen for the grid prompts: " + "; ".join(f"\"{k}\" → {v['label']} ({v['method']})" for k, v in cls.items()), SMALL)

# ---------------------------------------------------------------- U3
story.append(PageBreak())
p("U3 · CLIP ranking and comparison with diffusion", H1)
p("Every BigGAN sample is scored with CLIP ViT-B/32 (100 × cosine between image and prompt embeddings), and the gallery is "
  "sorted by this score. The same prompts were also rendered with sd-turbo (1-step text-to-image diffusion).")
for heading, rows in md_tables(RES / "diffusion_vs_biggan.md")[:1]:
    table(rows)
image(RES / "diffusion_vs_biggan.png", 11)

# ---------------------------------------------------------------- U4
story.append(PageBreak())
p("U4 · PyTorch truncation slider and grid", H1)
p("The Gradio app has a truncation slider (0.02–1.0), sample count and seed controls, and a <i>Truncation explorer</i> tab that "
  "renders the same seed at ψ = 0.1 … 1.0. Below is the required <b>5 prompts × 3 truncation values</b> grid (each cell = CLIP-best of 4).")
image(RES / "grid_5x3.png", 15)

# ---------------------------------------------------------------- U5
story.append(PageBreak())
p("U5 · Responsible AI: misuse, class imbalance, hallucination", H1)
p("<b>Guardrail:</b> input filter for violence, sexual content, hate, self-harm and real-person/deepfake requests; 200-character limit; "
  "weapon classes (assault rifle, revolver, rifle, missile …) are labelled, and blocked when combined with harmful-intent words.")
p("<b>Misuse note:</b> outputs are synthetic 128 px images of generic objects. They must not be passed off as real photos or used to "
  "deceive. The app is for education and research only.")
p("<b>Class imbalance:</b> ImageNet-1k has ~120 dog-breed classes but only 3 person-like classes (groom, scuba diver, ballplayer), "
  "and its photos skew towards Western web imagery. The probe below shows person prompts being forced onto unrelated objects.")
for heading, rows in tabs:
    if heading.startswith("People") or heading.startswith("Out-of"):
        p(heading, H2)
        table(rows)
p("<b>Failure cases:</b> v2 still maps \"a sour green apple\" to <i>lemon</i> instead of <i>Granny Smith</i> (the LLM reasons on "
  "\"sour\"). The embedding baseline maps \"an orange and black butterfly that migrates\" to <i>sulphur butterfly</i> (yellow): colour "
  "words are ignored by the nearest-label search. Compositional prompts (\"a cat riding a bicycle\" → <i>bicycle-built-for-two</i>) "
  "cannot be expressed by a single class at all, and abstract prompts (\"freedom\") trigger the low-confidence warning.")
p("<b>Licences:</b> BigGAN code MIT, weights from DeepMind TF-Hub (Apache-2.0); ImageNet non-commercial research only; CLIP MIT; "
  "all-MiniLM-L6-v2 Apache-2.0; sd-turbo Stability AI non-commercial licence (comparison only). The model card is in the repo.")
p("<b>Limits of the approach:</b> 1000 fixed classes, 128 px, no attributes or composition; mapping quality depends on the LLM; "
  "the CLIP score is itself biased by CLIP's web training data.")

out = ROOT / "report"
out.mkdir(exist_ok=True)
SimpleDocTemplate(str(out / "P39_Report.pdf"), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                  topMargin=1.4 * cm, bottomMargin=1.4 * cm, title="P39 BigGAN Studio Report").build(story)
print("wrote", out / "P39_Report.pdf")
