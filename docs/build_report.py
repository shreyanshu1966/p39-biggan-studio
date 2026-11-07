"""Builds report/P39_Report.pdf: a formatted project report generated from the files in results/.

Usage: python docs/build_report.py [--url LIVE_URL] [--github GITHUB_URL]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from html import escape
from pathlib import Path

import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent.parent
RES, SHOTS, WORK = ROOT / "results", ROOT / "docs" / "screenshots", ROOT / "report" / "_fig"
WORK.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="Shreyanshu Ramesh Maske")
ap.add_argument("--roll", default="39")
ap.add_argument("--klass", default="AIA5")
ap.add_argument("--url", default="https://p39-biggan-studio-2yj4yrxzre63hg8dhuqdkt.streamlit.app/")
ap.add_argument("--github", default="https://github.com/shreyanshu1966/p39-biggan-studio")
args = ap.parse_args()

# ------------------------------------------------------------------ fonts & styles
FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in [("Sans", "DejaVuSans"), ("Sans-B", "DejaVuSans-Bold"), ("Sans-I", "DejaVuSans-Oblique"),
                   ("Sans-BI", "DejaVuSans-BoldOblique")]:
    pdfmetrics.registerFont(TTFont(name, str(FD / f"{file}.ttf")))
addMapping("Sans", 0, 0, "Sans"); addMapping("Sans", 1, 0, "Sans-B")
addMapping("Sans", 0, 1, "Sans-I"); addMapping("Sans", 1, 1, "Sans-BI")

NAVY = colors.HexColor("#1f3b63")
ACCENT = colors.HexColor("#d6403a")
GREY = colors.HexColor("#5f6673")
LIGHT = colors.HexColor("#eef2f8")
LINE = colors.HexColor("#c9d2e0")

BODY = ParagraphStyle("body", fontName="Sans", fontSize=9.6, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3, alignment=TA_LEFT)
H1 = ParagraphStyle("h1", fontName="Sans-B", fontSize=15.5, leading=20, textColor=NAVY, spaceBefore=14, spaceAfter=8)
H2 = ParagraphStyle("h2", fontName="Sans-B", fontSize=11.5, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=5)
CAP = ParagraphStyle("cap", fontName="Sans-I", fontSize=8.2, leading=11, textColor=GREY, alignment=TA_CENTER,
                     spaceBefore=3, spaceAfter=10)
CELL = ParagraphStyle("cell", fontName="Sans", fontSize=8, leading=10.2)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Sans-B", textColor=colors.white)
CELLC = ParagraphStyle("cellc", parent=CELL, alignment=TA_CENTER)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.4, leading=11.5)
TASK = ParagraphStyle("task", parent=BODY, fontName="Sans-I", textColor=GREY, leftIndent=8, borderPadding=(4, 6, 4, 6),
                      backColor=LIGHT, spaceBefore=2, spaceAfter=8)

story: list = []
fig_no = [0]
tab_no = [0]


def P(text, style=BODY):
    story.append(Paragraph(text, style))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, BUL, bulletText="•"))
    story.append(Spacer(1, 4))


def heading(text, level=1):
    p = Paragraph(text, H1 if level == 1 else H2)
    p._toc = (level - 1, re.sub("<[^>]+>", "", text))
    story.append(CondPageBreak(4 * cm if level == 1 else 2.6 * cm))
    story.append(p)


def task(text):
    P(f"Task from the problem statement: {text}", TASK)


def table(rows, widths, caption=None, header=True, center_cols=(), keep=True, label=None):
    data = []
    for r_i, row in enumerate(rows):
        out = []
        for c_i, c in enumerate(row):
            if not isinstance(c, str):
                out.append(c)
                continue
            st = CELLB if (header and r_i == 0) else (CELLC if c_i in center_cols else CELL)
            out.append(Paragraph(c, st))
        data.append(out)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    cmds = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    for r in range(1 if header else 0, len(data)):
        if r % 2 == 0:
            cmds.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    t.setStyle(TableStyle(cmds))
    flow = [t]
    if caption:
        if label is None:
            tab_no[0] += 1
            label = str(tab_no[0])
        flow = [Paragraph(f"<b>Table {label}.</b> {caption}", CAP), t, Spacer(1, 10)]
    if keep:
        story.append(KeepTogether(flow))
    else:
        story.extend(flow)


def figure(path, width_cm, caption, max_h_cm=None):
    w, h = PILImage.open(path).size
    width = width_cm * cm
    height = width * h / w
    if max_h_cm and height > max_h_cm * cm:
        height = max_h_cm * cm
        width = height * w / h
    fig_no[0] += 1
    story.append(KeepTogether([Image(str(path), width=width, height=height),
                               Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", CAP)]))


def md_table(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            cells = [c.strip().replace("**", "") for c in line.strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
        elif rows:
            break
    return rows


# ------------------------------------------------------------------ data
METHOD = {"embedding": "Embedding search (no LLM)", "llm_zero": "LLM zero-shot",
          "llm_fewshot_v1": "LLM few-shot + retrieval (v1)", "llm_fewshot": "LLM few-shot + query expansion (v2)"}
summary = list(csv.DictReader(open(RES / "mapping_eval.csv", encoding="utf-8")))
details = json.loads((RES / "mapping_details.json").read_text(encoding="utf-8"))
trunc_rows = md_table(RES / "truncation_table.md")
diff_rows = md_table(RES / "diffusion_vs_biggan.md")
grid_classes = json.loads((RES / "grid_classes.json").read_text(encoding="utf-8"))
acc = {r["method"]: r for r in summary}


def pct(m, key="accuracy"):
    return f"{float(acc[m][key]) * 100:.0f}%"


# ------------------------------------------------------------------ figures rebuilt from result sheets
def crop_grid():
    """Crop the 5x3 grid cells out of results/grid_5x3.png (layout from eval/make_grid.py)."""
    im = PILImage.open(RES / "grid_5x3.png")
    cells = {}
    for r in range(5):
        for c in range(3):
            x, y = 230 + c * 160 + 2, 26 + r * 160 + 2
            p = WORK / f"grid_{r}_{c}.png"
            im.crop((x, y, x + 156, y + 156)).save(p)
            cells[r, c] = p
    return cells


def crop_diffusion():
    """Crop BigGAN / sd-turbo pairs out of results/diffusion_vs_biggan.png (layout from eval/compare_diffusion.py)."""
    im = PILImage.open(RES / "diffusion_vs_biggan.png")
    out = []
    for r in range(5):
        y = 24 + r * 200 + 2
        a, b = WORK / f"diff_{r}_g.png", WORK / f"diff_{r}_d.png"
        im.crop((262, y, 262 + 196, y + 196)).save(a)
        im.crop((462, y, 462 + 196, y + 196)).save(b)
        out.append((a, b))
    return out


# ------------------------------------------------------------------ page templates
class Doc(BaseDocTemplate):
    def afterFlowable(self, f):
        if hasattr(f, "_toc"):
            level, text = f._toc
            key = f"h{id(f)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level)
            self.notify("TOCEntry", (level, text, self.page, key))


def cover_page(canv, doc):
    canv.saveState()
    w, h = A4
    canv.setFillColor(NAVY)
    canv.rect(0, h - 7.2 * cm, w, 7.2 * cm, fill=1, stroke=0)
    canv.setFillColor(ACCENT)
    canv.rect(0, h - 7.45 * cm, w, 0.25 * cm, fill=1, stroke=0)
    canv.setFillColor(colors.white)
    canv.setFont("Sans", 11)
    canv.drawString(2 * cm, h - 2.2 * cm, "Generative AI · CA Mini-Project Report · Track D (Generation tools)")
    canv.setFont("Sans-B", 26)
    canv.drawString(2 * cm, h - 3.7 * cm, "P39 · Class-Conditional")
    canv.drawString(2 * cm, h - 4.85 * cm, "Image Studio (BigGAN)")
    canv.setFont("Sans", 12)
    canv.drawString(2 * cm, h - 6.0 * cm, "Free text → nearest ImageNet class (LLM) → BigGAN-deep-128 → CLIP ranking")
    canv.setFillColor(GREY)
    canv.setFont("Sans", 8)
    canv.drawCentredString(w / 2, 1.2 * cm, f"{dt.date.today():%d %B %Y} · all numbers produced by the scripts in eval/")
    canv.restoreState()


def normal_page(canv, doc):
    canv.saveState()
    w, h = A4
    canv.setStrokeColor(LINE)
    canv.setLineWidth(0.6)
    canv.line(2 * cm, h - 1.35 * cm, w - 2 * cm, h - 1.35 * cm)
    canv.setFont("Sans", 7.8)
    canv.setFillColor(GREY)
    canv.drawString(2 * cm, h - 1.15 * cm, "P39 · Class-Conditional Image Studio (BigGAN)")
    canv.drawRightString(w - 2 * cm, h - 1.15 * cm, f"{args.name} · Roll No. {args.roll} · {args.klass}")
    canv.line(2 * cm, 1.45 * cm, w - 2 * cm, 1.45 * cm)
    canv.drawCentredString(w / 2, 1.0 * cm, f"Page {doc.page}")
    canv.restoreState()


out_pdf = ROOT / "report" / "P39_Report.pdf"
doc = Doc(str(out_pdf), pagesize=A4, title="P39 Class-Conditional Image Studio (BigGAN)", author=args.name,
          leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
fw, fh = A4[0] - 4 * cm, A4[1] - 4 * cm
doc.addPageTemplates([
    PageTemplate("cover", [Frame(2 * cm, 2 * cm, fw, A4[1] - 10 * cm, id="c")], onPage=cover_page),
    PageTemplate("normal", [Frame(2 * cm, 2 * cm, fw, fh, id="n")], onPage=normal_page),
])
TW = fw  # text width

# ------------------------------------------------------------------ cover
story.append(Spacer(1, 0.6 * cm))
info = [["Student", args.name], ["Roll No.", args.roll], ["Class", args.klass],
        ["Course", "Generative AI (Units I–V)"], ["Problem", "P39 · Class-Conditional Image Studio (BigGAN)"],
        ["Live app", f'<link href="{args.url}" color="#1f3b63">{escape(args.url)}</link>'],
        ["Source code", f'<link href="{args.github}" color="#1f3b63">{escape(args.github)}</link>']]
t = Table([[Paragraph(f"<b>{k}</b>", CELL), Paragraph(v, CELL)] for k, v in info], colWidths=[3.2 * cm, TW - 3.2 * cm])
t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 0), (-1, -1), 5),
                       ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("BACKGROUND", (0, 0), (0, -1), LIGHT)]))
story.append(t)
story.append(Spacer(1, 0.7 * cm))
abstract = (
    "<b>Abstract.</b> A class-conditional GAN can only draw the 1000 ImageNet classes it was trained on, while people "
    "describe images in free text. This project bridges the two. A large language model (gpt-oss-20b) maps any "
    "description to the nearest ImageNet class, a pretrained <b>BigGAN-deep-128</b> generates samples with a "
    "user-controlled truncation ψ, and <b>CLIP</b> ranks the samples against the original text. Prompting the LLM "
    f"few-shot with retrieved candidates and query expansion raised text→class accuracy from {pct('embedding')} "
    f"(embedding search) to 93–97% over two runs on 30 labelled prompts, with 0% invalid classes vs "
    f"{pct('llm_zero', 'hallucination_rate')} for zero-shot. Raising ψ from 0.2 to 1.0 increased sample "
    "diversity about 4×, and a one-step diffusion model (sd-turbo) followed the text better (CLIP 32.2 vs 28.5) "
    "but was about 18× slower on CPU. The app is deployed on Streamlit Community Cloud and includes a safety "
    "guardrail, a misuse note, a class-imbalance probe and a model card.")
ab = Table([[Paragraph(abstract, BODY)]], colWidths=[TW])
ab.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
story.append(ab)
story.append(NextPageTemplate("normal"))
story.append(PageBreak())

# ------------------------------------------------------------------ contents
story.append(Paragraph("Contents", H1))
toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("t1", fontName="Sans-B", fontSize=10, leading=17, leftIndent=0),
                   ParagraphStyle("t2", fontName="Sans", fontSize=9, leading=14, leftIndent=14, textColor=GREY)]
story.append(toc)
story.append(PageBreak())

# ------------------------------------------------------------------ 1 introduction
heading("1. Introduction")
P("Allocation: Roll No. 39 → <b>P39 · Class-Conditional Image Studio (BigGAN)</b>, Track D (generation tools, "
  "so Unit IV is built deepest). <b>Goal:</b> generate images from free text via the nearest ImageNet class.")
heading("1.1 Problem statement and objectives", 2)
bullets([
    "Accept any free-text description and choose the most suitable of the 1000 ImageNet-1k classes.",
    "Generate class-conditional images with a pretrained BigGAN and expose the truncation trick as a control.",
    "Rank the generated samples with CLIP and compare BigGAN against a text-to-image diffusion model.",
    "Measure mapping accuracy, hallucination, diversity and speed, and document at least one failure case.",
    "Ship it as a public web app with a guardrail, a misuse note, licence notes and a model card.",
])
heading("1.2 Deliverables", 2)
table([["Deliverable", "Where"],
       ["Live studio app", f'<link href="{args.url}" color="#1f3b63">{escape(args.url)}</link>'],
       ["Public GitHub repository (README, model card, evaluation scripts)", escape(args.github)],
       ["5 prompts × 3 truncation values grid", "Figure 6, results/grid_5x3.png"],
       ["Measured results and failure cases", "Section 4, Section 5 and Appendix A"],
       ["2–3 minute demo video", "P39_Demo_Video.mp4 (submitted with this report)"]],
      [8.2 * cm, TW - 8.2 * cm], "Project deliverables.")

# ------------------------------------------------------------------ 2 system design
heading("2. System design")
P("Figure 1 shows the project's generative-AI lifecycle. At run time a request passes through five stages: an input "
  "guardrail; text→class mapping (the LLM first guesses class names, the MiniLM sentence encoder retrieves candidate "
  "labels for the text and the guesses, and the LLM picks one by index); a second guardrail check on the chosen "
  "class; generation by BigGAN with truncation ψ; and CLIP ranking against the original text.")
figure(ROOT / "docs" / "lifecycle.png", 17, "Generative-AI lifecycle of the project, from problem framing to governance.")
table([["Component", "Model / tool", "Role", "Licence"],
       ["Generator", "BigGAN-deep-128 (pytorch-pretrained-biggan)", "Class-conditional 128×128 synthesis", "MIT code, Apache-2.0 weights"],
       ["LLM (GPT-style)", "openai/gpt-oss-20b on Groq", "Query expansion and few-shot class choice", "Apache-2.0"],
       ["Encoder (BERT family)", "sentence-transformers/all-MiniLM-L6-v2", "Candidate retrieval, baseline, confidence", "Apache-2.0"],
       ["Ranker (multimodal)", "openai/clip-vit-base-patch32", "Image–text CLIP score", "MIT"],
       ["Comparison only", "stabilityai/sd-turbo", "Diffusion baseline (U3)", "Non-commercial research"],
       ["App and hosting", "Streamlit (Community Cloud), Gradio for local use", "UI and deployment", "Apache-2.0"]],
      [3.1 * cm, 5.2 * cm, 4.6 * cm, TW - 12.9 * cm], "Components of the pipeline.")

# ------------------------------------------------------------------ 3 implementation per unit
heading("3. Implementation by unit")
heading("3.1 Unit I · Foundations: truncation as a narrower distribution", 2)
task("Explain truncation as sampling from a narrower distribution.")
P("BigGAN's generator G(z, y) maps a latent vector z ∈ R<super>128</super> and a one-hot class y to an image; during "
  "training z ~ N(0, I). The <b>truncation trick</b> draws z from a normal distribution truncated to [−2, 2] and "
  "scales it by ψ ≤ 1. The samples therefore come from a narrower region around the mode of the prior, where the "
  "generator has seen the most data: images become more typical and cleaner but less varied. This is a fidelity–"
  "diversity (precision–recall) trade-off chosen at sampling time, with no retraining. "
  "<b>Generative vs discriminative:</b> BigGAN is generative (it samples x ~ p(x | y)), while CLIP and the "
  "sentence encoder are used discriminatively to score how well an image or label matches the text. "
  "<b>Prompt techniques compared:</b> zero-shot vs few-shot prompting (Section 3.2 and Table 4).")
heading("3.2 Unit II · GANs and LLMs: text → class, then BigGAN", 2)
task("Map text to a class with the LLM; generate with pretrained BigGAN.")
P("The LLM is adapted by <b>prompt-based learning</b>, not fine-tuning. Four mappers were compared: plain embedding "
  "search (nearest label to the text); a <b>zero-shot</b> prompt asking for a class name, which is validated against "
  "the 1000 labels so invented names count as hallucinations; <b>few-shot + retrieval (v1)</b>, with three worked "
  "examples and the top 20 labels from MiniLM, where the answer must be an index from that list; and <b>few-shot + "
  "query expansion (v2)</b>, where the LLM first guesses three class names and candidates are retrieved for each "
  "guess as well as for the text. Every v1 error happened because the right class was missing from the top 20 "
  "(\"the king of the jungle\" is not near \"lion\" in embedding space). v2 removes that bottleneck.")
P("<b>Attention and conditioning in BigGAN:</b> the class enters every residual block through conditional batch "
  "normalisation, whose gain and bias are predicted from [z, class embedding]. A self-attention block at 64×64 "
  "resolution lets distant regions of the image (two ears, several masts) stay consistent.")
figure(SHOTS / "lion.png", 15.5, "Studio tab: \"the king of the jungle roaring\" is mapped to <i>lion</i> (#291) "
       "and four BigGAN samples are shown, ranked by CLIP score.", max_h_cm=10)
heading("3.3 Unit III · GPT, BERT and multimodal: CLIP ranking vs diffusion", 2)
task("Rank outputs with CLIP; compare with a diffusion image.")
P("Each sample is scored with CLIP ViT-B/32 as 100 × cosine(image embedding, text embedding), and the gallery is "
  "sorted by this score. The same five prompts were rendered with <b>sd-turbo</b>, a one-step text-to-image diffusion "
  "model, and both outputs were scored against the original free text (Table 6 and Figure 7). The pipeline therefore "
  "combines a GPT-style generator (gpt-oss-20b), a BERT-family encoder (MiniLM) and an image–text model (CLIP).")
heading("3.4 Unit IV · Tools: PyTorch truncation slider and grid (Track D focus)", 2)
task("Build a truncation slider and grid in PyTorch.")
P("Generation runs in PyTorch on CPU (about 2 s for four 128 px images). The app has a truncation slider "
  "(0.02–1.0), sample-count and seed controls, and a <i>Truncation explorer</i> that renders the same seed at "
  "ψ = 0.1, 0.3, 0.5, 0.7 and 1.0, so only the truncation changes (Figure 3). The required 5 prompts × 3 truncation "
  "values grid is Figure 6. The BigGAN weights are downloaded from the official pytorch-pretrained-biggan release and "
  "loaded with <i>torch.load(weights_only=True)</i>, so no pickled code is executed.")
figure(SHOTS / "explorer.png", 15.5, "Truncation explorer: the same seed at ψ = 0.1 → 0.7 (rows). Low ψ gives "
       "near-identical dogs; higher ψ gives more varied poses and backgrounds.", max_h_cm=10.5)
heading("3.5 Unit V · Responsible AI: misuse note and class imbalance", 2)
task("Add a misuse note; discuss class imbalance.")
bullets([
    "<b>Guardrail:</b> the input filter blocks violence, sexual content, hate, self-harm and real-person/deepfake "
    "requests (200-character limit). Weapon classes (assault rifle, revolver, rifle, missile …) are labelled, and "
    "blocked when combined with harmful-intent words (Figure 4).",
    "<b>Misuse note:</b> outputs are synthetic 128 px images of generic objects. They must not be presented as real "
    "photographs or used to deceive. The app is for education and research only.",
    "<b>Class imbalance:</b> ImageNet-1k has about 120 dog-breed classes but only three person-like classes (groom, "
    "scuba diver, ballplayer), and its photos skew towards Western web imagery. The people probe (Table A3) shows "
    "\"a doctor\" → lab coat, \"a CEO in a meeting\" → suit and \"a farmer in India\" → tractor.",
    "<b>Hallucination test:</b> zero-shot prompting produced invalid class names for "
    f"{pct('llm_zero', 'hallucination_rate')} of prompts; the v2 mapper produced none, because every answer is "
    "checked against the label list.",
    "<b>Licences:</b> see Table 2. ImageNet and sd-turbo are non-commercial; a model card is in the repository.",
])
figure(SHOTS / "blocked.png", 13.5, "Guardrail: \"a revolver to attack a school\" is blocked after class mapping.")

# ------------------------------------------------------------------ 4 results
heading("4. Results")
heading("4.1 Summary of measured results", 2)
table([["Measure", "Result"],
       ["Text → class accuracy (30 labelled prompts)", f"{pct('embedding')} embedding → {pct('llm_zero')} zero-shot → "
        f"{pct('llm_fewshot_v1')} v1 → <b>93–97%</b> v2 (two runs)"],
       ["Invalid (hallucinated) classes", f"zero-shot {pct('llm_zero', 'hallucination_rate')} · v2 0%"],
       ["Truncation ψ 0.2 → 1.0", f"diversity {trunc_rows[-1][5]} → {trunc_rows[-1][7]} (≈4×); mean CLIP "
        f"{trunc_rows[-1][2]} → {trunc_rows[-1][4]}"],
       ["BigGAN vs sd-turbo (mean CLIP)", f"{diff_rows[-1][2]} vs {diff_rows[-1][3]}"],
       ["Generation speed (CPU)", "BigGAN ≈ 2 s for 4 images · sd-turbo ≈ 36 s for 1 image at 512 px"]],
      [6.2 * cm, TW - 6.2 * cm], "Headline results.")
heading("4.2 Text → class mapping", 2)
rows = [["Method", "Top-1 accuracy", "Invalid classes", "Mean latency"]]
for m in ["embedding", "llm_zero", "llm_fewshot_v1", "llm_fewshot"]:
    r = acc[m]
    rows.append([METHOD[m], f"{pct(m)} ({r['correct']}/{r['n']})", pct(m, "hallucination_rate"),
                 f"{float(r['mean_latency_ms']):,.0f} ms"])
table(rows, [7.4 * cm, 3.3 * cm, 3 * cm, TW - 13.7 * cm],
      "Mapping accuracy on 30 hand-labelled prompts (latest run; v2 scored 29/30 in the previous run). Latency "
      "includes free-tier API calls; v2 makes two LLM calls.", center_cols=(1, 2, 3))
P("The LLM methods are far more accurate than embedding search, which confuses surface words "
  "(\"a stringed instrument played with a bow\" → <i>bow</i>, \"the king of the jungle\" → <i>king snake</i>). Zero-shot "
  f"is fast and accurate when its answer is valid, but {pct('llm_zero', 'hallucination_rate')} of its answers were "
  "not real ImageNet names (\"piano\" instead of \"grand piano\") and had to be snapped to the nearest label. v2 "
  "combines the LLM's world knowledge with a constrained answer space. Every error is listed in Table A1.")
figure(SHOTS / "kingsnake.png", 15.5, "Embedding-only baseline on the same prompt picks <i>king snake</i>, showing "
       "why the LLM is needed.", max_h_cm=7)
heading("4.3 Truncation trade-off and the 5 × 3 grid", 2)
cells = crop_grid()
lab_w = 1.9 * cm
gw = (TW - lab_w) / 5 - 0.12 * cm
grid_rows = [[Paragraph("", CELL)] + [Paragraph(f"{escape(p)}<br/>→ <b>{escape(i['label'])}</b>", CELLC)
                                      for p, i in grid_classes.items()]]
for c, t_ in enumerate((0.2, 0.5, 1.0)):
    grid_rows.append([Paragraph(f"<b>ψ = {t_}</b>", CELL)] + [Image(str(cells[r, c]), width=gw, height=gw) for r in range(5)])
gt = Table(grid_rows, colWidths=[lab_w] + [gw + 0.12 * cm] * 5)
gt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY),
                        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
fig_no[0] += 1
story.append(KeepTogether([gt, Paragraph(f"<b>Figure {fig_no[0]}.</b> Required deliverable: 5 prompts × 3 truncation "
                                         "values (columns: prompt → class; rows: truncation ψ; each cell is the CLIP-best of 4 samples, same seed per prompt).", CAP)]))
rows = [["Prompt → class", "CLIP ψ 0.2", "CLIP ψ 0.5", "CLIP ψ 1.0", "Div. ψ 0.2", "Div. ψ 0.5", "Div. ψ 1.0"]]
for r in trunc_rows[1:]:
    rows.append([f"{escape(r[0])} → <i>{escape(r[1])}</i>" if r[1] else f"<b>{r[0]}</b>"] + r[2:])
table(rows, [5.9 * cm] + [(TW - 5.9 * cm) / 6] * 6,
      "Mean CLIP score and diversity (mean pairwise 1 − cosine of CLIP image embeddings) over 8 samples per cell.",
      center_cols=(1, 2, 3, 4, 5, 6))
P("Diversity rises steadily with ψ for every prompt (≈4× on average), while the CLIP score changes by less than one "
  "point. Truncation mostly controls variety, not relevance; relevance is decided by the class choice.")
heading("4.4 BigGAN vs diffusion", 2)
rows = [["Prompt", "BigGAN class", "BigGAN CLIP", "sd-turbo CLIP", "BigGAN time", "sd-turbo time"]]
rows += [[escape(r[0]), f"<i>{escape(r[1])}</i>" if r[1] else "", r[2], r[3], r[4], r[5]] for r in diff_rows[1:]]
rows[-1][0] = "<b>Mean</b>"
table(rows, [5.2 * cm, 3.4 * cm, 2 * cm, 2.1 * cm, 2 * cm, TW - 14.7 * cm],
      "CLIP score against the free-text prompt (BigGAN ψ = 0.4, best of 4). *First call includes loading CLIP.",
      center_cols=(2, 3, 4, 5))
pairs = crop_diffusion()
dw = (TW - lab_w) / 5 - 0.12 * cm
drows = [[Paragraph("", CELL)] + [Paragraph(f"{escape(r[0])}", CELLC) for r in diff_rows[1:6]]]
drows.append([Paragraph("<b>BigGAN</b>", CELL)] + [Image(str(g), width=dw, height=dw) for g, _ in pairs])
drows.append([Paragraph("<b>sd-turbo</b>", CELL)] + [Image(str(d), width=dw, height=dw) for _, d in pairs])
dtab = Table(drows, colWidths=[lab_w] + [dw + 0.12 * cm] * 5)
dtab.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY),
                          ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
fig_no[0] += 1
story.append(KeepTogether([dtab, Paragraph(f"<b>Figure {fig_no[0]}.</b> Same prompts rendered by BigGAN (top row, "
                                           "CLIP-best of 4) and sd-turbo (bottom row); CLIP scores are printed on each image.", CAP)]))
P("Diffusion follows the whole sentence (colours, actions, scenes) and wins on every prompt, by the widest margin on "
  "the compositional prompt \"a cat riding a bicycle\" (24.3 vs 36.1). BigGAN is limited to one class at 128 px but "
  "is about 18× faster on CPU and gives direct, interpretable control through the class and ψ.")

# ------------------------------------------------------------------ 5 failure cases
heading("5. Failure cases and limitations")
bullets([
    "<b>Composition is lost:</b> \"a cat riding a bicycle\" → <i>bicycle-built-for-two</i>; one class cannot "
    "express two objects and an action (Figure 7, last column).",
    "<b>Attribute reasoning:</b> v2 maps \"a sour green apple\" → <i>lemon</i> instead of <i>Granny Smith</i>; in one "
    "run \"a stringed instrument played with a bow under the chin\" → <i>cello</i>.",
    "<b>Out-of-vocabulary concepts:</b> \"a giraffe at sunset\" (giraffe is not an ImageNet-1k class) and \"the "
    "Eiffel Tower at night\" (→ <i>water tower</i>) are forced onto the nearest class (Table A2).",
    "<b>Abstract prompts:</b> \"freedom\" has no close label (best similarity 0.23), so the app shows a "
    "low-confidence warning (Figure 8).",
    "<b>Non-determinism:</b> the hosted LLM varied by one prompt between runs even at temperature 0.",
    "<b>General limits:</b> 128 px output, 1000 fixed classes, and CLIP's own web-data biases in the ranking score.",
])
figure(SHOTS / "freedom.png", 14.5, "Abstract prompt \"freedom\": no ImageNet class is close, so the app warns that "
       "the output may not match the text.", max_h_cm=9)

# ------------------------------------------------------------------ 6 deployment
heading("6. Deployment and reproducibility")
bullets([
    f"<b>Live app:</b> <link href='{args.url}' color='#1f3b63'>{escape(args.url)}</link> (Streamlit Community Cloud, "
    "file <i>streamlit_app.py</i>). The Groq API key is stored in the platform's secrets, never in the repository.",
    f"<b>Code:</b> {escape(args.github)}. <i>app.py</i> is the same app in Gradio for local use.",
    "<b>Reproduce the results:</b> <i>eval/run_mapping_eval.py</i>, <i>eval/make_grid.py</i> and "
    "<i>eval/compare_diffusion.py</i> regenerate every table and figure in this report; <i>docs/build_report.py</i> "
    "rebuilds this PDF.",
    "<b>Model card:</b> MODEL_CARD.md lists intended use, data, evaluation, biases and licences.",
])

# ------------------------------------------------------------------ 7 conclusion
heading("7. Conclusion and future work")
P("The studio shows that an LLM is a practical bridge between open-ended language and a closed-vocabulary generator. "
  "Prompt engineering alone (few-shot examples, retrieval and query expansion, with no fine-tuning) took class "
  "accuracy from 47% to 93–97% and removed hallucinated classes. The truncation slider makes the GAN's "
  "fidelity–diversity trade-off visible and measurable. The diffusion comparison marks the limit of the approach: "
  "class-conditional GANs are fast and controllable but cannot express attributes or compositions.")
P("<b>Future work:</b> blend two class embeddings for compositional prompts; let the LLM pick ψ from the wording "
  "(\"a typical …\" vs \"unusual …\"); upscale outputs with a super-resolution model; and replace the query-expansion "
  "call with a small fine-tuned classifier to cut latency.")

# ------------------------------------------------------------------ references
heading("References")
for ref in [
    "A. Brock, J. Donahue, K. Simonyan. <i>Large Scale GAN Training for High Fidelity Natural Image Synthesis</i> (BigGAN). ICLR 2019.",
    "A. Radford et al. <i>Learning Transferable Visual Models From Natural Language Supervision</i> (CLIP). ICML 2021.",
    "N. Reimers, I. Gurevych. <i>Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks</i>. EMNLP 2019.",
    "A. Sauer, D. Lorenz, A. Blattmann, R. Rombach. <i>Adversarial Diffusion Distillation</i> (SD-Turbo). 2023.",
    "O. Russakovsky et al. <i>ImageNet Large Scale Visual Recognition Challenge</i>. IJCV 2015.",
    "T. Brown et al. <i>Language Models are Few-Shot Learners</i>. NeurIPS 2020.",
    "OpenAI. <i>gpt-oss-20b model card</i>, 2025 (served via Groq).",
    "Hugging Face: pytorch-pretrained-BigGAN, transformers, sentence-transformers and diffusers libraries.",
]:
    story.append(Paragraph(ref, BUL, bulletText="–"))

# ------------------------------------------------------------------ appendix
story.append(PageBreak())
heading("Appendix A. Detailed mapping results")
errs = [d for d in details["labelled"] if not d["correct"]]
rows = [["Method", "Prompt", "Expected", "Predicted"]]
rows += [[METHOD[d["method"]], escape(d["prompt"]), escape(", ".join(d["gold"])), f"<i>{escape(d['pred'])}</i>"]
         for d in errs]
table(rows, [4.6 * cm, 5.6 * cm, 3.6 * cm, TW - 13.8 * cm],
      "Every wrong prediction on the 30 labelled prompts (latest run).", keep=False, label="A1")
order = ["embedding", "llm_zero", "llm_fewshot_v1", "llm_fewshot"]
for group, label, title in [("out_of_vocabulary", "A2", "Out-of-vocabulary prompts: class chosen by each method."),
                            ("people_bias_probe", "A3", "People / bias probe: ImageNet has almost no person classes.")]:
    by: dict = {}
    for d in details[group]:
        by.setdefault(d["prompt"], {})[d["method"]] = d
    rows = [["Prompt", "Embedding", "Zero-shot", "Few-shot v1", "Few-shot v2", "Best sim."]]
    for prompt, ms in by.items():
        best = next(iter(ms.values())).get("best_similarity", "")
        warn = " ⚠" if any(v.get("low_confidence") for v in ms.values()) else ""
        rows.append([escape(prompt)] + [escape(ms[m]["pred"]) if m in ms else "–" for m in order] + [f"{best}{warn}"])
    table(rows, [4.2 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, TW - 14.6 * cm], title, center_cols=(5,), label=label)
P("⚠ = low-confidence warning shown (best encoder similarity below 0.35).", SMALL)

doc.multiBuild(story)
print("wrote", out_pdf)
