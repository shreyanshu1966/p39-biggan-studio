"""Edits the final demo video with ffmpeg.

intro (brag.mp4, 20 s) -> captioned screen recording (waiting time sped up) -> end card,
with a soft synthesized music bed under the walkthrough.

Usage: python video/edit_demo.py <rec-dir> <out.mp4> [--live-url URL]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("rec")
ap.add_argument("out")
ap.add_argument("--live-url", default="")
a = ap.parse_args()
REC = Path(a.rec)
WORK = REC / "edit"
WORK.mkdir(exist_ok=True)
W, H, FPS = 1920, 1080, 30
BG, ACCENT, FG, MUTED = (14, 17, 23), (255, 75, 75), (250, 250, 250), (163, 168, 184)


def font(size, bold=False):
    for f in (["C:/Windows/Fonts/segoeuib.ttf"] if bold else []) + ["C:/Windows/Fonts/segoeui.ttf",
                                                                     "C:/Windows/Fonts/arial.ttf"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-2000:])


# Captions per recorded scene (same order as record_demo.py marks) + playback speed
CAPTIONS = [
    ("Type anything → nearest ImageNet class → BigGAN → CLIP ranking", 1.0),
    ("Prompt: “the king of the jungle roaring”", 1.0),
    ("The LLM picks lion (#291) · 4 BigGAN samples ranked by CLIP score", 1.0),
    ("Baseline: embedding search only, no LLM", 1.0),
    ("Without the LLM → king snake.  Accuracy: 47% vs 93–97% with the LLM", 1.0),
    ("Truncation slider ψ: 0.40 → 0.90", 1.4),
    ("High ψ samples the full latent distribution: more variety", 1.0),
    ("Truncation explorer: same seed, only ψ changes (0.1 → 1.0)", 2.2),
    ("ψ 0.1: near-identical · ψ 1.0: diverse  (diversity ≈ 4×)", 1.0),
    ("Guardrail: weapon class + harmful intent → blocked", 1.3),
    ("Failure case: “a cat riding a bicycle”", 1.0),
    ("→ bicycle-built-for-two: one ImageNet class can't hold the whole scene", 1.0),
    ("Abstract prompt “freedom” → low-confidence warning", 1.3),
    ("About tab: theory, ethics, licences and all results", 1.6),
]


def caption_png(text, path, step, total):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, H - 118, W, H], fill=BG + (232,))
    d.rectangle([0, H - 118, 10, H], fill=ACCENT + (255,))
    d.text((48, H - 59), text, font=font(44, True), fill=FG, anchor="lm")
    d.text((W - 40, H - 59), f"{step}/{total}", font=font(30), fill=MUTED, anchor="rm")
    chip = "P39 · BigGAN Studio · app demo"
    f = font(26, True)
    tw = d.textlength(chip, font=f)
    d.rounded_rectangle([W - tw - 70, 18, W - 22, 66], radius=24, fill=BG + (215,))
    d.text((W - 46, 42), chip, font=f, fill=FG, anchor="rm")
    im.save(path)


scenes = json.loads((REC / "scenes.json").read_text(encoding="utf-8"))
assert len(scenes) - 1 == len(CAPTIONS), (len(scenes), len(CAPTIONS))
segs = []
for i, ((_, start), (cap, speed)) in enumerate(zip(scenes[:-1], CAPTIONS)):
    end = scenes[i + 1][1]
    if i == 0:
        start = max(0.0, start - 1.0)
    png = WORK / f"cap{i:02d}.png"
    caption_png(cap, png, i + 1, len(CAPTIONS))
    seg = WORK / f"seg{i:02d}.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(REC / "raw.webm"),
         "-i", str(png), "-filter_complex",
         f"[0:v]setpts=(PTS-STARTPTS)/{speed},scale={W}:{H}:flags=lanczos,fps={FPS},format=yuv420p[v];[v][1:v]overlay=0:0",
         "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(seg)])
    segs.append(seg)
    print(f"seg {i:02d}: {start:6.1f}-{end:6.1f}s x{speed} {cap}")

# End card
card = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(card)
d.text((W // 2, 250), "BigGAN Studio", font=font(110, True), fill=FG, anchor="mm")
d.text((W // 2, 360), "P39 · Class-Conditional Image Studio · Generative AI mini-project", font=font(40), fill=MUTED, anchor="mm")
lines = [("Text → class accuracy", "47% embedding → 93–97% with the LLM"),
         ("Truncation ψ 0.2 → 1.0", "sample diversity ≈ 4×"),
         ("BigGAN vs sd-turbo", "CLIP 28.5 vs 32.2 · BigGAN ~18× faster on CPU")]
for k, (h, v) in enumerate(lines):
    y = 500 + k * 70
    d.text((W // 2 - 30, y), h, font=font(38, True), fill=ACCENT, anchor="rm")
    d.text((W // 2 + 30, y), v, font=font(38), fill=FG, anchor="lm")
links = ["github.com/shreyanshu1966/p39-biggan-studio"] + ([a.live_url.replace("https://", "")] if a.live_url else [])
for k, l in enumerate(links):
    d.text((W // 2, 790 + k * 60), l, font=font(40, True), fill=FG, anchor="mm")
d.text((W // 2, 960), "Shreyanshu Ramesh Maske · Roll No. 39 · AIA5", font=font(36), fill=MUTED, anchor="mm")
card.save(WORK / "endcard.png")
run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", "7", "-i", str(WORK / "endcard.png"),
     "-vf", f"fps={FPS},format=yuv420p,fade=in:st=0:d=0.6", "-c:v", "libx264", "-crf", "18", str(WORK / "end.mp4")])
segs.append(WORK / "end.mp4")

# Concatenate walkthrough + end card (video only), then add the music bed
lst = WORK / "list.txt"
lst.write_text("".join(f"file '{s.resolve().as_posix()}'\n" for s in segs), encoding="utf-8")
run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(WORK / "body_v.mp4")])
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            str(WORK / "body_v.mp4")], capture_output=True, text=True).stdout)
run([sys.executable, str(ROOT / "make_music.py"), "bed", str(WORK / "bed.wav"), f"{dur:.2f}"])
run(["ffmpeg", "-v", "error", "-y", "-i", str(WORK / "body_v.mp4"), "-i", str(WORK / "bed.wav"),
     "-filter_complex", f"[1:a]volume=0.45,afade=out:st={dur - 2.5:.2f}:d=2.5[a];[0:v]fade=in:st=0:d=0.5,fade=out:st={dur - 0.8:.2f}:d=0.8[v]",
     "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
     "-ar", "44100", "-shortest", str(WORK / "body.mp4")])

# Intro + body
intro = ROOT / "brag-output" / "brag.mp4"
run(["ffmpeg", "-v", "error", "-y", "-i", str(intro), "-i", str(WORK / "body.mp4"), "-filter_complex",
     "[0:v]fade=out:st=19.5:d=0.5,setsar=1[v0];[1:v]setsar=1[v1];[0:a]aresample=44100[a0];[1:a]aresample=44100[a1];"
     "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
     "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", a.out])
total = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.out],
                       capture_output=True, text=True).stdout.strip()
print("wrote", a.out, f"({float(total):.1f} s)")
