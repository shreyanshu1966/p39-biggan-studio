"""Records the app walkthrough with Playwright and logs scene timestamps for captions.

Usage: python video/record_demo.py <app-url> <out-dir>
Out:   <out-dir>/raw.webm, <out-dir>/scenes.json  [(caption, start_s), ...]
"""
import json
import shutil
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1600, 900
scenes: list[tuple[str, float]] = []
t0 = 0.0


def mark(caption: str) -> None:
    scenes.append((caption, round(time.time() - t0, 2)))
    print(f"{scenes[-1][1]:6.1f}s  {caption}", flush=True)


def wait_ready(pg, timeout=300000):
    pg.get_by_label("Describe an image", exact=True).wait_for(timeout=timeout)
    pg.wait_for_timeout(800)


def wait_idle(pg, timeout=180000):
    """Wait until Streamlit finishes the current run (status widget disappears)."""
    pg.wait_for_timeout(600)
    pg.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=timeout)
    pg.wait_for_timeout(500)


def scroll(pg, dy, steps=10, pause=60):
    pg.mouse.move(W * 0.6, H * 0.6)
    for _ in range(steps):
        pg.mouse.wheel(0, dy / steps)
        pg.wait_for_timeout(pause)


def scroll_top(pg):
    scroll(pg, -6000, steps=6, pause=30)


def type_prompt(pg, text, label="Describe an image"):
    box = pg.get_by_label(label, exact=True)
    box.click()
    box.press("Control+a")
    box.press("Delete")
    box.type(text, delay=55)
    box.press("Enter")
    pg.wait_for_timeout(400)


def choose_method(pg, name):
    pg.get_by_label("Text → class method").click() if pg.get_by_label("Text → class method").count() else \
        pg.locator('[data-testid="stSidebar"] [data-baseweb="select"]').first.click()
    pg.wait_for_timeout(400)
    pg.get_by_role("option", name=name).click()
    pg.wait_for_timeout(700)


def set_slider(pg, index, presses, key="ArrowRight"):
    thumb = pg.locator('[data-testid="stSidebar"] input[type="range"]').nth(index)
    thumb.focus()
    for _ in range(presses):
        pg.keyboard.press(key)
        pg.wait_for_timeout(45)
    pg.wait_for_timeout(500)


def wait_for_text(pg, expect, timeout=180000):
    """Wait until `expect` is on the page and no spinner text is showing."""
    pg.wait_for_timeout(700)
    pg.wait_for_function(
        """(e) => { const t = document.body.innerText;
                    return t.includes(e) && !/generating|Mapping text|Running/i.test(t); }""",
        arg=expect, timeout=timeout)
    pg.wait_for_timeout(900)


def generate(pg, expect):
    pg.get_by_role("button", name="Generate").click()
    wait_for_text(pg, expect)


with sync_playwright() as p:
    browser = p.chromium.launch()
    # warm-up (not recorded): first load downloads/loads the models
    warm = browser.new_page(viewport={"width": W, "height": H})
    warm.goto(URL)
    wait_ready(warm, timeout=900000)
    warm.close()

    ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(OUT / "tmp"),
                              record_video_size={"width": W, "height": H})
    pg = ctx.new_page()
    t0 = time.time()
    pg.goto(URL)
    wait_ready(pg)

    mark("Type anything: an LLM maps it to the nearest of 1000 ImageNet classes, BigGAN draws it, CLIP ranks it")
    pg.wait_for_timeout(3500)

    mark("Prompt: \"the king of the jungle roaring\", default method = LLM few-shot + query expansion")
    type_prompt(pg, "the king of the jungle roaring")
    generate(pg, "Chosen class: lion")
    mark("The LLM picks 'lion' (#291); BigGAN generates 4 samples, sorted by CLIP score")
    scroll(pg, 520)
    pg.wait_for_timeout(5000)

    scroll_top(pg)
    mark("Baseline: embedding-only mapping (no LLM), same prompt")
    choose_method(pg, "Embedding only (no LLM)")
    generate(pg, "Chosen class: king snake")
    scroll(pg, 520)
    mark("Without the LLM it picks 'king snake': 47% accuracy vs 93-97% with the LLM on 30 test prompts")
    pg.wait_for_timeout(5000)

    scroll_top(pg)
    choose_method(pg, "LLM few-shot + query expansion (default)")
    mark("Truncation slider: raise psi from 0.40 to 0.90 for more diverse samples")
    set_slider(pg, 0, 25)
    type_prompt(pg, "a mountain erupting with lava")
    generate(pg, "Chosen class: volcano")
    scroll(pg, 520)
    mark("High truncation = full latent distribution: more variety, some artefacts")
    pg.wait_for_timeout(5000)
    scroll_top(pg)
    set_slider(pg, 0, 25, key="ArrowLeft")

    mark("Truncation explorer: same seed, only psi changes (0.1 -> 1.0)")
    pg.get_by_role("tab", name="Truncation explorer").click()
    pg.wait_for_timeout(1200)
    pg.get_by_role("button", name="Build truncation grid").click()
    wait_for_text(pg, "Rows: truncation")
    scroll(pg, 300)
    mark("Low psi: clean but near-identical. High psi: diverse. Diversity rises about 4x (table)")
    pg.wait_for_timeout(6500)
    scroll(pg, 450)
    pg.wait_for_timeout(2500)
    scroll_top(pg)

    pg.get_by_role("tab", name="Studio").click()
    pg.wait_for_timeout(800)
    mark("Guardrail: a weapon class combined with harmful intent is blocked")
    type_prompt(pg, "a revolver to attack a school")
    generate(pg, "Blocked")
    pg.wait_for_timeout(3500)

    mark("Failure case: compositions can't be expressed by a single ImageNet class")
    type_prompt(pg, "a cat riding a bicycle")
    generate(pg, "Chosen class: bicycle-built-for-two")
    scroll(pg, 520)
    mark("'a cat riding a bicycle' becomes 'bicycle-built-for-two': the cat is lost (CLIP 24 vs 36 for diffusion)")
    pg.wait_for_timeout(5500)
    scroll_top(pg)

    mark("Abstract prompts get a low-confidence warning")
    type_prompt(pg, "freedom")
    generate(pg, "Low confidence")
    scroll(pg, 250)
    pg.wait_for_timeout(5000)
    scroll_top(pg)

    mark("About tab: truncation theory, attention, ethics, licences and all result tables")
    pg.get_by_role("tab", name="About · Ethics · Results").click()
    pg.wait_for_timeout(1500)
    for _ in range(14):
        scroll(pg, 380, steps=8, pause=70)
        pg.wait_for_timeout(700)
    mark("END")
    pg.wait_for_timeout(500)
    video = pg.video.path()
    ctx.close()
    browser.close()

shutil.move(video, OUT / "raw.webm")
shutil.rmtree(OUT / "tmp", ignore_errors=True)
(OUT / "scenes.json").write_text(json.dumps(scenes, indent=1), encoding="utf-8")
print("saved", OUT / "raw.webm")
