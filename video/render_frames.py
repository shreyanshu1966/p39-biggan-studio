"""Renders intro.html frame by frame (renderAt(t) is a pure function of time).

Usage: python video/render_frames.py <html> <out-dir> [--stills t1,t2,...] [--fps 30 --dur 20]
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("html")
ap.add_argument("out")
ap.add_argument("--stills", default="")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--dur", type=float, default=20)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(Path(a.html).resolve().as_uri())
    pg.evaluate("window.ready")
    if a.stills:
        for t in [float(x) for x in a.stills.split(",")]:
            pg.evaluate(f"renderAt({t})")
            pg.screenshot(path=str(out / f"still_{t:05.2f}.png"))
    else:
        n = int(round(a.fps * a.dur))
        for i in range(n):
            pg.evaluate(f"renderAt({i / a.fps})")
            pg.screenshot(path=str(out / f"f{i:05d}.jpg"), type="jpeg", quality=95)
    b.close()
print("done")
