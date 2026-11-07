# brag plan: P39 BigGAN Studio

**What it is:** type any sentence and get BigGAN images of the closest ImageNet class, ranked by CLIP.
**For:** Generative AI students and examiners who want to see class-conditional GANs, truncation and LLM prompting working together.
**Sets it apart:** an LLM turns fuzzy text ("the king of the jungle roaring") into an exact class (*lion*), which takes accuracy from 47% to 93–97%.
**Hook:** a sentence types itself, and four lions appear.
**Tone:** `default` (punchy, clean), matching the Streamlit look: dark #0e1117, accent #ff4b4b, white type.
**Share caption:** see `share-copy.txt`.

## Music
120 BPM (beat = 0.5 s, bar = 2 s), A minor: kick on every beat, soft hats on off-beats, a sub bass following Am–F–C–G, and a pad. Every cut lands on a bar line. A soft riser before the reveal and a low hit on the outro. Synthesized in Python, nothing downloaded.

## Storyboard (20 s, 1920×1080, 30 fps)
| # | Time | Scene | On screen |
|---|---|---|---|
| 1 | 0–4 s | Hook | "Type anything." Prompt box types *the king of the jungle roaring* with a blinking cursor |
| 2 | 4–8 s | Reveal | The box moves up, an arrow leads to the chip **lion · ImageNet #291**, and 4 real BigGAN lions pop in on beats with their CLIP scores |
| 3 | 8–12 s | Highlight: truncation | "One knob: truncation ψ". A slider sweeps 0.1 → 1.0 over 5 real volcano samples, labelled *faithful ↔ diverse* |
| 4 | 12–16 s | Highlight: accuracy | "The LLM picks the right class". Bars grow: embedding 47% → LLM 93–97%; "0 invalid classes" |
| 5 | 16–20 s | Outro | Mosaic of real BigGAN samples behind the title **BigGAN Studio**: "free text → ImageNet class → BigGAN → CLIP" · P39 · Roll No. 39 |

Transitions dip through the background (old content out, then new content in), so no muddy cross-fades. Every image is a real output of the app's BigGAN-deep-128; every number comes from `results/`.
