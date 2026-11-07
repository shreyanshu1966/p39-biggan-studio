"""Copies the summary tables from results/*.md into README.md (between the RESULTS markers)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def first_table(path: Path) -> str:
    if not path.exists():
        return "_not run yet_"
    lines, out, started = path.read_text(encoding="utf-8").splitlines(), [], False
    for ln in lines:
        if ln.startswith("|"):
            out.append(ln)
            started = True
        elif started:
            break
    return "\n".join(out)


block = "\n".join([
    "### Text → class mapping (30 labelled prompts)", first_table(RES / "mapping_eval.md"), "",
    "### Truncation trade-off", first_table(RES / "truncation_table.md"), "",
    "### BigGAN vs diffusion (CLIP score against the free text)", first_table(RES / "diffusion_vs_biggan.md"),
    "", "Full tables, error lists, out-of-vocabulary and people-bias probes: [`results/`](results/).",
])
readme = ROOT / "README.md"
text = readme.read_text(encoding="utf-8")
text = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
              lambda _: f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->", text, flags=re.S)
readme.write_text(text, encoding="utf-8")
print("README results updated")
