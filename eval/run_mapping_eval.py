"""U1/U2/U5 evaluation of the text -> ImageNet-class mapper.

Measures, for each method (embedding / llm_zero / llm_fewshot):
  * top-1 accuracy on 30 hand-labelled prompts
  * hallucination rate (LLM returned a class name / index that does not exist)
  * mean mapping latency
and logs what each method does on out-of-vocabulary prompts and a people-bias probe.

Run:  python eval/run_mapping_eval.py            (LLM methods need GROQ_API_KEY or HF_TOKEN)
Out:  results/mapping_eval.md, results/mapping_eval.csv, results/mapping_details.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from studio import mapper  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
data = json.loads((ROOT / "eval" / "prompts.json").read_text(encoding="utf-8"))

lower = {c.lower(): i for i, c in enumerate(mapper.CLASSES)}
for item in data["labelled"]:
    missing = [g for g in item["gold"] if g.lower() not in lower]
    assert not missing, f"gold label(s) not in ImageNet list: {missing}"
    item["gold_idx"] = {i for i, c in enumerate(mapper.CLASSES) if c.lower() in {g.lower() for g in item["gold"]}}

methods = ["embedding"] + (["llm_zero", "llm_fewshot_v1", "llm_fewshot"] if mapper.llm_available() else [])
if len(methods) == 1:
    print("No LLM key found: evaluating the embedding baseline only.")

mapper.map_text("warm-up", "embedding")  # load encoder + label cache outside the timing
summary, details = [], {"labelled": [], "out_of_vocabulary": [], "people_bias_probe": []}

for m in methods:
    correct = halluc = errors = 0
    lat = []
    for item in data["labelled"]:
        r = mapper.map_text(item["prompt"], m)
        if m != "embedding":
            time.sleep(0.3)  # stay under free-tier rate limits
        ok = r.idx in item["gold_idx"]
        correct += ok
        halluc += r.hallucinated
        errors += r.fallback.startswith("LLM error")
        lat.append(r.latency_ms)
        details["labelled"].append({"method": m, "prompt": item["prompt"], "gold": item["gold"],
                                    "pred": r.label, "correct": ok, "hallucinated": r.hallucinated,
                                    "reason": r.reason, "fallback": r.fallback})
    n = len(data["labelled"])
    summary.append({"method": m, "accuracy": correct / n, "correct": correct, "n": n,
                    "hallucination_rate": halluc / n, "llm_errors": errors,
                    "mean_latency_ms": sum(lat) / n})
    print(f"{m:12s} acc={correct}/{n}  halluc={halluc}  errors={errors}  latency={sum(lat) / n:.0f} ms")

for group in ("out_of_vocabulary", "people_bias_probe"):
    for p in data[group]:
        for m in methods:
            r = mapper.map_text(p, m)
            details[group].append({"method": m, "prompt": p, "pred": r.label, "similarity": round(r.similarity, 3), "best_similarity": round(r.best_similarity, 3),
                                   "low_confidence": r.low_confidence, "hallucinated": r.hallucinated,
                                   "reason": r.reason})

model_name = mapper.llm_config()["model"] if mapper.llm_available() else "-"
with open(OUT / "mapping_eval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
(OUT / "mapping_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")

md = [f"# Text → ImageNet class mapping evaluation\n\nLLM: `{model_name}` · encoder: `{mapper.EMBED_MODEL}` · "
      f"{len(data['labelled'])} labelled prompts\n",
      "| Method | Top-1 accuracy | Hallucination rate | LLM errors | Mean latency |", "|---|---|---|---|---|"]
for s in summary:
    md.append(f"| {s['method']} | {s['accuracy']:.0%} ({s['correct']}/{s['n']}) | {s['hallucination_rate']:.0%} | "
              f"{s['llm_errors']} | {s['mean_latency_ms']:.0f} ms |")
md += ["\n## Errors on labelled prompts\n", "| Method | Prompt | Gold | Predicted |", "|---|---|---|---|"]
md += [f"| {d['method']} | {d['prompt']} | {', '.join(d['gold'])} | {d['pred']} |" for d in details["labelled"] if not d["correct"]]
for group, title in (("out_of_vocabulary", "Out-of-vocabulary prompts (failure cases)"),
                     ("people_bias_probe", "People / bias probe (ImageNet has almost no person classes)")):
    md += [f"\n## {title}\n", "| Method | Prompt | Mapped to | Encoder sim | Low-confidence flag |", "|---|---|---|---|---|"]
    md += [f"| {d['method']} | {d['prompt']} | {d['pred']} | {d['similarity']} | {'yes' if d['low_confidence'] else 'no'} |"
           for d in details[group]]
(OUT / "mapping_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("wrote", OUT / "mapping_eval.md")
