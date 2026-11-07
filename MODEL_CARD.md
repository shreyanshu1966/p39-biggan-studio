# Model Card: P39 Class-Conditional Image Studio

## System overview
A pipeline, not a newly trained model: **free text → ImageNet class (LLM + sentence encoder) → BigGAN-deep-128 → CLIP ranking**.

| Component | Model | Role | Licence |
|---|---|---|---|
| Generator | BigGAN-deep-128 (Brock et al., 2019), via `pytorch-pretrained-biggan` | class-conditional 128×128 image synthesis | code MIT, weights from DeepMind TF-Hub (Apache-2.0) |
| Text → class (LLM) | `openai/gpt-oss-20b` on Groq (configurable via `LLM_MODEL`, e.g. `qwen/qwen3.8-27b`) | few-shot + retrieval class selection | Apache-2.0 (gpt-oss) / Groq terms |
| Text encoder (BERT family) | `sentence-transformers/all-MiniLM-L6-v2` | candidate retrieval, embedding baseline, confidence score | Apache-2.0 |
| Ranker | `openai/clip-vit-base-patch32` | CLIP score between prompt and images | MIT |
| Comparison only | `stabilityai/sd-turbo` | diffusion baseline in `eval/compare_diffusion.py` | Stability AI non-commercial research licence |

## Intended use
Education and research: demonstrating class-conditional GANs, the truncation trick, LLM prompt-based mapping and CLIP evaluation. **Not** for producing realistic images of people, deceptive media, or commercial use (ImageNet terms are non-commercial).

## Training data
BigGAN was trained on ImageNet-1k (ILSVRC-2012, about 1.28 M images, 1000 classes). No training or fine-tuning was done in this project.

## Evaluation
See `results/`:
- `mapping_eval.md`: top-1 accuracy, hallucination rate and latency of the three mapping methods on 30 labelled prompts, plus out-of-vocabulary and people-bias probes.
- `truncation_table.md`: CLIP score vs diversity at truncation 0.2 / 0.5 / 1.0.
- `diffusion_vs_biggan.md`: CLIP score of BigGAN vs sd-turbo on the same prompts.

## Known limitations and biases
- Only the 1000 ImageNet classes. Colours, counts, actions and scenes in the prompt are ignored.
- **Class imbalance**: about 120 of the 1000 classes are dog breeds, but only 3 depict people (groom, scuba diver, ballplayer). Person-related prompts are mapped to unrelated objects.
- ImageNet images skew towards Western, English-language web photos, and BigGAN reproduces those contexts.
- 128 px output, with occasional GAN artefacts at high truncation.
- The LLM can hallucinate class names (measured). The app validates every answer and falls back to the encoder.

## Safety measures
Input guardrail (violence, sexual, hate, self-harm, real-person/deepfake keywords, 200-character limit), output check for weapon classes, low-confidence warning when no ImageNet class is close, and API keys kept in platform secrets.
