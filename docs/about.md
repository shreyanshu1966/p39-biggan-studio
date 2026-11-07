## How it works
1. **Guardrail**: blocks violent, sexual, hateful, self-harm and real-person/deepfake prompts.
2. **Text → class**: an LLM (gpt-oss-20b on Groq) chooses the nearest of the 1000 ImageNet classes. It guesses likely class names, a BERT-family sentence encoder (all-MiniLM-L6-v2) retrieves candidate labels for the text and those guesses, and the LLM picks one by index from a few-shot prompt.
3. **Generation**: **BigGAN-deep-128**, a class-conditional GAN pretrained on ImageNet, turns noise *z* and the one-hot class *y* into a 128×128 image.
4. **Ranking**: **CLIP** scores every sample against your original text (100 × cosine similarity), and the best is shown first.

## Generative vs discriminative (U1)
A **discriminative** model learns *p(y | x)*: given an image, which class is it? (CLIP and the encoder play this role here.)
A **generative** model learns to sample *x ~ p(x | y)*: given a class, produce a new image. BigGAN's generator *G(z, y)* is trained adversarially against a discriminator *D(x, y)* that tries to tell real ImageNet photos from fakes.

## The truncation trick (U1)
BigGAN samples *z ~ N(0, I)* in a 128-dimensional latent space. With truncation *ψ*, values are drawn from a normal distribution **truncated to [−2, 2]** and then scaled by *ψ*, so all samples come from a **narrower region around the mode**.
- *ψ → 0*: samples sit close to the "average" image of the class: clean and faithful, but almost identical.
- *ψ = 1*: the full distribution: varied, but with more artefacts.

The **Truncation explorer** tab shows this with the same seed in every row. Mean CLIP-diversity rises from about 0.03 at ψ=0.2 to about 0.11 at ψ=1.0 (see results).

## Attention in BigGAN (U2)
BigGAN-deep has a **self-attention block** at the 64×64 stage (layer 8 in `config.json`). Each spatial position computes query/key/value projections and attends to all other positions. This lets the generator keep distant parts of the object consistent (for example a dog's two ears or a ship's masts), which purely local convolutions handle poorly. Class information enters every residual block through **conditional batch-norm**, whose gain and bias are predicted from *[z, class embedding]*.

## Prompt-based learning vs fine-tuning (U2)
We do not fine-tune the LLM. The mapping is learned **in context**:
- **Zero-shot**: "name the ImageNet class". It is fast, but the LLM often invents names that are not in the 1000 classes (hallucinations, which are counted).
- **Few-shot + retrieval (v1)**: three worked examples plus a numbered candidate list. The answer must be an index from the list, so invalid outputs can be detected and corrected. Its weak point is that when the encoder's top 20 miss the right class ("king of the jungle" is not near "lion"), the LLM cannot choose it.
- **Few-shot + query expansion (v2, default)**: the LLM first guesses 3 class names, candidates are retrieved for the text *and* each guess, then the LLM picks an index. On 30 labelled prompts accuracy rose from 47% (embedding) → 87% (v1) → **93–97% (v2, two runs: 28/30 and 29/30)**, with 0% invalid answers. The LLM is not fully deterministic even at temperature 0. Zero-shot reached 90% but invented non-existent class names 27% of the time.

## Responsible use (U5)
- **Misuse note**: outputs are 128 px synthetic images of generic ImageNet objects. They must not be presented as real photographs, used to mislead, or used to depict real people. ImageNet contains weapon classes. They are labelled when generated and blocked when combined with harmful-intent wording.
- **Class imbalance and bias**: ImageNet-1k has about 120 dog breeds but only 3 person-like classes (groom, scuba diver, ballplayer), and its photos skew towards Western, online contexts. Prompts about people ("a doctor", "a farmer in India") are forced onto unrelated objects (see the people-bias probe in the results). "Nearest class" is not the same as "what you asked for".
- **Hallucination**: the zero-shot LLM is measured for invalid class names, and the app checks every LLM answer against the label list.
- **Licences**: BigGAN code from `pytorch-pretrained-biggan` (MIT), with weights converted from DeepMind's TF-Hub release (Apache-2.0). ImageNet is for non-commercial research only. CLIP is MIT, all-MiniLM-L6-v2 is Apache-2.0, sd-turbo (comparison only) is under the Stability AI non-commercial research licence. **Use this app for education and research only.**

## Limits
128 px resolution, only 1000 fixed classes, attributes and compositions are ignored ("a *blue* dog", "a cat *riding a bicycle*"), and abstract prompts ("freedom") get a low-confidence warning.
