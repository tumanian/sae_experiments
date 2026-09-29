# SAE experiments

Small, local sparse-autoencoder experiments on GPT-2 small. Runs on an M1 in seconds.

## Setup (once)

```bash
python3.11 -m venv .venv && .venv/bin/pip install sae-lens transformer-lens torch
```

First run downloads GPT-2 small (~500 MB) and one SAE (~100 MB) into the HF cache.
Runs on CPU by default (~100 ms per forward pass, fully reproducible). `SAE_DEVICE=mps`
opts into the GPU — verified to match CPU to ~1e-4, but only ~20% faster for a model this small.

## The pieces

- **Model**: `gpt2-small` (12 layers, d_model=768), via TransformerLens.
- **SAE**: `gpt2-small-res-jb` — one SAE per layer trained on the residual stream
  (`blocks.L.hook_resid_pre`), 24,576 features each. Layer 8 is the default; it's
  mid-network where features are fairly abstract.
- **Labels**: [Neuronpedia](https://www.neuronpedia.org/gpt2-small) hosts auto-interp
  explanations for every feature. Scripts fetch and cache them so you can eyeball
  what a feature "means". Pass `--no-labels` to skip (offline / faster).

## The loop

1. **Inspect** — what fires on some text?
   ```bash
   .venv/bin/python sae/01_inspect.py --text "The Golden Gate Bridge is in San Francisco"
   ```
2. **Contrast** — what separates two texts? (a cheap way to find a candidate feature)
   ```bash
   .venv/bin/python sae/03_contrast.py --a "I love this, it's wonderful" --b "I hate this, it's awful"
   ```
3. **Steer** — clamp that feature during generation and see if behaviour changes.
   ```bash
   .venv/bin/python sae/02_steer.py --feature <id> --strength 4 8 16 --prompt "Today I"
   ```

4. **Steer with several features and score it** — the real experiment. `--scale` is a master
   volume knob; outputs are scored for target vocabulary, coherence (nll under the unsteered
   model), and closed-loop feature activation.
   ```bash
   .venv/bin/python sae/04_steer_multi.py --edit 9752:25 --edit 23807:25 --edit 4855:20 --edit 1032:20 \
       --prompt "San Francisco is a city" --scale 0 0.6 0.8 1 1.2 1.4 --samples 6 --quiet
   ```

Every run writes a JSON to `sae/results/` so you keep a log of what you tried.

## Worked example (what the first run found)

- `01_inspect` on *"The Golden Gate Bridge is in San Francisco"*: 257 of 24,576 features
  alive. Top hits: `20677` "cities like San Francisco" on `' San'`, `17608` "bridges" on
  `' Bridge'`, `8198` "gates" on `' Gate'`.
- `03_contrast` love-vs-hate reviews: `11978` "expressions of affection or strong liking"
  is +26.7 on the positive text; `8341` "words related to hate" is +23.4 on the negative one.
- `02_steer` feature `11978` on *"The weather today is"*: strength 10 → no change,
  20 → "so much better than it was", 40 → "so nice", "the best", "so glad", "perfect opportunity".
  So this feature is *causal*, not just a correlate — and ~20–40 is the useful range at layer 8.

- `04_steer_multi` "SF doom loop" (SF + drugs + fleeing + leaving, layer 8): doom vocabulary
  goes 0.4% → 4% at scale 0.6–1.0 with nll 2.3 → 2.8 (still fluent); at 1.4+ it collapses into
  function-word soup ("for the, the, to, and") — which 3-gram repetition does *not* catch, nll does.
  Closed-loop shows only the drugs feature re-activating in the outputs.
- **Ablation** of that steer (8 samples/row, scale 1): drugs-only gives 4.7% doom words at nll 2.49
  — matches the full four-feature steer (4.2%) at half the fluency cost. Dropping the SF feature
  *raises* doom words to 6.5% (the prompt already says SF; the topic feature is a passenger that
  competes with the frame). Fleeing + leaving without drugs: 2.8% at nll 2.98 — weak, expensive.
  Revised recipe: `--edit 23807:25` alone.
- Lesson from feature `3917`: constant activation (20.0 in every context) + a single dominant
  direct-logit token (`ctuary`) = token detector, not a concept. Ranking by peak activation
  selects for these at every layer; use contrast to find concepts.

- **Gemma 2 2B (Gemma Scope, layer 12, 16k)** on an A10G: same doom-vs-thriving contrast gives the
  same lexical answer ("leave", "empty", "drugs" vs "tourism", "full", "coffee") - so the lexical
  result was the method (word-swapped parallel paragraphs, peak-diff ranking), not GPT-2's size.
  Gotcha: Gemma 2 has a massive-activation outlier on the *first content token* (resid norm ~700 vs
  ~130), not just BOS; `feature_acts` now masks any position >4x the median norm.

- **Gemma has an urban-decline concept; GPT-2 doesn't.** `05_contrast_dataset` (12 varied decline
  sentences vs 12 matched growth ones, hit-rate floor 0.75) on Gemma layer 12 finds `7033`
  "economic crises" (promotes recession/bankruptcy/downturn/insolvency) and `15598` "decline or loss
  over time" (promotes Decline/lost/losing/Gone). Label and logit lens agree, and the promoted words
  appear nowhere in the dataset - it responds to the situation, not the wording. GPT-2's best was a
  thin "closures" feature. The growth side has no consistent feature in either model.
- Steering Gemma with `7033 + 15598` at 40 x scale 1.0-1.5 on "San Francisco is a city": baseline
  writes tourism and food trucks, steered writes "on the brink of financial collapse", "the
  once-great city is no longer the city of the future", "can't afford its famous cable cars" - at
  nll 2.0-2.1 vs 1.5 baseline, i.e. still fluent. Closed-loop activation rises 0.0 -> 4.0, so the
  generated text really does express the features we pushed (GPT-2 never showed this).
- Caveat: the doom *lexicon* score barely moves on Gemma (0.018 vs GPT-2's 0.047) because the
  lexicon was written for GPT-2's crude vocabulary (tents, drugs, homeless) and Gemma writes
  "financial collapse", "downturn", "bail them out". The closed-loop metric caught what the keyword
  metric missed - keyword scores are model-specific, feature scores aren't.
- The density column earns its place: Gemma feature `1041` tops the raw score table (11.6, hits 1.00)
  but has density 42.7% and a logit lens of 18th-century long-s text. Dense nuisance feature,
  disqualified on sight.

- **Is there a "more / less" meta-direction? No — it's per-concept.** `06_meta_direction` takes
  matched instruction pairs on gemma-2b-it layer 12, differences the residual at the last prompt
  position, and compares those directions across attributes.
  - Method validity: "be more negative" vs the token-disjoint "make it harsher" agree at **+0.55**
    mean across six attributes, so the extracted directions are real and reproducible.
  - Across *different* attributes, token-disjoint: mean cosine **-0.057**. No shared direction.
  - `negative` and `optimistic` **anti**-correlate at **-0.58** — the opposite of what a shared
    "crank it up" direction would produce.
  - Spectrum is flat (35/27/20/8/6/4%), and the top components are *semantic*, not degree:
    PC1 loads on formal+technical (register), PC2 splits negative vs optimistic (valence).
  - Confound worth knowing: with "more"/"less" left in the prompts, a deliberately mismatched
    `shuffled` pair scored +0.44..+0.58, as high as the real pairs — that shared variance is the
    literal degree *token*, not intensity. Token-disjoint paraphrases are required to test this.
  - Upshot: intensity lives *inside* a feature (its activation magnitude), which is what the
    `--scale` knob in `04_steer_multi` already assumes. "Be more X" is a bigger X, not X plus a
    separate amplifier.

## Things to try next

- Change `--layer` (0–11). Early layers = token/syntax features, later = semantic.
- Steer with negative strength to suppress a feature.
- Find a feature in `01_inspect.py`, then check whether it steers — many "clean"
  features don't do much causally, which is itself a finding.
- Look at density: how many features are alive per token? (`n_alive` in the output.)
