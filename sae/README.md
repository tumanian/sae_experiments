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

## Things to try next

- Change `--layer` (0–11). Early layers = token/syntax features, later = semantic.
- Steer with negative strength to suppress a feature.
- Find a feature in `01_inspect.py`, then check whether it steers — many "clean"
  features don't do much causally, which is itself a finding.
- Look at density: how many features are alive per token? (`n_alive` in the output.)
