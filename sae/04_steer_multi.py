"""Experiment 4 — steer with several features at once, sweep the strength, and *score* the outputs.

    python sae/04_steer_multi.py --edit 9752:25 --edit 23807:25 --edit 4855:20 --edit 1032:20 \
        --prompt "San Francisco is a city" --scale 0 0.5 1 1.5 2 --samples 4

Each --edit is FEATURE:STRENGTH. --scale multiplies all strengths (0 = unsteered baseline).
For every scale we generate N samples and report three numbers:

  lexicon   fraction of generated words that hit the target lexicon (--lexicon, comma-separated)
  closed    mean activation of the target features when the *unsteered* model re-reads the output.
            "Did we actually write text that expresses these features?" — the honest check.
  nll       per-token loss of the output under the *unsteered* model. Baseline prose ~2.3;
            >3.5 means it has degraded into word soup. Coherence guard. (repeat = 3-gram
            repetition rate, kept as a secondary check — it misses the soup failure mode.)

The lexicon/closed-loop vs nll curve across scales is the result.
"""
import argparse
import re

import torch

from common import feature_acts, hook_name, load, neuronpedia_label, save_result

DOOM_LEXICON = ("homeless,homelessness,empty,vacant,vacancy,fled,flee,fleeing,leaving,exodus,crime,drug,drugs,"
                "overdose,tent,tents,encampment,decline,decay,collapse,collapsing,crisis,closed,closing,shuttered,"
                "abandoned,blight,dying,dead,broken,poverty,violence,theft,unsafe,filth,needles,bankrupt,layoffs")


def parse_edit(s):
    f, v = s.split(":")
    return int(f), float(v)


def repeat_rate(text, n=3):
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def lexicon_rate(text, lexicon):
    words = re.findall(r"[a-z]+", text.lower())
    return 0.0 if not words else sum(w in lexicon for w in words) / len(words)


def nll(model, text):
    """Per-token cross-entropy of `text` under the unsteered model."""
    toks = model.to_tokens(text)
    logits = model(toks)[0, :-1]
    return torch.nn.functional.cross_entropy(logits, toks[0, 1:]).item()


def closed_loop(model, sae, generated, n_prompt_tokens, features):
    """Re-encode the generated text with no steering; mean activation of each target feature on new tokens."""
    _, acts = feature_acts(model, sae, generated)
    new = acts[n_prompt_tokens:]
    return {f: round(new[:, f].mean().item(), 2) for f in features}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edit", action="append", required=True, help="FEATURE:STRENGTH, repeatable")
    p.add_argument("--prompt", default="San Francisco is a city")
    p.add_argument("--scale", type=float, nargs="+", default=[0, 1])
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--tokens", type=int, default=50)
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--lexicon", default=DOOM_LEXICON)
    p.add_argument("--quiet", action="store_true", help="only print the summary table")
    args = p.parse_args()

    edits = dict(parse_edit(e) for e in args.edit)
    lexicon = set(args.lexicon.split(","))
    model, sae = load(args.layer)
    n_prompt = model.to_tokens(args.prompt).shape[1]

    print(f"\nprompt: {args.prompt!r}")
    for f, s in edits.items():
        print(f"  edit {f:>6} x{s:<5g} {neuronpedia_label(sae, f)[:80]}")
    base_vec = sum(s * sae.W_dec[f] for f, s in edits.items()).detach()

    runs = {}
    for scale in args.scale:
        vec = scale * base_vec

        def steer(resid, hook):
            resid[:, :, :] += vec
            return resid

        samples = []
        with model.hooks(fwd_hooks=[(hook_name(sae), steer)] if scale else []):
            for seed in range(args.samples):
                torch.manual_seed(seed)
                samples.append(model.generate(args.prompt, max_new_tokens=args.tokens, temperature=0.7, verbose=False))

        scored = []
        for text in samples:                       # scoring runs with hooks removed
            scored.append({"text": text,
                           "lexicon": lexicon_rate(text[len(args.prompt):], lexicon),
                           "nll": nll(model, text),
                           "repeat": repeat_rate(text),
                           "closed": closed_loop(model, sae, text, n_prompt, edits)})
        runs[scale] = scored
        if not args.quiet:
            print(f"\n--- scale {scale} ---")
            for i, s in enumerate(scored):
                print(f"[{i}] lex={s['lexicon']:.2f} nll={s['nll']:.1f}  {s['text'].replace(chr(10), ' ')}\n")

    def mean(xs):
        return sum(xs) / len(xs)

    print(f"\n{'scale':>6} {'lexicon':>8} {'nll':>5} {'repeat':>7}  closed-loop mean act per feature")
    for scale, scored in runs.items():
        closed = {f: mean([s["closed"][f] for s in scored]) for f in edits}
        print(f"{scale:>6g} {mean([s['lexicon'] for s in scored]):>8.3f} {mean([s['nll'] for s in scored]):>5.2f} "
              f"{mean([s['repeat'] for s in scored]):>7.3f}  "
              + "  ".join(f"{f}={v:.1f}" for f, v in closed.items()))

    save_result("steer_multi", {"prompt": args.prompt, "layer": args.layer, "edits": edits,
                                "runs": {str(k): v for k, v in runs.items()}})


if __name__ == "__main__":
    main()
