"""Experiment 6 — is there a content-independent "more / less" direction?

    python sae/06_meta_direction.py --model gemma-2b-it

Prompting "be more X" behaves a bit like steering. Two competing accounts:

  (a) per-concept  — "more negative" just means a bigger activation of the negativity feature.
                     Nothing extra needed; intensity is already inside each feature.
  (b) meta-direction — there is a shared "crank it up" direction that modulates whatever else is
                     active, and each attribute contributes only its own content on top.

Test: for each attribute build a matched instruction pair differing in ONE word (more/less), take
the residual-stream difference at the last prompt position, and compare those difference vectors
ACROSS attributes by cosine similarity.

  (a) predicts cosines near 0 between different attributes.
  (b) predicts a consistently high off-diagonal block.

CONTROLS matter here, because two prompts differing in one token share a lot of trivial structure:
  same-pair     the same attribute, reworded — an upper bound on how similar "the same thing" looks
  non-degree    instruction pairs that are not about degree at all (English/French, list/paragraph).
                If degree-pairs resemble each other no more than they resemble these, any shared
                component is just "an instruction changed", not "intensity".
  shuffled      more/less labels swapped within a pair, so the sign is meaningless -> sanity floor.
"""
import argparse
import itertools

import torch

from common import INSTRUCT, add_model_args, hook_name, load, save_result

TEMPLATE = "Write a paragraph about San Francisco. Be {word} {attr}."

# attribute -> (more-word, less-word). Kept to a single differing word per pair.
DEGREE = {
    "negative":   ("more", "less"),
    "formal":     ("more", "less"),
    "technical":  ("more", "less"),
    "detailed":   ("more", "less"),
    "optimistic": ("more", "less"),
    "emotional":  ("more", "less"),
}

# TOKEN-DISJOINT paraphrases of the same six requests: no "more"/"less" anywhere, so a shared
# component here cannot be the literal degree word. This is the version that actually tests (b).
PARAPHRASE = {
    "negative":   ("Make it harsher.",            "Make it gentler."),
    "formal":     ("Write it as a legal brief.",  "Write it as a text to a friend."),
    "technical":  ("Write it for engineers.",     "Write it for children."),
    "detailed":   ("Spell out every specific.",   "Keep it to the gist."),
    "optimistic": ("Make it hopeful.",            "Make it bleak."),
    "emotional":  ("Make it heartfelt.",          "Make it clinical."),
}

# Same attribute, different wording: how similar should "the same instruction" look?
SAME_PAIR = {
    "negative2": ("Write a paragraph about San Francisco. Make it harsher.",
                  "Write a paragraph about San Francisco. Make it gentler."),
}

# Instruction pairs that are not about degree. The key control.
NON_DEGREE = {
    "language":  ("Write a paragraph about San Francisco. Write in English.",
                  "Write a paragraph about San Francisco. Write in French."),
    "format":    ("Write a paragraph about San Francisco. Use a bulleted list.",
                  "Write a paragraph about San Francisco. Use flowing prose."),
    "person":    ("Write a paragraph about San Francisco. Use the first person.",
                  "Write a paragraph about San Francisco. Use the third person."),
}


def chat(model, text: str, instruct: bool) -> str:
    """Wrap in the model's chat template so an IT model sees a real instruction."""
    if not instruct:
        return text
    return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"


def last_resid(model, sae, text: str, instruct: bool) -> torch.Tensor:
    """Residual stream at the LAST prompt position — where the model has read the whole instruction."""
    toks = model.to_tokens(chat(model, text, instruct))
    _, cache = model.run_with_cache(toks, names_filter=hook_name(sae))
    return cache[hook_name(sae)][0, -1].float()          # [d_model]


def delta(model, sae, more_text, less_text, instruct) -> torch.Tensor:
    d = last_resid(model, sae, more_text, instruct) - last_resid(model, sae, less_text, instruct)
    return d / d.norm()                                   # unit vector: only direction matters


def main():
    p = argparse.ArgumentParser()
    add_model_args(p)
    p.add_argument("--no-labels", action="store_true")
    args = p.parse_args()

    model, sae = load(args.layer, args.model)
    instruct = args.model in INSTRUCT
    if not instruct:
        print(f"[warn] {args.model} is a base model; it does not follow instructions. "
              f"Results are about prompt *text*, not obeyed instructions.")

    deltas = {}
    for attr, (more, less) in DEGREE.items():
        deltas[attr] = delta(model, sae,
                             TEMPLATE.format(word=more, attr=attr),
                             TEMPLATE.format(word=less, attr=attr), instruct)
    for name, (a, b) in {**SAME_PAIR, **NON_DEGREE}.items():
        deltas[name] = delta(model, sae, a, b, instruct)
    # sanity floor: a pair whose sign is meaningless
    deltas["shuffled"] = delta(model, sae,
                               TEMPLATE.format(word="more", attr="negative"),
                               TEMPLATE.format(word="less", attr="formal"), instruct)

    names = list(deltas)
    M = torch.stack([deltas[n] for n in names])
    cos = M @ M.T

    print(f"\ncosine similarity of (more - less) directions at {hook_name(sae)}\n")
    w = max(len(n) for n in names)
    print(" " * (w + 2) + "".join(f"{n[:7]:>8}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>{w}}  " + "".join(f"{cos[i, j]:>8.2f}" for j in range(len(names))))

    # --- token-disjoint replication: same six attributes, no more/less words ---
    para = {}
    for attr, (hi, lo) in PARAPHRASE.items():
        para[attr] = delta(model, sae,
                           f"Write a paragraph about San Francisco. {hi}",
                           f"Write a paragraph about San Francisco. {lo}", instruct)
    P = torch.stack([para[a] for a in PARAPHRASE])
    pcos = P @ P.T
    off = [pcos[i, j].item() for i in range(len(P)) for j in range(len(P)) if i != j]
    ps = torch.linalg.svdvals(P)
    pfrac = (ps[0] ** 2 / (ps ** 2).sum()).item()
    # does each paraphrase point the same way as its more/less twin?
    twin = {a: (deltas[a] @ para[a]).item() for a in PARAPHRASE}

    deg = list(DEGREE)
    nondeg = list(NON_DEGREE)
    def block(rows, cols, same=False):
        vals = [cos[names.index(a), names.index(b)].item()
                for a, b in itertools.product(rows, cols) if not (same and a == b)]
        return sum(vals) / len(vals), min(vals), max(vals)

    dd = block(deg, deg, same=True)
    dn = block(deg, nondeg)
    nn = block(nondeg, nondeg, same=True)
    print(f"\nmean cosine, degree x degree      : {dd[0]:+.3f}   (range {dd[1]:+.2f} .. {dd[2]:+.2f})")
    print(f"mean cosine, degree x non-degree  : {dn[0]:+.3f}   (range {dn[1]:+.2f} .. {dn[2]:+.2f})")
    print(f"mean cosine, non-degree x non-deg : {nn[0]:+.3f}   (range {nn[1]:+.2f} .. {nn[2]:+.2f})")
    print(f"\n--> a meta-direction requires degree x degree to clearly exceed degree x non-degree.")
    print(f"    difference: {dd[0] - dn[0]:+.3f}")

    # How much of the degree block is one shared direction? (top singular value of the degree deltas)
    D = torch.stack([deltas[a] for a in deg])
    s = torch.linalg.svdvals(D)
    frac = (s[0] ** 2 / (s ** 2).sum()).item()
    print(f"\ntop singular direction explains {frac * 100:.1f}% of the variance across the "
          f"{len(deg)} degree deltas")
    print(f"(1/{len(deg)} = {100 / len(deg):.1f}% would be no shared structure at all)")

    print(f"\n--- token-disjoint paraphrases (no 'more'/'less' in any prompt) ---")
    print(f"mean off-diagonal cosine : {sum(off) / len(off):+.3f}   "
          f"(range {min(off):+.2f} .. {max(off):+.2f})")
    print(f"top singular direction   : {pfrac * 100:.1f}% of variance  (floor {100 / len(P):.1f}%)")
    print(f"\nagreement with the more/less version of the same attribute:")
    for a, v in twin.items():
        print(f"  {a:>11}  {v:+.2f}")
    print(f"  {'mean':>11}  {sum(twin.values()) / len(twin):+.2f}")

    save_result("meta_direction", {
        "paraphrase_mean_offdiag": sum(off) / len(off),
        "paraphrase_top_singular_frac": pfrac,
        "paraphrase_twin_agreement": twin,
        "model": args.model, "layer": hook_name(sae), "instruct": instruct,
        "names": names, "cosine": cos.tolist(),
        "degree_x_degree": dd[0], "degree_x_nondegree": dn[0], "nondegree_x_nondegree": nn[0],
        "top_singular_frac": frac,
    })


if __name__ == "__main__":
    main()
