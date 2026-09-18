"""Experiment 3 — which features separate text A from text B?

    python sae/03_contrast.py --a "The cat sat on the mat." --b "The dog sat on the mat."

Ranks features by (peak activation on A) - (peak activation on B) in both
directions. Good for finding a feature to steer with in experiment 2.
"""
import argparse

import torch

from common import feature_acts, load, neuronpedia_label, neuronpedia_url, save_result


def peaks(model, sae, text):
    _, acts = feature_acts(model, sae, text)
    acts[0] = 0
    return acts.max(dim=0).values


def show(sae, diff, title, top, labels):
    print(f"\n=== {title} ===")
    rows = []
    for val, idx in zip(*[t.tolist() for t in torch.topk(diff, top)]):
        if val <= 0:
            break
        label = neuronpedia_label(sae, idx) if labels else ""
        rows.append({"feature": idx, "delta": round(val, 2), "label": label})
        print(f"feature {idx:>6}  +{val:6.1f}  {label or '(no label)'}  {neuronpedia_url(sae, idx)}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--no-labels", action="store_true")
    args = p.parse_args()

    model, sae = load(args.layer)
    pa, pb = peaks(model, sae, args.a), peaks(model, sae, args.b)
    a_only = show(sae, pa - pb, "stronger in A", args.top, not args.no_labels)
    b_only = show(sae, pb - pa, "stronger in B", args.top, not args.no_labels)
    save_result("contrast", {"a": args.a, "b": args.b, "layer": args.layer,
                             "stronger_in_a": a_only, "stronger_in_b": b_only})


if __name__ == "__main__":
    main()
