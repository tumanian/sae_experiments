"""Experiment 1 — which SAE features fire on a piece of text?

    python sae/01_inspect.py --text "The Golden Gate Bridge is in San Francisco"
    python sae/01_inspect.py --text "..." --layer 6 --top 15 --no-labels

For each of the top-N features (ranked by peak activation, ignoring the BOS
token) we print the tokens it fires on and, if available, the Neuronpedia
auto-interp label so you can sanity-check what the feature "means".
"""
import argparse

import torch

from common import add_model_args, feature_acts, hook_name, load, neuronpedia_label, neuronpedia_url, save_result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    add_model_args(p)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--no-labels", action="store_true", help="skip Neuronpedia lookups")
    args = p.parse_args()

    model, sae = load(args.layer, args.model)
    tokens, acts = feature_acts(model, sae, args.text)
    acts[0] = 0  # BOS position has huge, uninformative activations

    peak, _ = acts.max(dim=0)                       # [d_sae]
    n_alive = int((peak > 0).sum())
    print(f"\n{n_alive} / {acts.shape[1]} features active on {len(tokens) - 1} tokens\n")

    top = torch.topk(peak, args.top)
    rows = []
    for val, idx in zip(top.values.tolist(), top.indices.tolist()):
        label = "" if args.no_labels else neuronpedia_label(sae, idx)
        col = acts[:, idx]
        firing = [(tokens[i], round(col[i].item(), 1)) for i in torch.argsort(col, descending=True)[:5] if col[i] > 0.05]
        rows.append({"feature": idx, "peak": round(val, 2), "label": label, "tokens": firing})
        print(f"feature {idx:>6}  peak {val:6.1f}  {label or '(no label)'}")
        print(f"        fires on: {', '.join(f'{t!r}={a}' for t, a in firing)}")
        print(f"        {neuronpedia_url(sae, idx)}\n")

    save_result("inspect", {"text": args.text, "model": args.model, "layer": hook_name(sae), "n_alive": n_alive, "top": rows})


if __name__ == "__main__":
    main()
