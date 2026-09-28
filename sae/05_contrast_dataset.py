"""Experiment 5 — find a *concept* feature with a contrastive dataset, not a sentence pair.

    python sae/05_contrast_dataset.py --model gemma-2-2b
    python sae/05_contrast_dataset.py --model gpt2 --dataset sae/datasets/doom.json --top 12

03_contrast compares two texts, so it mostly rediscovers the words you swapped ("empty" vs
"full"). This script compares two *sets* of texts that say the same thing in deliberately
different words. A feature that survives averaging over varied phrasings is a candidate concept;
a feature that only fires on one wording averages away.

Ranking (per feature, over each set):
  mean    mean of per-text peak activation  — how strongly it responds
  hits    fraction of texts in the set where it fires at all — how *general* it is
  score   mean(A) - mean(B), restricted to features with hits(A) >= --min-hits

The hits column is what separates a concept from a word: a lexical feature has high mean and low
hits (one text carries it), a concept has both. Each row also prints the Neuronpedia label, the
logit-lens tokens it promotes, and its density, so token detectors are visible at a glance.
"""
import argparse
import json
from pathlib import Path

import torch

from common import add_model_args, describe, feature_acts, load, neuronpedia_url, save_result

# Same claim, deliberately different vocabulary in every sentence: no shared content words to latch onto.
DEFAULT_DATASET = {
    "name": "urban-decline",
    "a": [
        "Downtown office towers stand vacant and the tax base is shrinking.",
        "Storefronts on the main commercial strip have been boarded up for years.",
        "Middle-class families keep moving out to the suburbs, and enrollment is falling.",
        "The transit system cut service again after fare revenue dropped.",
        "Property values in the core have not recovered since the recession.",
        "City hall is facing a budget deficit and is discussing layoffs.",
        "Sidewalks near the station feel unsafe after dark and few people linger.",
        "Investors have written off the central business district entirely.",
        "Once-grand hotels now sit shuttered behind plywood.",
        "Each year the census counts fewer residents than the year before.",
        "The convention center sits idle most months of the year.",
        "Local news covers another factory closing and the jobs that left with it.",
    ],
    "b": [
        "Downtown office towers are fully leased and the tax base keeps growing.",
        "Storefronts on the main commercial strip have a waiting list of tenants.",
        "Middle-class families keep moving in from the suburbs, and enrollment is rising.",
        "The transit system added service after fare revenue climbed.",
        "Property values in the core have doubled since the recession.",
        "City hall is running a surplus and is discussing new hires.",
        "Sidewalks near the station stay busy well past midnight.",
        "Investors are competing for anything available in the central business district.",
        "Historic hotels have been restored and are booked months ahead.",
        "Each year the census counts more residents than the year before.",
        "The convention center is booked solid through next season.",
        "Local news covers another plant opening and the jobs that came with it.",
    ],
}


def set_stats(model, sae, texts):
    """Return (mean_peak, hit_rate) over a list of texts, each [d_sae]."""
    peaks = []
    for t in texts:
        _, acts = feature_acts(model, sae, t)
        peaks.append(acts.max(dim=0).values)
    peaks = torch.stack(peaks)                      # [n_texts, d_sae]
    return peaks.mean(0), (peaks > 0).float().mean(0)


def show(sae, side, score, mean_a, hits_a, mean_b, hits_b, top, min_hits, labels):
    print(f"\n=== {side} ===")
    print(f"{'feat':>6} {'score':>7} {'mean':>7} {'hits':>5}  {'other':>6}")
    rows = []
    ranked = torch.argsort(score, descending=True)
    for idx in ranked:
        i = int(idx)
        if len(rows) >= top or score[i] <= 0:
            break
        if hits_a[i] < min_hits:                    # fires in too few texts to be a general concept
            continue
        rows.append({"feature": i, "score": round(score[i].item(), 2),
                     "mean": round(mean_a[i].item(), 2), "hits": round(hits_a[i].item(), 2),
                     "other_mean": round(mean_b[i].item(), 2), "other_hits": round(hits_b[i].item(), 2),
                     "info": describe(sae, i, labels)})
        r = rows[-1]
        print(f"{i:>6} {r['score']:>7.1f} {r['mean']:>7.1f} {r['hits']:>5.2f}  {r['other_mean']:>6.1f}")
        print(f"       {r['info'] or neuronpedia_url(sae, i)}")
    return rows


def main():
    p = argparse.ArgumentParser()
    add_model_args(p)
    p.add_argument("--dataset", type=Path, default=None, help="JSON with {name, a: [...], b: [...]}")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--min-hits", type=float, default=0.75,
                   help="require a feature to fire in this fraction of the set (0.75 = 9 of 12)")
    p.add_argument("--no-labels", action="store_true")
    args = p.parse_args()

    data = json.loads(args.dataset.read_text()) if args.dataset else DEFAULT_DATASET
    model, sae = load(args.layer, args.model)
    print(f"\ndataset {data['name']!r}: {len(data['a'])} A-texts vs {len(data['b'])} B-texts, "
          f"min-hits {args.min_hits}")

    mean_a, hits_a = set_stats(model, sae, data["a"])
    mean_b, hits_b = set_stats(model, sae, data["b"])

    a_rows = show(sae, "A (decline): consistent across phrasings", mean_a - mean_b,
                  mean_a, hits_a, mean_b, hits_b, args.top, args.min_hits, not args.no_labels)
    b_rows = show(sae, "B (growth): consistent across phrasings", mean_b - mean_a,
                  mean_b, hits_b, mean_a, hits_a, args.top, args.min_hits, not args.no_labels)

    save_result("contrast_dataset", {"model": args.model, "layer": sae.cfg.metadata.hook_name,
                                     "dataset": data, "min_hits": args.min_hits,
                                     "stronger_in_a": a_rows, "stronger_in_b": b_rows})


if __name__ == "__main__":
    main()
