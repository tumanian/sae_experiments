"""Experiment 7 — does "more X" work when X is a word the corpus has never seen?

    python sae/07_nonce_degree.py --model gemma-2b-it --samples 6

Two accounts of how "be more formal" works:
  H-local     the bigram "more formal" statistically precedes formal-looking text. A surface
              association, nothing reusable.
  H-abstract  "more" is a learned *operator* that turns up whatever attribute it is applied to.

Design: a matched pair of adjectives for the SAME target behaviour and the SAME metric, where the
only difference is whether the adjective exists in English.

    real   "Write it so it is exclamatory."     <- corpus has "more exclamatory"
    nonce  "Kravish writing ends every sentence with an exclamation mark. Write it so it is
            kravish."                            <- corpus has never seen "more kravish"

Both get the definition, so the model is never guessing what the word means; the only thing the
nonce condition lacks is any co-occurrence history for that word. Then compare compliance across
three degree conditions: `less` < plain < `more` is the signature of a working operator.

Compliance is counted with a regex, not judged, so the metric can't flatter the hypothesis.
"""
import argparse
import re

import torch

from common import chat_wrap, load_model, save_result

COLORS = r"(red|orange|yellow|green|blue|purple|violet|pink|brown|black|white|gray|grey|gold|silver|crimson|scarlet|azure|amber|teal)"


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) > 2]


def frac_exclaim(text):
    ss = sentences(text)
    return sum(s.endswith("!") for s in ss) / len(ss) if ss else 0.0


def frac_numeric(text):
    ss = sentences(text)
    pat = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million)\b", re.I)
    return sum(bool(pat.search(s)) for s in ss) / len(ss) if ss else 0.0


def frac_color_start(text):
    ss = sentences(text)
    pat = re.compile(r"^\W*" + COLORS, re.I)
    return sum(bool(pat.match(s)) for s in ss) / len(ss) if ss else 0.0


# metric name -> (definition of the behaviour, nonce adjective, real adjective, scorer)
ATTRS = {
    "exclaim": ("ends every sentence with an exclamation mark", "kravish",  "exclamatory", frac_exclaim),
    "numeric": ("includes a number in every sentence",          "zibbly",   "quantitative", frac_numeric),
    "color":   ("begins every sentence with a colour word",     "blorpy",   "chromatic",   frac_color_start),
}

# A partially-compliant example, so `more` has room to go up and `less` has room to go down.
SEEDS = {
    "exclaim": "Portland is a city of bridges! It rains often. The coffee is very good!",
    "numeric": "Portland has 12 bridges. The coffee is excellent. Roughly 650,000 people live there.",
    "color":   "Green moss covers the sidewalks. The coffee is excellent. Grey clouds sit over the river.",
}

TASK = "Now write three sentences about San Francisco in the same style{degree}."
DEGREES = {"less": ", but less {adj}", "plain": ", {adj}", "more": ", but more {adj}"}


def build_prompt(metric, kind, degree_key):
    definition, nonce, real, _ = ATTRS[metric]
    adj = nonce if kind == "nonce" else real
    # both conditions get the definition, so the ONLY difference is whether the adjective is a real
    # English word with co-occurrence history. The definition also cancels in the more-less gap.
    intro = f"{adj.capitalize()} writing {definition}. "
    degree = DEGREES[degree_key].format(adj=adj)
    return (f"{intro}Here is a passage that is somewhat {adj}:\n\n{SEEDS[metric]}\n\n"
            + TASK.format(degree=degree))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-2b-it")
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--tokens", type=int, default=70)
    p.add_argument("--show", action="store_true", help="print every generation")
    args = p.parse_args()

    model = load_model(args.model)
    results = {}

    for metric in ATTRS:
        scorer = ATTRS[metric][3]
        results[metric] = {}
        for kind in ("real", "nonce"):
            results[metric][kind] = {}
            for degree_key in DEGREES:
                prompt = build_prompt(metric, kind, degree_key)
                scores, texts = [], []
                for seed in range(args.samples):
                    torch.manual_seed(seed)
                    out = model.generate(chat_wrap(prompt, args.model), max_new_tokens=args.tokens,
                                         temperature=0.8, verbose=False)
                    gen = out[len(chat_wrap(prompt, args.model)):]
                    scores.append(scorer(gen))
                    texts.append(gen)
                results[metric][kind][degree_key] = {"mean": sum(scores) / len(scores),
                                                     "scores": scores, "texts": texts}
                if args.show:
                    print(f"\n--- {metric}/{kind}/{degree_key} (mean {sum(scores)/len(scores):.2f}) ---")
                    for t in texts[:2]:
                        print("   ", t.replace("\n", " ")[:170])

    print(f"\ncompliance rate ({args.samples} samples each), {args.model}\n")
    print(f"{'metric':>9} {'adj kind':>9} {'less':>7} {'plain':>7} {'more':>7}   {'more-less':>9}")
    summary = {}
    for metric in results:
        for kind in ("real", "nonce"):
            r = results[metric][kind]
            gap = r["more"]["mean"] - r["less"]["mean"]
            summary[f"{metric}/{kind}"] = gap
            print(f"{metric:>9} {kind:>9} {r['less']['mean']:>7.2f} {r['plain']['mean']:>7.2f} "
                  f"{r['more']['mean']:>7.2f}   {gap:>+9.2f}")

    # (more - less) conflates two effects. Report them separately: the interesting question is
    # whether `more` lifts compliance ABOVE the baseline, not just whether `less` lowers it.
    print("\nthe two contrasts, separated (mean over metrics):")
    for lbl, (x, y) in {"more vs plain": ("more", "plain"), "less vs plain": ("less", "plain")}.items():
        for kind in ("real", "nonce"):
            ds = [results[m][kind][x]["mean"] - results[m][kind][y]["mean"] for m in ATTRS]
            print(f"  {lbl:>13} {kind:>6}: " + " ".join(f"{d:+.2f}" for d in ds)
                  + f"   mean {sum(ds)/len(ds):+.3f}")

    real_gaps = [summary[f'{m}/real'] for m in ATTRS]
    nonce_gaps = [summary[f'{m}/nonce'] for m in ATTRS]
    print(f"\nmean (more - less) gap, real adjectives  : {sum(real_gaps)/len(real_gaps):+.3f}")
    print(f"mean (more - less) gap, nonce adjectives : {sum(nonce_gaps)/len(nonce_gaps):+.3f}")
    print("\n--> nonce gap ~ real gap  => 'more' is an operator, not a bigram statistic (H-abstract)")
    print("    nonce gap ~ 0          => the effect needs corpus history for that word (H-local)")
    print("    NB if plain compliance is ~0 or ~1 the test is saturated and says nothing.")

    save_result("nonce_degree", {"model": args.model, "samples": args.samples,
                                 "gaps": summary, "results": results})


if __name__ == "__main__":
    main()
