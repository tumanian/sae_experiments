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

from common import add_model_args, chat_wrap, hook_name, load, load_model, save_result

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

# For the REPRESENTATION test: 12 (nonce, real) pairs for the same behaviour. No generation, so no
# seeds or scorers are needed and we can afford many more pairs. The nonce words are arbitrary --
# deliberately giving no morphological hint of the meaning, so the definition is the only source.
REPR_PAIRS = [
    ("ends every sentence with an exclamation mark",        "kravish",  "exclamatory"),
    ("includes a number in every sentence",                 "zibbly",   "quantitative"),
    ("begins every sentence with a colour word",            "blorpy",   "chromatic"),
    ("ends every sentence with a question mark",            "quandric", "interrogative"),
    ("keeps every sentence under six words",                "thrimble", "terse"),
    ("makes every sentence at least thirty words long",     "vosker",   "verbose"),
    ("uses old-fashioned words such as thee and hence",     "glimf",    "archaic"),
    ("presents the content as numbered items",              "narptic",  "enumerated"),
    ("describes things by comparison to other things",      "welkish",  "figurative"),
    ("uses I and me throughout",                            "drupine",  "personal"),
    ("writes every word in capital letters",                "fanzo",    "capitalised"),
    ("starts several words in a row with the same letter",  "mibbly",   "alliterative"),
]

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


def repr_test(model_name, layer):
    """Representation-level version: does a DEFINED NONCE word get the same (more - less) direction
    as its real-English twin? Generation-free and deterministic, so it measures what the model
    understood rather than whether a 2B model managed to comply.
    """
    model, sae = load(layer, model_name)
    hook = hook_name(sae)

    def resid(text):
        toks = model.to_tokens(chat_wrap(text, model_name))
        _, cache = model.run_with_cache(toks, names_filter=hook)
        return cache[hook][0, -1].float()

    def prompt(definition, adj, degree):
        return (f"{adj.capitalize()} writing {definition}. "
                f"Write a paragraph about San Francisco, but {degree} {adj}.")

    def delta(definition, adj):
        d = resid(prompt(definition, adj, "more")) - resid(prompt(definition, adj, "less"))
        return d / d.norm()

    print(f"\nrepresentation-level nonce test at {hook}, {len(REPR_PAIRS)} pairs\n")
    nonce = torch.stack([delta(d, n) for d, n, _ in REPR_PAIRS])
    real = torch.stack([delta(d, r) for d, _, r in REPR_PAIRS])

    # These prompts are near-identical (one word differs), so every delta carries a large shared
    # "an instruction word changed here" component that inflates ALL cosines. Remove it by centering
    # over the whole set, then ask whether twins are *specifically* aligned.
    allv = torch.cat([nonce, real])
    allv = allv - allv.mean(0, keepdim=True)
    allv = allv / allv.norm(dim=1, keepdim=True)
    n = len(REPR_PAIRS)
    N, R = allv[:n], allv[n:]

    twin = [(N[i] @ R[i]).item() for i in range(n)]                       # same behaviour, nonce vs real
    cross = [(N[i] @ R[j]).item() for i in range(n) for j in range(n) if i != j]  # mismatched

    print(f"{'behaviour':>14} {'nonce':>10} {'real':>14} {'twin cos':>9}")
    for (d, nw, rw), t in zip(REPR_PAIRS, twin):
        print(f"{d.split()[0] + '..':>14} {nw:>10} {rw:>14} {t:>+9.2f}")

    import statistics as stat
    tm, cm = stat.mean(twin), stat.mean(cross)
    tci = 1.96 * stat.stdev(twin) / len(twin) ** 0.5
    cci = 1.96 * stat.stdev(cross) / len(cross) ** 0.5
    print(f"\ntwin  (nonce vs its own real twin) : {tm:+.3f} +- {tci:.3f}   n={len(twin)}")
    print(f"cross (nonce vs a DIFFERENT real)  : {cm:+.3f} +- {cci:.3f}   n={len(cross)}")
    print(f"separation                         : {tm - cm:+.3f}")
    print(f"twins above the cross mean         : {sum(t > cm for t in twin)}/{len(twin)}")
    print("\n--> twin >> cross  => the model composed the nonce word's meaning from its definition")
    print("    and applied the same degree operation to it (H-abstract).")
    print("    twin ~ cross    => the direction needs the real word's corpus history (H-local).")

    save_result("nonce_repr", {"model": model_name, "layer": hook, "n_pairs": n,
                               "pairs": [[d, nw, rw] for d, nw, rw in REPR_PAIRS],
                               "twin": twin, "cross": cross,
                               "twin_mean": tm, "cross_mean": cm, "separation": tm - cm})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-2b-it")
    p.add_argument("--repr", action="store_true",
                   help="representation-level test (no generation): cos(nonce delta, real delta)")
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--tokens", type=int, default=70)
    p.add_argument("--show", action="store_true", help="print every generation")
    args = p.parse_args()

    if args.repr:
        repr_test(args.model, args.layer)
        return

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
