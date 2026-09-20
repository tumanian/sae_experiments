"""Experiment 2 — steer generation by pushing on one SAE feature.

    python sae/02_steer.py --feature 12345 --strength 8 --prompt "I went for a walk and"

We add `strength * W_dec[feature]` to the residual stream at every position
during generation, and print the unsteered vs. steered completions side by
side. Too little strength = no effect; too much = word salad. Sweep it.
"""
import argparse

import torch

from common import add_model_args, hook_name, load, neuronpedia_label, save_result


def generate(model, prompt, max_new_tokens, seed):
    torch.manual_seed(seed)
    return model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.7, verbose=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", type=int, required=True)
    p.add_argument("--strength", type=float, nargs="+", default=[8.0], help="one or more coefficients to try")
    p.add_argument("--prompt", default="I went for a walk and")
    add_model_args(p)
    p.add_argument("--tokens", type=int, default=40)
    p.add_argument("--samples", type=int, default=2)
    args = p.parse_args()

    model, sae = load(args.layer, args.model)
    label = neuronpedia_label(sae, args.feature)
    print(f"\nfeature {args.feature}: {label or '(no label)'}\nprompt: {args.prompt!r}\n")

    direction = sae.W_dec[args.feature].detach()   # [d_model]

    outputs = {"baseline": [], "steered": {}}
    print("--- baseline ---")
    for s in range(args.samples):
        out = generate(model, args.prompt, args.tokens, seed=s)
        outputs["baseline"].append(out)
        print(f"[{s}] {out}\n")

    for strength in args.strength:
        def steer(resid, hook):
            resid[:, :, :] += (strength * direction).to(resid.dtype)
            return resid

        print(f"--- steered, strength={strength} ---")
        outputs["steered"][strength] = []
        with model.hooks(fwd_hooks=[(hook_name(sae), steer)]):
            for s in range(args.samples):
                out = generate(model, args.prompt, args.tokens, seed=s)
                outputs["steered"][strength].append(out)
                print(f"[{s}] {out}\n")

    save_result("steer", {"feature": args.feature, "label": label, "model": args.model, "layer": hook_name(sae),
                          "prompt": args.prompt, **outputs})


if __name__ == "__main__":
    main()
