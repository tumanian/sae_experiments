"""Shared helpers for the SAE experiments.

Everything here is intentionally small: load a model + SAE once, turn text into
per-token feature activations, and (optionally) look up a human-readable label
for a feature on Neuronpedia.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

import warnings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unsupported ops fall back to CPU on Apple Silicon
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")    # verified: MPS matches CPU to ~1e-4 for gpt2-small
warnings.filterwarnings("ignore", message="(?s).*model_from_pretrained_kwargs")  # we already load the model correctly

import torch
from sae_lens import SAE, HookedSAETransformer

# name -> (TransformerLens model, sae_lens release, sae_id template, default layer, dtype)
PRESETS = {
    "gpt2":       ("gpt2-small",  "gpt2-small-res-jb",               "blocks.{L}.hook_resid_pre",       8,  "float32"),
    "gemma-2-2b": ("gemma-2-2b",  "gemma-scope-2b-pt-res-canonical", "layer_{L}/width_16k/canonical",   12, "bfloat16"),
}
DEFAULT_MODEL = os.environ.get("SAE_MODEL", "gpt2")
HERE = Path(__file__).parent
CACHE_DIR = HERE / ".cache" / "neuronpedia"
RESULTS_DIR = HERE / "results"


def device() -> str:
    """CUDA if present (cloud GPU box), else CPU. On a Mac, CPU is the default: gpt2-small is
    ~100ms/forward there and bit-reproducible; SAE_DEVICE=mps opts in (only ~20% faster on M1)."""
    if "SAE_DEVICE" in os.environ:
        return os.environ["SAE_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(layer: int | None = None, model_name: str = DEFAULT_MODEL):
    """Load a model and its residual-stream SAE for `layer` (preset default if None)."""
    tl_name, release, sae_id_tpl, default_layer, dtype = PRESETS[model_name]
    layer = default_layer if layer is None else layer
    dev = device()
    t0 = time.time()
    sae = SAE.from_pretrained(release=release, sae_id=sae_id_tpl.format(L=layer), device=dev)
    # load the model exactly the way the SAE was trained on it (no extra weight processing)
    model = HookedSAETransformer.from_pretrained_no_processing(
        tl_name, device=dev, dtype=getattr(torch, dtype), **(sae.cfg.metadata.model_from_pretrained_kwargs or {}))
    print(f"[load] {tl_name} + {release}/layer{layer} on {dev} "
          f"({sae.cfg.d_sae} features, {dtype}) in {time.time() - t0:.1f}s")
    return model, sae


def add_model_args(parser):
    """Shared CLI flags: --model preset and --layer (None = preset default)."""
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(PRESETS))
    parser.add_argument("--layer", type=int, default=None)


def hook_name(sae) -> str:
    return sae.cfg.metadata.hook_name


def feature_acts(model, sae, text: str):
    """Return (token_strings, acts) where acts is [seq, d_sae] of SAE feature activations."""
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens, names_filter=hook_name(sae))
    resid = cache[hook_name(sae)][0]            # [seq, d_model]
    acts = sae.encode(resid.to(sae.dtype))      # [seq, d_sae]  (model may be bf16, SAE is fp32)
    return model.to_str_tokens(tokens[0]), acts.detach()


def neuronpedia_url(sae, feature: int) -> str:
    return f"https://www.neuronpedia.org/{sae.cfg.metadata.neuronpedia_id}/{feature}"


def neuronpedia_label(sae, feature: int) -> str:
    """Fetch the top auto-interp explanation for a feature (cached on disk). Empty string on failure."""
    np_id = sae.cfg.metadata.neuronpedia_id            # e.g. "gpt2-small/8-res-jb"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{np_id.replace('/', '-')}-{feature}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
    else:
        url = f"https://www.neuronpedia.org/api/feature/{np_id}/{feature}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sae-experiments"})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.load(r)
            cache_file.write_text(json.dumps(data))
        except Exception:
            return ""
    exps = data.get("explanations") or []
    return exps[0].get("description", "") if exps else ""


def save_result(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {path.relative_to(HERE.parent)}")
    return path
