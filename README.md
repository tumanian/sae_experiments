# sae_experiments

Small sparse-autoencoder interpretability experiments, Goodfire-style: find a feature,
check what it means, test whether it's causal. Runs on a laptop CPU in seconds.

- [sae/README.md](sae/README.md) — the experiments and how to run them
- [cloud/aws/README.md](cloud/aws/README.md) — optional: run the same scripts on an EC2 box

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r sae/requirements.txt
.venv/bin/python sae/01_inspect.py --text "The Golden Gate Bridge is in San Francisco"
```
