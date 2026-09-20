#!/usr/bin/env bash
# Sync sae/ to the box, run a script there, pull results back.
#   cloud/aws/run.sh sae/01_inspect.py --text "The Golden Gate Bridge"
#   cloud/aws/run.sh sae/02_steer.py --feature 11978 --strength 20 40
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
IP=$(cat "$HERE/.ip")
SSH="ssh -i $HOME/.ssh/sae-cpu.pem -o StrictHostKeyChecking=accept-new ubuntu@$IP"

until $SSH test -f .bootstrap-done 2>/dev/null; do echo "waiting for bootstrap..."; sleep 15; done

rsync -az -e "ssh -i $HOME/.ssh/sae-cpu.pem" --exclude .cache --exclude results --exclude __pycache__ "$ROOT/sae/" "ubuntu@$IP:sae/"
# HF token (for gated models like Gemma): from $HF_TOKEN or the local `hf auth login`; written to the box's HF cache over stdin
TOKEN=${HF_TOKEN:-$(hf auth token 2>/dev/null || true)}
if [ -n "$TOKEN" ]; then printf '%s' "$TOKEN" | $SSH "mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token"; fi
$SSH "cd ~ && ~/venv/bin/python $(printf '%q ' "$@")"
rsync -az -e "ssh -i $HOME/.ssh/sae-cpu.pem" "ubuntu@$IP:sae/results/" "$ROOT/sae/results/"
echo "[synced] results -> sae/results/"
