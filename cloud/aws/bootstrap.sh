#!/bin/bash
# EC2 user-data: runs once as root on first boot. Installs the torch stack for the ubuntu user.
# Picks the CUDA wheel if an NVIDIA GPU is present (Deep Learning AMI), else the small CPU wheel.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -q && apt-get install -yq python3-venv python3-pip rsync
if command -v nvidia-smi >/dev/null 2>&1; then TORCH_INDEX=https://download.pytorch.org/whl/cu128; else TORCH_INDEX=https://download.pytorch.org/whl/cpu; fi
sudo -u ubuntu bash -c "
  cd ~ && python3 -m venv venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q torch==2.14.0 --index-url $TORCH_INDEX
  ~/venv/bin/pip install -q sae-lens==6.51.0 transformer-lens==3.9.0
"
# swap: model loading peaks in CPU RAM (Gemma 2 2B needs ~15 GB just to load); g5.xlarge only has 16 GB
if ! swapon --show | grep -q swapfile; then
  fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
touch /home/ubuntu/.bootstrap-done
