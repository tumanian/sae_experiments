#!/bin/bash
# EC2 user-data: runs once as root on first boot. Installs a CPU-only torch stack for the ubuntu user.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -q && apt-get install -yq python3-venv python3-pip rsync
sudo -u ubuntu bash -c '
  cd ~ && python3 -m venv venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # CPU wheel: ~200MB, not the 3GB CUDA one
  ~/venv/bin/pip install -q sae-lens==6.51.0 transformer-lens==3.9.0
'
touch /home/ubuntu/.bootstrap-done
