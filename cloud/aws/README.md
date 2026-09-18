# Run the SAE experiments on an AWS CPU box

GPT-2 small doesn't need a GPU, so this uses a plain `c7i.xlarge` (4 vCPU / 8 GB,
~$0.18/hr in us-east-1) — no GPU quota request needed. Ubuntu 24.04, CPU-only torch.

## One-time

1. Credentials: `aws configure` (access key, secret, region `us-east-1`). The IAM user
   needs EC2 + `ssm:GetParameter` (the `AmazonEC2FullAccess` policy plus SSM read is enough).
2. That's it — `up.sh` creates the key pair (`~/.ssh/sae-cpu.pem`) and a security group
   that only allows SSH from your current IP.

## Loop

```bash
cloud/aws/up.sh                                                    # ~1 min to boot, ~3 min bootstrap
cloud/aws/run.sh sae/01_inspect.py --text "The Golden Gate Bridge"  # syncs sae/, runs, pulls results back
cloud/aws/run.sh sae/02_steer.py --feature 11978 --strength 20 40
cloud/aws/down.sh                                                  # terminate — do this, it bills hourly
```

`run.sh` rsyncs `sae/` up (so local edits are live), runs the command with `~/venv/bin/python`,
and rsyncs `sae/results/` back down. The HF model cache stays on the box between runs
(lost on `down.sh`; re-downloads ~600 MB on the next `up.sh`).

State (`.instance-id`, `.ip`) is kept in this folder and gitignored. If your IP changes,
re-run `up.sh`'s security-group line or just `down.sh && up.sh`.
