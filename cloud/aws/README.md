# Run the SAE experiments on an AWS box

Default is a CPU `c7i.xlarge` (~$0.18/hr, no quota needed). For real models set
`INSTANCE_TYPE=g5.xlarge` (A10G 24 GB, ~$1/hr): `up.sh` then uses the AWS Deep Learning
Base AMI (NVIDIA driver preinstalled) and `bootstrap.sh` installs CUDA torch.

## One-time

1. **Credentials.** `aws login` (browser sign-in) or `aws configure` with an access key.
   Region `us-east-1`. Permissions: EC2 full access plus `ssm:GetParameter` (used to look
   up the current AMI id).
2. **GPU quota** (only for `g*` instances). New accounts have 0. Service Quotas console ->
   EC2 -> "Running On-Demand G and VT instances" -> request 8 vCPUs. Usually approved in
   hours, sometimes a day. `up.sh` fails with `VcpuLimitExceeded` until then.
3. **Hugging Face token** (for Gemma / Llama, not GPT-2): accept the model license on the
   HF page, then `export HF_TOKEN=...` before `run.sh`; it's forwarded to the box.

`up.sh` creates the key pair (`~/.ssh/sae-cpu.pem`) and a security group that only allows
SSH from your current IP.

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
