#!/usr/bin/env bash
# Stop (not terminate) the box: no compute charge, disk kept (~$8/mo for 100 GB), model cache preserved.
set -euo pipefail
cd "$(dirname "$0")"
export AWS_REGION=${AWS_REGION:-$(aws configure get region 2>/dev/null || echo us-east-1)}
ID=$(cat .instance-id)
aws ec2 stop-instances --instance-ids "$ID" --query 'StoppingInstances[0].CurrentState.Name' --output text
rm -f .ip
echo "stopped $ID — start.sh brings it back with the HF cache intact"
