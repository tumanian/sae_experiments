#!/usr/bin/env bash
# Terminate the box. Key pair and security group are kept (free) so up.sh is fast next time.
set -euo pipefail
cd "$(dirname "$0")"
export AWS_REGION=${AWS_REGION:-us-east-1}
ID=$(cat .instance-id)
aws ec2 terminate-instances --instance-ids "$ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text
rm -f .instance-id .ip
echo "terminated $ID"
