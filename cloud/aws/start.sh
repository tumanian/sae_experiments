#!/usr/bin/env bash
# Start a stopped box and refresh its (new) public IP.
set -euo pipefail
cd "$(dirname "$0")"
export AWS_REGION=${AWS_REGION:-$(aws configure get region 2>/dev/null || echo us-east-1)}
ID=$(cat .instance-id)
aws ec2 start-instances --instance-ids "$ID" >/dev/null
aws ec2 wait instance-running --instance-ids "$ID"
IP=$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$IP" > .ip
MYIP=$(curl -s https://checkip.amazonaws.com)
SG=$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 || true
echo "up: ssh -i $HOME/.ssh/sae-cpu.pem ubuntu@$IP"
