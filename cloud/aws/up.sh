#!/usr/bin/env bash
# Launch a CPU EC2 box for the SAE experiments. Idempotent-ish: reuses key pair and security group.
#   cloud/aws/up.sh                      # c7i.xlarge (4 vCPU / 8 GB, ~$0.18/hr) in us-east-1
#   INSTANCE_TYPE=m7i.large cloud/aws/up.sh
set -euo pipefail
cd "$(dirname "$0")"
REGION=${AWS_REGION:-us-east-1}
INSTANCE_TYPE=${INSTANCE_TYPE:-c7i.xlarge}
NAME=sae-cpu
KEY_FILE=$HOME/.ssh/$NAME.pem
export AWS_REGION=$REGION

if [ -f .instance-id ]; then
  echo "instance $(cat .instance-id) already recorded in cloud/aws/.instance-id — run down.sh first"; exit 1
fi

if ! aws ec2 describe-key-pairs --key-names "$NAME" >/dev/null 2>&1; then
  aws ec2 create-key-pair --key-name "$NAME" --query KeyMaterial --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  echo "created key pair -> $KEY_FILE"
fi

SG=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$NAME" --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --group-name "$NAME" --description "SAE cpu box: ssh only" --query GroupId --output text)
fi
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 || true

AMI=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)

ID=$(aws ec2 run-instances \
  --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --key-name "$NAME" --security-group-ids "$SG" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3}' \
  --user-data file://bootstrap.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "$ID" > .instance-id
echo "launched $ID ($INSTANCE_TYPE, $AMI) — waiting for it to come up"
aws ec2 wait instance-running --instance-ids "$ID"
IP=$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$IP" > .ip
echo "up: ssh -i $KEY_FILE ubuntu@$IP"
echo "bootstrap (pip install) takes ~3 min; run.sh waits for it."
