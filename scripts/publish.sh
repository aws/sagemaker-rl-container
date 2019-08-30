#!/bin/bash
#
# Pushes a locally tagged image to ECR repo.

set -euo pipefail

source scripts/shared.sh

parse_publish_std_args "$@"

echo "logging into ECR."
$(aws ecr get-login --no-include-email --registry-id $aws_account) &>/dev/null || echo 'warning: ecr login failed'
docker tag $local_image_tag $new_ecr_image_tag

echo "pushing local_image: $local_image_tag to ecr_repo: $new_ecr_image_tag "
docker push $new_ecr_image_tag

docker logout https://$aws_account.dkr.ecr.$aws_region.amazonaws.com &>/dev/null

