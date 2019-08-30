#!/bin/bash
#
# Build the docker images.

set -euo pipefail

source scripts/shared.sh

parse_build_std_args "$@"

echo "pulling previous_image: $base_image_ecr_repo for layer cache... "
$(aws ecr get-login --no-include-email --registry-id $base_image_account_id) &>/dev/null || echo 'warning: ecr login failed'
docker pull $base_image_ecr_repo &>/dev/null || echo 'warning: base image pull failed'
docker logout https://$base_image_account_id.dkr.ecr.$aws_region.amazonaws.com &>/dev/null

echo "building image with version_number: $version_number, and tagging as $local_image_tag ... "
docker build \
    --cache-from $base_image_ecr_repo \
    -f vw/docker/$version_number/Dockerfile \
    -t $local_image_tag .

