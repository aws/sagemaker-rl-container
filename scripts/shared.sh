#!/bin/bash
#
# Utility functions for build/test scripts.

function error() {
    >&2 echo $1
    >&2 echo "build usage: $0 [--version-number <major-version>] [--local-image-tag <tag-for-new-image>] [--base-image-ecr-repo <base ecr repo to cache from>] [--region <aws-region>]"
    >&2 echo "publish usage: $0 [--local-image-tag <local image tag to push>] [--new-ecr-image-repo <ecr repo to push to>] [--region <aws-region>]"
    exit 1
}

function get_default_region() {
    if [ -n "${AWS_DEFAULT_REGION:-}" ]; then
        echo "$AWS_DEFAULT_REGION"
    else
        aws configure get region
    fi
}

function get_aws_account() {
    aws sts get-caller-identity --query 'Account' --output text
}

function parse_build_std_args() {
    # defaults
    aws_region=$(get_default_region)

    while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -b|--base-image-account-id)
        base_image_account_id="$2"
        shift
        shift
        ;;
        -v|--version-number)
        version_number="$2"
        shift
        shift
        ;;
        -r|--region)
        aws_region="$2"
        shift
        shift
        ;;
        -l|--local-image-tag)
        local_image_tag="$2"
        shift
        shift
        ;;
        -b|--base-image-ecr-repo)
        base_image_ecr_repo="$2"
        shift
        shift
        ;;
        *) # unknown option
        error "unknown option: $1"
        shift
        ;;
    esac
    done

    [[ -z "${base_image_account_id// }" ]] && error 'missing base-image-account-id'
    [[ -z "${version_number// }" ]] && error 'missing version-number'
    [[ -z "${aws_region// }" ]] && error 'missing aws region'
    [[ -z "${local_image_tag// }" ]] && error 'missing local tag name for the built image'
    [[ -z "${base_image_ecr_repo// }" ]] && error 'missing base image to build from'

    true
}

function parse_publish_std_args() {
    # defaults
    aws_region=$(get_default_region)
    aws_account=$(get_aws_account)

    while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -l|--local-image-tag)
        local_image_tag="$2"
        shift
        shift
        ;;
        -r|--region)
        aws_region="$2"
        shift
        shift
        ;;
        -n|--new-ecr-image-tag)
        new_ecr_image_tag="$2"
        shift
        shift
        ;;
        *) # unknown option
        error "unknown option: $1"
        shift
        ;;
    esac
    done
    
    [[ -z "${aws_account// }" ]] && error 'missing aws aws_account'
    [[ -z "${aws_region// }" ]] && error 'missing aws region'
    [[ -z "${local_image_tag// }" ]] && error 'missing local tag name for the built image'
    [[ -z "${new_ecr_image_tag// }" ]] && error 'missing base image to build from'

    true
}

