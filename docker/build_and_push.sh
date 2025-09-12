#!/bin/bash
set -e

# This script builds a Docker image and pushes it to a specified registry.
# Usage: ./build_and_push.sh build-only | push-only | build-and-push (default)
ACTION=${1:-build-and-push}

if [[ "$ACTION" != "build-only" && "$ACTION" != "push-only" && "$ACTION" != "build-and-push" ]]; then
    echo "Invalid argument: $ACTION"
    echo "Usage: ./build_and_push.sh build-only | push-only | build-and-push (default)"
    exit 1
fi

# Container Parameters
CONTAINER_NAME="daai-dgs"
CONTAINER_TAG="12.1"

# ECR Repository Setup
REGION="ap-southeast-2"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY="daai/dgs"
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
ECR_CONTAINER_NAME="$ECR_URI/$ECR_REPOSITORY"

# AWS Official Image URI
AWS_ECR_URI="763104351884.dkr.ecr.$REGION.amazonaws.com"

echo "Local Container Name: $CONTAINER_NAME"
echo "ECR Repository: $ECR_CONTAINER_NAME"
echo "Tag: $CONTAINER_TAG"

# Login to both AWS Official ECR and our ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ECR_URI
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI

# build the Docker image
if [[ "$ACTION" == "build-only" ]] || [[ "$ACTION" == "build-and-push" ]]; then

    docker build -t $CONTAINER_NAME:$CONTAINER_TAG -f docker/Dockerfile.sagemaker .
    # Tag the Docker image
    docker tag $CONTAINER_NAME:$CONTAINER_TAG $ECR_CONTAINER_NAME:$CONTAINER_TAG
    docker tag $CONTAINER_NAME:$CONTAINER_TAG $ECR_CONTAINER_NAME:latest
fi

if [[ "$ACTION" == "push-only" ]] || [[ "$ACTION" == "build-and-push" ]]; then
    
    # Push the Docker image to ECR
    docker push --all-tags $ECR_CONTAINER_NAME
fi
