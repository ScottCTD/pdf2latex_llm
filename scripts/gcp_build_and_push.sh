#!/bin/bash

# Exit on error
set -e

PROJECT_ID=$1
REGION=$2
REPO_NAME=$3

if [ -z "$PROJECT_ID" ] || [ -z "$REGION" ] || [ -z "$REPO_NAME" ]; then
    echo "Usage: ./gcp_build_and_push.sh <PROJECT_ID> <REGION> <REPO_NAME>"
    exit 1
fi

echo "Building and pushing images for Project: $PROJECT_ID, Region: $REGION, Repo: $REPO_NAME"

# Configure docker auth
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and Push Training Image
TRAIN_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/pdf2latex-train:latest"
echo "Building Training Image: $TRAIN_IMAGE"
docker build -t $TRAIN_IMAGE -f Dockerfile.train .
docker push $TRAIN_IMAGE

# Build and Push Serving Image
SERVE_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/pdf2latex-serve:latest"
echo "Building Serving Image: $SERVE_IMAGE"
docker build -t $SERVE_IMAGE -f Dockerfile.serve .
docker push $SERVE_IMAGE

echo "Images pushed successfully."
