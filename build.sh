#!/bin/bash

# Set image name
IMAGE_NAME=${1:-invis-image}


# Build the Docker image
sudo docker build --build-arg IMAGE_NAME=nvidia/cuda --build-arg TARGETARCH=amd64 -t $IMAGE_NAME .

# Notify that the build is complete
echo "Docker image $IMAGE_NAME built successfully."

