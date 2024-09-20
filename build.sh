#!/bin/bash

# Default to GPU version unless 'cpu' is specified
if [[ $1 == "cpu" ]]; then
    IMAGE_NAME="invis-image-cpu"
    DOCKERFILE="Dockerfile.cpu"
else
    IMAGE_NAME="invis-image"
    DOCKERFILE="Dockerfile"
fi

# Build the Docker image
sudo docker build -f $DOCKERFILE -t $IMAGE_NAME .

# Notify that the build is complete
echo "Docker image $IMAGE_NAME built successfully."
