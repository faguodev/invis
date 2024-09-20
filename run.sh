#!/bin/bash

# Allow connections from local clients
xhost +local:

# Default container name
CONTAINER_NAME=${2:-container01}

# Check if the CPU image is being run
if [[ $1 == "cpu" ]]; then
    IMAGE_NAME="invis-image-cpu"
    GPU_FLAG=""
else
    IMAGE_NAME="invis-image"
    GPU_FLAG="--gpus all"
fi

# Run the Docker Container
sudo docker run --rm $GPU_FLAG -it \
    -e DISPLAY=$DISPLAY \
    -e XDG_RUNTIME_DIR=/tmp/runtime-root \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name $CONTAINER_NAME $IMAGE_NAME
