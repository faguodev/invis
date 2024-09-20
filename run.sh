#!/bin/bash

# Allow connections from local clients
xhost +local:

# Default container name
CONTAINER_NAME=${1:-container01}

# Run the temporary Docker Container
sudo docker run --rm --gpus all -it \
    -e DISPLAY=$DISPLAY \
    -e XDG_RUNTIME_DIR=/tmp/runtime-root \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name $CONTAINER_NAME invis-image

