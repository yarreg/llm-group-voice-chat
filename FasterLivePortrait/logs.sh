#!/bin/bash

CONTAINER_NAME="faster_live_portrait"

# check if the container is running
if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container $CONTAINER_NAME is not running."
    exit 1
fi

docker logs $CONTAINER_NAME