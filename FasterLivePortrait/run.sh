#!/bin/bash

CONTAINER_NAME="faster_live_portrait"

# Stop and remove any existing container with the same name
docker rm -f $CONTAINER_NAME > /dev/null 2>&1 || true


CONTAINER_ID=$(docker run -it -d \
    --name $CONTAINER_NAME \
    -v $(pwd)/api_v2.py:/root/FasterLivePortrait/api_v2.py \
    -v $(pwd)/src:/root/FasterLivePortrait/src \
    -v $(pwd)/configs:/root/FasterLivePortrait/configs \
    -p 8081:8081 \
    --gpus all \
    flp1_data \
    bash -ic "python /root/FasterLivePortrait/api_v2.py")

# Wait until the server is up for 30 seconds.
printf "Waiting for the server to start."
for i in {1..6}; do
    if curl -fsS --max-time 2 http://localhost:8081/ping >/dev/null 2>&1; then
        echo "OK"
        break
    fi
    printf '.'

    # Check if docker container is still running
    if [ ! "$(docker ps -q -f id=$CONTAINER_ID)" ]; then
        printf "\n"
        echo "Docker container has stopped unexpectedly. Exiting."
        echo "Logs: $(docker logs $CONTAINER_ID)"
        exit 1
    fi
    sleep 5
done

exit 0