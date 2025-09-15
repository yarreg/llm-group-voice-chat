#!/bin/bash

AVATAR="assets/avatar.jpg"
VOICE="assets/voice.wav"
OUTPUT="output.mp4"

# time with milliseconds
START_TIME=$(date +%s%3N)

rm -f $OUTPUT

curl -v -X POST "http://localhost:8081/predict_audio/" \
  -F "source_image=@$AVATAR" \
  -F "driving_audio=@$VOICE" \
  --output $OUTPUT

echo "Generation completed. Time taken: $(( $(date +%s%3N) - $START_TIME )) ms" 