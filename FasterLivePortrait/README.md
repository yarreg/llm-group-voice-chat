# FasterLivePortrait API Service

FastAPI service for generating talking portrait videos from images and audio using FasterLivePortrait with TensorRT acceleration.

## Features

- Real-time talking portrait generation
- Audio-driven lip synchronization
- TensorRT GPU acceleration
- REST API interface
- Docker containerization

## Quick Start

### Build docker image
```bash
docker build -t flp_api .
```

### Using Run Script
```bash
./run.sh
```

### Generate Example
```bash
./gen.sh
```

## API Endpoints

### POST /predict_audio/
Generate talking portrait video.

**Request:**
- `source_image`: Source image file
- `driving_audio`: Audio file
- Optional parameters for animation control
    - check `api_v2.py` for full list

**Response:** MP4 video file

### GET /ping
Health check endpoint.

## Example Usage
```bash
curl -X POST "http://localhost:8083/predict_audio/" \
  -F "source_image=@assets/male1.png" \
  -F "driving_audio=@assets/voice.wav" \
  -o output.mp4
```

## Configuration
Configuration files in `configs/` directory:
- `trt_infer.yaml`: Main inference settings
- Model paths and parameters

## Requirements
- NVIDIA GPU with CUDA
- Docker
- GPU memory: 8GB+ recommended
