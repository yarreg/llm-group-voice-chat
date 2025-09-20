# F5-TTS API Service

This service provides a FastAPI-based Text-to-Speech (TTS) API using the F5-TTS model with Russian accent support.

## API Service (api.py)

The `api.py` file implements a FastAPI web server that provides text-to-speech functionality with the following features:

- **Text-to-Speech Conversion**: Converts input text to audio using the F5-TTS model
- **Multiple Voice Support**: Supports different voice profiles configured via `voices.yaml`
- **Russian Accent Processing**: Includes ruaccent integration for proper Russian pronunciation
- **REST API Endpoints**:
  - `GET /ping` - Health check endpoint
  - `GET /voices` - Returns list of available voices
  - `POST /tts` - Main TTS endpoint that accepts text and voice parameters, returns audio file
- **Automatic Speed Adjustment**: Dynamically adjusts speech speed based on text length
- **Temporary File Management**: Handles temporary audio file creation and cleanup

The service runs on port 8080 by default and supports CUDA acceleration for faster inference.

## Voice Configuration (voices.yaml)
The `voices.yaml` file defines available voice profiles for the TTS service. Each voice entry includes:
- `file`: Path to the reference audio file for the voice
- `text`: Sample text used for voice adaptation 
