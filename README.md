# Group LLM Chat with Talking Portraits

A comprehensive multi-agent conversation system with real-time talking portrait generation, combining AI-powered dialogue with animated avatar visualization.

## Project Overview

This project integrates three main components to create an interactive conversation experience with animated talking avatars:

- **Chat Application** - FastAPI-based web application that orchestrates multi-agent conversations using Microsoft Autogen
- **FasterLivePortrait** - Real-time talking portrait generation service with TensorRT acceleration
- **TTS Service** - Text-to-speech service for voice generation using F5 TTS project

## Key Features

- Multi-agent conversation engine with configurable speakers and voices
- Real-time talking portrait generation from static images and audio
- Audio-driven lip synchronization with GPU acceleration
- Web-based interface with WebSocket support for live updates
- YAML-driven configuration for flexible setup
- Docker containerization for easy deployment

## Architecture

The system consists of three interconnected services:

1. **Chat Application** (`chat/`) - Main orchestrator that:
   - Manages conversation flow between multiple AI agents
   - Handles user interactions and conversation state
   - Coordinates with TTS and FasterLivePortrait services
   - Provides web interface with Bootstrap-based UI

2. **FasterLivePortrait** (`FasterLivePortrait/`) - Video generation service that:
   - Creates talking portrait videos from static images
   - Uses TensorRT for GPU-accelerated processing
   - Provides REST API for video generation
   - Supports audio-driven lip synchronization

3. **TTS Service** (`TTS/`) - Text-to-speech service that:
   - Converts text to natural-sounding speech
   - Supports multiple voices and languages
   - Provides audio output for portrait animation

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker and Docker Compose
- 8GB+ GPU memory recommended

### Setup

1. Configure the chat application by following the instructions in the `chat/` README.md
2. Update configuration files with your LLM credentials and service endpoints
    ```bash
    make build-flp
    make build-tts
    ```
4. Start all services using Docker Compose:
   ```bash
   docker-compose up -d
   ```
5. Open the web interface at `http://localhost:8084` (or the host/port specified in your `chat/config/config.yaml` configuration file).

## API Endpoints

### Chat Application
- `POST /api/conversations/start` - Start conversation generation
- `POST /api/conversations/stop` - Stop ongoing conversation
- `GET /api/conversations/{id}/state` - Get conversation state
- `POST /api/user_message` - Inject user message
- `WS /ws` - WebSocket for real-time updates

### FasterLivePortrait
- `POST /predict_audio/` - Generate talking portrait video
- `GET /ping` - Health check

## Configuration

The system uses YAML configuration files for flexible setup:
- `config/app.yaml` - Main chat application settings
- `config/scene.yaml` - Conversation scene configuration
- `FasterLivePortrait/configs/` - Video generation settings
- `TTS/voices.yaml` - Voice configuration

## Requirements

- NVIDIA GPU with CUDA support
- Docker and Docker Compose
- Python 3.10+
- 8GB+ GPU memory recommended
- OpenAI-compatible LLM service credentials

