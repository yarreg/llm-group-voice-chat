# Chat Application

This folder contains the FastAPI web application that orchestrates the multi-agent conversation demo described in the repository README.

## Features

- FastAPI backend with REST + WebSocket API
- YAML driven configuration (`app.yaml`, `scene.yaml`)
- Multi-actor conversation engine powered by Microsoft Autogen with configurable speakers and voices
- Media pipeline that talks to the bundled TTS (`/TTS`) and FastLivePortrait (`/FasterLivePortrait`) services
- Bootstrap based frontend that renders actor cards, chat bubbles, and controls conversation playback

## Quickstart

1. Install Python dependencies (ideally in a virtualenv):

   ```bash
   pip install -e "."
   pip install -e ".[dev]" # for development with linting, etc.
   ```

2. Prepare configuration files. The `chat/config` directory contains sample YAMLs:

   ```bash
   cp chat/config/app.sample.yaml config/app.yaml
   cp chat/config/scene.sample.yaml config/scene.yaml
   ```

   Update the files to point to valid LLM credentials, TTS/FLP endpoints, and avatar images.

3. Launch the application:

   ```bash
   python -m chat --app-config config/app.yaml --scene-config config/scene.yaml
   ```

   The server listens on the host/port specified in `app.yaml` (overridable via CLI flags). Open the printed URL in the browser to see the UI.

## API Overview

- `POST /api/conversations/start` – starts generation for the configured scene
- `POST /api/conversations/stop` – stops any ongoing work
- `POST /api/conversations/restart` – resets queues and restarts from seed messages
- `GET /api/conversations/{id}/state` – snapshot of current state
- `POST /api/user_message` – inject a user-authored message for a specific actor
- `GET /api/assets/{conversation_id}/{turn_id}.mp4` – served MP4 for a turn
- `GET /api/thumbs/{actor_id}.png` – avatar thumbnails
- `WS /ws` – server-sent events that drive the frontend

## Notes

- Dialogue is generated through Microsoft Autogen (`pyautogen` + `autogen-ext`) using the configured OpenAI-compatible model clients.
- Media files are written under `chat/generated_media/<conversation_id>/` and cleaned manually if needed.

## Development Notes

- This application uses ruff for linting and formatting. Run `ruff check .` and `ruff format .` as needed.
- Use `mypy .` to perform static type checking on the codebase.
- The `--reload` flag can be passed to the CLI for auto-reloading during development.
