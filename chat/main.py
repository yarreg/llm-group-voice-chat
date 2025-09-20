"""Command line entrypoint to serve the JoyVASA chat application."""

import argparse
from pathlib import Path

import uvicorn

from app import create_app
from config import load_app_config


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JoyVASA Conversation Server")
    parser.add_argument("--app-config", required=True, type=Path, help="Path to app.yaml configuration")
    parser.add_argument("--scene-config", required=True, type=Path, help="Path to scene.yaml configuration")
    parser.add_argument("--host", type=str, default=None, help="Override host (defaults to config value)")
    parser.add_argument("--port", type=int, default=None, help="Override port (defaults to config value)")
    parser.add_argument("--reload", action="store_true", help="Enable FastAPI reload (development only)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    app = create_app(app_config_path=args.app_config, scene_config_path=args.scene_config)
    app_cfg = load_app_config(args.app_config)  # reuse to pick server defaults
    host = args.host or app_cfg.server.host
    port = args.port or app_cfg.server.port

    uvicorn.run(app, host=host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
