import logging
from pathlib import Path
from typing import cast

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from config import AppConfig, SceneConfig, load_app_config, load_scene_config
from conversation import ConversationManager
from events import EventBroker
from media import MediaPipeline

logger = logging.getLogger(__name__)


class ApplicationContext:
    def __init__(self, app_config: AppConfig, scene_config: SceneConfig, media_root: Path):
        self.app_config = app_config
        self.scene_config = scene_config
        self.broker = EventBroker()
        self.media_pipeline = MediaPipeline(app_config, media_root)
        self.manager = ConversationManager(app_config, scene_config, self.media_pipeline, self.broker)

    async def shutdown(self) -> None:
        await self.manager.stop(reason="shutdown")
        await self.media_pipeline.shutdown()


def create_app(*, app_config_path: Path, scene_config_path: Path) -> FastAPI:
    app_config = load_app_config(app_config_path)
    scene_config = load_scene_config(scene_config_path)

    logging.basicConfig(
        level=getattr(logging, app_config.logging.level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    media_root = Path(__file__).parent / "generated_media"
    context = ApplicationContext(app_config, scene_config, media_root)

    app = FastAPI(title="JoyVASA Conversation", version="0.1.0")
    app.state.context = context

    static_dir = Path(__file__).parent / "static"
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:  # pragma: no cover - lifecycle
        await context.shutdown()

    def get_manager() -> ConversationManager:
        return cast(ConversationManager, app.state.context.manager)

    def get_scene_config() -> SceneConfig:
        return cast(SceneConfig, app.state.context.scene_config)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "conversation": context.scene_config.conversation,
                "actors": context.scene_config.actors,
            },
        )

    @app.post("/api/conversations/start")
    async def start_conversation(manager: ConversationManager = Depends(get_manager)) -> dict[str, str]:
        await manager.start()
        return {"conversation_id": manager.state.config.conversation.id}

    @app.post("/api/conversations/stop")
    async def stop_conversation(manager: ConversationManager = Depends(get_manager)) -> dict[str, bool]:
        await manager.stop()
        return {"ok": True}

    @app.post("/api/conversations/restart")
    async def restart_conversation(manager: ConversationManager = Depends(get_manager)) -> dict[str, str]:
        await manager.restart()
        return {"conversation_id": manager.state.config.conversation.id}

    @app.get("/api/conversations/{conversation_id}/state")
    async def get_state(conversation_id: str, manager: ConversationManager = Depends(get_manager)) -> dict[str, object]:
        expected = manager.state.config.conversation.id
        if conversation_id != expected:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return manager.state.to_dict()

    @app.post("/api/user_message")
    async def user_message(
        payload: dict[str, str], manager: ConversationManager = Depends(get_manager)
    ) -> dict[str, bool]:
        speaker_id = payload.get("speaker_id")
        text = payload.get("text")
        if not speaker_id or not text:
            raise HTTPException(status_code=400, detail="speaker_id and text are required")
        try:
            await manager.inject_user_message(actor_id=speaker_id, text=text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    @app.get("/api/assets/{conversation_id}/{filename}")
    async def get_asset(
        conversation_id: str, filename: str, manager: ConversationManager = Depends(get_manager)
    ) -> FileResponse:
        expected = manager.state.config.conversation.id
        if conversation_id != expected:
            raise HTTPException(status_code=404, detail="Unknown conversation")
        media_path = context.media_pipeline.resolve_media_path(conversation_id, filename)
        if not media_path.exists():
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(media_path)

    @app.get("/api/thumbs/{actor_id}.png")
    async def get_thumbnail(actor_id: str, scene_config: SceneConfig = Depends(get_scene_config)) -> FileResponse:
        actor = next((actor for actor in scene_config.actors if actor.id == actor_id), None)
        if not actor:
            raise HTTPException(status_code=404, detail="Actor not found")
        return FileResponse(actor.avatar_image)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, manager: ConversationManager = Depends(get_manager)) -> None:
        await websocket.accept()
        subscription = await context.broker.subscribe()
        try:
            await websocket.send_json({"type": "conversation.state", **manager.state.to_dict()})
            async for event in context.broker.stream(subscription):
                await websocket.send_json(event)
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected")
        finally:
            await context.broker.unsubscribe(subscription)

    return app
