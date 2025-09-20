"""Media pipeline coordinating TTS and FastLivePortrait services."""

import asyncio
import io
import logging
import math
import mimetypes
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional, TypeVar, cast

import httpx

from config import ActorConfig, AppConfig

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class MediaResult:
    media_path: Path
    media_url: str
    duration_ms: int


class MediaPipeline:
    def __init__(self, app_config: AppConfig, media_root: Path):
        self._config = app_config
        self._media_root = media_root
        self._media_root.mkdir(parents=True, exist_ok=True)
        self._client = httpx.AsyncClient()

    async def shutdown(self) -> None:
        await self._client.aclose()

    def resolve_media_path(self, conversation_id: str, filename: str) -> Path:
        return self._media_root / conversation_id / filename

    @property
    def media_root(self) -> Path:
        return self._media_root

    async def generate(
        self,
        *,
        actor: ActorConfig,
        text: str,
        conversation_id: str,
        turn_id: str,
        progress: Callable[[str, int], Awaitable[None]] | None = None,
    ) -> MediaResult:
        logger.debug("Starting media pipeline for turn %s", turn_id)
        avatar_path = Path(actor.avatar_image).expanduser().resolve()
        if not avatar_path.exists():
            raise FileNotFoundError(f"Avatar image not found: {avatar_path}")

        async def _tts_call() -> tuple[bytes, float]:
            if progress:
                await progress("tts", 0)
            data, duration_sec = await self._request_tts(text=text, voice_id=self._resolve_voice(actor))
            if progress:
                await progress("tts", 100)
            return data, duration_sec

        audio_bytes, duration_sec = await self._with_retries(_tts_call, stage="tts")

        async def _flp_call() -> bytes:
            if progress:
                await progress("flp", 0)
            data = await self._request_flp(avatar_path=avatar_path, audio_bytes=audio_bytes)
            if progress:
                await progress("flp", 100)
            return data

        video_bytes = await self._with_retries(_flp_call, stage="flp")

        media_path = self._write_media(conversation_id, turn_id, video_bytes)
        media_url = f"/api/assets/{conversation_id}/{media_path.name}"
        return MediaResult(media_path=media_path, media_url=media_url, duration_ms=math.ceil(duration_sec * 1000))

    async def _with_retries(
        self,
        func: Callable[[], Awaitable[T]],
        stage: str,
    ) -> T:
        retry_cfg = self._config.pipeline.retries
        attempts = retry_cfg.attempts
        backoff = retry_cfg.backoff_sec
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await func()
            except Exception as exc:  # pragma: no cover - network errors dominated
                last_exc = exc
                logger.warning("%s stage failed on attempt %s/%s: %s", stage, attempt, attempts, exc)
                if attempt == attempts:
                    break
                await asyncio.sleep(backoff * math.pow(2, attempt - 1))
        assert last_exc is not None
        raise last_exc

    async def _request_tts(self, text: str, voice_id: str) -> tuple[bytes, float]:
        url = self._config.tts.base_url.rstrip("/") + "/tts"
        timeout = self._config.pipeline.timeouts.tts_sec
        logger.debug("POST %s (voice=%s, text_len=%s)", url, voice_id, len(text))
        response = await self._client.post(
            url,
            json={"text": text, "voice": voice_id},
            timeout=timeout,
        )
        response.raise_for_status()

        duration_sec = float(response.headers.get("X-Duration-Sec", 0))
        return response.content, duration_sec

    async def _request_flp(self, avatar_path: Path, audio_bytes: bytes) -> bytes:
        url = self._config.flp.base_url.rstrip("/") + "/predict_audio/"
        timeout = self._config.pipeline.timeouts.flp_sec
        logger.debug("POST %s with avatar=%s", url, avatar_path)

        with avatar_path.open("rb") as avatar_fp:
            files = {
                "source_image": (
                    avatar_path.name,
                    cast(IO[bytes], avatar_fp),
                    mimetypes.guess_type(str(avatar_path))[0] or "image/png",
                ),
                "driving_audio": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
            }

            response = await self._client.post(url, files=files, timeout=timeout)
            response.raise_for_status()
            return response.content

    def _resolve_voice(self, actor: ActorConfig) -> str:
        if actor.voice and actor.voice.voice_id:
            return actor.voice.voice_id
        return self._config.tts.default_voice.voice_id

    def _write_media(self, conversation_id: str, turn_id: str, data: bytes) -> Path:
        conv_dir = self._media_root / conversation_id
        conv_dir.mkdir(parents=True, exist_ok=True)
        media_path = conv_dir / f"{turn_id}.mp4"
        media_path.write_bytes(data)
        return media_path
