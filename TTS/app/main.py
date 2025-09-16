import os
import time
import re
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .config import get_config
from .logging import setup_logging, get_logger, RequestLogger
from .tts_pipeline import get_tts_pipeline, TTSInferenceError
from .voices import VoiceError

# Initialize logging
logger = get_logger(__name__)

# Global app instance
api = FastAPI(
    title="TTS Service API",
    description="Text-to-Speech service using F5-TTS with Russian accent support",
    version="1.0.0"
)


# Request models
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Text to synthesize")
    voice: str = Field(..., description="Voice ID to use")
    format: Optional[str] = Field(default="wav", pattern=r"^(wav|mp3)$", description="Audio format")
    sample_rate: Optional[int] = Field(default=24000, ge=8000, le=48000, description="Sample rate")
    normalize: Optional[bool] = Field(default=True, description="Normalize audio")
    seed: Optional[int] = Field(default=None, ge=0, description="Random seed")


class VoiceResponse(BaseModel):
    voices: list[str]


class VoiceDetailResponse(BaseModel):
    voices: list[Dict[str, str]]


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    voices_loaded: int
    model_loaded: bool
    vocoder_loaded: bool
    accent_enabled: bool
    cuda_device_name: Optional[str] = None
    cuda_memory_allocated: Optional[int] = None
    cuda_memory_reserved: Optional[int] = None


# Middleware setup
def setup_middleware(app: FastAPI):
    """Setup FastAPI middleware"""
    config = get_config()

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request, call_next):
        start_time = time.time()
        request_id = f"req_{int(time.time() * 1000)}_{os.getpid()}"

        # Add request ID to logger context
        request_logger = RequestLogger(request_id)

        request_logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client=request.client.host if request.client else None
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            request_logger.info(
                "Request completed",
                status_code=response.status_code,
                process_time_ms=int(process_time * 1000)
            )

            return response

        except Exception as e:
            process_time = time.time() - start_time
            request_logger.error(
                "Request failed",
                error=str(e),
                process_time_ms=int(process_time * 1000)
            )
            raise


# Exception handlers
def setup_exception_handlers(app: FastAPI):
    """Setup custom exception handlers"""

    @app.exception_handler(TTSInferenceError)
    async def tts_inference_exception_handler(request, exc: TTSInferenceError):
        logger.error(f"TTS inference error: {exc}", exc_info=True)
        return HTTPException(
            status_code=500,
            detail=f"TTS inference failed: {str(exc)}"
        )

    @app.exception_handler(VoiceError)
    async def voice_exception_handler(request, exc: VoiceError):
        logger.error(f"Voice error: {exc}", exc_info=True)
        return HTTPException(
            status_code=404,
            detail=f"Voice error: {str(exc)}"
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc: ValueError):
        logger.warning(f"Validation error: {exc}")
        return HTTPException(
            status_code=400,
            detail=f"Validation error: {str(exc)}"
        )


# API endpoints
@api.get("/ping", response_model=Dict[str, str])
async def ping():
    """Health check endpoint"""
    try:
        # Check if TTS pipeline is initialized
        tts_pipeline = get_tts_pipeline()
        health_status = tts_pipeline.health_check()

        if health_status["status"] != "ok":
            raise HTTPException(
                status_code=503,
                detail="Service not ready"
            )

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Service not ready"
        )


@api.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check endpoint"""
    try:
        tts_pipeline = get_tts_pipeline()
        health_status = tts_pipeline.health_check()

        return HealthResponse(**health_status)

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Service not ready"
        )


@api.get("/voices", response_model=VoiceResponse)
async def list_voices():
    """List available voices"""
    try:
        tts_pipeline = get_tts_pipeline()
        voices = tts_pipeline.get_available_voices()

        return VoiceResponse(voices=voices)

    except Exception as e:
        logger.error(f"Failed to list voices: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to list voices"
        )


@api.get("/voices/detailed", response_model=VoiceDetailResponse)
async def list_voices_detailed():
    """List available voices with detailed information"""
    try:
        tts_pipeline = get_tts_pipeline()
        voices = tts_pipeline.get_available_voices_detailed()

        return VoiceDetailResponse(voices=voices)

    except Exception as e:
        logger.error(f"Failed to list voices: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to list voices"
        )


@api.get("/tts")
async def text_to_speech(
    text: str = Query(..., min_length=1, max_length=2000, description="Text to synthesize"),
    voice: str = Query(..., description="Voice ID to use"),
    format: str = Query(default="wav", pattern=r"^(wav|mp3)$", description="Audio format"),
    sample_rate: Optional[int] = Query(default=24000, ge=8000, le=48000, description="Sample rate"),
    normalize: bool = Query(default=True, description="Normalize audio"),
    seed: Optional[int] = Query(default=None, ge=0, description="Random seed")
):
    """Synthesize text to speech"""
    request_logger = RequestLogger()

    try:
        # Validate voice ID format
        if not re.match(r"^RU_(Male|Female)_[a-z_]+$", voice, re.IGNORECASE):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice ID format: {voice}"
            )

        # Get TTS pipeline
        tts_pipeline = get_tts_pipeline()

        # Generate speech
        audio_bytes = tts_pipeline.synthesize_speech(
            text=text,
            voice_id=voice,
            format=format,
            sample_rate=sample_rate,
            normalize=normalize,
            seed=seed
        )

        # Determine content type and filename
        if format.lower() == "wav":
            content_type = "audio/wav"
            file_extension = "wav"
        else:  # mp3
            content_type = "audio/mpeg"
            file_extension = "mp3"

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{voice}_{timestamp}.{file_extension}"

        # Return response
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=\"{filename}\"",
                "Content-Length": str(len(audio_bytes))
            }
        )

    except HTTPException:
        raise
    except TTSInferenceError as e:
        request_logger.error("TTS inference failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"TTS synthesis failed: {str(e)}"
        )
    except VoiceError as e:
        request_logger.error("Voice error", error=str(e))
        raise HTTPException(
            status_code=404,
            detail=f"Voice not available: {str(e)}"
        )
    except Exception as e:
        request_logger.error("Unexpected error", error=str(e))
        logger.error(f"Unexpected error in TTS endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@api.post("/tts")
async def text_to_speech_post(request: TTSRequest):
    """Synthesize text to speech (POST endpoint)"""
    request_logger = RequestLogger()

    try:
        # Validate voice ID format
        if not re.match(r"^RU_(Male|Female)_[a-z_]+$", request.voice, re.IGNORECASE):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice ID format: {request.voice}"
            )

        # Get TTS pipeline
        tts_pipeline = get_tts_pipeline()

        # Generate speech
        audio_bytes = tts_pipeline.synthesize_speech(
            text=request.text,
            voice_id=request.voice,
            format=request.format,
            sample_rate=request.sample_rate,
            normalize=request.normalize,
            seed=request.seed
        )

        # Determine content type and filename
        if request.format.lower() == "wav":
            content_type = "audio/wav"
            file_extension = "wav"
        else:  # mp3
            content_type = "audio/mpeg"
            file_extension = "mp3"

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{request.voice}_{timestamp}.{file_extension}"

        # Return response
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=\"{filename}\"",
                "Content-Length": str(len(audio_bytes))
            }
        )

    except HTTPException:
        raise
    except TTSInferenceError as e:
        request_logger.error("TTS inference failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"TTS synthesis failed: {str(e)}"
        )
    except VoiceError as e:
        request_logger.error("Voice error", error=str(e))
        raise HTTPException(
            status_code=404,
            detail=f"Voice not available: {str(e)}"
        )
    except Exception as e:
        request_logger.error("Unexpected error", error=str(e))
        logger.error(f"Unexpected error in TTS endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


# Startup and shutdown events
@api.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting TTS service...")

    try:
        # Setup logging
        setup_logging()

        # Initialize components
        from .voices import initialize_voices
        from .accents import initialize_accents
        from .audio_utils import initialize_audio_utils
        from .tts_pipeline import initialize_tts_pipeline

        # Initialize all components
        initialize_voices()
        initialize_accents()
        initialize_audio_utils()
        initialize_tts_pipeline()

        logger.info("TTS service started successfully")

    except Exception as e:
        logger.error(f"Failed to start TTS service: {e}", exc_info=True)
        if get_config().f5_tts.fail_fast:
            raise
        else:
            logger.warning("Continuing with partial initialization")


@api.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down TTS service...")
    # Add any cleanup logic here
    logger.info("TTS service shutdown complete")


# Setup middleware and exception handlers
setup_middleware(api)
setup_exception_handlers(api)


# CLI entry point
def main():
    """Main entry point for running the service"""
    config = get_config()

    logger.info(f"Starting TTS service on {config.server.host}:{config.server.port}")

    uvicorn.run(
        "app.main:api",
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        log_config=None,  # Use our custom logging
        access_log=False,  # Disable uvicorn access logging
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()