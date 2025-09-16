import io
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

api = FastAPI(
    title="TTS Service API (Test Mode)",
    description="Text-to-Speech service test implementation",
    version="1.0.0"
)

class VoiceResponse(BaseModel):
    voices: List[str]

class VoiceDetailResponse(BaseModel):
    voices: List[Dict[str, str]]

class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    voices_loaded: int
    model_loaded: bool
    vocoder_loaded: bool
    accent_enabled: bool
    test_mode: bool

# Test voices
TEST_VOICES = [
    "RU_Female_abramova_tatjyana",
    "RU_Male_baranov_mihail",
    "RU_Female_aksyuta_tatjyana"
]

# Middleware setup
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "ok"}

@api.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check endpoint"""
    return HealthResponse(
        status="ok",
        device="cpu",
        cuda_available=False,
        voices_loaded=len(TEST_VOICES),
        model_loaded=False,
        vocoder_loaded=False,
        accent_enabled=False,
        test_mode=True
    )

@api.get("/voices", response_model=VoiceResponse)
async def list_voices():
    """List available voices"""
    return VoiceResponse(voices=TEST_VOICES)

@api.get("/voices/detailed", response_model=VoiceDetailResponse)
async def list_voices_detailed():
    """List available voices with detailed information"""
    voices = []
    for voice_id in TEST_VOICES:
        voices.append({
            "name": voice_id,
            "files": {
                "voice": f"voices/{voice_id}.mp3",
                "text": f"voices/{voice_id}.txt"
            }
        })
    return VoiceDetailResponse(voices=voices)

@api.get("/tts")
async def text_to_speech(
    text: str = Query(..., min_length=1, max_length=2000, description="Text to synthesize"),
    voice: str = Query(..., description="Voice ID to use"),
    format: str = Query(default="wav", regex=r"^(wav|mp3)$", description="Audio format"),
    sample_rate: Optional[int] = Query(default=24000, ge=8000, le=48000, description="Sample rate"),
    normalize: bool = Query(default=True, description="Normalize audio"),
    seed: Optional[int] = Query(default=None, ge=0, description="Random seed")
):
    """Synthesize text to speech (test version)"""

    # Validate voice ID
    if voice not in TEST_VOICES:
        raise HTTPException(status_code=404, detail=f"Voice not available: {voice}")

    # Generate test audio (silent WAV for testing)
    if format.lower() == "wav":
        # Create simple WAV header + silent audio
        sample_rate = sample_rate or 24000
        duration = 2.0  # 2 seconds of silence
        num_samples = int(sample_rate * duration)

        # WAV header
        wav_header = io.BytesIO()
        wav_header.write(b'RIFF')
        wav_header.write((36 + num_samples * 2).to_bytes(4, 'little'))  # file size
        wav_header.write(b'WAVE')
        wav_header.write(b'fmt ')
        wav_header.write((16).to_bytes(4, 'little'))  # fmt chunk size
        wav_header.write((1).to_bytes(2, 'little'))   # PCM format
        wav_header.write((1).to_bytes(2, 'little'))   # mono
        wav_header.write(sample_rate.to_bytes(4, 'little'))  # sample rate
        wav_header.write((sample_rate * 2).to_bytes(4, 'little'))  # byte rate
        wav_header.write((2).to_bytes(2, 'little'))   # block align
        wav_header.write((16).to_bytes(2, 'little'))  # bits per sample
        wav_header.write(b'data')
        wav_header.write((num_samples * 2).to_bytes(4, 'little'))  # data size

        # Silent audio data (16-bit PCM)
        silent_data = b'\x00\x00' * num_samples

        audio_bytes = wav_header.getvalue() + silent_data
        content_type = "audio/wav"
        file_extension = "wav"
    else:
        # For MP3, return a minimal MP3 file (silent)
        audio_bytes = (
            b'\xff\xfb\x90\x44\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        )
        content_type = "audio/mpeg"
        file_extension = "mp3"

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{voice}_{timestamp}.{file_extension}"

    # Log the request
    print(f"[TEST] TTS request: text='{text}', voice='{voice}', format='{format}'")

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=\"{filename}\"",
            "Content-Length": str(len(audio_bytes))
        }
    )

if __name__ == "__main__":
    print("Starting TTS Test Server...")
    print("Available voices:", TEST_VOICES)
    print("Server will be available at: http://localhost:8000")
    print("\nTest commands:")
    print("curl -s http://localhost:8000/ping")
    print("curl -s http://localhost:8000/voices")
    print('curl -G "http://localhost:8000/tts" --data-urlencode "text=Привет мир" --data-urlencode "voice=RU_Female_abramova_tatjyana" -o test.wav')

    uvicorn.run(
        "test_main:api",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )