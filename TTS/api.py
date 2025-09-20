#!/usr/bin/env python3

import os
import argparse
import tempfile
import soundfile
import logging
from typing import Optional

import yaml
import ruaccent
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT
from f5_tts.model.utils import seed_everything


class TTSConfig:
    def __init__(self):
        self.device = "cuda"
        self.seed = 42421
        self.weights_path = (
            self.get_models_path("F5TTS_v1_Base_v2/model_last_inference.safetensors")
        )
        self.vocab_path = self.get_models_path("F5TTS_v1_Base/vocab.txt")
        self.ruaccent_path = self.get_models_path("ruaccent")
        self.vocoder_path = self.get_models_path("vocos")
        self.accent_dict = {
            "реке": "р+еке",
        }
        self.voices_config_path = "voices.yaml"
        self.voices_dir = "voices"
        self.voices_config = self.load_voices_config()
        
    def get_models_path(self, name):
        # TODO: add support for searching in hf cache
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "models", name)

    def load_voices_config(self):
        try:
            with open(self.voices_config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            return config.get("voices", {})
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error loading voices config: {str(e)}"
            )


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    use_accentizer: Optional[bool] = True


class TTSAPI:
    def __init__(self, config: TTSConfig, app: FastAPI):
        self.config = config
        self.app = app
        self.logger = logging.getLogger(__name__)
        self._initialize_models()
        self._register_routes()

    def _initialize_models(self):
        self.logger.info("Initializing TTS models...")

        seed_everything(self.config.seed)
        self.logger.debug(f"Set seed to {self.config.seed}")

        self.logger.info(f"Loading vocoder from {self.config.vocoder_path}")
        self.vocoder = load_vocoder(
            device=self.config.device,
            is_local=True,
            local_path=self.config.vocoder_path,
        )
        self.logger.info("Vocoder loaded successfully")

        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
        self.logger.info(f"Loading TTS model from {self.config.weights_path}")
        self.model_obj = load_model(
            DiT, model_cfg, self.config.weights_path, vocab_file=self.config.vocab_path
        )
        self.logger.info("TTS model loaded successfully")

        self.logger.info("Loading ruaccent accentizer")
        self.accentizer = ruaccent.RUAccent()
        self.accentizer.load(
            omograph_model_size="turbo3.1",
            device=self.config.device,
            use_dictionary=True,
            tiny_mode=False,
            custom_dict=self.config.accent_dict,
            workdir=self.config.ruaccent_path,
        )
        self.logger.info("Accentizer loaded successfully")
        self.logger.info("All models initialized successfully")

    def _register_routes(self):
        self.app.add_api_route("/ping", self.ping_handler, methods=["GET"])
        self.app.add_api_route("/voices", self.get_voices_handler, methods=["GET"])
        self.app.add_api_route("/tts", self.tts_handler, methods=["POST"])

    def _get_temporary_file(self):
        with tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="tts_", delete=False
        ) as temp_file:
            file_path = temp_file.name
            return file_path, os.path.basename(file_path)

    def _cleanup_tts_file(self, file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)

    def _get_tts_speed(self, text: str) -> float:
        number_of_words = len(text.split())
        if number_of_words < 3:
            return 0.5
        elif number_of_words < 7:
            return 0.7
        elif number_of_words < 10:
            return 0.9
        return 1.0

    async def ping_handler(self):
        return {"status": "ok", "message": "F5-TTS API is running"}

    async def get_voices_handler(self):
        return {"voices": list(self.config.voices_config.keys())}

    async def tts_handler(self, request: TTSRequest, background_tasks: BackgroundTasks):
        try:
            self.logger.info(f"TTS request: voice='{request.voice}', text_length={len(request.text)}, accentizer={request.use_accentizer}")
            self.logger.debug(f"Request text: {request.text}")

            if request.voice not in self.config.voices_config:
                self.logger.error(f"Voice '{request.voice}' not found")
                raise HTTPException(
                    status_code=400,
                    detail=f"Voice '{request.voice}' not found. Available voices: {list(self.config.voices_config.keys())}",
                )

            voice_config = self.config.voices_config[request.voice]
            ref_file = os.path.join(self.config.voices_dir, voice_config["file"])
            ref_text = voice_config["text"]

            self.logger.debug(f"Using voice: {request.voice}, reference file: {ref_file}")

            if not os.path.exists(ref_file):
                self.logger.error(f"Reference audio file not found: {ref_file}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Reference audio file not found: {ref_file}",
                )

            output_path, output_filename = self._get_temporary_file()
            self.logger.debug(f"Created temporary file: {output_path}")

            ref = (ref_file, ref_text)
            ref_file_processed, ref_text_processed = preprocess_ref_audio_text(
                ref[0], ref[1]
            )
            text_to_process = request.text

            if request.use_accentizer:
                self.logger.debug("Applying accentizer")
                text_to_process = self.accentizer.process_all(text_to_process)

            if not text_to_process.endswith("."):
                text_to_process += "."

            self.logger.debug(f"Processed text: {text_to_process}")

            speed = self._get_tts_speed(text_to_process)
            self.logger.debug(f"Calculated speed: {speed}")

            self.logger.info("Starting TTS inference...")
            wav, sr, _ = infer_process(
                ref_file_processed,
                ref_text_processed,
                text_to_process,
                self.model_obj,
                self.vocoder,
                cross_fade_duration=0.15,
                nfe_step=64,
                speed=speed,
                device=self.config.device,
            )
            duration_sec = len(wav) / float(sr)
            self.logger.info(f"TTS inference completed. Audio shape: {wav.shape}, sample rate: {sr}, duration: {duration_sec} sec")

            soundfile.write(output_path, wav, sr)
            self.logger.debug(f"Audio saved to: {output_path}")

            background_tasks.add_task(self._cleanup_tts_file, output_path)
            self.logger.debug(f"Scheduled cleanup for: {output_path}")

            self.logger.info(f"Returning audio file: {output_filename}")
            
            
            # set in header file duration
            headers = {"X-Duration-Sec": str(duration_sec)}
            
            return FileResponse(
                path=output_path, media_type="audio/wav", filename=output_filename, headers=headers
            )

        except Exception as e:
            self.logger.error(f"TTS processing error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"TTS processing error: {str(e)}"
            )


def main():
    default_port = int(os.environ.get("PORT", "8080"))
    default_host = os.environ.get("HOST", "0.0.0.0")
    
    parser = argparse.ArgumentParser(description="F5-TTS API Server")
    parser.add_argument("--host", default=default_host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=default_port, help="Port to bind to")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--loglevel", default="info", choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    app = FastAPI(title="F5-TTS API", version="1.0.0")
    tts_config = TTSConfig()
    tts_api = TTSAPI(tts_config, app)

    logger.info(f"Starting F5-TTS API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level=args.loglevel)


if __name__ == "__main__":
    main()
