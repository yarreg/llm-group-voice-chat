import io
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torchaudio

from .config import get_config
from .voices import get_voice_manager, VoiceError
from .accents import get_accent_processor, AccentError
from .audio_utils import get_audio_utils, AudioProcessingError
from .logging import get_logger, log_execution_time, RequestLogger

logger = get_logger(__name__)


class TTSInferenceError(Exception):
    """Custom exception for TTS inference errors"""
    pass


class TTSPipeline:
    """Main TTS pipeline combining all components"""

    def __init__(self):
        self.config = get_config()
        self.voice_manager = get_voice_manager()
        self.accent_processor = get_accent_processor()
        self.audio_utils = get_audio_utils()

        # F5-TTS model and vocoder
        self.model = None
        self.vocoder = None
        self.device = self._get_device()

        # Load models
        self._load_models()

    def _get_device(self) -> str:
        """Get best available device for inference"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("Using CPU device")
        return device

    def _load_models(self) -> None:
        """Load F5-TTS model and vocoder"""
        try:
            logger.info("Loading F5-TTS models...")

            # Import F5-TTS
            from f5_tts.model import DiT
            from f5_tts.infer.utils_infer import load_vocoder

            # Get model paths
            models_dir = Path(self.config.f5_tts.models_dir)
            vocab_path = models_dir / "vocab.txt"
            weights_path = models_dir / "model.safetensors"

            if not vocab_path.exists():
                raise TTSInferenceError(f"Vocab file not found: {vocab_path}")
            if not weights_path.exists():
                raise TTSInferenceError(f"Model weights not found: {weights_path}")

            # Load model
            self.model = DiT(
                text_num_embeds=len(open(vocab_path).readlines()),
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                conv_layers=4,
                dropout=0.1
            )

            # Load weights
            try:
                from safetensors import safe_open
                state_dict = {}
                with safe_open(weights_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            except ImportError:
                # Fallback to torch.load if safetensors not available
                state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)

            # Remove 'ema_model.' prefix from keys if present
            if any(key.startswith('ema_model.') for key in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('ema_model.'):
                        new_key = key[len('ema_model.'):]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict

            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Load vocoder
            self.vocoder = load_vocoder(self.config.f5_tts.vocoder_name, self.device)

            logger.info("F5-TTS models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load F5-TTS models: {e}")
            if self.config.f5_tts.fail_fast:
                raise TTSInferenceError(f"Failed to load F5-TTS models: {e}")
            else:
                logger.warning("Continuing without TTS models")

    def validate_inputs(self, text: str, voice_id: str) -> Tuple[bool, Optional[str]]:
        """Validate TTS inputs"""
        if not text or not text.strip():
            return False, "Text is empty"

        if len(text) > self.config.limits.max_text_chars:
            return False, f"Text too long: {len(text)} > {self.config.limits.max_text_chars}"

        if not self.voice_manager.is_voice_available(voice_id):
            return False, f"Voice not available: {voice_id}"

        return True, None

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for TTS"""
        # Apply accent processing if enabled
        if self.config.accent.enabled:
            try:
                text = self.accent_processor.process_text(text)
            except Exception as e:
                logger.warning(f"Accent processing failed: {e}")

        # Basic text cleaning
        text = text.strip()
        if not text:
            raise TTSInferenceError("Text became empty after preprocessing")

        return text

    def chunk_text(self, text: str, max_chars: int = 200) -> list:
        """Split text into chunks for processing"""
        # Simple sentence-based chunking
        sentences = text.split('.')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Add period back if not the last sentence
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += sentence + '.'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '.'

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    @log_execution_time(logger, "TTS inference")
    def synthesize_speech(self, text: str, voice_id: str,
                         format: str = "wav",
                         sample_rate: Optional[int] = None,
                         normalize: bool = True,
                         seed: Optional[int] = None) -> bytes:
        """Main TTS synthesis method"""
        request_logger = RequestLogger()

        try:
            # Validate inputs
            valid, error_msg = self.validate_inputs(text, voice_id)
            if not valid:
                raise TTSInferenceError(error_msg)

            # Log request
            request_logger.info(
                "TTS request started",
                voice_id=voice_id,
                text_length=len(text),
                format=format,
                sample_rate=sample_rate or self.config.f5_tts.default_sample_rate
            )

            # Preprocess text
            processed_text = self.preprocess_text(text)
            request_logger.info("Text preprocessed successfully")

            # Get voice data
            try:
                ref_audio, ref_text = self.voice_manager.get_processed_voice_data(voice_id)
            except Exception as e:
                raise TTSInferenceError(f"Failed to get voice data for {voice_id}: {e}")

            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            # Perform inference
            start_time = time.time()
            audio_data = self._run_inference(processed_text, ref_audio, ref_text)
            inference_time = time.time() - start_time

            # Process audio (normalize, resample)
            orig_sr = self.config.f5_tts.default_sample_rate
            target_sr = sample_rate or self.config.f5_tts.default_sample_rate

            audio_data = self.audio_utils.process_audio(
                audio_data, orig_sr, target_sr, normalize
            )

            # Encode to requested format
            if format.lower() == "wav":
                audio_bytes = self.audio_utils.encode_wav(audio_data, target_sr)
                content_type = "audio/wav"
                file_extension = "wav"
            elif format.lower() == "mp3":
                audio_bytes = self.audio_utils.encode_mp3(audio_data, target_sr)
                content_type = "audio/mpeg"
                file_extension = "mp3"
            else:
                raise TTSInferenceError(f"Unsupported audio format: {format}")

            # Calculate audio duration
            duration = self.audio_utils.get_audio_duration(audio_data, target_sr)

            # Log completion
            request_logger.info(
                "TTS request completed",
                audio_duration_ms=int(duration * 1000),
                inference_time_ms=int(inference_time * 1000),
                audio_size_bytes=len(audio_bytes)
            )

            logger.info(
                f"TTS synthesis completed: {voice_id}, {duration:.2f}s, "
                f"{inference_time:.2f}s inference, {len(audio_bytes)} bytes"
            )

            return audio_bytes

        except Exception as e:
            request_logger.error("TTS request failed", error=str(e))
            logger.error(f"TTS synthesis failed: {e}")
            raise TTSInferenceError(f"TTS synthesis failed: {e}")

    def _run_inference(self, text: str, ref_audio: np.ndarray, ref_text: str) -> np.ndarray:
        """Run F5-TTS inference"""
        try:
            from f5_tts.infer.utils_infer import infer_process

            # Prepare reference audio
            if len(ref_audio.shape) == 1:
                ref_audio = ref_audio.reshape(1, -1)

            # Get inference parameters from config
            params = {
                "nfe_step": self.config.f5_tts.nfe_step,
                "cfg_strength": self.config.f5_tts.cfg_strength,
                "sway_sampling_coef": self.config.f5_tts.sway_sampling_coef,
                "speed": self.config.f5_tts.speed,
                "fix_duration": self.config.f5_tts.fix_duration,
            }

            # Run inference
            result = infer_process(
                self.model,
                self.vocoder,
                ref_audio,
                ref_text,
                text,
                device=self.device,
                **params
            )

            # Convert to numpy
            if isinstance(result, torch.Tensor):
                result = result.cpu().numpy()

            # Ensure 1D array
            if len(result.shape) > 1:
                result = result.squeeze()

            return result

        except Exception as e:
            raise TTSInferenceError(f"F5-TTS inference failed: {e}")

    def get_available_voices(self) -> list:
        """Get list of available voices"""
        return self.voice_manager.list_voices()

    def get_available_voices_detailed(self) -> list:
        """Get detailed list of available voices"""
        return self.voice_manager.list_voices_detailed()

    def health_check(self) -> dict:
        """Perform health check of TTS pipeline"""
        status = {
            "status": "ok",
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "voices_loaded": len(self.voice_manager.list_voices()),
            "model_loaded": self.model is not None,
            "vocoder_loaded": self.vocoder is not None,
            "accent_enabled": self.config.accent.enabled,
        }

        # Add GPU info if available
        if torch.cuda.is_available():
            status["cuda_device_name"] = torch.cuda.get_device_name()
            status["cuda_memory_allocated"] = torch.cuda.memory_allocated()
            status["cuda_memory_reserved"] = torch.cuda.memory_reserved()

        return status


# Global TTS pipeline instance
_tts_pipeline: Optional[TTSPipeline] = None


def get_tts_pipeline() -> TTSPipeline:
    """Get global TTS pipeline instance"""
    global _tts_pipeline
    if _tts_pipeline is None:
        _tts_pipeline = TTSPipeline()
    return _tts_pipeline


def initialize_tts_pipeline() -> None:
    """Initialize TTS pipeline"""
    logger.info("Initializing TTS pipeline...")

    try:
        # Create TTS pipeline
        pipeline = get_tts_pipeline()

        # Perform health check
        health_status = pipeline.health_check()
        logger.info(f"TTS pipeline health check: {health_status}")

        if health_status["status"] != "ok":
            raise TTSInferenceError("TTS pipeline health check failed")

        logger.info("TTS pipeline initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize TTS pipeline: {e}")
        if get_config().f5_tts.fail_fast:
            raise TTSInferenceError(f"Failed to initialize TTS pipeline: {e}")
        else:
            logger.warning("Continuing with incomplete TTS pipeline initialization")


def synthesize_speech(text: str, voice_id: str, **kwargs) -> bytes:
    """Convenience function for speech synthesis"""
    pipeline = get_tts_pipeline()
    return pipeline.synthesize_speech(text, voice_id, **kwargs)