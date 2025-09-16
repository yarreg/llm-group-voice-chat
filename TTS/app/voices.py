import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import hashlib

from .config import get_config
from .download import DownloadError
from .logging import get_logger

logger = get_logger(__name__)


class VoiceError(Exception):
    """Custom exception for voice-related errors"""
    pass


class VoiceInfo:
    """Information about a voice"""

    def __init__(self, voice_id: str, voice_path: Path, text_path: Path):
        self.voice_id = voice_id
        self.voice_path = voice_path
        self.text_path = text_path
        self.processed = False
        self.cache_path: Optional[Path] = None

    def validate(self) -> bool:
        """Validate that voice files exist and are readable"""
        if not self.voice_path.exists():
            logger.error(f"Voice file not found: {self.voice_path}")
            return False

        if not self.text_path.exists():
            logger.error(f"Text file not found: {self.text_path}")
            return False

        try:
            # Try to read audio file
            import soundfile as sf
            sf.read(str(self.voice_path))
        except Exception as e:
            logger.error(f"Cannot read voice audio file {self.voice_path}: {e}")
            return False

        try:
            # Try to read text file
            with open(self.text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if not text:
                logger.error(f"Text file is empty: {self.text_path}")
                return False
        except Exception as e:
            logger.error(f"Cannot read voice text file {self.text_path}: {e}")
            return False

        return True

    def get_reference_text(self) -> str:
        """Get reference text for this voice"""
        try:
            with open(self.text_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise VoiceError(f"Cannot read reference text for {self.voice_id}: {e}")

    def get_cache_key(self) -> str:
        """Generate cache key for this voice"""
        # Use file paths, sizes, and modification times as cache key
        voice_stat = self.voice_path.stat()
        text_stat = self.text_path.stat()

        key_data = f"{self.voice_id}_{self.voice_path}_{voice_stat.st_size}_{voice_stat.st_mtime}_{text_stat.st_size}_{text_stat.st_mtime}"
        return hashlib.md5(key_data.encode()).hexdigest()


class VoiceManager:
    """Manages voice processing and caching"""

    def __init__(self):
        self.config = get_config()
        self.voices: Dict[str, VoiceInfo] = {}
        self.cache_dir = Path(self.config.f5_tts.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_voices(self) -> None:
        """Load all voices from config"""
        logger.info("Loading voices from config...")

        voices_dir = Path(self.config.f5_tts.voices_dir)

        for voice_id, voice_files in self.config.f5_tts.voices.items():
            logger.info(f"Loading voice: {voice_id}")

            # Determine file paths
            if Path(voice_files["voice"]).is_absolute():
                voice_path = Path(voice_files["voice"])
            else:
                voice_path = voices_dir / f"{voice_id}.mp3"

            if Path(voice_files["text"]).is_absolute():
                text_path = Path(voice_files["text"])
            else:
                text_path = voices_dir / f"{voice_id}.txt"

            # Create voice info
            voice_info = VoiceInfo(voice_id, voice_path, text_path)

            # Validate voice files
            if not voice_info.validate():
                error_msg = f"Voice {voice_id} validation failed"
                logger.error(error_msg)
                if self.config.f5_tts.fail_fast:
                    raise VoiceError(error_msg)
                continue

            self.voices[voice_id] = voice_info
            logger.info(f"Voice {voice_id} loaded successfully")

        logger.info(f"Loaded {len(self.voices)} voices")

    def get_voice(self, voice_id: str) -> VoiceInfo:
        """Get voice by ID"""
        if voice_id not in self.voices:
            raise VoiceError(f"Voice not found: {voice_id}")
        return self.voices[voice_id]

    def list_voices(self) -> List[str]:
        """List all available voice IDs"""
        return list(self.voices.keys())

    def list_voices_detailed(self) -> List[Dict[str, str]]:
        """List all voices with detailed information"""
        result = []
        for voice_id, voice_info in self.voices.items():
            result.append({
                "name": voice_id,
                "files": {
                    "voice": str(voice_info.voice_path),
                    "text": str(voice_info.text_path)
                }
            })
        return result

    def preprocess_all_voices(self) -> None:
        """Preprocess all voices for faster inference"""
        if not self.config.f5_tts.preprocess_on_start:
            logger.info("Voice preprocessing disabled, skipping")
            return

        logger.info("Preprocessing all voices...")

        try:
            # Import here to avoid dependency issues if not installed
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text

            for voice_id, voice_info in self.voices.items():
                logger.info(f"Preprocessing voice: {voice_id}")

                try:
                    # Check cache
                    cache_key = voice_info.get_cache_key()
                    cache_path = self.cache_dir / f"{voice_id}_{cache_key}.pkl"

                    if cache_path.exists():
                        logger.info(f"Using cached preprocessing for {voice_id}")
                        voice_info.cache_path = cache_path
                        voice_info.processed = True
                        continue

                    # Preprocess reference audio and text
                    ref_audio, ref_text = preprocess_ref_audio_text(
                        str(voice_info.voice_path),
                        voice_info.get_reference_text()
                    )

                    # Cache the result
                    cache_data = {
                        "ref_audio": ref_audio,
                        "ref_text": ref_text,
                        "voice_id": voice_id
                    }

                    with open(cache_path, 'wb') as f:
                        pickle.dump(cache_data, f)

                    voice_info.cache_path = cache_path
                    voice_info.processed = True

                    logger.info(f"Voice {voice_id} preprocessed successfully")

                except Exception as e:
                    logger.error(f"Failed to preprocess voice {voice_id}: {e}")
                    if self.config.f5_tts.fail_fast:
                        raise VoiceError(f"Failed to preprocess voice {voice_id}: {e}")
                    else:
                        logger.warning(f"Continuing without preprocessing for {voice_id}")

            logger.info("All voices preprocessed successfully")

        except ImportError as e:
            logger.error(f"Failed to import F5-TTS preprocessing: {e}")
            if self.config.f5_tts.fail_fast:
                raise VoiceError(f"Failed to import F5-TTS preprocessing: {e}")
            else:
                logger.warning("Continuing without voice preprocessing")

    def get_processed_voice_data(self, voice_id: str) -> Tuple:
        """Get processed voice data for inference"""
        voice_info = self.get_voice(voice_id)

        if voice_info.cache_path and voice_info.cache_path.exists():
            try:
                with open(voice_info.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                return cache_data["ref_audio"], cache_data["ref_text"]
            except Exception as e:
                logger.warning(f"Failed to load cached data for {voice_id}: {e}")

        # Fallback to real-time processing
        try:
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
            ref_audio, ref_text = preprocess_ref_audio_text(
                str(voice_info.voice_path),
                voice_info.get_reference_text()
            )
            return ref_audio, ref_text
        except Exception as e:
            raise VoiceError(f"Failed to process voice {voice_id}: {e}")

    def is_voice_available(self, voice_id: str) -> bool:
        """Check if voice is available"""
        return voice_id in self.voices

    def validate_voice_id(self, voice_id: str) -> bool:
        """Validate voice ID format"""
        return bool(re.match(r"^RU_(Male|Female)_[a-z_]+$", voice_id, re.IGNORECASE))


# Global voice manager instance
_voice_manager: Optional[VoiceManager] = None


def get_voice_manager() -> VoiceManager:
    """Get global voice manager instance"""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = VoiceManager()
    return _voice_manager


def initialize_voices() -> None:
    """Initialize voice system"""
    logger.info("Initializing voice system...")

    try:
        # Download required files if needed
        from .download import download_models_and_voices
        download_models_and_voices()

        # Load voices
        voice_manager = get_voice_manager()
        voice_manager.load_voices()

        # Preprocess voices
        voice_manager.preprocess_all_voices()

        logger.info("Voice system initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize voice system: {e}")
        if get_config().f5_tts.fail_fast:
            raise
        else:
            logger.warning("Continuing with partial voice system initialization")