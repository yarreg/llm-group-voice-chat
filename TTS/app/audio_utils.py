import io
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment

from .config import get_config
from .logging import get_logger

logger = get_logger(__name__)


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass


class AudioUtils:
    """Utilities for audio processing"""

    def __init__(self):
        self.config = get_config()

    def calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS level of audio"""
        if len(audio) == 0:
            return -np.inf

        # Convert to float if needed
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)

        # Calculate RMS
        rms = np.sqrt(np.mean(audio ** 2))

        # Convert to dB
        if rms == 0:
            return -np.inf
        else:
            return 20 * np.log10(rms)

    def normalize_rms(self, audio: np.ndarray, target_rms: float = -18.0) -> np.ndarray:
        """Normalize audio to target RMS level"""
        if len(audio) == 0:
            return audio

        # Calculate current RMS
        current_rms = self.calculate_rms(audio)

        # If already at target RMS (or silent), return as-is
        if current_rms == -np.inf or abs(current_rms - target_rms) < 0.1:
            return audio

        # Calculate gain needed
        gain_db = target_rms - current_rms
        gain_linear = 10 ** (gain_db / 20)

        # Apply gain
        normalized = audio * gain_linear

        # Clip to prevent distortion
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            logger.warning(f"Audio clipped during normalization (max: {max_val:.3f})")
            normalized = normalized / max_val

        return normalized

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio

        try:
            # Convert to torch tensor
            if len(audio.shape) == 1:
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            else:
                audio_tensor = torch.from_numpy(audio).float()

            # Resample
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            resampled = resampler(audio_tensor)

            # Convert back to numpy
            if len(audio.shape) == 1:
                return resampled.squeeze(0).numpy()
            else:
                return resampled.numpy()

        except Exception as e:
            raise AudioProcessingError(f"Failed to resample audio: {e}")

    def cross_fade(self, audio1: np.ndarray, audio2: np.ndarray, duration: float, sr: int) -> np.ndarray:
        """Cross-fade two audio segments"""
        if duration <= 0:
            return np.concatenate([audio1, audio2])

        fade_samples = int(duration * sr)

        # Ensure we have enough samples for cross-fade
        fade_samples = min(fade_samples, len(audio1), len(audio2))

        if fade_samples <= 0:
            return np.concatenate([audio1, audio2])

        # Create fade curves
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        fade_in = np.linspace(0.0, 1.0, fade_samples)

        # Apply fades
        audio1_end = audio1[-fade_samples:] * fade_out
        audio2_start = audio2[:fade_samples] * fade_in

        # Combine
        cross_faded = audio1_end + audio2_start

        # Construct final audio
        result = np.concatenate([
            audio1[:-fade_samples],
            cross_faded,
            audio2[fade_samples:]
        ])

        return result

    def encode_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Encode audio as WAV"""
        try:
            # Create buffer
            buffer = io.BytesIO()

            # Write WAV
            sf.write(buffer, audio, sample_rate, format='WAV')

            return buffer.getvalue()

        except Exception as e:
            raise AudioProcessingError(f"Failed to encode WAV: {e}")

    def encode_mp3(self, audio: np.ndarray, sample_rate: int, bitrate: str = "192k") -> bytes:
        """Encode audio as MP3"""
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                sf.write(tmp_wav.name, audio, sample_rate, format='WAV')

                # Convert to MP3 using pydub
                wav_audio = AudioSegment.from_wav(tmp_wav.name)
                mp3_audio = wav_audio.export(format='mp3', bitrate=bitrate)

                # Read MP3 data
                mp3_data = mp3_audio.read()

                # Clean up
                Path(tmp_wav.name).unlink()

                return mp3_data

        except Exception as e:
            raise AudioProcessingError(f"Failed to encode MP3: {e}")

    def process_audio(self, audio: np.ndarray, orig_sr: int,
                     target_sr: Optional[int] = None,
                     normalize: bool = True,
                     target_rms: Optional[float] = None) -> np.ndarray:
        """Process audio with normalization and resampling"""
        config = self.config

        # Set defaults from config
        if target_sr is None:
            target_sr = config.f5_tts.default_sample_rate
        if target_rms is None:
            target_rms = config.f5_tts.target_rms

        processed_audio = audio

        # Resample if needed
        if orig_sr != target_sr:
            logger.info(f"Resampling audio from {orig_sr}Hz to {target_sr}Hz")
            processed_audio = self.resample_audio(processed_audio, orig_sr, target_sr)

        # Normalize if requested
        if normalize:
            logger.info(f"Normalizing audio to {target_rms}dB RMS")
            processed_audio = self.normalize_rms(processed_audio, target_rms)

        return processed_audio

    def get_audio_duration(self, audio: np.ndarray, sample_rate: int) -> float:
        """Get duration of audio in seconds"""
        return len(audio) / sample_rate

    def validate_audio(self, audio: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Validate audio data"""
        if len(audio) == 0:
            return False, "Audio is empty"

        if np.isnan(audio).any():
            return False, "Audio contains NaN values"

        if np.isinf(audio).any():
            return False, "Audio contains infinite values"

        if np.max(np.abs(audio)) > 1.0:
            logger.warning(f"Audio levels exceed 0dBFS (max: {np.max(np.abs(audio)):.3f})")

        return True, None

    def load_audio_file(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data with sample rate"""
        try:
            audio, sr = sf.read(str(file_path))

            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            return audio, sr

        except Exception as e:
            raise AudioProcessingError(f"Failed to load audio file {file_path}: {e}")

    def save_audio_file(self, audio: np.ndarray, sample_rate: int, file_path: Union[str, Path]) -> None:
        """Save audio to file"""
        try:
            sf.write(str(file_path), audio, sample_rate, format='WAV')
        except Exception as e:
            raise AudioProcessingError(f"Failed to save audio file {file_path}: {e}")

    def chunk_audio(self, audio: np.ndarray, chunk_duration: float, sr: int,
                   overlap_duration: float = 0.0) -> list:
        """Split audio into chunks with overlap"""
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap_duration * sr)

        if chunk_samples <= 0:
            return [audio]

        chunks = []
        start = 0

        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            # Pad last chunk if needed
            if len(chunk) < chunk_samples and start + chunk_samples < len(audio):
                padding = np.zeros(chunk_samples - len(chunk))
                chunk = np.concatenate([chunk, padding])

            chunks.append(chunk)

            start += chunk_samples - overlap_samples

        return chunks

    def merge_chunks(self, chunks: list, overlap_duration: float, sr: int) -> np.ndarray:
        """Merge audio chunks with cross-fade"""
        if len(chunks) <= 1:
            return chunks[0] if chunks else np.array([])

        result = chunks[0]
        overlap_samples = int(overlap_duration * sr)

        for chunk in chunks[1:]:
            if overlap_samples > 0 and len(result) > overlap_samples:
                # Cross-fade
                result = self.cross_fade(result, chunk, overlap_duration, sr)
            else:
                # Simple concatenation
                result = np.concatenate([result, chunk])

        return result


# Global audio utils instance
_audio_utils: Optional[AudioUtils] = None


def get_audio_utils() -> AudioUtils:
    """Get global audio utils instance"""
    global _audio_utils
    if _audio_utils is None:
        _audio_utils = AudioUtils()
    return _audio_utils


def initialize_audio_utils() -> None:
    """Initialize audio utilities"""
    logger.info("Initializing audio utilities...")

    try:
        # Create audio utils instance
        audio_utils = get_audio_utils()

        # Test with sample audio
        test_audio = np.random.randn(16000)  # 1 second of noise
        processed = audio_utils.process_audio(test_audio, 16000, 24000, True, -18.0)

        logger.info(f"Audio utilities test successful: processed {len(processed)} samples")

        logger.info("Audio utilities initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize audio utilities: {e}")
        if get_config().f5_tts.fail_fast:
            raise AudioProcessingError(f"Failed to initialize audio utilities: {e}")
        else:
            logger.warning("Continuing without audio utilities")