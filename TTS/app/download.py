import os
import shutil
from pathlib import Path
from typing import Optional
import requests
from huggingface_hub import hf_hub_download, HfFileSystem
from fsspec import filesystem
import hashlib

from .config import get_config
from .logging import get_logger

logger = get_logger(__name__)


class DownloadError(Exception):
    """Custom exception for download errors"""
    pass


def get_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def download_http_url(url: str, local_path: Path, chunk_size: int = 8192) -> None:
    """Download file from HTTP/HTTPS URL"""
    logger.info(f"Downloading from HTTP URL: {url} -> {local_path}")

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            # Create parent directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        logger.info(f"Successfully downloaded {url} to {local_path}")

    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        # Clean up partial download
        if local_path.exists():
            local_path.unlink()
        raise DownloadError(f"Failed to download {url}: {e}")


def download_hf_file(hf_path: str, local_path: Path, repo_type: str = None) -> None:
    """Download file from Hugging Face Hub using HfFileSystem.resolve_path"""
    logger.info(f"Downloading from Hugging Face: {hf_path} -> {local_path}")

    try:
        # Parse hf:// path
        if hf_path.startswith("hf://"):
            hf_path = hf_path[5:]  # Remove hf:// prefix
        elif hf_path.startswith("hg://"):
            hf_path = hf_path[5:]  # Remove hg:// alias prefix

        # Split into repo_id and filename
        if "/" not in hf_path:
            raise DownloadError(f"Invalid Hugging Face path: {hf_path}")

        repo_id, filename = hf_path.split("/", 1)

        # Create parent directory if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Try to auto-detect repo_type if not specified
        if repo_type is None:
            fs = HfFileSystem()
            try:
                # Resolve the Hugging Face path to get proper repo_id, filename and repo_type
                full_path = f"hf://{hf_path}"
                resolved_path = fs.resolve_path(full_path)

                # Use the resolved information for download
                downloaded_path = hf_hub_download(
                    repo_id=resolved_path.repo_id,
                    filename=resolved_path.path_in_repo,
                    repo_type=resolved_path.repo_type,
                    local_files_only=False
                )

                logger.info(f"Successfully resolved and downloaded with repo_type: {resolved_path.repo_type}")

            except Exception as resolve_error:
                logger.warning(f"Failed to resolve path with HfFileSystem: {resolve_error}")
                # Try different repo types as fallback
                for possible_repo_type in ["model", "dataset", "space"]:
                    try:
                        downloaded_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            repo_type=possible_repo_type,
                            local_files_only=False
                        )
                        logger.info(f"Successfully found file with repo_type: {possible_repo_type}")
                        break
                    except Exception as try_error:
                        logger.debug(f"Failed with repo_type {possible_repo_type}: {try_error}")
                        continue
                else:
                    raise DownloadError(f"Could not find file in any repo_type for {hf_path}")
        else:
            # Download from Hugging Face Hub with specified repo_type
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_files_only=False
            )

        # Copy to target location (avoid symlinks)
        shutil.copy2(downloaded_path, local_path)
        # Remove temporary file
        if Path(downloaded_path).exists():
            Path(downloaded_path).unlink()

        logger.info(f"Successfully downloaded {hf_path} to {local_path}")

    except Exception as e:
        logger.error(f"Failed to download from Hugging Face {hf_path}: {e}")
        # Clean up partial download
        if local_path.exists():
            local_path.unlink()
        raise DownloadError(f"Failed to download from Hugging Face {hf_path}: {e}")


def copy_local_file(src_path: Path, dst_path: Path) -> None:
    """Copy local file to destination"""
    logger.info(f"Copying local file: {src_path} -> {dst_path}")

    try:
        # Create parent directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file (this will follow symlinks and copy the actual content)
        shutil.copy2(src_path, dst_path)

        logger.info(f"Successfully copied {src_path} to {dst_path}")

    except Exception as e:
        logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
        raise DownloadError(f"Failed to copy {src_path} to {dst_path}: {e}")


def resolve_path(path: str, base_dir: Optional[Path] = None) -> tuple[str, Path]:
    """Resolve path and return path type and local path"""
    config = get_config()

    if base_dir is None:
        base_dir = Path.cwd()

    path_obj = Path(path)

    # Check if it's already a local absolute path
    if path_obj.is_absolute():
        if path_obj.exists():
            return "local", path_obj
        else:
            raise DownloadError(f"Local file does not exist: {path_obj}")

    # Check if it's a remote URL
    if path.startswith(("http://", "https://")):
        return "http", base_dir / "downloads" / path_obj.name

    # Check if it's a Hugging Face path
    if path.startswith(("hf://", "hg://")):
        return "hf", base_dir / "downloads" / path_obj.name

    # Treat as local relative path
    local_path = base_dir / path_obj
    if local_path.exists():
        return "local", local_path
    else:
        raise DownloadError(f"Local file does not exist: {local_path}")


def ensure_file_available(source_path: str, target_dir: Path, filename: Optional[str] = None) -> Path:
    """Ensure file is available locally, download if needed"""
    config = get_config()

    if filename is None:
        filename = Path(source_path).name

    target_path = target_dir / filename

    # Check if file already exists and is valid
    if target_path.exists():
        try:
            # Try to read first few bytes to verify file is not corrupted
            with open(target_path, 'rb') as f:
                f.read(1024)
            logger.info(f"File already exists: {target_path}")
            return target_path
        except Exception as e:
            logger.warning(f"Existing file appears corrupted: {target_path}, re-downloading")
            target_path.unlink()

    # Resolve and download file
    try:
        path_type, resolved_path = resolve_path(source_path, target_dir.parent)

        if path_type == "local":
            copy_local_file(resolved_path, target_path)
        elif path_type == "http":
            download_http_url(source_path, target_path)
        elif path_type == "hf":
            download_hf_file(source_path, target_path)
        else:
            raise DownloadError(f"Unsupported path type: {path_type}")

        return target_path

    except Exception as e:
        logger.error(f"Failed to ensure file availability for {source_path}: {e}")
        if config.f5_tts.fail_fast:
            raise
        return None


def download_models_and_voices() -> None:
    """Download all required models and voice files"""
    config = get_config()

    if not config.f5_tts.download_on_start:
        logger.info("Download on start disabled, skipping downloads")
        return

    logger.info("Starting download of models and voices...")

    try:
        # Create directories
        models_dir = Path(config.f5_tts.models_dir)
        voices_dir = Path(config.f5_tts.voices_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        voices_dir.mkdir(parents=True, exist_ok=True)

        # Download vocab and weights
        logger.info("Downloading F5-TTS models...")
        vocab_path = ensure_file_available(config.f5_tts.vocab, models_dir, "vocab.txt")
        weights_path = ensure_file_available(config.f5_tts.weights, models_dir, "model.safetensors")

        if vocab_path is None or weights_path is None:
            raise DownloadError("Failed to download required model files")

        logger.info(f"Models downloaded successfully: vocab={vocab_path}, weights={weights_path}")

        # Download voice files
        logger.info("Downloading voice files...")
        for voice_id, voice_files in config.f5_tts.voices.items():
            logger.info(f"Processing voice: {voice_id}")

            # Download voice audio file
            voice_path = ensure_file_available(
                voice_files["voice"],
                voices_dir,
                f"{voice_id}.mp3"
            )

            # Download voice text file
            text_path = ensure_file_available(
                voice_files["text"],
                voices_dir,
                f"{voice_id}.txt"
            )

            if voice_path is None or text_path is None:
                raise DownloadError(f"Failed to download voice files for {voice_id}")

            logger.info(f"Voice {voice_id} downloaded successfully")

        logger.info("All models and voices downloaded successfully")

    except Exception as e:
        logger.error(f"Failed to download models and voices: {e}")
        if config.f5_tts.fail_fast:
            raise DownloadError(f"Failed to download models and voices: {e}")
        else:
            logger.warning("Continuing despite download errors (fail_fast=False)")