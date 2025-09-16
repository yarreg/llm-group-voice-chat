import os
import re
from pathlib import Path
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, validator
import yaml


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO", pattern=r"DEBUG|INFO|WARNING|ERROR")


class F5TTSConfig(BaseModel):
    vocab: str
    weights: str
    models_dir: str = "models"
    voices_dir: str = "voices"
    cache_dir: str = "cache"
    download_on_start: bool = True
    preprocess_on_start: bool = True
    fail_fast: bool = True
    vocoder_name: str = "vocos"
    default_sample_rate: int = 24000
    target_rms: float = -18.0
    cross_fade_duration: float = 0.08
    nfe_step: int = 16
    cfg_strength: float = 2.0
    sway_sampling_coef: float = 1.0
    speed: float = 1.0
    fix_duration: Optional[float] = None
    voices: Dict[str, Dict[str, str]] = {}

    @validator("voices")
    def validate_voices(cls, v):
        for voice_id, files in v.items():
            if not re.match(r"^RU_(Male|Female)_[a-z_]+$", voice_id, re.IGNORECASE):
                raise ValueError(f"Invalid voice_id format: {voice_id}")
            if "voice" not in files or "text" not in files:
                raise ValueError(f"Voice {voice_id} must have both 'voice' and 'text' keys")
        return v


class AccentConfig(BaseModel):
    enabled: bool = True
    overrides: Dict[str, str] = {}


class LimitsConfig(BaseModel):
    max_text_chars: int = 2000
    timeout_s: int = 120


class Config(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    f5_tts: F5TTSConfig
    accent: AccentConfig = Field(default_factory=AccentConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_absolute_path(self, path: str, base_dir: Optional[str] = None) -> Path:
        """Convert relative paths to absolute paths"""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj

        if base_dir is None:
            base_dir = Path.cwd()
        else:
            base_dir = Path(base_dir)

        return base_dir / path_obj


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance"""
    global _config
    if _config is None:
        config_path = os.getenv("CONFIG_PATH", "config.yaml")
        _config = Config.from_yaml(config_path)
    return _config


def set_config(config: Config) -> None:
    """Set global config instance"""
    global _config
    _config = config