import os
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator

HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)


class PipelineTimeouts(BaseModel):
    tts_sec: int = Field(default=60, ge=1)
    flp_sec: int = Field(default=120, ge=1)


class PipelineRetries(BaseModel):
    attempts: int = Field(default=3, ge=1)
    backoff_sec: float = Field(default=1.5, gt=0)


class PipelineConfig(BaseModel):
    timeouts: PipelineTimeouts = PipelineTimeouts()
    retries: PipelineRetries = PipelineRetries()


class LoggingConfig(BaseModel):
    level: str = Field(default="info")

    @field_validator("level")
    @classmethod
    def _validate_level(cls, value: str) -> str:
        allowed = {"debug", "info", "warning", "error"}
        value_lower = value.lower()
        if value_lower not in allowed:
            raise ValueError(f"Unsupported logging level '{value}'. Allowed: {sorted(allowed)}")
        return value_lower


class LLMModelConfig(BaseModel):
    model_name: str
    client_params: dict[str, object] = Field(default_factory=dict)

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True


class LLMConfig(BaseModel):
    default_model: str
    models: list[LLMModelConfig]

    @field_validator("models")
    @classmethod
    def _ensure_models(cls, value: list[LLMModelConfig]) -> list[LLMModelConfig]:
        if not value:
            raise ValueError("At least one LLM model must be configured")
        seen = set()
        for model in value:
            if model.model_name in seen:
                raise ValueError(f"Duplicate llm model name '{model.model_name}'")
            seen.add(model.model_name)
        return value

    def get_model(self, name: str | None = None) -> LLMModelConfig:
        target = name or self.default_model
        for model in self.models:
            if model.model_name == target:
                return model
        raise KeyError(f"LLM model '{target}' not configured")


class TTSVoiceConfig(BaseModel):
    voice_id: str


class TTSConfig(BaseModel):
    base_url: str
    default_voice: TTSVoiceConfig


class FLPConfig(BaseModel):
    base_url: str


class AppConfig(BaseModel):
    schema_version: int = Field(..., alias="schemaVersion")
    server: ServerConfig = ServerConfig()
    pipeline: PipelineConfig = PipelineConfig()
    logging: LoggingConfig = LoggingConfig()
    llm: LLMConfig
    tts: TTSConfig
    flp: FLPConfig

    class Config:
        allow_population_by_field_name = True


class ConversationConfig(BaseModel):
    id: str
    title: str
    mode: str = Field("round_robin")
    max_turns: int = Field(..., ge=1)
    start_speaker: str
    termination_regex: str | None = None
    allow_user_injection: bool = True
    allow_interrupt: bool = True

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        allowed = {"round_robin", "free"}
        if value not in allowed:
            raise ValueError(f"Unsupported conversation mode '{value}'. Allowed: {sorted(allowed)}")
        return value


class ScriptMessage(BaseModel):
    role: str
    content: str
    speaker: str | None = None

    @field_validator("role")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        allowed = {"system", "user", "assistant"}
        if value not in allowed:
            raise ValueError(f"Unsupported script role '{value}'. Allowed: {sorted(allowed)}")
        return value


class ActorLLMConfig(BaseModel):
    provider: str | None = None
    model: str | None = None


class ActorVoiceConfig(BaseModel):
    voice_id: str | None = None


class ActorConfig(BaseModel):
    id: str
    name: str
    color: str
    avatar_image: str
    system_prompt: str
    extra_prompt: str | None = None
    llm: ActorLLMConfig | None = None
    llm_model: str | None = None
    voice: ActorVoiceConfig | None = None

    @field_validator("color")
    @classmethod
    def _validate_color(cls, value: str) -> str:
        if not HEX_COLOR_RE.match(value):
            raise ValueError(f"Invalid color '{value}', expected format #RRGGBB")
        return value


class ScenePrompts(BaseModel):
    global_system: str


class SceneScript(BaseModel):
    seed_messages: list[ScriptMessage] = Field(default_factory=list)


class SceneConfig(BaseModel):
    schema_version: int = Field(..., alias="schemaVersion")
    conversation: ConversationConfig
    prompts: ScenePrompts
    actors: list[ActorConfig]
    script: SceneScript = Field(default_factory=SceneScript)

    class Config:
        allow_population_by_field_name = True

    @field_validator("actors")
    @classmethod
    def _validate_actors(cls, value: list[ActorConfig], info: ValidationInfo) -> list[ActorConfig]:
        if len(value) < 2:
            raise ValueError("At least two actors must be configured")
        ids = set()
        names = set()
        for actor in value:
            if actor.id in ids:
                raise ValueError(f"Duplicate actor id '{actor.id}'")
            if actor.name in names:
                raise ValueError(f"Duplicate actor name '{actor.name}'")
            ids.add(actor.id)
            names.add(actor.name)

        conversation: ConversationConfig = info.data.get("conversation")  # type: ignore[assignment]
        if conversation:
            actor_names = {actor.name for actor in value}
            if conversation.start_speaker not in actor_names:
                raise ValueError(f"start_speaker '{conversation.start_speaker}' is not one of configured actor names")
        return value


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def load_app_config(path: Path) -> AppConfig:
    raw = _load_yaml(path)
    app_config = AppConfig.parse_obj(raw)

    # Validate that each configured model has an API key available
    missing_keys: list[str] = []
    for model in app_config.llm.models:
        params = model.client_params or {}
        api_key = params.get("api_key")
        env_var = params.get("api_key_env")
        if not api_key and env_var:
            api_key = os.getenv(str(env_var))
        if not api_key:
            missing_keys.append(model.model_name)

    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing API keys for LLM models: {missing}")

    return app_config


def load_scene_config(path: Path) -> SceneConfig:
    raw = _load_yaml(path)
    scene_config = SceneConfig.parse_obj(raw)

    config_dir = path.parent
    missing_files: list[str] = []
    for actor in scene_config.actors:
        avatar_path = (config_dir / actor.avatar_image).resolve()
        if not avatar_path.exists():
            missing_files.append(actor.avatar_image)
        else:
            actor.avatar_image = str(avatar_path)
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Actor avatar files not found: {missing}")

    return scene_config
