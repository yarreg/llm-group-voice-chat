"""Autogen-backed text generation using OpenAI chat completion client."""

import os
from collections.abc import Iterable
from typing import Any, Optional, cast

from autogen_core._cancellation_token import CancellationToken
from autogen_core.models import AssistantMessage, ModelInfo, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import ActorConfig, AppConfig


class AutoGenEngine:
    """Generates dialogue turns for scene actors via Autogen model clients."""

    def __init__(self, app_config: AppConfig):
        self._app_config = app_config
        self._clients: dict[str, OpenAIChatCompletionClient] = {}
        self._system_prompts: dict[str, str] = {}
        self._register_model_clients()

    def _register_model_clients(self) -> None:
        for model in self._app_config.llm.models:
            # Work with a mutable, loosely-typed copy for client construction
            params: dict[str, Any] = dict(model.client_params or {})
            api_key = params.pop("api_key", None)
            env_var = params.pop("api_key_env", None)
            if not api_key and env_var:
                api_key = os.getenv(str(env_var))
            if not api_key:
                raise ValueError(f"Missing API key for LLM model '{model.model_name}'")

            model_name = cast(str, params.pop("model", None)) or model.model_name
            client_kwargs: dict[str, Any] = {
                "model": model_name,
                "api_key": api_key,
            }

            base_url = params.pop("api_base", None) or params.pop("base_url", None)
            if base_url:
                client_kwargs["base_url"] = base_url
            organization = params.pop("organization", None)
            if organization:
                client_kwargs["organization"] = organization

            supports_function_calling = bool(params.pop("supports_function_calling", True))
            supports_vision = bool(params.pop("supports_vision", False))
            supports_json_output = bool(params.pop("supports_json_output", False))
            supports_structured_output = bool(params.pop("supports_structured_output", False))
            multiple_system_messages = params.pop("supports_multiple_system_messages", None)
            # Ensure the value is a plain string to satisfy ModelInfo typing
            family_value: str = str(params.pop("model_family", "unknown"))

            model_info: ModelInfo = {
                "vision": supports_vision,
                "function_calling": supports_function_calling,
                "json_output": supports_json_output,
                "structured_output": supports_structured_output,
                "family": family_value,
            }
            if multiple_system_messages is not None:
                model_info["multiple_system_messages"] = bool(multiple_system_messages)

            client_kwargs["model_info"] = model_info

            optional_kwargs = [
                "timeout",
                "max_retries",
                "default_headers",
                "add_name_prefixes",
                "include_name_in_message",
                "parallel_tool_calls",
                "user",
            ]
            for key in optional_kwargs:
                if key in params:
                    client_kwargs[key] = params.pop(key)

            client_kwargs.update(params)

            # Some parameters are accepted by the runtime client but not captured
            # precisely in type hints; cast to Any to keep mypy satisfied.
            client_ctor = cast(Any, OpenAIChatCompletionClient)
            self._clients[model.model_name] = client_ctor(**client_kwargs)

    @staticmethod
    def _sanitize_source(source: Optional[str]) -> str:
        # Normalize None/empty values to a default non-empty label.
        return source if source else "agent"

    def register_actor(self, actor: ActorConfig, system_prompt: str) -> None:
        resolved_model = self._resolve_model_name(actor)
        if resolved_model not in self._clients:
            raise KeyError(f"LLM model '{resolved_model}' referenced by actor '{actor.name}' is not configured")
        self._system_prompts[actor.id] = system_prompt

    async def generate(self, actor: ActorConfig, messages: list[dict[str, str]]) -> str:
        client_key = self._resolve_model_name(actor)
        client = self._clients[client_key]
        autogen_messages = self._build_messages(actor.id, messages)

        result = await client.create(autogen_messages, cancellation_token=CancellationToken())
        content = result.content
        if isinstance(content, list):
            return "\n".join(str(item) for item in content).strip()
        return str(content).strip()

    def _build_messages(
        self, actor_id: str, messages: Iterable[dict[str, str]]
    ) -> list[SystemMessage | AssistantMessage | UserMessage]:
        system_prompt = self._system_prompts.get(actor_id, "")
        converted: list[SystemMessage | AssistantMessage | UserMessage] = (
            [SystemMessage(content=system_prompt)] if system_prompt else []
        )
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            source = message.get("source") or message.get("name")
            source = self._sanitize_source(source)
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AssistantMessage(content=content, source=source))
            else:
                converted.append(UserMessage(content=content, source=source))
        return converted

    def _resolve_model_name(self, actor: ActorConfig) -> str:
        if actor.llm and actor.llm.model:
            return actor.llm.model
        if actor.llm_model:
            return actor.llm_model
        return self._app_config.llm.default_model


__all__ = ["AutoGenEngine"]
