import logging
import os
from typing import Any, Protocol

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class LlmServiceError(RuntimeError):
    """Raised when an LLM provider cannot produce a usable response."""


class MissingLlmApiKeyError(LlmServiceError):
    """Raised when the configured LLM provider API key is missing."""


class UnsupportedLlmProviderError(LlmServiceError):
    """Raised when the configured LLM provider is not implemented."""


class LlmService(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def is_available(self) -> bool: ...

    async def chat(self, messages: list[dict[str, Any]]) -> str: ...

    async def chat_with_images(self, messages: list[dict[str, Any]]) -> str: ...


class PolzaLlmService:
    def __init__(self, settings: Settings) -> None:
        self._provider = "polza"
        self._base_url = settings.llm_base_url.rstrip("/")
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self._timeout = settings.llm_request_timeout_seconds
        self._api_key_env_var = settings.llm_api_key_env_var

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    async def is_available(self) -> bool:
        try:
            api_key = self._require_api_key()
            async with httpx.AsyncClient(timeout=min(self._timeout, 10.0)) as client:
                response = await client.get(
                    f"{self._base_url}/models",
                    headers=self._headers(api_key),
                )
                response.raise_for_status()
            return True
        except MissingLlmApiKeyError as exc:
            logger.warning("LLM availability check failed: %s", exc)
            return False
        except httpx.HTTPError as exc:
            logger.warning(
                "LLM availability check failed for provider=%s model=%s: %s",
                self._provider,
                self._model,
                exc,
            )
            return False

    async def chat(self, messages: list[dict[str, Any]]) -> str:
        payload = self.build_chat_payload(messages)
        return await self._post_chat_payload(payload)

    async def chat_with_images(self, messages: list[dict[str, Any]]) -> str:
        payload = self.build_chat_payload(messages)
        return await self._post_chat_payload(payload)

    def build_chat_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }

    async def _post_chat_payload(self, payload: dict[str, Any]) -> str:
        api_key = self._require_api_key()

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            logger.error(
                "LLM request timed out for provider=%s model=%s timeout=%ss",
                self._provider,
                self._model,
                self._timeout,
            )
            raise LlmServiceError("LLM provider request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            detail = _extract_openai_error_detail(exc.response)
            logger.error(
                "LLM provider returned HTTP %s for provider=%s model=%s. Detail: %s",
                exc.response.status_code,
                self._provider,
                self._model,
                detail,
            )
            raise LlmServiceError(f"LLM provider returned an error: {detail}") from exc
        except httpx.HTTPError as exc:
            logger.error("LLM request failed for provider=%s model=%s: %s", self._provider, self._model, exc)
            raise LlmServiceError("LLM provider is unavailable.") from exc

        data = response.json()
        content = _extract_chat_completion_content(data)
        if not content:
            logger.error(
                "LLM provider returned an empty assistant response for provider=%s model=%s",
                self._provider,
                self._model,
            )
            raise LlmServiceError("LLM provider returned an empty response.")

        return content

    def _require_api_key(self) -> str:
        api_key = os.getenv(self._api_key_env_var)
        if not api_key:
            logger.error("Missing LLM API key in environment variable %s.", self._api_key_env_var)
            raise MissingLlmApiKeyError(f"Missing LLM API key. Set {self._api_key_env_var}.")
        return api_key

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }


def create_llm_service(settings: Settings) -> LlmService:
    provider = settings.llm_provider.strip().lower()
    if provider == "polza":
        return PolzaLlmService(settings)

    raise UnsupportedLlmProviderError(f"Unsupported LLM provider: {settings.llm_provider}")


def _extract_chat_completion_content(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    text = first_choice.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    return None


def _extract_openai_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        data = None

    if isinstance(data, dict):
        error = data.get("error") or data.get("detail")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code") or error.get("type")
            if isinstance(message, str) and message.strip():
                return message.strip()
        if isinstance(error, str) and error.strip():
            return error.strip()

    text = response.text.strip()
    if text:
        return text[:500]

    return f"HTTP {response.status_code}"
