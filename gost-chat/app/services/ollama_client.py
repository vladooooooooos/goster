import logging
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached or returns an error."""


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout = settings.ollama_request_timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
                return True
        except httpx.HTTPError as exc:
            logger.warning("Ollama availability check failed: %s", exc)
            return False

    async def chat(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _extract_ollama_error_detail(exc.response)
            logger.error("Ollama returned an HTTP error: %s. Detail: %s", exc, detail)
            raise OllamaUnavailableError(f"Ollama returned an error: {detail}") from exc
        except httpx.HTTPError as exc:
            logger.error("Ollama request failed: %s", exc)
            raise OllamaUnavailableError("Ollama is unavailable. Check that it is running locally.") from exc

        data = response.json()
        content = data.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            logger.error("Ollama response did not include assistant content: %s", data)
            raise OllamaUnavailableError("Ollama returned an invalid response.")

        return content


def _extract_ollama_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        data = None

    if isinstance(data, dict):
        error = data.get("error") or data.get("detail")
        if isinstance(error, str) and error.strip():
            return error.strip()

    text = response.text.strip()
    if text:
        return text

    return f"HTTP {response.status_code}"
