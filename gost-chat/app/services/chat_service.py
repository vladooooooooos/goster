import logging

from app.services.llm_service import LlmService

logger = logging.getLogger(__name__)


class EmptyMessageError(ValueError):
    """Raised when the user sends an empty chat message."""


class ChatService:
    def __init__(self, llm_service: LlmService) -> None:
        self._llm_service = llm_service
        self._messages: list[dict[str, str]] = []

    @property
    def model(self) -> str:
        return self._llm_service.model

    @property
    def provider(self) -> str:
        return self._llm_service.provider

    async def is_model_available(self) -> bool:
        return await self._llm_service.is_available()

    async def send_message(self, message: str) -> str:
        user_message = message.strip()
        if not user_message:
            raise EmptyMessageError("Message cannot be empty.")

        self._messages.append({"role": "user", "content": user_message})
        logger.info(
            "Sending chat request to LLM provider=%s model=%s with %s in-memory messages.",
            self.provider,
            self.model,
            len(self._messages),
        )

        assistant_response = await self._llm_service.chat(self._messages)
        self._messages.append({"role": "assistant", "content": assistant_response})

        return assistant_response
