from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.orchestration.chat_store import ChatStore
from app.orchestration.tool_contracts import ToolContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatTurnResponse:
    session_id: str
    message_id: str
    answer: str
    model: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)


class EmptyChatMessageError(ValueError):
    """Raised when a session chat message is empty."""


class ChatOrchestrator:
    def __init__(self, store: ChatStore, executor: Any, model: str, history_limit: int = 12) -> None:
        self._store = store
        self._executor = executor
        self._model = model
        self._history_limit = history_limit

    async def send_message(self, session_id: str, message: str, top_k: int = 12) -> ChatTurnResponse:
        user_content = message.strip()
        if not user_content:
            raise EmptyChatMessageError("Message cannot be empty.")
        user_message = self._store.append_message(session_id, "user", user_content)
        history = [
            {"role": item.role, "content": item.content, "created_at": item.created_at}
            for item in self._store.recent_messages(session_id, self._history_limit)
        ]
        context = ToolContext(session_id=session_id, message_id=user_message.id, history=history)
        logger.info("Running chat orchestration for session_id=%s message_id=%s", session_id, user_message.id)
        rag_result = await self._executor.execute(
            "document_rag",
            {"query": _effective_query(history), "top_k": top_k},
            context,
        )
        tool_events = [*rag_result.events]
        if rag_result.ok:
            answer = str(rag_result.content.get("answer") or "")
            citations = list(rag_result.content.get("citations") or [])
            visual_evidence = list(rag_result.content.get("visual_evidence") or [])
        else:
            answer = "The document tool failed. Please try again or inspect backend logs."
            citations = []
            visual_evidence = []
        asset_result = await self._executor.execute("visual_asset", {"visual_evidence": visual_evidence}, context)
        tool_events.extend(asset_result.events)
        attachments = list(asset_result.content.get("attachments") or []) if asset_result.ok else []
        assistant_message = self._store.append_message(
            session_id,
            "assistant",
            answer,
            citations=citations,
            attachments=attachments,
            tool_trace=tool_events,
        )
        return ChatTurnResponse(
            session_id=session_id,
            message_id=assistant_message.id,
            answer=answer,
            model=self._model,
            citations=citations,
            attachments=attachments,
            tool_events=tool_events,
        )


def _effective_query(history: list[dict[str, Any]]) -> str:
    user_messages = [str(item["content"]) for item in history if item.get("role") == "user"]
    return "\n".join(user_messages[-3:])
