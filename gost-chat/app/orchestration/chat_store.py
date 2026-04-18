from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatSession:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ChatMessage:
    id: str
    session_id: str
    role: Role
    content: str
    created_at: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)


class ChatSessionNotFoundError(KeyError):
    """Raised when a chat session does not exist."""


class ChatStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def create_session(self, title: str = "New chat") -> ChatSession:
        data = self._load()
        now = _now()
        session = ChatSession(id=f"session-{uuid.uuid4()}", title=title, created_at=now, updated_at=now)
        data["sessions"][session.id] = asdict(session)
        data["messages"][session.id] = []
        self._save(data)
        return session

    def get_session(self, session_id: str) -> ChatSession:
        data = self._load()
        session = data["sessions"].get(session_id)
        if not session:
            raise ChatSessionNotFoundError(session_id)
        return ChatSession(**session)

    def append_message(
        self,
        session_id: str,
        role: Role,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        tool_trace: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        data = self._load()
        if session_id not in data["sessions"]:
            raise ChatSessionNotFoundError(session_id)
        now = _now()
        message = ChatMessage(
            id=f"message-{uuid.uuid4()}",
            session_id=session_id,
            role=role,
            content=content,
            created_at=now,
            citations=citations or [],
            attachments=attachments or [],
            tool_trace=tool_trace or [],
        )
        data["messages"].setdefault(session_id, []).append(asdict(message))
        data["sessions"][session_id]["updated_at"] = now
        self._save(data)
        return message

    def list_messages(self, session_id: str) -> list[ChatMessage]:
        self.get_session(session_id)
        data = self._load()
        return [ChatMessage(**message) for message in data["messages"].get(session_id, [])]

    def recent_messages(self, session_id: str, limit: int) -> list[ChatMessage]:
        return self.list_messages(session_id)[-limit:]

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"sessions": {}, "messages": {}}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"sessions": {}, "messages": {}}
        if not isinstance(data, dict):
            return {"sessions": {}, "messages": {}}
        sessions = data.get("sessions") if isinstance(data.get("sessions"), dict) else {}
        messages = data.get("messages") if isinstance(data.get("messages"), dict) else {}
        return {"sessions": sessions, "messages": messages}

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, self._path)


def _now() -> str:
    return datetime.now(UTC).isoformat()
