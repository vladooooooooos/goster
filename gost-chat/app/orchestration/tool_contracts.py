from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolContext:
    session_id: str
    message_id: str
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    ok: bool
    content: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


class BaseTool(Protocol):
    @property
    def definition(self) -> ToolDefinition: ...

    async def run(self, payload: dict[str, Any], context: ToolContext) -> ToolResult: ...
