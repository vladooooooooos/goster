from __future__ import annotations

import logging
import time
from typing import Any

from app.orchestration.tool_contracts import ToolContext, ToolResult
from app.orchestration.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, name: str, payload: dict[str, Any], context: ToolContext) -> ToolResult:
        started = time.perf_counter()
        logger.info(
            "Tool execution started: tool=%s session_id=%s message_id=%s",
            name,
            context.session_id,
            context.message_id,
        )
        try:
            result = await self._registry.get(name).run(payload, context)
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception("Tool execution failed: tool=%s session_id=%s", name, context.session_id)
            return ToolResult(
                tool_name=name,
                ok=False,
                error=str(exc),
                events=[{"tool_name": name, "status": "error", "duration_ms": duration_ms, "error": str(exc)}],
            )
        duration_ms = (time.perf_counter() - started) * 1000
        event = {"tool_name": name, "status": "ok" if result.ok else "error", "duration_ms": duration_ms}
        if result.error:
            event["error"] = result.error
        logger.info("Tool execution completed: tool=%s status=%s duration_ms=%.2f", name, event["status"], duration_ms)
        return ToolResult(
            tool_name=result.tool_name,
            ok=result.ok,
            content=result.content,
            error=result.error,
            events=[*result.events, event],
        )
