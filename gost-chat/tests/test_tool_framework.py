import asyncio
import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.orchestration.tool_contracts import ToolContext, ToolDefinition, ToolResult
from app.orchestration.tool_executor import ToolExecutor
from app.orchestration.tool_registry import DuplicateToolError, ToolRegistry, UnknownToolError


class EchoTool:
    @property
    def definition(self):
        return ToolDefinition(
            name="echo",
            description="Return the provided message.",
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
        )

    async def run(self, payload, context):
        return ToolResult(tool_name="echo", ok=True, content={"message": payload["message"]})


class FailingTool:
    @property
    def definition(self):
        return ToolDefinition(name="fail", description="Raise an error.", input_schema={"type": "object"})

    async def run(self, payload, context):
        raise RuntimeError("boom")


class ToolFrameworkTest(unittest.TestCase):
    def test_registry_returns_definitions_and_rejects_duplicates(self):
        registry = ToolRegistry()
        registry.register(EchoTool())

        self.assertEqual(registry.get("echo").definition.name, "echo")
        self.assertEqual(registry.definitions()[0].name, "echo")
        with self.assertRaises(DuplicateToolError):
            registry.register(EchoTool())

    def test_registry_rejects_unknown_tool(self):
        registry = ToolRegistry()

        with self.assertRaises(UnknownToolError):
            registry.get("missing")

    def test_executor_returns_success_result_with_event(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        executor = ToolExecutor(registry)

        result = asyncio.run(
            executor.execute("echo", {"message": "hello"}, ToolContext(session_id="s1", message_id="m1"))
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.content["message"], "hello")
        self.assertEqual(result.tool_name, "echo")
        self.assertEqual(result.events[0]["status"], "ok")
        self.assertGreaterEqual(result.events[0]["duration_ms"], 0)

    def test_executor_converts_exceptions_to_failed_result(self):
        registry = ToolRegistry()
        registry.register(FailingTool())
        executor = ToolExecutor(registry)

        result = asyncio.run(executor.execute("fail", {}, ToolContext(session_id="s1", message_id="m1")))

        self.assertFalse(result.ok)
        self.assertEqual(result.error, "boom")
        self.assertEqual(result.events[0]["status"], "error")


if __name__ == "__main__":
    unittest.main()
