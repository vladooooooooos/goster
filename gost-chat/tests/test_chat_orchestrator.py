import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.orchestration.chat_orchestrator import ChatOrchestrator
from app.orchestration.chat_store import ChatStore
from app.orchestration.tool_contracts import ToolContext, ToolResult


class FakeExecutor:
    def __init__(self):
        self.calls = []

    async def execute(self, name, payload, context):
        self.calls.append((name, payload, context))
        if name == "document_rag":
            return ToolResult(
                tool_name="document_rag",
                ok=True,
                content={
                    "answer": "Grounded answer [1].",
                    "citations": [{"chunk_id": "c1"}],
                    "visual_evidence": [{"block_id": "v1", "document_id": "doc-1", "crop_url": "/crops/v1.png"}],
                },
                events=[{"tool_name": "document_rag", "status": "ok"}],
            )
        if name == "visual_asset":
            return ToolResult(
                tool_name="visual_asset",
                ok=True,
                content={"attachments": [{"asset_id": "doc-1:v1", "url": "/crops/v1.png"}]},
                events=[{"tool_name": "visual_asset", "status": "ok"}],
            )
        raise AssertionError(name)


class ChatOrchestratorTest(unittest.TestCase):
    def test_send_message_persists_messages_and_returns_rag_payload(self):
        with tempfile.TemporaryDirectory(dir=APP_ROOT / "data") as temp_dir:
            store = ChatStore(Path(temp_dir) / "sessions.json")
            session = store.create_session()
            executor = FakeExecutor()
            orchestrator = ChatOrchestrator(store, executor, model="gemma-test", history_limit=4)

            response = asyncio.run(orchestrator.send_message(session.id, "What is required?", top_k=12))

            messages = store.list_messages(session.id)
            self.assertEqual(messages[0].role, "user")
            self.assertEqual(messages[1].role, "assistant")
            self.assertEqual(response.answer, "Grounded answer [1].")
            self.assertEqual(response.citations[0]["chunk_id"], "c1")
            self.assertEqual(response.attachments[0]["asset_id"], "doc-1:v1")
            self.assertEqual(response.session_id, session.id)
            self.assertEqual(response.model, "gemma-test")
            self.assertEqual([call[0] for call in executor.calls], ["document_rag", "visual_asset"])

    def test_follow_up_call_includes_recent_history(self):
        with tempfile.TemporaryDirectory(dir=APP_ROOT / "data") as temp_dir:
            store = ChatStore(Path(temp_dir) / "sessions.json")
            session = store.create_session()
            store.append_message(session.id, "user", "Initial question")
            store.append_message(session.id, "assistant", "Initial answer")
            executor = FakeExecutor()
            orchestrator = ChatOrchestrator(store, executor, model="gemma-test", history_limit=3)

            asyncio.run(orchestrator.send_message(session.id, "And the connectors?", top_k=12))

            document_call = executor.calls[0]
            context = document_call[2]
            self.assertIsInstance(context, ToolContext)
            self.assertEqual(
                [item["content"] for item in context.history],
                ["Initial question", "Initial answer", "And the connectors?"],
            )


if __name__ == "__main__":
    unittest.main()
