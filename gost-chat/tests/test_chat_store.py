import sys
import tempfile
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.config import Settings
from app.orchestration.chat_store import ChatStore


class ChatStoreTest(unittest.TestCase):
    def test_settings_expose_chat_store_defaults(self):
        settings = Settings()

        self.assertEqual(settings.chat_store_path.as_posix(), "data/chat_sessions.json")
        self.assertEqual(settings.chat_history_limit, 12)
        self.assertEqual(settings.agent_max_tool_loops, 3)

    def test_creates_session_and_persists_messages(self):
        with tempfile.TemporaryDirectory(dir=APP_ROOT / "data") as temp_dir:
            path = Path(temp_dir) / "sessions.json"
            store = ChatStore(path)

            session = store.create_session(title="First chat")
            user = store.append_message(session.id, "user", "Question?")
            assistant = store.append_message(
                session.id,
                "assistant",
                "Answer.",
                citations=[{"chunk_id": "c1"}],
                attachments=[{"asset_id": "a1"}],
                tool_trace=[{"tool_name": "document_rag"}],
            )

            reloaded = ChatStore(path)
            loaded = reloaded.get_session(session.id)
            messages = reloaded.list_messages(session.id)

            self.assertEqual(loaded.title, "First chat")
            self.assertEqual(messages[0].id, user.id)
            self.assertEqual(messages[1].id, assistant.id)
            self.assertEqual(messages[1].citations[0]["chunk_id"], "c1")
            self.assertEqual(messages[1].attachments[0]["asset_id"], "a1")
            self.assertEqual(messages[1].tool_trace[0]["tool_name"], "document_rag")

    def test_recent_messages_respects_limit(self):
        with tempfile.TemporaryDirectory(dir=APP_ROOT / "data") as temp_dir:
            store = ChatStore(Path(temp_dir) / "sessions.json")
            session = store.create_session()
            for index in range(5):
                store.append_message(session.id, "user", f"message {index}")

            recent = store.recent_messages(session.id, limit=2)

            self.assertEqual([message.content for message in recent], ["message 3", "message 4"])


if __name__ == "__main__":
    unittest.main()
