import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.api.chat import router
from app.orchestration.chat_store import ChatStore


class FakeOrchestrator:
    model = "gemma-test"

    async def send_message(self, session_id, message, top_k=12):
        return type(
            "Response",
            (),
            {
                "session_id": session_id,
                "message_id": "message-1",
                "answer": f"answer to {message}",
                "model": self.model,
                "citations": [{"chunk_id": "c1"}],
                "attachments": [{"asset_id": "a1"}],
                "tool_events": [{"tool_name": "document_rag"}],
            },
        )()


class ChatApiSessionsTest(unittest.TestCase):
    def test_create_session_and_send_message(self):
        app = FastAPI()
        with tempfile.TemporaryDirectory(dir=APP_ROOT / "data") as temp_dir:
            store = ChatStore(Path(temp_dir) / "sessions.json")
            app.state.chat_store = store
            app.state.chat_orchestrator = FakeOrchestrator()
            app.include_router(router)
            client = TestClient(app)

            create_response = client.post("/chat/sessions", json={"title": "Ship rules"})
            self.assertEqual(create_response.status_code, 200)
            session_id = create_response.json()["session_id"]

            message_response = client.post(
                f"/chat/sessions/{session_id}/messages",
                json={"message": "Question?", "top_k": 12},
            )

            self.assertEqual(message_response.status_code, 200)
            payload = message_response.json()
            self.assertEqual(payload["session_id"], session_id)
            self.assertEqual(payload["answer"], "answer to Question?")
            self.assertEqual(payload["citations"][0]["chunk_id"], "c1")
            self.assertEqual(payload["attachments"][0]["asset_id"], "a1")


if __name__ == "__main__":
    unittest.main()
