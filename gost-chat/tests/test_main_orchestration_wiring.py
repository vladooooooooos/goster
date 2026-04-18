import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import app.main as main


class MainOrchestrationWiringTest(unittest.TestCase):
    def test_main_exposes_chat_orchestration_services(self):
        self.assertTrue(hasattr(main.app.state, "chat_store"))
        self.assertTrue(hasattr(main.app.state, "tool_registry"))
        self.assertTrue(hasattr(main.app.state, "tool_executor"))
        self.assertTrue(hasattr(main.app.state, "chat_orchestrator"))

        tool_names = [definition.name for definition in main.app.state.tool_registry.definitions()]
        self.assertIn("document_rag", tool_names)
        self.assertIn("visual_crop", tool_names)
        self.assertIn("visual_asset", tool_names)


if __name__ == "__main__":
    unittest.main()
