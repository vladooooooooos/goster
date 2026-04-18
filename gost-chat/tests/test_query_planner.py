import unittest
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.services.query_planner import QueryPlanner


class QueryPlannerTest(unittest.TestCase):
    def test_simple_query_creates_one_text_task(self):
        plan = QueryPlanner().plan("What is required for potable water?")

        self.assertEqual(len(plan.tasks), 1)
        self.assertFalse(plan.tasks[0].needs_visual)
        self.assertEqual(plan.complexity, "simple")

    def test_visual_layout_query_without_photo_word_needs_visual(self):
        plan = QueryPlanner().plan("sink and mixer layout")

        self.assertEqual(len(plan.tasks), 1)
        self.assertTrue(plan.tasks[0].needs_visual)
        self.assertIn("visual_evidence", plan.tasks[0].intent)

    def test_russian_how_are_located_query_needs_visual(self):
        plan = QueryPlanner().plan(
            "\u043a\u0430\u043a \u0440\u0430\u0441\u043f\u043e\u043b\u0430\u0433\u0430\u044e\u0442\u0441\u044f "
            "\u0440\u0430\u043a\u043e\u0432\u0438\u043d\u044b \u0438 \u0441\u043c\u0435\u0441\u0438\u0442\u0435\u043b\u0438 "
            "\u043d\u0430 \u0441\u0443\u0434\u043d\u0435"
        )

        self.assertEqual(len(plan.tasks), 1)
        self.assertTrue(plan.tasks[0].needs_visual)
        self.assertIn("visual_evidence", plan.tasks[0].intent)

    def test_compound_query_creates_multiple_tasks(self):
        plan = QueryPlanner().plan("show sink layout and explain elements")

        self.assertEqual(len(plan.tasks), 2)
        self.assertTrue(plan.tasks[0].needs_visual)
        self.assertFalse(plan.tasks[1].needs_visual)
        self.assertEqual(plan.complexity, "compound")

    def test_two_visual_targets_create_two_visual_tasks(self):
        plan = QueryPlanner().plan("show sink layout and shower grid layout")

        self.assertEqual(len(plan.tasks), 2)
        self.assertTrue(all(task.needs_visual for task in plan.tasks))


if __name__ == "__main__":
    unittest.main()
