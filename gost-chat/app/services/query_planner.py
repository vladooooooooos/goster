from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal


QueryComplexity = Literal["simple", "compound"]

_VISUAL_TERMS = (
    "image",
    "picture",
    "photo",
    "figure",
    "drawing",
    "diagram",
    "scheme",
    "table",
    "formula",
    "layout",
    "arrangement",
    "view",
    "\u0444\u043e\u0442\u043e",
    "\u0444\u043e\u0442\u043e\u0433\u0440\u0430\u0444",
    "\u043a\u0430\u0440\u0442\u0438\u043d",
    "\u0438\u0437\u043e\u0431\u0440\u0430\u0436",
    "\u0440\u0438\u0441\u0443\u043d",
    "\u0447\u0435\u0440\u0442\u0435\u0436",
    "\u0441\u0445\u0435\u043c",
    "\u0434\u0438\u0430\u0433\u0440\u0430\u043c",
    "\u0442\u0430\u0431\u043b\u0438\u0446",
    "\u0444\u043e\u0440\u043c\u0443\u043b",
    "\u0440\u0430\u0441\u043f\u043e\u043b\u043e\u0436",
    "\u0432\u0438\u0434",
)
_EXPLANATION_TERMS = (
    "explain",
    "describe",
    "list",
    "\u043e\u0431\u044a\u044f\u0441\u043d",
    "\u043e\u043f\u0438\u0448",
    "\u043e\u043f\u0438\u0441\u0430",
    "\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b",
)
_VISUAL_TARGET_TERMS = (
    "sink",
    "mixer",
    "shower",
    "grid",
    "bath",
    "washbasin",
    "\u0440\u0430\u043a\u043e\u0432\u0438\u043d",
    "\u0441\u043c\u0435\u0441\u0438\u0442\u0435\u043b",
    "\u0434\u0443\u0448",
    "\u0441\u0435\u0442\u043a",
    "\u0432\u0430\u043d\u043d",
    "\u0443\u043c\u044b\u0432\u0430\u043b",
)


@dataclass(frozen=True)
class QueryTask:
    id: str
    text: str
    intent: str
    needs_visual: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QueryPlan:
    query: str
    tasks: list[QueryTask]
    complexity: QueryComplexity

    @property
    def needs_visual(self) -> bool:
        return any(task.needs_visual for task in self.tasks)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "tasks": [task.to_dict() for task in self.tasks],
            "complexity": self.complexity,
            "needs_visual": self.needs_visual,
        }


class QueryPlanner:
    def plan(self, query: str) -> QueryPlan:
        normalized = " ".join(query.split())
        segments = _split_query(normalized)
        tasks = [_task_from_segment(index + 1, segment) for index, segment in enumerate(segments)]
        tasks = _split_mixed_visual_and_text_task(tasks)
        complexity: QueryComplexity = "compound" if len(tasks) > 1 else "simple"
        return QueryPlan(query=normalized, tasks=tasks, complexity=complexity)


def _split_query(query: str) -> list[str]:
    if "sink and mixer" in query.casefold() or "\u0440\u0430\u043a\u043e\u0432\u0438\u043d" in query.casefold() and "\u0441\u043c\u0435\u0441\u0438\u0442\u0435\u043b" in query.casefold():
        return [query]
    parts = [
        part.strip()
        for part in re.split(r"\s+(?:and|also|plus)\s+|[;?!.]+", query, flags=re.IGNORECASE)
        if part.strip()
    ]
    if len(parts) <= 1:
        parts = _split_multiple_visual_targets(query)
    return parts or [query]


def _split_multiple_visual_targets(query: str) -> list[str]:
    normalized = query.casefold()
    if not _contains_any(normalized, _VISUAL_TERMS):
        return [query]
    targets = [term for term in _VISUAL_TARGET_TERMS if term in normalized]
    if len(targets) < 2:
        return [query]
    return [f"{target} layout" for target in targets[:4]]


def _task_from_segment(index: int, segment: str) -> QueryTask:
    normalized = segment.casefold()
    needs_visual = _contains_any(normalized, _VISUAL_TERMS)
    intent = "visual_evidence" if needs_visual else "text_answer"
    if _contains_any(normalized, _EXPLANATION_TERMS):
        intent = "visual_evidence_and_text" if needs_visual else "text_explanation"
    return QueryTask(id=f"task-{index}", text=segment, intent=intent, needs_visual=needs_visual)


def _split_mixed_visual_and_text_task(tasks: list[QueryTask]) -> list[QueryTask]:
    if len(tasks) != 1:
        return tasks
    task = tasks[0]
    normalized = task.text.casefold()
    if task.needs_visual and _contains_any(normalized, _EXPLANATION_TERMS):
        return [
            QueryTask(id="task-1", text=task.text, intent="visual_evidence", needs_visual=True),
            QueryTask(id="task-2", text=task.text, intent="text_explanation", needs_visual=False),
        ]
    return tasks


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def _contains_term(text: str, term: str) -> bool:
    if term.isascii() and term.replace("_", "").isalpha():
        return bool(re.search(rf"\b{re.escape(term)}\b", text))
    return term in text
