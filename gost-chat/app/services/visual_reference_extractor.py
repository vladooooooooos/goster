from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class VisualReferenceMention:
    kind: str
    value: str
    normalized_label: str
    raw_text: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "value": self.value,
            "normalized_label": self.normalized_label,
            "raw_text": self.raw_text,
        }


class VisualReferenceExtractor:
    def extract(self, text: str) -> list[VisualReferenceMention]:
        if not text.strip():
            return []
        mentions: list[VisualReferenceMention] = []
        seen: set[str] = set()
        for match in _REFERENCE_RE.finditer(text):
            kind = _normalize_kind(match.group("kind"))
            value = _normalize_value(match.group("value"))
            if not kind or not value:
                continue
            normalized_label = f"{kind} {value}"
            if normalized_label in seen:
                continue
            seen.add(normalized_label)
            mentions.append(
                VisualReferenceMention(
                    kind=kind,
                    value=value,
                    normalized_label=normalized_label,
                    raw_text=match.group(0).strip(),
                )
            )
        return mentions


_REFERENCE_RE = re.compile(
    r"(?P<kind>"
    r"fig(?:ure)?|drawing|table|formula|appendix|"
    r"\u0440\u0438\u0441\.?|\u0440\u0438\u0441\u0443\u043d\u043e\u043a|"
    r"\u0447\u0435\u0440\u0442\.?|\u0447\u0435\u0440\u0442\u0435\u0436|"
    r"\u0442\u0430\u0431\u043b\u0438\u0446\u0430|"
    r"\u0444\u043e\u0440\u043c\u0443\u043b\u0430|"
    r"\u043f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u0435"
    r")\s*"
    r"(?P<value>\(?[A-Za-z0-9\u0410-\u042f\u0430-\u044f\u0401\u0451]+(?:[.\-][A-Za-z0-9\u0410-\u042f\u0430-\u044f\u0401\u0451]+)*\)?)",
    re.IGNORECASE,
)


def _normalize_kind(value: str) -> str:
    normalized = value.casefold().rstrip(".")
    if normalized in {"fig", "figure", "\u0440\u0438\u0441", "\u0440\u0438\u0441\u0443\u043d\u043e\u043a"}:
        return "figure"
    if normalized in {"drawing", "\u0447\u0435\u0440\u0442", "\u0447\u0435\u0440\u0442\u0435\u0436"}:
        return "drawing"
    if normalized in {"table", "\u0442\u0430\u0431\u043b\u0438\u0446\u0430"}:
        return "table"
    if normalized in {"formula", "\u0444\u043e\u0440\u043c\u0443\u043b\u0430"}:
        return "formula"
    if normalized in {"appendix", "\u043f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u0435"}:
        return "appendix"
    return ""


def _normalize_value(value: str) -> str:
    normalized = value.strip().strip("()").casefold()
    return normalized.translate(str.maketrans({
        "\u0430": "a",
        "\u0432": "b",
        "\u0435": "e",
        "\u043a": "k",
        "\u043c": "m",
        "\u043d": "h",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0442": "t",
        "\u0445": "x",
    }))
