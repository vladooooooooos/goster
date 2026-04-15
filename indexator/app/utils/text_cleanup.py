"""Safe text cleanup helpers for raw PDF extraction."""

from __future__ import annotations

import re


_HORIZONTAL_SPACE_RE = re.compile(r"[ \t]+")
_REPEATED_EMPTY_LINES_RE = re.compile(r"\n\s*\n+")


def clean_text(text: str) -> str:
    """Apply conservative cleanup that should not change document meaning."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [_HORIZONTAL_SPACE_RE.sub(" ", line).strip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    normalized = _REPEATED_EMPTY_LINES_RE.sub("\n\n", normalized)
    return normalized.strip()
