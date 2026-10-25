"""Vision provider interface and the deterministic parser for the model's text output.

Providers are synchronous on purpose: they run on one dedicated inference thread (see
vision/monitor.py) that loads the model once and keeps it in memory.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..schemas import CookingStatus


class VisionProvider(ABC):
    name: str = "base"

    def load(self) -> None:
        """Load the model. Called once, on the inference thread, before any assess()."""

    def on_session_start(self) -> None:
        """Hook for providers that keep per-session state (the mock's scripted progression)."""

    @abstractmethod
    def assess(self, jpeg: bytes) -> str:
        """Return the model's raw text for one JPEG frame (LLM/inference.py output format)."""


@dataclass(frozen=True)
class ParsedObservation:
    food: str | None
    status: CookingStatus
    evidence: str
    recommendation: str
    parse_ok: bool


_FIELD = r"^[\s*_#>-]*{name}[\s*_]*:[\s*_]*(?P<value>.+?)[\s*_]*$"
_FOOD_RE = re.compile(_FIELD.format(name=r"food\s+identified"), re.I | re.M)
_STATUS_RE = re.compile(_FIELD.format(name=r"cooking\s+status"), re.I | re.M)
_EVIDENCE_RE = re.compile(_FIELD.format(name=r"visual\s+evidence"), re.I | re.M)
_RECOMMENDATION_RE = re.compile(_FIELD.format(name=r"recommendation"), re.I | re.M)
_UNKNOWN_FOOD = {"unknown", "none", "n/a", "na", "unclear", "no food", "not visible", "-"}


def _field(regex: re.Pattern, text: str) -> str:
    m = regex.search(text)
    return m.group("value").strip().strip("[]").strip() if m else ""


# Loose spellings accepted on the "Cooking Status:" line only (never in free text, where
# "not burnt" or "no charring" would be misread). Checked in order.
_STATUS_KEYWORDS = [
    (CookingStatus.CANNOT_DETERMINE, ("CANNOT", "UNDETERMIN", "UNCLEAR", "UNKNOWN")),
    (CookingStatus.POSSIBLY_OVERCOOKED_OR_BURNT, ("OVERCOOK", "BURNT", "BURNED", "BURNING", "CHARRED")),
    (CookingStatus.APPEARS_UNDERCOOKED, ("UNDERCOOK", "RAW", "UNDERDONE")),
    (CookingStatus.APPEARS_COOKED, ("COOKED", "DONE")),
]


def _status_from(text: str, loose: bool = False) -> CookingStatus | None:
    normalized = re.sub(r"[\s/-]+", "_", text.upper())
    # Longest names first so APPEARS_UNDERCOOKED is not read as APPEARS_COOKED.
    for status in sorted(CookingStatus, key=lambda s: -len(s.value)):
        if status.value in normalized:
            return status
    if loose:
        for status, words in _STATUS_KEYWORDS:
            if any(w in normalized for w in words):
                return status
    return None


def parse_observation(raw: str) -> ParsedObservation:
    """Parse the four labelled fields. Anything unparseable becomes CANNOT_DETERMINE.

    Only ``Cooking Status`` (and the food name) feed decisions; free text is informational.
    """
    food = _field(_FOOD_RE, raw)
    if food.lower().rstrip(".") in _UNKNOWN_FOOD:
        food = ""
    status_text = _field(_STATUS_RE, raw)
    status = _status_from(status_text, loose=True) if status_text else None
    parse_ok = status is not None
    if status is None:
        # The model ignored the format; accept a status token anywhere, else give up.
        status = _status_from(raw) or CookingStatus.CANNOT_DETERMINE
    return ParsedObservation(
        food=food or None,
        status=status,
        evidence=_field(_EVIDENCE_RE, raw),
        recommendation=_field(_RECOMMENDATION_RE, raw),
        parse_ok=parse_ok,
    )
