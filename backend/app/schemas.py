"""Pydantic models: visual observations, alerts, and the HTTP API models."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel

DISPLAY_TEXT_MAX = 24


def display_text(text: str) -> str:
    """Fit text to the BOX screen: one line, plain ASCII, at most 24 characters."""
    text = " ".join(text.split()).encode("ascii", "ignore").decode()
    return text[:DISPLAY_TEXT_MAX].rstrip()


class CookingStatus(str, Enum):
    """The four statuses the vision model (LLM/inference.py COOKING_PROMPT) may report."""

    APPEARS_COOKED = "APPEARS_COOKED"
    APPEARS_UNDERCOOKED = "APPEARS_UNDERCOOKED"
    POSSIBLY_OVERCOOKED_OR_BURNT = "POSSIBLY_OVERCOOKED_OR_BURNT"
    CANNOT_DETERMINE = "CANNOT_DETERMINE"


class VisualObservation(BaseModel):
    """What the vision model saw in one frame. Facts only; no commands, no timers.

    ``recommendation`` is the model's free text. It is kept for debugging but never spoken,
    because the model sometimes claims food is "safe to eat" despite being told not to.
    """

    food: str | None
    status: CookingStatus
    evidence: str = ""
    recommendation: str = ""
    parse_ok: bool = True
    frame_seq: int
    frame_ts: float  # monotonic time the frame was captured
    observed_at: float  # wall-clock time the result was produced
    latency_s: float = 0.0
    raw: str = ""


class AlertType(str, Enum):
    OVERCOOK_WARNING = "OVERCOOK_WARNING"  # vision: food may be burning (independent of any timer)
    TIMER_DONE = "TIMER_DONE"  # the countdown reached zero; says nothing about doneness
    MONITORING_LOST = "MONITORING_LOST"  # camera stale during a session: nobody is watching the pan


ALERT_SEVERITY = {AlertType.OVERCOOK_WARNING: 3, AlertType.TIMER_DONE: 2, AlertType.MONITORING_LOST: 1}


class Alert(BaseModel):
    id: int
    type: AlertType
    text: str  # short, for the screen
    message: str  # spoken
    audio_id: str | None = None
    created_at: float  # wall clock

    @property
    def severity(self) -> int:
        return ALERT_SEVERITY[self.type]


class TimerSuggestion(BaseModel):
    """A typical cook time from the lookup table. A suggestion only, never a doneness guarantee."""

    food: str
    seconds: int
    category: str | None = None
    safe_temp_f: int | None = None


class Phase(str, Enum):
    IDLE = "IDLE"
    ANALYZING = "ANALYZING"  # session started, waiting for the first usable observation
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"  # a timer suggestion awaits thumbs-up
    MONITORING = "MONITORING"  # watching the food, no timer
    COOKING = "COOKING"  # watching the food, timer running
    DONE = "DONE"  # timer reached zero (still watching until acknowledged)
    ERROR = "ERROR"


ACTIVE_PHASES = {Phase.ANALYZING, Phase.AWAITING_CONFIRMATION, Phase.MONITORING, Phase.COOKING, Phase.DONE}

TimerStatus = Literal["none", "running", "paused", "expired"]


class DeviceState(BaseModel):
    """Compact state the ESP32-S3-BOX-3 polls. Flat fields so ArduinoJson parsing stays trivial."""

    version: int
    state: Phase
    display_text: str
    # Timer: re-anchor the local countdown from remaining_ms on *every* response.
    timer: TimerStatus
    remaining_ms: int
    remaining_s: int
    total_s: int
    paused: bool
    # Latest spoken clip (informational speech). Play when it changes.
    audio_id: str | None
    # Highest-priority unacknowledged alert. Play alert_audio_id when alert_id changes,
    # keep alerting until POST /device/ack.
    alert: bool
    alert_id: int | None
    alert_type: AlertType | None
    alert_text: str | None
    alert_audio_id: str | None
    alerts_pending: int
    # Latest visual condition and whether the camera+model are actually watching right now.
    condition: CookingStatus | None
    monitoring: bool
    busy: bool
    session_id: str | None


class DeviceStateUnchanged(BaseModel):
    changed: Literal[False] = False
    version: int
    timer: TimerStatus
    remaining_ms: int
    remaining_s: int


class ButtonRequest(BaseModel):
    button: Literal["ack", "cancel", "pause", "resume", "plus_1m", "minus_1m"]


class AckRequest(BaseModel):
    alert_id: int


class CommandRequest(BaseModel):
    transcript: str | None = None


class TimerControlRequest(BaseModel):
    action: Literal["start", "cancel", "adjust", "pause", "resume"]
    seconds: int | None = None
    label: str | None = None
