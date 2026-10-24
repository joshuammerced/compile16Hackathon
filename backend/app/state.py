"""Thread-safe, persistent cooking-session state and the timer state machine.

All mutation goes through ``StateStore.mutate()``, which holds a lock and, on exit:
  * bumps ``version`` iff the device-visible projection changed (the countdown ticking is not
    a change; the device re-anchors from ``remaining_ms`` on every response instead), and
  * saves the session to disk so timers and pending alerts survive a backend restart.

There is a single active session today (one stove), but sessions are keyed by id so more
timers can be added later without changing the device contract for the default session.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import (
    ACTIVE_PHASES,
    Alert,
    AlertType,
    CookingStatus,
    DeviceState,
    Phase,
    TimerStatus,
    TimerSuggestion,
    VisualObservation,
    display_text,
)

log = logging.getLogger(__name__)

Clock = Callable[[], float]
DEFAULT_SESSION = "stove1"
IDLE_TEXT = "Show me food"
OBSERVATION_HISTORY = 30


class Timer:
    """Countdown driven by server-side monotonic time.

    Remaining time is computed on read from ``started_at`` / ``duration_s`` / pause bookkeeping,
    so it is correct no matter how irregularly the device polls.
    """

    def __init__(self, duration_s: int, label: str = "", clock: Clock = time.monotonic):
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        self._clock = clock
        self.label = label
        self.duration_s = int(duration_s)
        self.started_at = clock()
        self._paused_at: float | None = None
        self._paused_total = 0.0

    @property
    def paused(self) -> bool:
        return self._paused_at is not None

    def elapsed(self) -> float:
        now = self._paused_at if self._paused_at is not None else self._clock()
        return now - self.started_at - self._paused_total

    def remaining(self) -> float:
        return max(0.0, self.duration_s - self.elapsed())

    @property
    def ends_at(self) -> float | None:
        """Monotonic deadline, or None while paused."""
        if self.paused:
            return None
        return self.started_at + self._paused_total + self.duration_s

    @property
    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def status(self) -> TimerStatus:
        if self.expired:
            return "expired"
        return "paused" if self.paused else "running"

    def pause(self) -> bool:
        if self.paused or self.expired:
            return False
        self._paused_at = self._clock()
        return True

    def resume(self) -> bool:
        if self._paused_at is None:
            return False
        self._paused_total += self._clock() - self._paused_at
        self._paused_at = None
        return True

    def adjust(self, delta_s: int) -> None:
        """Add (or subtract) time. Subtracting past zero simply expires the timer."""
        self.duration_s = max(0, self.duration_s + int(delta_s))

    # --- persistence: monotonic time does not survive a restart, so store remaining time ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "duration_s": self.duration_s,
            "remaining_s": self.remaining(),
            "paused": self.paused,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], downtime_s: float, clock: Clock = time.monotonic) -> Timer:
        """Rebuild a timer; a running timer keeps counting through the downtime."""
        t = cls(max(1, int(d["duration_s"])), d.get("label", ""), clock)
        t.duration_s = int(d["duration_s"])
        remaining = max(0.0, float(d["remaining_s"]) - (0.0 if d.get("paused") else downtime_s))
        now = clock()
        t.started_at = now - (t.duration_s - remaining)  # so that elapsed() == duration - remaining
        if d.get("paused"):
            t._paused_at = now
        return t


@dataclass
class CookingSession:
    session_id: str = DEFAULT_SESSION
    phase: Phase = Phase.IDLE
    display_text: str = IDLE_TEXT
    # Cooking-session identity. ``run_id`` is None when idle; ``generation`` increments on every
    # start/end so in-flight vision results from an older session can be recognized and dropped.
    run_id: str | None = None
    generation: int = 0
    started_wall: float | None = None
    timer: Timer | None = None
    suggestion: TimerSuggestion | None = None
    food: str | None = None
    observations: deque[VisualObservation] = field(default_factory=lambda: deque(maxlen=OBSERVATION_HISTORY))
    analyze_attempts: int = 0
    # Alerts: pending until acknowledged by id. Acked ids are remembered so acks are idempotent.
    alerts: list[Alert] = field(default_factory=list)
    acked_ids: deque[int] = field(default_factory=lambda: deque(maxlen=100))
    next_alert_id: int = 1
    last_overcook_alert_at: float | None = None  # monotonic
    looks_done_notified: bool = False
    # Liveness of visual monitoring.
    camera_stale_since: float | None = None  # monotonic
    vision_live: bool = False
    audio_id: str | None = None
    error: str | None = None
    history: list[tuple[float, Phase, str]] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.phase in ACTIVE_PHASES

    @property
    def condition(self) -> CookingStatus | None:
        return self.observations[-1].status if self.observations else None

    def transition(self, phase: Phase, reason: str) -> None:
        if phase == self.phase:
            return
        log.info("state %s: %s -> %s (%s)", self.session_id, self.phase.value, phase.value, reason)
        self.history.append((time.time(), phase, reason))
        del self.history[:-50]
        self.phase = phase

    # --- session lifecycle ----------------------------------------------------------

    def start_session(self, reason: str) -> None:
        self._reset_session()
        self.run_id = uuid.uuid4().hex[:8]
        self.started_wall = time.time()
        self.display_text = "Looking..."
        self.transition(Phase.ANALYZING, reason)
        log.info("cooking session %s started (generation %d)", self.run_id, self.generation)

    def end_session(self, reason: str) -> None:
        if self.run_id:
            log.info("cooking session %s ended: %s", self.run_id, reason)
        self._reset_session()
        self.alerts.clear()
        self.display_text = IDLE_TEXT
        self.transition(Phase.IDLE, reason)

    def _reset_session(self) -> None:
        self.generation += 1
        self.run_id = None
        self.started_wall = None
        self.timer = None
        self.suggestion = None
        self.food = None
        self.observations.clear()
        self.analyze_attempts = 0
        self.last_overcook_alert_at = None
        self.looks_done_notified = False
        self.camera_stale_since = None
        self.vision_live = False
        self.error = None

    # --- timer operations -----------------------------------------------------------

    def start_timer(self, seconds: int, label: str, clock: Clock = time.monotonic) -> None:
        if self.run_id is None:
            self.start_session("timer started")
        if self.timer is not None:
            log.info("replacing running timer %r with %r", self.timer.label, label)
        self.timer = Timer(seconds, display_text(label) or "Timer", clock)
        self.suggestion = None
        self.looks_done_notified = False
        self.resolve_alerts(AlertType.TIMER_DONE)
        self.display_text = self.timer.label
        self.transition(Phase.COOKING, f"timer started {seconds}s")

    def pause(self) -> bool:
        return bool(self.timer and self.timer.pause())

    def resume(self) -> bool:
        return bool(self.timer and self.timer.resume())

    def adjust(self, delta_s: int) -> bool:
        if self.timer is None or self.phase != Phase.COOKING:
            return False
        self.timer.adjust(delta_s)
        return True

    def stop_timer(self, reason: str) -> None:
        """Drop the timer but keep watching the food."""
        self.timer = None
        self.suggestion = None
        self.resolve_alerts(AlertType.TIMER_DONE)
        self.display_text = display_text(self.food or "Watching")
        self.transition(Phase.MONITORING, reason)

    def finish(self) -> None:
        label = self.timer.label if self.timer else ""
        self.display_text = "DONE"
        self.transition(Phase.DONE, f"timer expired ({label})")

    def timer_due(self) -> bool:
        return self.phase == Phase.COOKING and self.timer is not None and self.timer.expired

    # --- alerts ---------------------------------------------------------------------

    def add_alert(self, type_: AlertType, text: str, message: str) -> Alert:
        alert = Alert(
            id=self.next_alert_id,
            type=type_,
            text=display_text(text),
            message=message,
            created_at=time.time(),
        )
        self.next_alert_id += 1
        self.alerts.append(alert)
        log.info("alert %d raised: %s %r", alert.id, alert.type.value, alert.message)
        return alert

    def top_alert(self) -> Alert | None:
        if not self.alerts:
            return None
        return max(self.alerts, key=lambda a: (a.severity, -a.id))

    def pending(self, type_: AlertType) -> Alert | None:
        return next((a for a in self.alerts if a.type == type_), None)

    def ack(self, alert_id: int) -> Alert | None | bool:
        """Acknowledge by id. Returns the Alert, True if it was already acked, None if unknown."""
        for i, alert in enumerate(self.alerts):
            if alert.id == alert_id:
                del self.alerts[i]
                self.acked_ids.append(alert_id)
                log.info("alert %d acknowledged (%s)", alert_id, alert.type.value)
                return alert
        return True if alert_id in self.acked_ids else None

    def resolve_alerts(self, type_: AlertType) -> list[Alert]:
        """Clear alerts that no longer apply (e.g. camera came back)."""
        gone = [a for a in self.alerts if a.type == type_]
        for a in gone:
            self.alerts.remove(a)
            self.acked_ids.append(a.id)
            log.info("alert %d resolved (%s)", a.id, a.type.value)
        return gone

    # --- read model -----------------------------------------------------------------

    def remaining_ms(self) -> int:
        if self.timer is None or self.phase not in (Phase.COOKING, Phase.DONE):
            return 0
        return int(self.timer.remaining() * 1000)

    def timer_status(self) -> TimerStatus:
        if self.timer is None or self.phase not in (Phase.COOKING, Phase.DONE):
            return "none"
        return self.timer.status()

    def total_s(self) -> int:
        return self.timer.duration_s if self.timer else 0

    def device_view(self, version: int) -> DeviceState:
        ms = self.remaining_ms()
        top = self.top_alert()
        return DeviceState(
            version=version,
            state=self.phase,
            display_text=self.display_text,
            timer=self.timer_status(),
            remaining_ms=ms,
            remaining_s=-(-ms // 1000),  # round up so the display shows 0:01 until truly zero
            total_s=self.total_s(),
            paused=bool(self.timer and self.timer.paused),
            audio_id=self.audio_id,
            alert=top is not None,
            alert_id=top.id if top else None,
            alert_type=top.type if top else None,
            alert_text=top.text if top else None,
            alert_audio_id=top.audio_id if top else None,
            alerts_pending=len(self.alerts),
            condition=self.condition,
            monitoring=self.vision_live and self.active,
            busy=self.phase == Phase.ANALYZING,
            session_id=self.run_id,
        )

    # --- persistence ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "display_text": self.display_text,
            "run_id": self.run_id,
            "generation": self.generation,
            "started_wall": self.started_wall,
            "timer": self.timer.to_dict() if self.timer else None,
            "suggestion": self.suggestion.model_dump() if self.suggestion else None,
            "food": self.food,
            "observations": [o.model_dump(mode="json", exclude={"raw"}) for o in list(self.observations)[-5:]],
            "alerts": [a.model_dump(mode="json") for a in self.alerts],
            "acked_ids": list(self.acked_ids),
            "next_alert_id": self.next_alert_id,
            "looks_done_notified": self.looks_done_notified,
            "audio_id": self.audio_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], downtime_s: float, clock: Clock) -> CookingSession:
        s = cls(session_id=d.get("session_id", DEFAULT_SESSION))
        s.phase = Phase(d["phase"])
        s.display_text = d.get("display_text", IDLE_TEXT)
        s.run_id = d.get("run_id")
        s.generation = int(d.get("generation", 0)) + 1  # anything in flight before the restart is stale
        s.started_wall = d.get("started_wall")
        s.timer = Timer.from_dict(d["timer"], downtime_s, clock) if d.get("timer") else None
        s.suggestion = TimerSuggestion(**d["suggestion"]) if d.get("suggestion") else None
        s.food = d.get("food")
        # Monotonic frame timestamps from the previous process are meaningless; keep the
        # history for context but it never counts as fresh.
        for o in d.get("observations", []):
            s.observations.append(VisualObservation(**{**o, "frame_ts": float("-inf")}))
        s.alerts = [Alert(**a) for a in d.get("alerts", [])]
        s.acked_ids.extend(d.get("acked_ids", []))
        s.next_alert_id = int(d.get("next_alert_id", 1))
        s.looks_done_notified = bool(d.get("looks_done_notified", False))
        s.audio_id = None  # audio clips live in memory and did not survive
        if s.phase in (Phase.ANALYZING, Phase.ERROR):
            s.phase = Phase.MONITORING if s.run_id else Phase.IDLE
        return s


class SessionPersistence:
    """Atomic JSON snapshot of the sessions (write temp file, then rename)."""

    def __init__(self, path: Path, wall: Clock = time.time):
        self.path = Path(path)
        self.wall = wall

    def save(self, sessions: dict[str, CookingSession], version: int) -> None:
        data = {
            "saved_at": self.wall(),
            "version": version,
            "sessions": {k: s.to_dict() for k, s in sessions.items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self.path)

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.error("could not read saved state %s (%s); starting fresh", self.path, exc)
            return None


class StateStore:
    def __init__(self, clock: Clock = time.monotonic, persistence: SessionPersistence | None = None):
        self.clock = clock
        self.persistence = persistence
        self._lock = threading.RLock()
        self._version = 0
        self.sessions: dict[str, CookingSession] = {DEFAULT_SESSION: CookingSession()}
        if persistence is not None:
            self._restore(persistence)
        self._last_view = {k: self._view_key(s) for k, s in self.sessions.items()}

    def _restore(self, persistence: SessionPersistence) -> None:
        data = persistence.load()
        if not data:
            return
        now = persistence.wall()
        downtime = max(0.0, now - float(data.get("saved_at", now)))
        try:
            self.sessions = {
                k: CookingSession.from_dict(d, downtime, self.clock) for k, d in data["sessions"].items()
            }
        except Exception:  # noqa: BLE001 - a corrupt snapshot must not stop the stove timer service
            log.exception("saved state is unusable; starting fresh")
            return
        self._version = int(data.get("version", 0)) + 1
        s = self.sessions.get(DEFAULT_SESSION)
        if s is not None:
            log.info(
                "restored session %s: %s, %d pending alert(s), timer %s, after %.0fs downtime",
                s.run_id,
                s.phase.value,
                len(s.alerts),
                f"{s.timer.remaining():.0f}s left" if s.timer else "none",
                downtime,
            )

    @staticmethod
    def _view_key(s: CookingSession) -> dict:
        # Everything the device sees except the ticking countdown.
        return s.device_view(0).model_dump(exclude={"version", "remaining_ms", "remaining_s", "timer"}) | {
            "timer_running": s.timer_status() in ("running", "paused"),
        }

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @contextmanager
    def mutate(self, session_id: str = DEFAULT_SESSION) -> Iterator[CookingSession]:
        """Lock, yield the session for mutation; bump the version if the device view changed."""
        with self._lock:
            session = self.sessions[session_id]
            try:
                yield session
            finally:
                key = self._view_key(session)
                if key != self._last_view.get(session_id):
                    self._last_view[session_id] = key
                    self._version += 1
                if self.persistence is not None:
                    try:
                        self.persistence.save(self.sessions, self._version)
                    except OSError:
                        log.exception("failed to persist state")

    @contextmanager
    def read(self, session_id: str = DEFAULT_SESSION) -> Iterator[CookingSession]:
        with self._lock:
            yield self.sessions[session_id]

    def snapshot(self, session_id: str = DEFAULT_SESSION) -> DeviceState:
        with self._lock:
            return self.sessions[session_id].device_view(self._version)
