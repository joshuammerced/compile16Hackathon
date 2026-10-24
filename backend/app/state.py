"""Thread-safe session state and the cooking timer state machine.

All mutation goes through ``StateStore.mutate()``, which holds a lock and bumps the global
version on exit, so device polling can cheaply detect "nothing changed".

There is a single active session today (one stove), but sessions are keyed by id so more
timers can be added later without changing the device contract for the default session.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .schemas import CookingDecision, DeviceState, Phase

log = logging.getLogger(__name__)

Clock = Callable[[], float]
DEFAULT_SESSION = "stove1"


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


@dataclass
class CookingSession:
    session_id: str = DEFAULT_SESSION
    phase: Phase = Phase.IDLE
    display_text: str = "Show me food"
    timer: Timer | None = None
    pending: CookingDecision | None = None
    current: CookingDecision | None = None
    audio_id: str | None = None
    alert: bool = False
    busy: bool = False
    error: str | None = None
    history: list[tuple[float, Phase, str]] = field(default_factory=list)

    def transition(self, phase: Phase, reason: str) -> None:
        if phase == self.phase:
            return
        log.info("state %s: %s -> %s (%s)", self.session_id, self.phase.value, phase.value, reason)
        self.history.append((time.time(), phase, reason))
        del self.history[:-50]
        self.phase = phase

    # --- timer operations ---------------------------------------------------

    def start_timer(self, seconds: int, label: str, clock: Clock = time.monotonic) -> None:
        if self.timer is not None:
            log.info("replacing running timer %r with %r", self.timer.label, label)
        self.timer = Timer(seconds, label, clock)
        self.pending = None
        self.alert = False
        self.display_text = label
        self.transition(Phase.COOKING, f"timer started {seconds}s")

    def cancel_timer(self, reason: str = "cancelled") -> bool:
        had_timer = self.timer is not None
        self.timer = None
        self.pending = None
        self.current = None
        self.alert = False
        self.display_text = "Cancelled"
        self.transition(Phase.IDLE, reason)
        return had_timer

    def pause(self) -> bool:
        return bool(self.timer and self.timer.pause())

    def resume(self) -> bool:
        return bool(self.timer and self.timer.resume())

    def adjust(self, delta_s: int) -> bool:
        if self.timer is None or self.phase != Phase.COOKING:
            return False
        self.timer.adjust(delta_s)
        return True

    def finish(self) -> None:
        label = self.timer.label if self.timer else ""
        self.display_text = "DONE"
        self.alert = True
        self.transition(Phase.DONE, f"timer expired ({label})")

    def acknowledge(self) -> bool:
        if self.phase not in (Phase.DONE, Phase.ERROR):
            return False
        self.alert = False
        self.timer = None
        self.current = None
        self.error = None
        self.display_text = "Show me food"
        self.transition(Phase.IDLE, "acknowledged")
        return True

    def timer_due(self) -> bool:
        return self.phase == Phase.COOKING and self.timer is not None and self.timer.expired

    # --- read model -----------------------------------------------------------

    def remaining_s(self) -> int:
        if self.timer is None or self.phase not in (Phase.COOKING, Phase.DONE):
            return 0
        # Round up so the display shows 0:01 until it truly hits zero.
        return int(-(-self.timer.remaining() // 1))

    def total_s(self) -> int:
        return self.timer.duration_s if self.timer else 0


class StateStore:
    def __init__(self, clock: Clock = time.monotonic):
        self.clock = clock
        self._lock = threading.RLock()
        self._version = 0
        self.sessions: dict[str, CookingSession] = {DEFAULT_SESSION: CookingSession()}

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @contextmanager
    def mutate(self, session_id: str = DEFAULT_SESSION) -> Iterator[CookingSession]:
        """Lock, yield the session for mutation, then bump the version."""
        with self._lock:
            session = self.sessions[session_id]
            try:
                yield session
            finally:
                self._version += 1

    @contextmanager
    def read(self, session_id: str = DEFAULT_SESSION) -> Iterator[CookingSession]:
        with self._lock:
            yield self.sessions[session_id]

    def snapshot(self, session_id: str = DEFAULT_SESSION) -> DeviceState:
        with self._lock:
            s = self.sessions[session_id]
            return DeviceState(
                version=self._version,
                state=s.phase,
                display_text=s.display_text,
                remaining_s=s.remaining_s(),
                total_s=s.total_s(),
                audio_id=s.audio_id,
                alert=s.alert,
                paused=bool(s.timer and s.timer.paused),
                busy=s.busy,
            )
