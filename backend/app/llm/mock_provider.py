"""Mock vision model that returns canned observations, so everything runs without the real model.

It emits exactly the text format of LLM/inference.py, so the real parser runs on it. The
default script walks one session through undercooked -> cooked -> burning, which exercises the
"looks done" notice and the overcook warning in the demo.
"""

from __future__ import annotations

import threading
import time

from ..schemas import CookingStatus
from .base import VisionProvider

U, C, B, X = (
    CookingStatus.APPEARS_UNDERCOOKED,
    CookingStatus.APPEARS_COOKED,
    CookingStatus.POSSIBLY_OVERCOOKED_OR_BURNT,
    CookingStatus.CANNOT_DETERMINE,
)

DEFAULT_SCRIPT: list[tuple[str | None, CookingStatus]] = (
    [("Fried egg", U)] * 4 + [("Fried egg", C)] * 6 + [("Fried egg", B)] * 100
)

EVIDENCE = {
    U: "The egg white is still translucent around the yolk.",
    C: "The white is opaque and set with lightly browned edges.",
    B: "The edges are dark brown to black and there is visible smoke.",
    X: "The food is not clearly visible.",
}


def format_response(food: str | None, status: CookingStatus) -> str:
    return (
        f"Food Identified: {food or 'Unknown'}\n\n"
        f"Cooking Status: {status.value}\n\n"
        f"Visual Evidence: {EVIDENCE[status]}\n\n"
        "Recommendation: Keep an eye on it."
    )


class MockVisionProvider(VisionProvider):
    name = "mock"

    def __init__(self, script: list[tuple[str | None, CookingStatus]] | None = None, latency_s: float = 0.05):
        self.script = list(script or DEFAULT_SCRIPT)
        self.latency_s = latency_s
        self.calls = 0
        self._step = 0
        self._lock = threading.Lock()

    def on_session_start(self) -> None:
        with self._lock:
            self._step = 0

    def assess(self, jpeg: bytes) -> str:
        time.sleep(self.latency_s)  # runs on the inference thread, like the real model
        with self._lock:
            food, status = self.script[min(self._step, len(self.script) - 1)]
            self._step += 1
            self.calls += 1
        return format_response(food, status)
