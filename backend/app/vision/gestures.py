"""MediaPipe gesture worker thread with debouncing.

The worker runs the MediaPipe Tasks Gesture Recognizer on the latest camera frame at a capped
rate, debounces the result, and hands confirmed gestures to the asyncio orchestrator through
``loop.call_soon_threadsafe(queue.put_nowait, ...)`` so no async code runs on this thread.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from ..ingest.stream_reader import StreamReader

log = logging.getLogger(__name__)

# MediaPipe canned gestures we act on. Everything else ("None", "Victory", ...) resets the hold.
TRACKED_GESTURES = {"Thumb_Up", "Open_Palm", "Thumb_Down"}


@dataclass(frozen=True)
class GestureEvent:
    gesture: str  # MediaPipe category name, e.g. "Thumb_Up"
    score: float
    timestamp: float


class GestureDebouncer:
    """Fires a gesture only after ``hold_frames`` consecutive confident sightings, then cools down.

    Pure logic (no MediaPipe, no threads) so it can be tested with synthetic results.
    """

    def __init__(self, hold_frames: int = 5, min_score: float = 0.6, cooldown_s: float = 3.0):
        self.hold_frames = max(1, hold_frames)
        self.min_score = min_score
        self.cooldown_s = cooldown_s
        self._candidate: str | None = None
        self._count = 0
        self._last_fire = float("-inf")

    def reset(self) -> None:
        self._candidate = None
        self._count = 0

    def update(self, gesture: str | None, score: float, now: float) -> GestureEvent | None:
        if gesture not in TRACKED_GESTURES or score < self.min_score:
            self.reset()
            return None
        if gesture != self._candidate:
            self._candidate = gesture
            self._count = 0
        self._count += 1
        if self._count < self.hold_frames:
            return None
        if now - self._last_fire < self.cooldown_s:
            return None
        self._last_fire = now
        # Require a fresh hold before the same gesture can fire again.
        self.reset()
        return GestureEvent(gesture=gesture, score=score, timestamp=now)


class GestureWorker:
    def __init__(
        self,
        reader: StreamReader,
        model_path: Path,
        fps: float = 10.0,
        hold_frames: int = 5,
        min_score: float = 0.6,
        cooldown_s: float = 3.0,
        input_width: int = 320,
    ):
        self.reader = reader
        self.model_path = Path(model_path)
        self.period = 1.0 / max(fps, 0.1)
        self.input_width = input_width
        self.debouncer = GestureDebouncer(hold_frames, min_score, cooldown_s)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[GestureEvent] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._label: tuple[str, float] | None = None
        self.status = "not started"

    # --- public API -------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[GestureEvent]) -> None:
        self._loop, self._queue = loop, queue
        self._thread = threading.Thread(target=self._run, name="gesture-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def current_label(self) -> tuple[str, float] | None:
        """Latest raw (undebounced) top gesture, for the debug overlay."""
        with self._lock:
            return self._label

    # --- internals ---------------------------------------------------------------

    def _emit(self, event: GestureEvent) -> None:
        log.info("gesture event: %s (score %.2f)", event.gesture, event.score)
        if self._loop is not None and self._queue is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def _create_recognizer(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )
        return mp, mp_vision.GestureRecognizer.create_from_options(options)

    def _run(self) -> None:
        if not self.model_path.exists():
            self.status = f"model missing: {self.model_path} (run scripts/download_models.py)"
            log.error("gesture worker disabled: %s", self.status)
            return
        try:
            mp, recognizer = self._create_recognizer()
        except Exception as exc:  # noqa: BLE001 - mediapipe missing or incompatible
            self.status = f"mediapipe unavailable: {exc}"
            log.error("gesture worker disabled: %s", self.status)
            return

        self.status = "running"
        log.info("gesture worker started at %.1f fps", 1.0 / self.period)
        last_seq = -1
        last_ts_ms = 0
        with recognizer:
            while not self._stop.is_set():
                tick = time.monotonic()
                frame = self.reader.latest()
                if frame is not None and frame.seq != last_seq:
                    last_seq = frame.seq
                    # VIDEO mode requires strictly increasing timestamps.
                    last_ts_ms = max(last_ts_ms + 1, int(tick * 1000))
                    try:
                        gesture, score = self._recognize(mp, recognizer, frame.image, last_ts_ms)
                    except Exception:  # noqa: BLE001 - never let one bad frame kill the worker
                        log.exception("gesture inference failed")
                        gesture, score = None, 0.0
                    with self._lock:
                        self._label = (gesture, score) if gesture else None
                    event = self.debouncer.update(gesture, score, tick)
                    if event is not None:
                        self._emit(event)
                self._stop.wait(max(0.0, self.period - (time.monotonic() - tick)))
        self.status = "stopped"

    def _recognize(self, mp, recognizer, image, ts_ms: int) -> tuple[str | None, float]:
        h, w = image.shape[:2]
        if w > self.input_width:
            scale = self.input_width / w
            image = cv2.resize(image, (self.input_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = recognizer.recognize_for_video(mp_image, ts_ms)
        if not result.gestures or not result.gestures[0]:
            return None, 0.0
        top = result.gestures[0][0]
        return top.category_name, float(top.score)
