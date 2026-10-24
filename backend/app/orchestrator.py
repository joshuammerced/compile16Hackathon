"""Orchestrator: turns gestures, voice and button presses into state transitions.

Everything here runs on the asyncio event loop. Background threads (camera ingest, gestures)
only hand events over via an asyncio.Queue fed with ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

from .audio.stt import STTProvider
from .audio.tts import Speaker
from .ingest.stream_reader import StreamReader
from .llm.base import LLMRouter, RouterResult
from .schemas import DISPLAY_TEXT_MAX, CookingDecision, DeviceState, Phase
from .state import CookingSession, StateStore
from .vision.gestures import GestureEvent, GestureWorker

log = logging.getLogger(__name__)

TIMER_TICK_S = 0.2


def fmt_duration(seconds: int) -> str:
    """Spoken duration: 90 -> '1 minute 30 seconds'."""
    m, s = divmod(int(seconds), 60)
    parts = []
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s or not m:
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    return " ".join(parts)


class Orchestrator:
    def __init__(
        self,
        store: StateStore,
        reader: StreamReader,
        router: LLMRouter,
        stt: STTProvider,
        speaker: Speaker,
        gestures: GestureWorker | None = None,
        auto_start_timer: bool = True,
    ):
        self.store = store
        self.reader = reader
        self.router = router
        self.stt = stt
        self.speaker = speaker
        self.gestures = gestures
        self.auto_start_timer = auto_start_timer

        self.last_result: RouterResult | None = None
        self.last_transcript: str | None = None
        self._analyzing = False
        self._queue: asyncio.Queue[GestureEvent] | None = None
        self._tasks: list[asyncio.Task] = []
        self._background: set[asyncio.Task] = set()

    # --- lifecycle ----------------------------------------------------------------

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self.reader.start()
        if self.gestures is not None:
            self.gestures.start(loop, self._queue)
        self._tasks = [
            asyncio.create_task(self._consume_gestures(), name="gesture-consumer"),
            asyncio.create_task(self._watch_timer(), name="timer-watcher"),
        ]
        warm_up = getattr(self.stt, "warm_up", None)
        if warm_up is not None:
            # Whisper takes seconds to load (and downloads on first run); don't make the first voice request wait.
            self._spawn(asyncio.to_thread(warm_up))
        log.info("orchestrator started")

    async def stop(self) -> None:
        for task in [*self._tasks, *self._background]:
            task.cancel()
        await asyncio.gather(*self._tasks, *self._background, return_exceptions=True)
        if self.gestures is not None:
            await asyncio.to_thread(self.gestures.stop)
        await asyncio.to_thread(self.reader.stop)
        log.info("orchestrator stopped")

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def snapshot(self) -> DeviceState:
        return self.store.snapshot()

    # --- audio ----------------------------------------------------------------------

    async def say(self, text: str, kind: str = "info", chime: str | None = None) -> str:
        clip_id = await self.speaker.speak(text, fallback_kind=kind, chime=chime)
        with self.store.mutate() as s:
            s.audio_id = clip_id
        return clip_id

    # --- background loops -------------------------------------------------------------

    async def _consume_gestures(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                await self.handle_gesture(event.gesture)
            except Exception:  # noqa: BLE001 - keep consuming
                log.exception("gesture handling failed")

    async def _watch_timer(self) -> None:
        while True:
            await asyncio.sleep(TIMER_TICK_S)
            with self.store.read() as s:
                due = s.timer_due()
            if not due:
                continue
            with self.store.mutate() as s:
                if not s.timer_due():
                    continue
                s.finish()
                text = self._done_text(s)
            await self.say(text, kind="done", chime="done")

    @staticmethod
    def _done_text(s: CookingSession) -> str:
        label = s.timer.label if s.timer else "Your food"
        text = f"Time's up. {label} should be done."
        if s.current and s.current.safe_internal_temp_f:
            text += f" Check it has reached {s.current.safe_internal_temp_f} degrees."
        return text

    # --- inputs -----------------------------------------------------------------------

    async def handle_gesture(self, gesture: str) -> DeviceState:
        with self.store.read() as s:
            phase = s.phase
        if gesture == "Thumb_Up":
            if phase == Phase.AWAITING_CONFIRMATION:
                return await self.confirm()
            if phase == Phase.DONE:
                return await self.button("ack")
            return await self.analyze(None, source="gesture")
        if gesture == "Open_Palm":
            if phase == Phase.DONE:
                return await self.button("ack")
            if phase in (Phase.COOKING, Phase.AWAITING_CONFIRMATION):
                return await self.button("cancel")
        if gesture == "Thumb_Down" and phase == Phase.AWAITING_CONFIRMATION:
            return await self.reject()
        log.info("gesture %s ignored in state %s", gesture, phase.value)
        return self.snapshot()

    async def voice(self, audio: bytes, transcript: str | None = None) -> DeviceState:
        if transcript is None:
            try:
                transcript = await self.stt.transcribe(audio)
            except Exception:  # noqa: BLE001
                log.exception("STT failed")
                transcript = ""
        log.info("voice transcript: %r", transcript)
        if not transcript.strip():
            await self.say("Sorry, I didn't hear anything.", kind="error")
            return self.snapshot()
        return await self.analyze(transcript, source="voice")

    async def analyze(self, transcript: str | None, source: str = "control") -> DeviceState:
        if self._analyzing:
            log.info("analyze from %s ignored: one is already in flight", source)
            return self.snapshot()
        self._analyzing = True
        try:
            with self.store.mutate() as s:
                prev_phase = s.phase
                s.busy = True
                if prev_phase != Phase.COOKING:
                    s.transition(Phase.ANALYZING, f"analyze ({source})")
                    s.display_text = "Thinking..."
                timer_label = s.timer.label if s.timer else None
                remaining = s.remaining_s()

            frame = await asyncio.to_thread(self.reader.get_snapshot)
            jpeg = await asyncio.to_thread(frame.to_jpeg) if frame is not None else None
            self.last_transcript = transcript
            result = await self.router.decide(jpeg, transcript, prev_phase.value, timer_label, remaining)
            self.last_result = result

            with self.store.mutate() as s:
                s.busy = False
                text = self._apply_decision(s, result.decision, prev_phase)
            await self.say(text)
        except Exception as exc:  # noqa: BLE001 - surface as ERROR state instead of crashing
            log.exception("analyze failed")
            with self.store.mutate() as s:
                s.busy = False
                s.error = str(exc)
                if s.phase != Phase.COOKING:
                    s.display_text = "Error"
                    s.transition(Phase.ERROR, f"analyze failed: {exc}")
            await self.say("Sorry, something went wrong.", kind="error")
        finally:
            self._analyzing = False
        return self.snapshot()

    def _apply_decision(self, s: CookingSession, d: CookingDecision, prev_phase: Phase) -> str:
        """Apply a validated decision to the session (lock held). Returns the text to speak."""
        if d.action == "start_timer" and d.recommended_cook_seconds:
            label = d.display_text or (d.food_name or "Timer")[:DISPLAY_TEXT_MAX]
            if self.auto_start_timer:
                s.current = d
                s.start_timer(d.recommended_cook_seconds, label, self.store.clock)
                return d.spoken_feedback
            s.pending = d
            mins = max(1, round(d.recommended_cook_seconds / 60))
            s.display_text = f"{label[: DISPLAY_TEXT_MAX - 6]}? {mins}m"
            s.transition(Phase.AWAITING_CONFIRMATION, "suggestion awaiting thumbs-up")
            return f"{d.spoken_feedback} Thumbs up to start the timer, thumbs down to skip."

        if d.action == "cancel_timer":
            if s.timer is None and s.pending is None:
                self._restore(s, prev_phase, "nothing to cancel")
                return "There's no timer running."
            s.cancel_timer("cancelled by voice")
            return d.spoken_feedback or "Timer cancelled."

        if d.action == "adjust_timer":
            if d.adjust_seconds and s.adjust(d.adjust_seconds):
                verb = "Added" if d.adjust_seconds > 0 else "Took off"
                return f"{verb} {fmt_duration(abs(d.adjust_seconds))}. {fmt_duration(s.remaining_s())} left."
            self._restore(s, prev_phase, "nothing to adjust")
            return "There's no timer running to change."

        # ask_clarification / no_food_detected / answer_question: no timer change.
        self._restore(s, prev_phase, d.action)
        if s.phase != Phase.COOKING:
            s.display_text = d.display_text
        return d.spoken_feedback

    @staticmethod
    def _restore(s: CookingSession, prev_phase: Phase, reason: str) -> None:
        if prev_phase in (Phase.COOKING, Phase.AWAITING_CONFIRMATION):
            s.transition(prev_phase, reason)
        else:
            s.alert = False
            s.transition(Phase.IDLE, reason)

    async def confirm(self) -> DeviceState:
        with self.store.mutate() as s:
            d = s.pending
            if s.phase != Phase.AWAITING_CONFIRMATION or d is None or not d.recommended_cook_seconds:
                return self.snapshot()
            s.current = d
            s.start_timer(d.recommended_cook_seconds, d.display_text or "Timer", self.store.clock)
            text = f"Starting {fmt_duration(d.recommended_cook_seconds)} for {d.food_name or 'your food'}."
        await self.say(text)
        return self.snapshot()

    async def reject(self) -> DeviceState:
        with self.store.mutate() as s:
            if s.phase != Phase.AWAITING_CONFIRMATION:
                return self.snapshot()
            s.pending = None
            s.display_text = "Show me food"
            s.transition(Phase.IDLE, "suggestion rejected")
        await self.say("Okay, skipped. Show me the food again when you're ready.")
        return self.snapshot()

    async def button(self, button: str) -> DeviceState:
        """BOX buttons. State changes apply immediately; speech is generated in the background."""
        text: str | None = None
        with self.store.read() as s:
            phase = s.phase
        if button == "ack":
            with self.store.mutate() as s:
                if not s.acknowledge():
                    return self.snapshot()
        elif button == "cancel":
            if phase not in (Phase.COOKING, Phase.AWAITING_CONFIRMATION, Phase.DONE):
                return self.snapshot()
            with self.store.mutate() as s:
                s.cancel_timer("cancel button")
            text = "Timer cancelled."
        elif button in ("pause", "resume"):
            with self.store.mutate() as s:
                ok = s.pause() if button == "pause" else s.resume()
                remaining = s.remaining_s()
            if ok:
                text = f"Paused with {fmt_duration(remaining)} left." if button == "pause" else "Resumed."
        elif button in ("plus_1m", "minus_1m"):
            delta = 60 if button == "plus_1m" else -60
            with self.store.mutate() as s:
                ok = s.adjust(delta)
                remaining = s.remaining_s()
            if ok:
                text = f"{fmt_duration(remaining)} left."
        else:
            raise ValueError(f"unknown button {button!r}")
        if text:
            self._spawn(self.say(text))
        return self.snapshot()

    async def timer_control(self, action: str, seconds: int | None, label: str | None) -> DeviceState:
        """Manual timer control for the debug UI."""
        if action == "start":
            if not seconds or seconds <= 0:
                raise ValueError("start requires positive seconds")
            with self.store.mutate() as s:
                s.current = None
                s.start_timer(seconds, (label or "Timer")[:DISPLAY_TEXT_MAX], self.store.clock)
            self._spawn(self.say(f"Timer set for {fmt_duration(seconds)}."))
            return self.snapshot()
        if action == "adjust":
            if not seconds:
                raise ValueError("adjust requires non-zero seconds")
            with self.store.mutate() as s:
                s.adjust(seconds)
            return self.snapshot()
        mapping = {"cancel": "cancel", "pause": "pause", "resume": "resume"}
        return await self.button(mapping[action])
