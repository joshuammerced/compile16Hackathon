"""Orchestrator: routes inputs (gestures, voice, buttons) and vision results to the state.

Everything here runs on the asyncio event loop, and nothing on it waits for the vision model
or for TTS:
  * vision inference runs on its own thread (vision/monitor.py) and arrives as callbacks,
  * speech is synthesized by a background task fed by a priority queue (alerts first),
so device polls, buttons and the timer watcher always respond immediately.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Coroutine
from typing import Any

from .audio.stt import STTProvider
from .audio.tts import Speaker
from .commands import Command, parse_command
from .controller import Controller, Effects, fmt_duration
from .llm import food_safety
from .schemas import AlertType, CookingStatus, DeviceState, Phase, display_text
from .state import CookingSession, StateStore
from .vision.gestures import GestureEvent, GestureWorker
from .vision.monitor import InferenceJob, VisionMonitor

log = logging.getLogger(__name__)

TIMER_TICK_S = 0.2
ALERT_SOUND = {AlertType.OVERCOOK_WARNING: "warning", AlertType.TIMER_DONE: "done", AlertType.MONITORING_LOST: "error"}
CONDITION_WORDS = {
    CookingStatus.APPEARS_COOKED: "looks cooked",
    CookingStatus.APPEARS_UNDERCOOKED: "doesn't look cooked yet",
    CookingStatus.POSSIBLY_OVERCOOKED_OR_BURNT: "may be burning",
    CookingStatus.CANNOT_DETERMINE: "is hard to see, so I can't tell how it's doing",
}
HELP = (
    "You can say: what is this, set a timer for 5 minutes, add a minute, pause, "
    "how does it look, or stop watching."
)


class Orchestrator:
    def __init__(
        self,
        store: StateStore,
        reader,
        monitor: VisionMonitor,
        controller: Controller,
        stt: STTProvider,
        speaker: Speaker,
        gestures: GestureWorker | None = None,
    ):
        self.store = store
        self.reader = reader
        self.monitor = monitor
        self.controller = controller
        self.stt = stt
        self.speaker = speaker
        self.gestures = gestures

        self.last_transcript: str | None = None
        self.last_command: Command | None = None
        self._queue: asyncio.Queue[GestureEvent] | None = None
        self._speech: asyncio.PriorityQueue | None = None
        self._speech_seq = itertools.count()
        self._tasks: list[asyncio.Task] = []
        self._background: set[asyncio.Task] = set()

    # --- lifecycle ----------------------------------------------------------------

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._speech = asyncio.PriorityQueue()
        self.reader.start()
        if self.gestures is not None:
            self.gestures.start(loop, self._queue)
        self.monitor.start(self._probe, self._on_observation, self._on_camera)
        self._tasks = [
            asyncio.create_task(self._consume_gestures(), name="gesture-consumer"),
            asyncio.create_task(self._watch_timer(), name="timer-watcher"),
            asyncio.create_task(self._speech_loop(), name="speech"),
        ]
        warm_up = getattr(self.stt, "warm_up", None)
        if warm_up is not None:
            # Whisper takes seconds to load (and downloads on first run); don't make the first voice request wait.
            self._spawn(asyncio.to_thread(warm_up))
        # Alerts restored from disk lost their in-memory audio: re-synthesize it.
        with self.store.read() as s:
            for alert in s.alerts:
                if alert.audio_id is None or self.speaker.store.get(alert.audio_id) is None:
                    self._enqueue_alert(alert.id)
        log.info("orchestrator started")

    async def stop(self) -> None:
        for task in [*self._tasks, *self._background]:
            task.cancel()
        await asyncio.gather(*self._tasks, *self._background, return_exceptions=True)
        await self.monitor.stop()
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

    # --- speech (never awaited by request handlers) ----------------------------------

    def say(self, text: str) -> None:
        """Queue informational speech for the current session; dropped if the session changes first."""
        assert self._speech is not None
        with self.store.read() as s:
            generation = s.generation
        self._speech.put_nowait((1, next(self._speech_seq), "say", text, generation))

    def _enqueue_alert(self, alert_id: int) -> None:
        assert self._speech is not None
        self._speech.put_nowait((0, next(self._speech_seq), "alert", alert_id, None))

    def _apply(self, fx: Effects) -> None:
        for alert in fx.alerts:
            self._enqueue_alert(alert.id)
        for text in fx.say:
            self.say(text)

    async def _speech_loop(self) -> None:
        assert self._speech is not None
        while True:
            _prio, _seq, kind, payload, generation = await self._speech.get()
            try:
                if kind == "say":
                    with self.store.read() as s:
                        if generation != s.generation:
                            log.info("dropping speech from an earlier session: %r", payload)
                            continue
                    clip_id = await self.speaker.speak(payload)
                    with self.store.mutate() as s:
                        s.audio_id = clip_id
                else:
                    await self._voice_alert(payload)
            except Exception:  # noqa: BLE001 - speech must never take the loop down
                log.exception("speech failed")

    async def _voice_alert(self, alert_id: int) -> None:
        with self.store.read() as s:
            alert = next((a for a in s.alerts if a.id == alert_id), None)
        if alert is None:
            return  # acknowledged before we got to it
        sound = ALERT_SOUND[alert.type]
        clip_id = await self.speaker.speak(alert.message, fallback_kind=sound, chime=sound, pin=True)
        with self.store.mutate() as s:
            live = next((a for a in s.alerts if a.id == alert_id), None)
            if live is None:
                self.speaker.store.unpin(clip_id)
            else:
                if live.audio_id:
                    self.speaker.store.unpin(live.audio_id)
                live.audio_id = clip_id

    # --- vision callbacks ------------------------------------------------------------

    def _probe(self) -> tuple[bool, int]:
        with self.store.read() as s:
            return s.active, s.generation

    async def _on_observation(self, job: InferenceJob, obs) -> bool:
        with self.store.mutate() as s:
            if not s.active or s.generation != job.generation:
                log.info("vision job %d dropped: session changed (gen %d -> %d)", job.id, job.generation, s.generation)
                return False
            fx = self.controller.on_observation(s, obs)
        self._apply(fx)
        return True

    async def _on_camera(self, ok: bool) -> None:
        with self.store.read() as s:
            if ok and s.vision_live and s.camera_stale_since is None:
                return  # steady state: nothing to change, nothing to persist
        with self.store.mutate() as s:
            fx = self.controller.on_camera(s, ok)
        self._apply(fx)

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
                fx = self.controller.on_timer_done(s)
            self._apply(fx)

    # --- session actions ------------------------------------------------------------

    def _begin_session(self, s: CookingSession, reason: str) -> None:
        s.start_session(reason)
        self.monitor.provider.on_session_start()
        self.monitor.cancel_current("new session")
        self.monitor.kick()

    async def start_session(self, source: str) -> DeviceState:
        with self.store.read() as s:
            active = s.active
        if active:
            return await self.status()
        if not self.monitor.available:
            log.warning("start_session from %s refused: vision %s", source, self.monitor.status)
            self.say("I can't watch the food right now because the camera model isn't ready. "
                     "Timers still work: say, set a timer for 5 minutes.")  # fmt: skip
            return self.snapshot()
        with self.store.mutate() as s:
            self._begin_session(s, f"start ({source})")
        self.say("Let me take a look.")
        return self.snapshot()

    async def end_session(self, reason: str) -> DeviceState:
        with self.store.mutate() as s:
            if not s.active and not s.alerts:
                return self.snapshot()
            for alert in s.alerts:
                self.speaker.store.unpin(alert.audio_id)
            s.end_session(reason)
        self.monitor.cancel_current(reason)
        self.say("Okay, I've stopped the timer and stopped watching.")
        return self.snapshot()

    async def confirm(self) -> DeviceState:
        with self.store.mutate() as s:
            sug = s.suggestion
            if s.phase != Phase.AWAITING_CONFIRMATION or sug is None:
                return self.snapshot()
            s.start_timer(sug.seconds, sug.food, self.store.clock)
        self.say(f"Starting a {fmt_duration(sug.seconds)} reminder for the {sug.food}. "
                 "I'll keep watching it, but check it's cooked before eating.")  # fmt: skip
        return self.snapshot()

    async def reject(self) -> DeviceState:
        with self.store.mutate() as s:
            if s.phase != Phase.AWAITING_CONFIRMATION:
                return self.snapshot()
            s.stop_timer("suggestion rejected")
            food = s.food or "food"
        self.say(f"Okay, no timer. I'll keep watching the {food}.")
        return self.snapshot()

    async def ack(self, alert_id: int) -> tuple[DeviceState, bool]:
        """Acknowledge an alert by id. Idempotent; returns (state, known)."""
        with self.store.mutate() as s:
            result = s.ack(alert_id)
            if result is None:
                return self.snapshot(), False
            if result is True:
                return self.snapshot(), True  # already acknowledged
            self.speaker.store.unpin(result.audio_id)
            if result.type == AlertType.TIMER_DONE and s.phase == Phase.DONE:
                # The countdown is over but the food is still on the stove: keep watching.
                s.timer = None
                s.display_text = display_text(s.food or "Watching")
                s.transition(Phase.MONITORING, "timer alert acknowledged")
        return self.snapshot(), True

    def status_text(self, s: CookingSession) -> str:
        if not s.active:
            return "Nothing is cooking right now. Show me the food and give a thumbs up."
        food = self.controller.food_name(s)
        parts = []
        obs = self.controller.fresh_observation(s)
        if not s.vision_live or not self.monitor.available:
            parts.append("I can't see the stove right now.")
        elif obs is not None:
            parts.append(f"The {food} {CONDITION_WORDS[obs.status]}.")
        else:
            parts.append(f"I'm watching the {food}.")
        if s.timer is not None and s.phase == Phase.COOKING:
            parts.append(f"{fmt_duration(int(s.timer.remaining()))} left on the timer"
                         + (", paused." if s.timer.paused else "."))  # fmt: skip
        elif s.phase == Phase.DONE:
            parts.append("The timer has finished.")
        return " ".join(parts)

    async def status(self) -> DeviceState:
        with self.store.read() as s:
            text = self.status_text(s)
        self.say(text)
        return self.snapshot()

    # --- inputs -----------------------------------------------------------------------

    async def handle_gesture(self, gesture: str) -> DeviceState:
        with self.store.read() as s:
            phase, top = s.phase, s.top_alert()
        log.info("gesture %s in state %s", gesture, phase.value)
        if gesture in ("Thumb_Up", "Open_Palm") and top is not None:
            return (await self.ack(top.id))[0]  # first, any gesture acknowledges the alert
        if gesture == "Thumb_Up":
            if phase == Phase.AWAITING_CONFIRMATION:
                return await self.confirm()
            if phase in (Phase.IDLE, Phase.ERROR):
                return await self.start_session("gesture")
            return await self.status()
        if gesture == "Open_Palm" and phase != Phase.IDLE:
            return await self.end_session("open palm")
        if gesture == "Thumb_Down" and phase == Phase.AWAITING_CONFIRMATION:
            return await self.reject()
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
            self.say("Sorry, I didn't hear anything.")
            return self.snapshot()
        return await self.command(transcript)

    async def command(self, transcript: str) -> DeviceState:
        cmd = parse_command(transcript)
        self.last_transcript, self.last_command = transcript, cmd
        log.info("command %r -> %s%s", transcript, cmd.intent, f" ({cmd.seconds}s)" if cmd.seconds else "")
        with self.store.read() as s:
            phase, has_timer = s.phase, s.timer is not None

        match cmd.intent:
            case "start_monitoring":
                return await self.start_session("voice")
            case "end_session":
                return await self.end_session("voice")
            case "cancel":
                if has_timer and phase == Phase.COOKING:
                    with self.store.mutate() as s:
                        s.stop_timer("timer cancelled by voice")
                    self.say("Timer cancelled. I'm still watching the food; say stop watching when you're done.")
                    return self.snapshot()
                return await self.end_session("voice cancel")
            case "set_timer":
                return await self.timer_control("start", cmd.seconds, None)
            case "add_time" | "subtract_time":
                delta = cmd.seconds if cmd.intent == "add_time" else -(cmd.seconds or 0)
                if phase == Phase.DONE and delta > 0:
                    return await self.timer_control("start", delta, None)
                with self.store.mutate() as s:
                    ok = s.adjust(delta)
                    remaining = int(s.timer.remaining()) if s.timer else 0
                if not ok:
                    self.say("There's no timer running to change.")
                else:
                    verb = "Added" if delta > 0 else "Took off"
                    self.say(f"{verb} {fmt_duration(abs(delta))}. {fmt_duration(remaining)} left.")
                return self.snapshot()
            case "pause" | "resume":
                return await self.button(cmd.intent)
            case "confirm":
                if phase == Phase.AWAITING_CONFIRMATION:
                    return await self.confirm()
                if phase in (Phase.IDLE, Phase.ERROR):
                    return await self.start_session("voice")
                return await self.status()
            case "reject":
                return await self.reject()
            case "status":
                return await self.status()
            case "cook_time_question":
                self.say(self._cook_time_answer(cmd.food or ""))
                return self.snapshot()
        self.say(f"Sorry, I didn't understand. {HELP}")
        return self.snapshot()

    @staticmethod
    def _cook_time_answer(food: str) -> str:
        info = food_safety.classify(food)
        if info.typical_s is None:
            return f"I don't have a typical time for {food}. Check the package or recipe."
        reminder = food_safety.thermometer_reminder(info)
        return f"{food.capitalize()} usually takes about {fmt_duration(info.typical_s)}." + (
            f" {reminder}" if reminder else ""
        )

    async def button(self, button: str) -> DeviceState:
        """BOX buttons. State changes apply immediately; speech follows in the background."""
        text: str | None = None
        if button == "ack":
            with self.store.mutate() as s:
                top = s.top_alert()
                if top is None and s.phase == Phase.ERROR:
                    s.end_session("error acknowledged")
            if top is not None:
                return (await self.ack(top.id))[0]
            return self.snapshot()
        if button == "cancel":
            return await self.end_session("cancel button")
        if button in ("pause", "resume"):
            with self.store.mutate() as s:
                ok = s.pause() if button == "pause" else s.resume()
                remaining = int(s.timer.remaining()) if s.timer else 0
            if ok:
                text = f"Paused with {fmt_duration(remaining)} left." if button == "pause" else "Resumed."
        elif button in ("plus_1m", "minus_1m"):
            delta = 60 if button == "plus_1m" else -60
            with self.store.mutate() as s:
                ok = s.adjust(delta)
                remaining = int(s.timer.remaining()) if s.timer else 0
            if ok:
                text = f"{fmt_duration(remaining)} left."
        else:
            raise ValueError(f"unknown button {button!r}")
        if text:
            self.say(text)
        return self.snapshot()

    async def timer_control(self, action: str, seconds: int | None, label: str | None) -> DeviceState:
        if action == "start":
            if not seconds or seconds <= 0:
                raise ValueError("start requires positive seconds")
            with self.store.mutate() as s:
                if not s.active:
                    self._begin_session(s, "timer set")
                s.start_timer(seconds, label or s.food or "Timer", self.store.clock)
            watching = " I'll keep an eye on the food too." if self.monitor.available else ""
            self.say(f"Timer set for {fmt_duration(seconds)}.{watching}")
            return self.snapshot()
        if action == "adjust":
            if not seconds:
                raise ValueError("adjust requires non-zero seconds")
            with self.store.mutate() as s:
                s.adjust(seconds)
            return self.snapshot()
        if action == "cancel":
            with self.store.mutate() as s:
                if s.timer is not None:
                    s.stop_timer("timer cancelled")
            return self.snapshot()
        return await self.button(action)
