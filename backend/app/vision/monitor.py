"""Continuous visual monitoring during an active cooking session.

* ``InferenceWorker``: one daemon thread that loads the vision model once, keeps it in memory,
  and runs one inference at a time. Its single job slot refuses new work while busy, so frames
  are never queued and the event loop is never blocked.
* ``VisionMonitor``: an asyncio loop that, while a session is active, picks the *latest* frame,
  rejects it if stale (too old, or the same frame as last time), submits it if the worker is
  free, and validates the result before handing it to the orchestrator.

Every job carries the session generation and frame timestamp. A result is dropped if it was
cancelled, belongs to an earlier session, or describes a frame that is too old by the time the
model finished, so a slow inference can never overwrite fresher knowledge.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..ingest.stream_reader import Frame
from ..llm.base import VisionProvider, parse_observation
from ..schemas import VisualObservation

log = logging.getLogger(__name__)


@dataclass
class InferenceJob:
    id: int
    generation: int
    frame_seq: int
    frame_ts: float
    submitted_at: float
    cancelled: bool = False


@dataclass
class MonitorStats:
    submitted: int = 0
    completed: int = 0
    applied: int = 0
    failed: int = 0
    rejected_stale: int = 0
    rejected_cancelled: int = 0
    skipped_busy: int = 0
    skipped_stale_frame: int = 0
    skipped_repeat_frame: int = 0
    last_latency_s: float | None = None
    last_error: str | None = None
    rejections: list[str] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.rejections.append(reason)
        del self.rejections[:-20]


class InferenceWorker:
    """Single thread that owns the model. ``try_submit`` never queues: busy means no."""

    def __init__(self, provider: VisionProvider):
        self.provider = provider
        self.status = "not started"
        self._cv = threading.Condition()
        self._job: tuple[Frame, asyncio.AbstractEventLoop, asyncio.Future] | None = None
        self._busy = False
        self._stop = False
        self._thread: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def busy(self) -> bool:
        with self._cv:
            return self._busy

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="vision-inference", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None:
            # An in-progress generate() can't be interrupted; the thread is a daemon, so don't wait long.
            self._thread.join(timeout)

    def try_submit(self, frame: Frame) -> asyncio.Future | None:
        loop = asyncio.get_running_loop()
        with self._cv:
            if self._busy or not self.ready or self._stop:
                return None
            fut = loop.create_future()
            self._job = (frame, loop, fut)
            self._busy = True
            self._cv.notify()
            return fut

    @staticmethod
    def _deliver(fut: asyncio.Future, result: object, error: BaseException | None) -> None:
        if fut.done():  # awaiting task was cancelled
            return
        if error is not None:
            fut.set_exception(error)
        else:
            fut.set_result(result)

    def _run(self) -> None:
        self.status = "loading"
        log.info("loading vision model (%s)...", self.provider.name)
        try:
            self.provider.load()
        except Exception as exc:  # noqa: BLE001 - e.g. MLX unavailable on this machine
            self.status = f"unavailable: {exc}"
            log.error("vision model unavailable, visual monitoring disabled: %s", exc)
            return
        self.status = "ready"
        log.info("vision model ready (%s)", self.provider.name)
        while True:
            with self._cv:
                while self._job is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                frame, loop, fut = self._job
                self._job = None
            result, error = None, None
            t0 = time.perf_counter()
            try:
                raw = self.provider.assess(frame.to_jpeg())
                result = (raw, time.perf_counter() - t0)
            except Exception as exc:  # noqa: BLE001
                error = exc
            with self._cv:
                self._busy = False
            if not loop.is_closed():
                loop.call_soon_threadsafe(self._deliver, fut, result, error)


SessionProbe = Callable[[], tuple[bool, int]]  # -> (session active, generation)
ResultHandler = Callable[[InferenceJob, VisualObservation], Awaitable[bool]]
CameraHandler = Callable[[bool], Awaitable[None]]


class VisionMonitor:
    def __init__(
        self,
        reader,
        provider: VisionProvider,
        interval_s: float = 4.0,
        frame_max_age_s: float = 2.0,
        result_max_age_s: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.reader = reader
        self.worker = InferenceWorker(provider)
        self.interval_s = interval_s
        self.frame_max_age_s = frame_max_age_s
        self.result_max_age_s = result_max_age_s
        self.clock = clock
        self.stats = MonitorStats()
        self.last_observation: VisualObservation | None = None

        self._probe: SessionProbe | None = None
        self._on_result: ResultHandler | None = None
        self._on_camera: CameraHandler | None = None
        self._kick = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._job_task: asyncio.Task | None = None
        self._current: InferenceJob | None = None
        self._last_seq = -1
        self._ids = itertools.count(1)

    @property
    def provider(self) -> VisionProvider:
        return self.worker.provider

    @property
    def available(self) -> bool:
        return self.worker.ready

    @property
    def status(self) -> str:
        return self.worker.status

    def start(self, probe: SessionProbe, on_result: ResultHandler, on_camera: CameraHandler) -> None:
        self._probe, self._on_result, self._on_camera = probe, on_result, on_camera
        self.worker.start()
        self._task = asyncio.create_task(self._loop(), name="vision-monitor")

    async def stop(self) -> None:
        for task in (self._task, self._job_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(t for t in (self._task, self._job_task) if t), return_exceptions=True)
        await asyncio.to_thread(self.worker.stop)

    def kick(self) -> None:
        """Look now instead of waiting for the next interval (e.g. a session just started)."""
        self._kick.set()

    def cancel_current(self, reason: str) -> None:
        """Mark the in-flight job cancelled; its result will be discarded when it arrives."""
        if self._current is not None and not self._current.cancelled:
            self._current.cancelled = True
            log.info("vision job %d cancelled: %s", self._current.id, reason)

    # --- loop -------------------------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._kick.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass
            self._kick.clear()
            try:
                await self.tick()
            except Exception:  # noqa: BLE001 - the monitor must keep running
                log.exception("vision monitor tick failed")

    async def tick(self) -> None:
        assert self._probe is not None and self._on_camera is not None
        active, generation = self._probe()
        if not active or not self.worker.ready:
            return
        frame = self.reader.latest()
        now = self.clock()
        if frame is None or now - frame.timestamp > self.frame_max_age_s:
            self.stats.skipped_stale_frame += 1
            await self._on_camera(False)
            return
        await self._on_camera(True)
        if self._job_task is not None and not self._job_task.done():
            self.stats.skipped_busy += 1  # never queue: the next tick will take the newest frame
            return
        if frame.seq == self._last_seq:
            self.stats.skipped_repeat_frame += 1
            return
        fut = self.worker.try_submit(frame)
        if fut is None:
            self.stats.skipped_busy += 1
            return
        self._last_seq = frame.seq
        job = InferenceJob(next(self._ids), generation, frame.seq, frame.timestamp, now)
        self._current = job
        self.stats.submitted += 1
        self._job_task = asyncio.create_task(self._await_job(job, fut), name=f"vision-job-{job.id}")

    async def _await_job(self, job: InferenceJob, fut: asyncio.Future) -> None:
        try:
            raw, latency = await fut
        except Exception as exc:  # noqa: BLE001
            self.stats.failed += 1
            self.stats.last_error = str(exc)
            log.warning("vision job %d failed: %s", job.id, exc)
            return
        finally:
            if self._current is job:
                self._current = None
        self.stats.completed += 1
        self.stats.last_latency_s = latency
        parsed = parse_observation(raw)
        obs = VisualObservation(
            food=parsed.food,
            status=parsed.status,
            evidence=parsed.evidence,
            recommendation=parsed.recommendation,
            parse_ok=parsed.parse_ok,
            frame_seq=job.frame_seq,
            frame_ts=job.frame_ts,
            observed_at=time.time(),
            latency_s=latency,
            raw=raw,
        )
        log.info(
            "vision job %d: %s / %s in %.2fs%s",
            job.id,
            obs.food,
            obs.status.value,
            latency,
            "" if parsed.parse_ok else " (unparseable output)",
        )
        self.last_observation = obs
        age = self.clock() - job.frame_ts
        if job.cancelled:
            self.stats.rejected_cancelled += 1
            self.stats.reject(f"job {job.id}: cancelled")
            return
        if age > self.result_max_age_s:
            self.stats.rejected_stale += 1
            self.stats.reject(f"job {job.id}: frame {age:.1f}s old when result arrived")
            log.warning("vision job %d rejected: frame was %.1fs old", job.id, age)
            return
        assert self._on_result is not None
        if await self._on_result(job, obs):
            self.stats.applied += 1
        else:
            self.stats.rejected_stale += 1
            self.stats.reject(f"job {job.id}: session changed (generation {job.generation})")
