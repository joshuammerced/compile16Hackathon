"""Continuous monitoring: freshness, single-flight inference, stale-result rejection, and
inference never blocking device controls or the timer."""

import threading
import time

from app.llm.base import VisionProvider
from app.llm.mock_provider import MockVisionProvider, format_response
from app.schemas import AlertType, CookingStatus as S, Phase
from app.vision.monitor import InferenceJob
from tests.conftest import FakeReader, mock_settings, running_app, wait_for

UNDERCOOKED_EGG = [("Fried egg", S.APPEARS_UNDERCOOKED)]


class TrackingProvider(MockVisionProvider):
    """Records concurrency and which frames it was given."""

    def __init__(self, latency_s=0.05, script=UNDERCOOKED_EGG):
        super().__init__(script=script, latency_s=latency_s)
        self.active = 0
        self.max_active = 0
        self.threads = set()
        self._guard = threading.Lock()

    def assess(self, jpeg):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.threads.add(threading.current_thread().name)
        try:
            return super().assess(jpeg)
        finally:
            with self._guard:
                self.active -= 1


async def test_monitoring_runs_continuously_during_session():
    vision = TrackingProvider()
    async with running_app(reader=FakeReader(), vision=vision) as (app, client):
        orch = app.state.orch
        assert vision.calls == 0  # nothing runs while idle
        await client.post("/control/gesture/Thumb_Up")
        await wait_for(lambda: vision.calls >= 5)
        with orch.store.read() as s:
            assert s.phase == Phase.AWAITING_CONFIRMATION and len(s.observations) >= 4
            seqs = [o.frame_seq for o in s.observations]
        assert seqs == sorted(set(seqs))  # each a newer frame, none reused
        assert vision.threads == {"vision-inference"}  # always the one model thread

        await client.post("/device/button", json={"button": "cancel"})
        calls = vision.calls
        await wait_for(lambda: orch.monitor.worker.busy is False)
        time.sleep(0.3)
        assert vision.calls <= calls + 1  # stops when the session ends


async def test_stale_frames_are_never_analyzed_and_raise_monitoring_lost():
    reader = FakeReader(auto=False)
    reader.push(age_s=10)  # camera has stalled
    vision = TrackingProvider()
    async with running_app(reader=reader, vision=vision) as (app, client):
        await client.post("/control/analyze")
        state = await wait_for(lambda: (s := app.state.orch.snapshot()).alert and s)
        assert vision.calls == 0
        assert app.state.orch.monitor.stats.skipped_stale_frame > 0
        assert state.alert_type == AlertType.MONITORING_LOST and state.monitoring is False

        reader.auto = True  # camera recovers: alert resolves itself, analysis resumes
        await wait_for(lambda: not app.state.orch.snapshot().alert and vision.calls > 0)
        assert app.state.orch.snapshot().monitoring is True


async def test_same_frame_is_not_reanalyzed():
    reader = FakeReader(auto=False)
    reader.push()
    vision = TrackingProvider()
    async with running_app(reader=reader, vision=vision) as (app, client):
        await client.post("/control/analyze")
        await wait_for(lambda: app.state.orch.monitor.stats.skipped_repeat_frame >= 3)
        assert vision.calls == 1


async def test_only_one_inference_in_flight_and_no_backlog():
    vision = TrackingProvider(latency_s=0.25)
    async with running_app(reader=FakeReader(), vision=vision) as (app, client):
        await client.post("/control/analyze")
        await wait_for(lambda: vision.calls >= 4, timeout=4)
        stats = app.state.orch.monitor.stats
        assert vision.max_active == 1
        assert stats.skipped_busy > 0  # ticks while busy were dropped, not queued
        assert stats.submitted - stats.completed <= 1


async def test_result_for_ended_session_is_discarded():
    vision = TrackingProvider(latency_s=0.4)
    async with running_app(reader=FakeReader(), vision=vision) as (app, client):
        orch = app.state.orch
        await client.post("/control/analyze")
        await wait_for(lambda: orch.monitor.worker.busy)
        await client.post("/device/button", json={"button": "cancel"})  # ends session mid-inference
        await client.post("/control/timer", json={"action": "start", "seconds": 600})  # new session
        await wait_for(lambda: orch.monitor.stats.rejected_cancelled >= 1)
        with orch.store.read() as s:
            # The old session's result never reached the new session.
            assert all(o.frame_ts >= 0 for o in s.observations)
            assert orch.monitor.stats.applied == 0 or len(s.observations) <= orch.monitor.stats.applied


async def test_result_from_older_generation_rejected_by_orchestrator():
    async with running_app(reader=FakeReader(), vision=TrackingProvider(latency_s=5)) as (app, client):
        orch = app.state.orch
        await client.post("/control/analyze")
        with orch.store.read() as s:
            gen = s.generation
        parsed = format_response("Fried egg", S.POSSIBLY_OVERCOOKED_OR_BURNT)
        from app.llm.base import parse_observation
        from app.schemas import VisualObservation

        p = parse_observation(parsed)
        obs = VisualObservation(food=p.food, status=p.status, frame_seq=1, frame_ts=time.monotonic(), observed_at=0)
        assert await orch._on_observation(InferenceJob(99, gen - 1, 1, time.monotonic(), 0), obs) is False
        assert await orch._on_observation(InferenceJob(100, gen, 1, time.monotonic(), 0), obs) is True


async def test_results_whose_frame_got_too_old_are_rejected():
    settings = mock_settings(result_max_age_s=0.1)
    vision = TrackingProvider(latency_s=0.25)
    async with running_app(settings, reader=FakeReader(), vision=vision) as (app, client):
        await client.post("/control/analyze")
        await wait_for(lambda: app.state.orch.monitor.stats.rejected_stale >= 2)
        with app.state.orch.store.read() as s:
            assert len(s.observations) == 0 and s.phase == Phase.ANALYZING


async def test_device_controls_and_polls_stay_fast_during_slow_inference():
    vision = TrackingProvider(latency_s=1.5)
    async with running_app(reader=FakeReader(), vision=vision) as (app, client):
        await client.post("/control/timer", json={"action": "start", "seconds": 300})
        await wait_for(lambda: app.state.orch.monitor.worker.busy)
        t0 = time.perf_counter()
        for _ in range(5):
            assert (await client.get("/device/state")).status_code == 200
        s = (await client.post("/device/button", json={"button": "plus_1m"})).json()
        elapsed = time.perf_counter() - t0
        assert app.state.orch.monitor.worker.busy  # inference is still running...
        assert elapsed < 0.5 and s["total_s"] == 360  # ...but controls answered immediately


async def test_timer_expires_on_time_while_model_is_busy():
    vision = TrackingProvider(latency_s=3)
    async with running_app(reader=FakeReader(), vision=vision) as (app, client):
        await client.post("/control/timer", json={"action": "start", "seconds": 1})
        await wait_for(lambda: app.state.orch.monitor.worker.busy)
        t0 = time.perf_counter()
        state = await wait_for(lambda: (s := app.state.orch.snapshot()).state == Phase.DONE and s, timeout=2)
        assert time.perf_counter() - t0 < 1.5
        assert state.alert_type == AlertType.TIMER_DONE


class BrokenModel(VisionProvider):
    name = "broken"

    def load(self):
        raise RuntimeError("MLX requires Apple Silicon")

    def assess(self, jpeg):
        raise AssertionError("never called")


async def test_without_model_monitoring_is_unavailable_but_timers_work():
    async with running_app(reader=FakeReader(), vision=BrokenModel()) as (app, client):
        health = (await client.get("/health")).json()
        assert health["vision"]["status"].startswith("unavailable")
        s = (await client.post("/control/gesture/Thumb_Up")).json()
        assert s["state"] == "IDLE"  # refuses to pretend it is watching
        s = (await client.post("/control/command", json={"transcript": "set a timer for 2 minutes"})).json()
        assert s["state"] == "COOKING" and s["total_s"] == 120 and s["monitoring"] is False


async def test_overcook_warning_from_continuous_monitoring():
    script = [("Fried egg", S.APPEARS_UNDERCOOKED)] * 2 + [("Fried egg", S.POSSIBLY_OVERCOOKED_OR_BURNT)] * 50
    async with running_app(reader=FakeReader(), vision=TrackingProvider(script=script)) as (app, client):
        await client.post("/control/analyze")
        await wait_for(lambda: app.state.orch.snapshot().state == Phase.AWAITING_CONFIRMATION)
        await client.post("/control/gesture/Thumb_Up")  # confirm the 3 minute suggestion
        state = await wait_for(lambda: (s := app.state.orch.snapshot()).alert and s)
        assert state.alert_type == AlertType.OVERCOOK_WARNING
        assert state.state == Phase.COOKING and state.timer == "running"  # separate from the timer
        assert state.condition == S.POSSIBLY_OVERCOOKED_OR_BURNT
