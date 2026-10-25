from __future__ import annotations

import asyncio
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.ingest.stream_reader import Frame  # noqa: E402


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


class FakeReader:
    """Stands in for StreamReader. ``auto=True`` behaves like a live camera (a new frame on
    every read); otherwise frames are pushed explicitly, with an optional age."""

    source = "fake"
    stream_url = ""
    fps = 10.0

    def __init__(self, auto: bool = True):
        self.auto = auto
        self.connected = True
        self._seq = 0
        self._frame: Frame | None = None
        self._lock = threading.Lock()

    def push(self, age_s: float = 0.0) -> Frame:
        with self._lock:
            self._seq += 1
            self._frame = Frame(np.zeros((240, 320, 3), np.uint8), time.monotonic() - age_s, self._seq)
            return self._frame

    def freeze(self) -> None:
        """Camera stops delivering: the last frame just gets older."""
        self.auto = False

    def latest(self) -> Frame | None:
        if self.auto and self.connected:
            return self.push()
        return self._frame

    def get_snapshot(self, max_age_s: float = 2.0) -> Frame | None:
        return self.latest()

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def last_frame_age_s(self) -> float | None:
        return None if self._frame is None else time.monotonic() - self._frame.timestamp

    def start(self) -> None: ...

    def stop(self, timeout: float = 0) -> None: ...


def mock_settings(**overrides) -> Settings:
    base = dict(
        cam_source="none",
        cam_capture_url="",
        vision_provider="mock",
        stt_provider="mock",
        tts_provider="beep",
        gesture_enabled=False,
        auto_start_timer=False,
        state_file=None,
        monitor_interval_s=0.05,
        camera_lost_after_s=0.3,
        log_level="WARNING",
    )
    base.update(overrides)
    # _env_file=None: tests must not pick up a developer's backend/.env.
    return Settings(_env_file=None, **base)


@asynccontextmanager
async def running_app(settings: Settings | None = None, reader=None, vision=None):
    """An app with its lifespan running, plus an httpx.AsyncClient."""
    from app.main import create_app

    app = create_app(settings or mock_settings(), reader=reader, vision=vision)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # The vision worker loads on its own thread; wait until it is usable.
            await wait_for(lambda: app.state.orch.monitor.status != "loading" and app.state.orch.monitor.status != "not started")
            yield app, client


async def wait_for(predicate, timeout: float = 3.0, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return result
        await asyncio.sleep(interval)
    raise AssertionError("condition not met in time")


@pytest.fixture
async def app_client():
    """Full mock mode with a live fake camera."""
    async with running_app(reader=FakeReader(auto=True)) as pair:
        yield pair
