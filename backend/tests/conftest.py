from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402


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


def mock_settings(**overrides) -> Settings:
    base = dict(
        cam_source="none",
        cam_capture_url="",
        llm_provider="mock",
        stt_provider="mock",
        tts_provider="beep",
        gesture_enabled=False,
        auto_start_timer=True,
        log_level="WARNING",
    )
    base.update(overrides)
    # _env_file=None: tests must not pick up a developer's backend/.env.
    return Settings(_env_file=None, **base)


@pytest.fixture
async def app_client():
    """An app in full mock mode with its lifespan running, plus an httpx.AsyncClient."""
    from app.main import create_app

    app = create_app(mock_settings())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield app, client
