"""FastAPI app. The lifespan starts and stops the camera, gesture, vision and timer workers."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .audio.stt import create_stt
from .audio.tts import AudioClipStore, Speaker, create_tts
from .config import Settings, get_settings
from .controller import Controller, Rules
from .ingest.stream_reader import StreamReader
from .llm import create_vision_provider
from .llm.base import VisionProvider
from .orchestrator import Orchestrator
from .routes import control, device, health
from .state import SessionPersistence, StateStore
from .vision.gestures import GestureWorker
from .vision.monitor import VisionMonitor


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_orchestrator(
    settings: Settings,
    reader=None,
    vision: VisionProvider | None = None,
) -> Orchestrator:
    """Wire the components. ``reader`` and ``vision`` can be injected (tests)."""
    if reader is None:
        reader = StreamReader(settings.cam_source, settings.cam_stream_url, settings.cam_capture_url)
    gestures = None
    if settings.gesture_enabled and settings.cam_source != "none":
        gestures = GestureWorker(
            reader,
            settings.gesture_model_path,
            fps=settings.gesture_fps,
            hold_frames=settings.gesture_hold_frames,
            min_score=settings.gesture_min_score,
            cooldown_s=settings.gesture_cooldown_s,
            input_width=settings.gesture_input_width,
        )
    store = StateStore(persistence=SessionPersistence(settings.state_file) if settings.state_file else None)
    monitor = VisionMonitor(
        reader,
        vision or create_vision_provider(settings),
        interval_s=settings.monitor_interval_s,
        frame_max_age_s=settings.frame_max_age_s,
        result_max_age_s=settings.result_max_age_s,
        clock=store.clock,
    )
    rules = Rules(
        overcook_confirm=settings.overcook_confirm,
        overcook_repeat_s=settings.overcook_repeat_s,
        looks_done_confirm=settings.looks_done_confirm,
        camera_lost_after_s=settings.camera_lost_after_s,
        auto_start_timer=settings.auto_start_timer,
    )
    return Orchestrator(
        store=store,
        reader=reader,
        monitor=monitor,
        controller=Controller(rules, store.clock),
        stt=create_stt(settings.stt_provider, settings.whisper_model, settings.mock_transcript),
        speaker=Speaker(create_tts(settings.tts_provider, settings.tts_voice), AudioClipStore(10)),
        gestures=gestures,
    )


def create_app(settings: Settings | None = None, reader=None, vision: VisionProvider | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    log = logging.getLogger("app")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        orch = build_orchestrator(settings, reader=reader, vision=vision)
        app.state.orch = orch
        app.state.settings = settings
        log.info(
            "starting: camera=%s vision=%s stt=%s tts=%s auto_start=%s state=%s",
            settings.cam_source if settings.cam_source != "stream" else settings.cam_stream_url,
            orch.monitor.provider.name,
            orch.stt.name,
            orch.speaker.provider.name,
            settings.auto_start_timer,
            settings.state_file or "not persisted",
        )
        await orch.start()
        try:
            yield
        finally:
            await orch.stop()

    app = FastAPI(title="Cooking Helper Backend", version="2.0.0", lifespan=lifespan)
    app.include_router(device.router)
    app.include_router(control.router)
    app.include_router(health.router)
    return app


app = create_app()
