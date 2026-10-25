"""Kokoro-82M text-to-speech service (mlx-audio). Returns ESP32-ready WAV (16 kHz mono 16-bit).

    python services/kokoro_service.py          # :8103
    POST /v1/speak  JSON {"text": "...", "voice": "af_heart", "speed": 0.9}
"""

import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common import ModelWorker, add_common_routes, configure_logging, log


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KOKORO_")
    host: str = "0.0.0.0"
    port: int = 8103
    model: str = "mlx-community/Kokoro-82M-bf16"
    max_queue: int = 8
    timeout_s: float = 60.0


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice: str = Field("af_heart", pattern=r"^[a-z]{2}_[a-z]+$")
    speed: float = Field(0.9, ge=0.5, le=2.0)


def create_app(settings: Settings | None = None, tts=None, speak_fn=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal tts, speak_fn
        if speak_fn is None:
            from voice_pipeline import kokoro_wav, load_tts

            t0 = time.perf_counter()
            tts = load_tts(settings.model)
            speak_fn = kokoro_wav
            # Warm up: the first call builds the G2P pipeline and fetches the voice file.
            warm = speak_fn(tts, "Ready.")
            log.info("kokoro: loaded %s in %.1fs (%d bytes warm-up)", settings.model, time.perf_counter() - t0, len(warm))
        app.state.model_name = settings.model if tts is not None else "mock"
        app.state.worker = ModelWorker("kokoro", 1, settings.max_queue, settings.timeout_s)
        yield
        app.state.worker.shutdown()

    app = FastAPI(title="kokoro-service", lifespan=lifespan)
    add_common_routes(app, "kokoro")

    @app.post("/v1/speak")
    async def speak(req: SpeakRequest):
        t0 = time.perf_counter()
        wav = await app.state.worker.run(speak_fn, tts, req.text, req.voice, req.speed)
        return Response(wav, media_type="audio/wav", headers={"X-Seconds": f"{time.perf_counter() - t0:.2f}"})

    return app


if __name__ == "__main__":
    configure_logging()
    s = Settings()
    uvicorn.run(create_app(s), host=s.host, port=s.port, workers=1)
