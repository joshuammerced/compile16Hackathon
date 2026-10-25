"""Whisper speech-to-text service (faster-whisper, CPU int8).

    python services/whisper_service.py         # :8102
    POST /v1/transcribe  multipart: audio=<WAV, any rate/channels>
"""

import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic_settings import BaseSettings, SettingsConfigDict

from common import ModelWorker, add_common_routes, configure_logging, log


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WHISPER_")
    host: str = "0.0.0.0"
    port: int = 8102
    model: str = "base.en"
    workers: int = 1  # CPU model, so >1 can transcribe in parallel (costs RAM and cores)
    max_queue: int = 8
    timeout_s: float = 60.0


def create_app(settings: Settings | None = None, stt=None, transcribe_fn=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal stt, transcribe_fn
        if transcribe_fn is None:
            from voice_pipeline import load_stt, transcribe_wav

            t0 = time.perf_counter()
            stt = load_stt(settings.model, settings.workers)
            transcribe_fn = transcribe_wav
            log.info("whisper: loaded %s in %.1fs", settings.model, time.perf_counter() - t0)
        app.state.model_name = settings.model if stt is not None else "mock"
        app.state.worker = ModelWorker("whisper", settings.workers, settings.max_queue, settings.timeout_s)
        yield
        app.state.worker.shutdown()

    app = FastAPI(title="whisper-service", lifespan=lifespan)
    add_common_routes(app, "whisper")

    @app.post("/v1/transcribe")
    async def transcribe(audio: UploadFile = File(...)):
        data = await audio.read()
        if not data:
            raise HTTPException(400, "empty audio")
        t0 = time.perf_counter()
        try:
            text = await app.state.worker.run(transcribe_fn, stt, data)
        except RuntimeError as e:  # soundfile.LibsndfileError: not a WAV it can decode
            raise HTTPException(400, f"audio is not a readable WAV: {e}")
        return {"text": text, "seconds": round(time.perf_counter() - t0, 2)}

    return app


if __name__ == "__main__":
    configure_logging()
    s = Settings()
    uvicorn.run(create_app(s), host=s.host, port=s.port, workers=1)
