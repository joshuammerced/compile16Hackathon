"""Qwen3-VL-4B service: camera frame + what the cook said -> short spoken-style reply.

    python services/qwen_service.py            # :8101
    POST /v1/think  multipart: image=<jpeg/png>, said=<text, optional>
"""

import io
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic_settings import BaseSettings, SettingsConfigDict

from common import ModelWorker, add_common_routes, configure_logging, log


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QWEN_")
    host: str = "0.0.0.0"
    port: int = 8101
    model: str = ""  # empty = MODEL_ID from inference.py
    max_tokens: int = 120
    max_queue: int = 8
    timeout_s: float = 120.0


def create_app(settings: Settings | None = None, vlm=None, reply_fn=None) -> FastAPI:
    """vlm / reply_fn are injectable so tests can run without the real model."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal vlm, reply_fn
        if reply_fn is None:
            from inference import MODEL_ID, CookingVLM
            from voice_pipeline import qwen_reply

            t0 = time.perf_counter()
            vlm = CookingVLM(settings.model or MODEL_ID)
            reply_fn = qwen_reply
            log.info("qwen: loaded %s in %.1fs", vlm.model_id, time.perf_counter() - t0)
        app.state.model_name = getattr(vlm, "model_id", "mock")
        app.state.worker = ModelWorker("qwen", 1, settings.max_queue, settings.timeout_s)
        yield
        app.state.worker.shutdown()

    app = FastAPI(title="qwen-service", lifespan=lifespan)
    add_common_routes(app, "qwen")

    @app.post("/v1/think")
    async def think(image: UploadFile = File(...), said: str = Form("")):
        try:
            frame = Image.open(io.BytesIO(await image.read()))
            frame.load()
        except (UnidentifiedImageError, OSError) as e:
            raise HTTPException(400, f"image is not readable: {e}")
        t0 = time.perf_counter()
        result = await app.state.worker.run(reply_fn, vlm, frame, said.strip(), settings.max_tokens)
        return {
            "reply": result.text,
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "seconds": round(time.perf_counter() - t0, 2),
        }

    return app


if __name__ == "__main__":
    configure_logging()
    s = Settings()
    uvicorn.run(create_app(s), host=s.host, port=s.port, workers=1)
