"""Gateway the ESP32 talks to. Holds no models: whisper -> qwen -> kokoro over HTTP.

    python services/gateway.py                 # :8100
    POST /v1/respond       multipart: image=<jpeg>, audio=<WAV> or said=<text>  -> audio/wav
                           headers X-Transcript / X-Reply (URL-encoded)
    POST /v1/respond.json  same input -> {"transcript", "reply", "audio_b64", "timings"}

Each request is its own coroutine, so while one request is in Qwen the next can already be
in Whisper; the per-model queues do the rest.
"""

import base64
import time
from contextlib import asynccontextmanager
from urllib.parse import quote

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic_settings import BaseSettings, SettingsConfigDict

from common import configure_logging, log


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_")
    host: str = "0.0.0.0"
    port: int = 8100
    whisper_url: str = "http://127.0.0.1:8102"
    qwen_url: str = "http://127.0.0.1:8101"
    kokoro_url: str = "http://127.0.0.1:8103"
    voice: str = "af_heart"
    speed: float = 0.9
    # Generous: a call may sit in that service's queue behind other requests.
    timeout_s: float = 300.0


class UpstreamError(Exception):
    def __init__(self, status: int, message: str, retry_after: str | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


async def call(client: httpx.AsyncClient, service: str, url: str, **kwargs) -> httpx.Response:
    """POST to a service, turning every failure into an UpstreamError that names the service."""
    try:
        r = await client.post(url, **kwargs)
    except httpx.TimeoutException:
        raise UpstreamError(504, f"{service} timed out") from None
    except httpx.TransportError as e:
        raise UpstreamError(502, f"{service} unreachable ({type(e).__name__})") from None
    if r.status_code == 503:
        raise UpstreamError(503, f"{service} busy, retry later", r.headers.get("Retry-After", "5"))
    if r.status_code == 504:
        raise UpstreamError(504, f"{service} timed out")
    if 400 <= r.status_code < 500:
        try:
            detail = r.json().get("detail") or r.json().get("error")
        except ValueError:
            detail = r.text[:200]
        raise UpstreamError(400, f"{service} rejected the request: {detail}")
    if r.status_code >= 500:
        raise UpstreamError(502, f"{service} failed (HTTP {r.status_code})")
    return r


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """transport is injectable so tests can fake the three services."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(timeout=settings.timeout_s, transport=transport)
        yield
        await app.state.client.aclose()

    app = FastAPI(title="cooking-gateway", lifespan=lifespan)

    @app.exception_handler(UpstreamError)
    async def upstream_error(_: Request, e: UpstreamError):
        headers = {"Retry-After": e.retry_after} if e.retry_after else None
        return JSONResponse({"error": str(e)}, status_code=e.status, headers=headers)

    async def run_pipeline(image: UploadFile, audio: UploadFile | None, said: str) -> dict:
        client: httpx.AsyncClient = app.state.client
        timings = {}
        frame = await image.read()
        if not frame:
            raise HTTPException(400, "empty image")
        if audio is not None:
            t0 = time.perf_counter()
            r = await call(
                client, "whisper", f"{settings.whisper_url}/v1/transcribe",
                files={"audio": ("audio.wav", await audio.read(), "audio/wav")},
            )
            said = r.json()["text"]
            timings["whisper"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        r = await call(
            client, "qwen", f"{settings.qwen_url}/v1/think",
            files={"image": (image.filename or "frame.jpg", frame, image.content_type or "image/jpeg")},
            data={"said": said},
        )
        reply = r.json()["reply"]
        timings["qwen"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        r = await call(
            client, "kokoro", f"{settings.kokoro_url}/v1/speak",
            json={"text": reply, "voice": settings.voice, "speed": settings.speed},
        )
        timings["kokoro"] = round(time.perf_counter() - t0, 2)
        log.info("respond: said=%r reply=%r timings=%s", said, reply, timings)
        return {"transcript": said, "reply": reply, "wav": r.content, "timings": timings}

    @app.post("/v1/respond")
    async def respond(image: UploadFile = File(...), audio: UploadFile | None = File(None), said: str = Form("")):
        out = await run_pipeline(image, audio, said)
        return Response(
            out["wav"],
            media_type="audio/wav",
            headers={
                "X-Transcript": quote(out["transcript"]),
                "X-Reply": quote(out["reply"]),
                "X-Timings": ",".join(f"{k}={v}" for k, v in out["timings"].items()),
            },
        )

    @app.post("/v1/respond.json")
    async def respond_json(image: UploadFile = File(...), audio: UploadFile | None = File(None), said: str = Form("")):
        out = await run_pipeline(image, audio, said)
        return {
            "transcript": out["transcript"],
            "reply": out["reply"],
            "audio_b64": base64.b64encode(out["wav"]).decode(),
            "timings": out["timings"],
        }

    @app.get("/health")
    async def health():
        return {"service": "gateway", "status": "up"}

    @app.get("/ready")
    async def ready():
        services = {}
        for name, url in (("whisper", settings.whisper_url), ("qwen", settings.qwen_url), ("kokoro", settings.kokoro_url)):
            try:
                r = await app.state.client.get(f"{url}/ready", timeout=3)
                services[name] = r.json()
            except (httpx.HTTPError, ValueError) as e:
                services[name] = {"ready": False, "error": type(e).__name__}
        ok = all(s.get("ready") for s in services.values())
        return JSONResponse({"ready": ok, "services": services}, status_code=200 if ok else 503)

    return app


if __name__ == "__main__":
    configure_logging()
    s = Settings()
    uvicorn.run(create_app(s), host=s.host, port=s.port, workers=1)
