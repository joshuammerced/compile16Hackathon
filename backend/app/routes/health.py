from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    o = request.app.state.orch
    age = o.reader.last_frame_age_s
    return {
        "ok": True,
        "camera": {
            "source": o.reader.source,
            "url": o.reader.stream_url if o.reader.source == "stream" else None,
            "connected": o.reader.is_connected,
            "fps": round(o.reader.fps, 1),
            "last_frame_age_s": None if age is None else round(age, 2),
        },
        "gestures": {
            "alive": bool(o.gestures and o.gestures.alive),
            "status": o.gestures.status if o.gestures else "disabled",
        },
        "providers": {
            "llm": o.router.provider.name,
            "llm_model": getattr(o.router.provider, "model", None),
            "stt": o.stt.name,
            "tts": o.speaker.provider.name,
        },
        "state": o.snapshot().model_dump(),
    }
