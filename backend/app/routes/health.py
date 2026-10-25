from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    o = request.app.state.orch
    age = o.reader.last_frame_age_s
    stats = asdict(o.monitor.stats)
    stats.pop("rejections")
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
        "vision": {
            "provider": o.monitor.provider.name,
            "model": getattr(o.monitor.provider, "model_id", None),
            "status": o.monitor.status,
            "busy": o.monitor.worker.busy,
            "interval_s": o.monitor.interval_s,
            **stats,
        },
        "providers": {"stt": o.stt.name, "tts": o.speaker.provider.name},
        "state": o.snapshot().model_dump(),
    }
