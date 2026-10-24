"""Endpoints the ESP32-S3-BOX-3 calls. Small JSON, no chunked responses."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect

from ..orchestrator import Orchestrator
from ..schemas import AckRequest, ButtonRequest, DeviceState, DeviceStateUnchanged

router = APIRouter(tags=["device"])

MAX_VOICE_BYTES = 16000 * 2 * 30  # 30 s of 16 kHz 16-bit mono


def orch(request: Request) -> Orchestrator:
    return request.app.state.orch


@router.get("/device/state", response_model=DeviceState | DeviceStateUnchanged)
async def device_state(request: Request, since_version: int | None = None):
    """Poll every ~500 ms. ``version`` changes only when something the device shows changes;
    the countdown is carried separately in ``remaining_ms`` on every response (changed or not),
    so re-anchor the local countdown from it each time."""
    state = orch(request).snapshot()
    if since_version is not None and since_version == state.version:
        return DeviceStateUnchanged(
            version=state.version, timer=state.timer, remaining_ms=state.remaining_ms, remaining_s=state.remaining_s
        )
    return state


@router.get("/device/audio/{audio_id}.wav")
async def device_audio(request: Request, audio_id: str):
    wav = orch(request).speaker.store.get(audio_id)
    if wav is None:
        raise HTTPException(status_code=404, detail="audio clip expired or unknown")
    # A plain Response sets Content-Length and is never chunked.
    return Response(content=wav, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@router.post("/device/voice", response_model=DeviceState)
async def device_voice(request: Request):
    """Body: WAV bytes (16 kHz, 16-bit, mono), or headerless s16le PCM at 16 kHz.
    Optional header X-Transcript skips speech-to-text (handy for testing)."""
    audio = await request.body()
    transcript = request.headers.get("x-transcript")
    if not audio and transcript is None:
        raise HTTPException(status_code=400, detail="empty audio body")
    if len(audio) > MAX_VOICE_BYTES:
        raise HTTPException(status_code=413, detail="audio longer than 30 s")
    return await orch(request).voice(audio, transcript)


@router.post("/device/button", response_model=DeviceState)
async def device_button(request: Request, body: ButtonRequest):
    return await orch(request).button(body.button)


@router.post("/device/ack", response_model=DeviceState)
async def device_ack(request: Request, body: AckRequest):
    """Acknowledge an alert by id. Idempotent: acking an already-acked id returns 200."""
    state, known = await orch(request).ack(body.alert_id)
    if not known:
        raise HTTPException(status_code=404, detail=f"unknown alert id {body.alert_id}")
    return state


@router.get("/device/alerts")
async def device_alerts(request: Request):
    """All unacknowledged alerts, most important first."""
    with orch(request).store.read() as s:
        alerts = sorted(s.alerts, key=lambda a: (-a.severity, a.id))
        return {"alerts": [a.model_dump(mode="json", include={"id", "type", "text", "message", "audio_id"}) for a in alerts]}


@router.websocket("/ws/device")
async def device_ws(websocket: WebSocket):
    """Pushes the full state JSON whenever the version changes (and once per second while a
    timer runs, so remaining_ms stays fresh). Polling works without this."""
    await websocket.accept()
    o: Orchestrator = websocket.app.state.orch
    last_version = -1
    last_remaining = -1
    try:
        while True:
            state = o.snapshot()
            if state.version != last_version or state.remaining_s != last_remaining:
                await websocket.send_text(state.model_dump_json())
                last_version, last_remaining = state.version, state.remaining_s
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError):
        return
