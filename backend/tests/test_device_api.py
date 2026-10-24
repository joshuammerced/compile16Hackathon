import asyncio
import io
import wave

from app.audio.wav import encode_wav
from app.schemas import Phase

import numpy as np


async def test_initial_state_and_unchanged_poll(app_client):
    _, client = app_client
    r = await client.get("/device/state")
    assert r.status_code == 200
    s = r.json()
    assert s["state"] == "IDLE" and s["alert"] is False
    assert set(s) >= {"version", "state", "display_text", "remaining_s", "total_s", "audio_id", "alert"}

    r = await client.get("/device/state", params={"since_version": s["version"]})
    assert r.json() == {"changed": False, "version": s["version"], "remaining_s": 0}


async def test_analyze_starts_timer_with_audio(app_client):
    app, client = app_client
    # No camera in tests: a transcript naming the food drives the mock.
    r = await client.post("/control/analyze", json={"transcript": "chicken in a pan"})
    s = r.json()
    assert s["state"] == "COOKING"
    assert s["total_s"] > 0 and 0 < s["remaining_s"] <= s["total_s"]
    assert s["audio_id"]

    r = await client.get(f"/device/audio/{s['audio_id']}.wav")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert int(r.headers["content-length"]) == len(r.content)
    w = wave.open(io.BytesIO(r.content))
    assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)


async def test_buttons(app_client):
    _, client = app_client
    await client.post("/control/timer", json={"action": "start", "seconds": 300, "label": "Pasta"})
    s = (await client.post("/device/button", json={"button": "plus_1m"})).json()
    assert s["total_s"] == 360
    s = (await client.post("/device/button", json={"button": "minus_1m"})).json()
    assert s["total_s"] == 300
    s = (await client.post("/device/button", json={"button": "pause"})).json()
    assert s["paused"] is True
    s = (await client.post("/device/button", json={"button": "resume"})).json()
    assert s["paused"] is False
    s = (await client.post("/device/button", json={"button": "cancel"})).json()
    assert s["state"] == "IDLE"
    r = await client.post("/device/button", json={"button": "explode"})
    assert r.status_code == 422


async def test_timer_expiry_done_alert_ack(app_client):
    app, client = app_client
    store = app.state.orch.store
    await client.post("/control/timer", json={"action": "start", "seconds": 1, "label": "Egg"})
    with store.mutate() as s:
        s.timer.started_at -= 5  # pretend time passed
    for _ in range(30):
        await asyncio.sleep(0.1)
        s = (await client.get("/device/state")).json()
        if s["state"] == "DONE" and s["audio_id"]:
            break
    assert s["state"] == "DONE" and s["alert"] is True and s["display_text"] == "DONE"
    assert (await client.get(f"/device/audio/{s['audio_id']}.wav")).status_code == 200

    s = (await client.post("/device/button", json={"button": "ack"})).json()
    assert s["state"] == "IDLE" and s["alert"] is False


async def test_voice_with_transcript_header_adjusts_timer(app_client):
    _, client = app_client
    await client.post("/control/timer", json={"action": "start", "seconds": 300})
    wav = encode_wav(np.zeros(1600, dtype=np.float32))
    r = await client.post(
        "/device/voice",
        content=wav,
        headers={"Content-Type": "audio/wav", "X-Transcript": "add two minutes"},
    )
    s = r.json()
    assert s["state"] == "COOKING" and s["total_s"] == 420


async def test_voice_cancel_without_frame(app_client):
    _, client = app_client
    await client.post("/control/timer", json={"action": "start", "seconds": 300})
    r = await client.post("/device/voice", content=b"\x00\x00" * 800, headers={"X-Transcript": "cancel"})
    assert r.json()["state"] == "IDLE"


async def test_voice_empty_transcript_is_harmless(app_client):
    _, client = app_client
    # Mock STT returns "" by default.
    r = await client.post("/device/voice", content=b"\x00\x00" * 800, headers={"Content-Type": "audio/wav"})
    assert r.status_code == 200 and r.json()["state"] == "IDLE"


async def test_confirmation_flow_with_gestures(app_client):
    app, client = app_client
    orch = app.state.orch
    orch.auto_start_timer = False
    await client.post("/control/analyze", json={"transcript": "egg in a pan"})
    assert orch.snapshot().state == Phase.AWAITING_CONFIRMATION
    s = (await client.post("/control/gesture/Thumb_Up")).json()
    assert s["state"] == "COOKING"

    await client.post("/control/gesture/Open_Palm")
    await client.post("/control/analyze", json={"transcript": "pasta boiling"})
    s = (await client.post("/control/gesture/Thumb_Down")).json()
    assert s["state"] == "IDLE"


async def test_analyze_ignored_while_in_flight(app_client):
    app, client = app_client
    orch = app.state.orch
    calls = 0
    real = orch.router.provider.generate

    async def slow(req):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.3)
        return await real(req)

    orch.router.provider.generate = slow
    await asyncio.gather(orch.analyze("chicken"), orch.analyze("chicken"), orch.handle_gesture("Thumb_Up"))
    assert calls == 1


async def test_health_and_debug(app_client):
    _, client = app_client
    h = (await client.get("/health")).json()
    assert h["providers"] == {"llm": "mock", "llm_model": None, "stt": "mock", "tts": "beep"}
    assert h["camera"]["connected"] is False
    await client.post("/control/analyze", json={"transcript": "chicken"})
    d = (await client.get("/debug/last_decision")).json()
    assert d["result"]["decision"]["action"] == "start_timer"
    assert (await client.get("/")).status_code == 200
    assert (await client.get("/frame.jpg")).status_code == 503
