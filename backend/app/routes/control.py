"""Debug / manual-control endpoints and the demo page at /."""

from __future__ import annotations

import asyncio
from dataclasses import asdict

import cv2
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from ..orchestrator import Orchestrator
from ..schemas import CommandRequest, DeviceState, TimerControlRequest

router = APIRouter(tags=["control"])


def orch(request: Request) -> Orchestrator:
    return request.app.state.orch


@router.get("/frame.jpg")
async def frame_jpg(request: Request):
    frame = await asyncio.to_thread(orch(request).reader.get_snapshot)
    if frame is None:
        raise HTTPException(status_code=503, detail="no camera frame available")
    return Response(content=await asyncio.to_thread(frame.to_jpeg), media_type="image/jpeg")


def _annotate(o: Orchestrator, image):
    image = image.copy()
    label = o.gestures.current_label if o.gestures else None
    text = f"{label[0]} {label[1]:.2f}" if label else "no gesture"
    state = o.snapshot()
    cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    status = f"{state.state.value} {state.remaining_s // 60}:{state.remaining_s % 60:02d}"
    h = image.shape[0]
    cv2.putText(image, status, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, status, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else None


@router.get("/debug/stream")
async def debug_stream(request: Request):
    """MJPEG of the latest frames with the detected gesture label drawn on top."""
    o = orch(request)

    async def frames():
        last_seq = -1
        while not await request.is_disconnected():
            frame = o.reader.latest()
            if frame is not None and frame.seq != last_seq:
                last_seq = frame.seq
                jpeg = await asyncio.to_thread(_annotate, o, frame.image)
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/control/analyze", response_model=DeviceState)
async def control_analyze(request: Request, body: CommandRequest | None = None):
    """No body: same as a thumbs-up (start watching). With a transcript: same as speaking it."""
    if body and body.transcript:
        return await orch(request).command(body.transcript)
    return await orch(request).start_session("control")


@router.post("/control/command", response_model=DeviceState)
async def control_command(request: Request, body: CommandRequest):
    """Run a typed command exactly as if it had been spoken to the BOX."""
    return await orch(request).command(body.transcript or "")


@router.post("/control/gesture/{gesture}", response_model=DeviceState)
async def control_gesture(request: Request, gesture: str):
    """Simulate a debounced gesture: Thumb_Up, Open_Palm or Thumb_Down."""
    if gesture not in ("Thumb_Up", "Open_Palm", "Thumb_Down"):
        raise HTTPException(status_code=400, detail="unknown gesture")
    return await orch(request).handle_gesture(gesture)


@router.post("/control/timer", response_model=DeviceState)
async def control_timer(request: Request, body: TimerControlRequest):
    try:
        return await orch(request).timer_control(body.action, body.seconds, body.label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/debug/vision")
async def debug_vision(request: Request, limit: int = 10):
    """Recent observations applied to the session, the raw last model output, and monitor stats."""
    o = orch(request)
    with o.store.read() as s:
        observations = [obs.model_dump(mode="json") for obs in list(s.observations)[-limit:]]
        session = {
            "run_id": s.run_id,
            "generation": s.generation,
            "phase": s.phase.value,
            "food": s.food,
            "suggestion": s.suggestion.model_dump() if s.suggestion else None,
            "alerts": [a.model_dump(mode="json") for a in s.alerts],
            "history": [(t, p.value, r) for t, p, r in s.history[-15:]],
        }
    last = o.monitor.last_observation
    return {
        "status": o.monitor.status,
        "stats": asdict(o.monitor.stats),
        "last_result": last.model_dump(mode="json") if last else None,
        "observations": observations,
        "session": session,
        "last_command": {"transcript": o.last_transcript, "intent": o.last_command.intent if o.last_command else None},
    }


@router.get("/debug/last_decision", include_in_schema=False)
async def last_decision(request: Request):
    return await debug_vision(request, limit=1)


INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cooking Helper</title>
<style>
 :root{--bg:#111;--fg:#eee;--muted:#999;--card:#1d1d1d;--accent:#ff8a3d;--warn:#ff3d3d;--ok:#3ddc84}
 body{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg)}
 main{max-width:1000px;margin:0 auto;padding:16px;display:grid;gap:16px;grid-template-columns:1fr}
 @media(min-width:820px){main{grid-template-columns:3fr 2fr}}
 .card{background:var(--card);border-radius:12px;padding:16px;min-width:0}
 img{width:100%;border-radius:8px;background:#000;min-height:180px}
 #timer{font-size:72px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1}
 #label{font-size:22px;margin:8px 0}
 #phase,#cond{color:var(--muted);letter-spacing:.06em;font-size:14px}
 #alert{display:none;background:var(--warn);color:#fff;border-radius:10px;padding:12px;margin:10px 0;font-weight:600}
 #alert.TIMER_DONE{background:var(--accent);color:#111}
 #alert.MONITORING_LOST{background:#666}
 #alert button{float:right;background:#fff;color:#111}
 .alerting #timer{animation:blink 1s infinite}
 @keyframes blink{50%{opacity:.3}}
 .row{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
 button{background:#333;color:var(--fg);border:0;border-radius:8px;padding:10px 14px;font-size:15px;cursor:pointer}
 button.primary{background:var(--accent);color:#111}
 input[type=text]{flex:1;min-width:0;padding:10px;border-radius:8px;border:0;background:#2a2a2a;color:var(--fg)}
 pre{white-space:pre-wrap;word-break:break-word;font-size:12px;color:var(--muted);max-height:300px;overflow:auto}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#555;margin-right:6px}
 .live .dot{background:var(--ok)}
</style></head><body><main>
<section class="card"><img src="/debug/stream" alt="camera stream with gesture overlay">
<pre id="health"></pre></section>
<section class="card" id="panel">
 <div id="phase">-</div>
 <div id="cond"><span class="dot"></span><span id="condtext">not watching</span></div>
 <div id="alert"><button onclick="ackAlert()">Ack</button><span id="alerttext"></span></div>
 <div id="timer">0:00</div><div id="label"></div>
 <div class="row"><button class="primary" onclick="post('/control/analyze')">Start watching (thumbs-up)</button>
  <button onclick="btn('cancel')">Stop</button></div>
 <div class="row"><button onclick="btn('pause')">Pause</button><button onclick="btn('resume')">Resume</button>
  <button onclick="btn('plus_1m')">+1 min</button><button onclick="btn('minus_1m')">-1 min</button></div>
 <div class="row"><button onclick="post('/control/gesture/Thumb_Up')">Thumb_Up</button>
  <button onclick="post('/control/gesture/Thumb_Down')">Thumb_Down</button>
  <button onclick="post('/control/gesture/Open_Palm')">Open_Palm</button></div>
 <div class="row"><input type="text" id="say" placeholder="Say something, e.g. set a timer for 3 minutes">
  <button onclick="say()">Send</button></div>
 <div class="row"><label><input type="checkbox" id="sound" checked> play audio</label></div>
 <pre id="vision"></pre>
</section></main>
<script>
let lastAudio=null,lastAlertAudio=null,version=null,alertId=null,deadline=0,running=false;
const fmt=ms=>{const s=Math.ceil(ms/1000);return Math.floor(s/60)+':'+String(s%60).padStart(2,'0');};
const $=id=>document.getElementById(id);
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});if(r.ok)render(await r.json());}
const btn=b=>post('/device/button',{button:b});
const ackAlert=()=>{if(alertId!==null)post('/device/ack',{alert_id:alertId});};
const say=()=>{const t=$('say').value;if(t){post('/control/command',{transcript:t});$('say').value='';}};
function play(id){if(id&&$('sound').checked)new Audio('/device/audio/'+id+'.wav').play().catch(()=>{});}
function sync(s){deadline=performance.now()+s.remaining_ms;running=s.timer==='running';if(!running)$('timer').textContent=s.state==='DONE'?'DONE':fmt(s.remaining_ms);}
function render(s){if(!s)return;sync(s);if(s.changed===false)return;
 version=s.version;$('phase').textContent=s.state+(s.paused?' (paused)':'')+(s.busy?' (looking...)':'');
 $('label').textContent=s.display_text;
 $('panel').classList.toggle('live',s.monitoring);
 $('condtext').textContent=s.monitoring?('watching: '+(s.condition||'...')):(s.session_id?'camera/model not available':'not watching');
 $('alert').style.display=s.alert?'block':'none';$('alert').className=s.alert_type||'';
 $('alerttext').textContent=s.alert?(s.alert_text+(s.alerts_pending>1?'  (+'+(s.alerts_pending-1)+')':'')):'';
 $('panel').classList.toggle('alerting',s.alert);
 alertId=s.alert_id;
 if(s.alert_audio_id&&s.alert_audio_id!==lastAlertAudio){if(lastAlertAudio!==null||version>0)play(s.alert_audio_id);lastAlertAudio=s.alert_audio_id;}
 if(s.audio_id&&s.audio_id!==lastAudio){if(lastAudio!==null)play(s.audio_id);lastAudio=s.audio_id;}
 loadVision();}
function tick(){if(running)$('timer').textContent=fmt(Math.max(0,deadline-performance.now()));requestAnimationFrame(tick);}
async function poll(){try{const q=version===null?'':'?since_version='+version;render(await (await fetch('/device/state'+q)).json());}catch(e){}setTimeout(poll,500);}
async function loadVision(){try{const d=await (await fetch('/debug/vision?limit=3')).json();
 $('vision').textContent=JSON.stringify({status:d.status,last:d.last_result&&{food:d.last_result.food,status:d.last_result.status,evidence:d.last_result.evidence,latency_s:d.last_result.latency_s},stats:d.stats,command:d.last_command},null,1);}catch(e){}}
async function health(){try{const h=await (await fetch('/health')).json();$('health').textContent=JSON.stringify({camera:h.camera,gestures:h.gestures,vision:{provider:h.vision.provider,status:h.vision.status,busy:h.vision.busy}},null,1);}catch(e){}setTimeout(health,3000);}
poll();health();tick();setInterval(loadVision,4000);
</script></body></html>"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return INDEX_HTML
