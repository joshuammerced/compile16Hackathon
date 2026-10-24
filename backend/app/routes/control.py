"""Debug / manual-control endpoints and the demo page at /."""

from __future__ import annotations

import asyncio

import cv2
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from ..orchestrator import Orchestrator
from ..schemas import AnalyzeRequest, DeviceState, TimerControlRequest

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
async def control_analyze(request: Request, body: AnalyzeRequest | None = None):
    transcript = body.transcript if body else None
    return await orch(request).analyze(transcript or None, source="control")


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


@router.get("/debug/last_decision")
async def last_decision(request: Request):
    o = orch(request)
    if o.last_result is None:
        return {"transcript": None, "result": None}
    return {"transcript": o.last_transcript, "result": o.last_result.as_debug()}


INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cooking Helper</title>
<style>
 :root{--bg:#111;--fg:#eee;--muted:#999;--card:#1d1d1d;--accent:#ff8a3d}
 body{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg)}
 main{max-width:960px;margin:0 auto;padding:16px;display:grid;gap:16px;grid-template-columns:1fr}
 @media(min-width:800px){main{grid-template-columns:3fr 2fr}}
 .card{background:var(--card);border-radius:12px;padding:16px}
 img{width:100%;border-radius:8px;background:#000;min-height:180px}
 #timer{font-size:72px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1}
 #label{font-size:22px;margin:8px 0}
 #phase{color:var(--muted);letter-spacing:.08em}
 .alert #timer{color:var(--accent);animation:blink 1s infinite}
 @keyframes blink{50%{opacity:.3}}
 .row{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
 button{background:#333;color:var(--fg);border:0;border-radius:8px;padding:10px 14px;font-size:15px;cursor:pointer}
 button.primary{background:var(--accent);color:#111}
 input{flex:1;min-width:0;padding:10px;border-radius:8px;border:0;background:#2a2a2a;color:var(--fg)}
 pre{white-space:pre-wrap;font-size:12px;color:var(--muted);max-height:260px;overflow:auto}
</style></head><body><main>
<section class="card"><img src="/debug/stream" alt="camera stream with gesture overlay">
<pre id="health"></pre></section>
<section class="card" id="panel">
 <div id="phase">-</div><div id="timer">0:00</div><div id="label"></div>
 <div class="row"><button class="primary" onclick="post('/control/analyze')">Analyze (thumbs-up)</button>
  <button onclick="btn('cancel')">Cancel</button><button onclick="btn('ack')">Ack</button></div>
 <div class="row"><button onclick="btn('pause')">Pause</button><button onclick="btn('resume')">Resume</button>
  <button onclick="btn('plus_1m')">+1 min</button><button onclick="btn('minus_1m')">-1 min</button></div>
 <div class="row"><button onclick="post('/control/gesture/Thumb_Up')">Thumb_Up</button>
  <button onclick="post('/control/gesture/Thumb_Down')">Thumb_Down</button>
  <button onclick="post('/control/gesture/Open_Palm')">Open_Palm</button></div>
 <div class="row"><input id="say" placeholder="Type what you'd say, e.g. add two minutes">
  <button onclick="say()">Send</button></div>
 <div class="row"><label><input type="checkbox" id="sound" checked style="flex:none"> play audio</label></div>
 <pre id="decision"></pre>
</section></main>
<script>
let lastAudio=null, version=null;
const fmt=s=>Math.floor(s/60)+':'+String(s%60).padStart(2,'0');
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});render(await r.json());loadDecision();}
const btn=b=>post('/device/button',{button:b});
const say=()=>{const t=document.getElementById('say').value;if(t)post('/control/analyze',{transcript:t});};
function render(s){if(!s||s.version===undefined||s.changed===false){if(s&&s.remaining_s!==undefined)document.getElementById('timer').textContent=fmt(s.remaining_s);return;}
 version=s.version;document.getElementById('phase').textContent=s.state+(s.paused?' (paused)':'')+(s.busy?' (thinking)':'');
 document.getElementById('timer').textContent=s.state==='DONE'?'DONE':fmt(s.remaining_s);
 document.getElementById('label').textContent=s.display_text;
 document.getElementById('panel').classList.toggle('alert',s.alert);
 if(s.audio_id&&s.audio_id!==lastAudio){if(lastAudio!==null&&document.getElementById('sound').checked)new Audio('/device/audio/'+s.audio_id+'.wav').play().catch(()=>{});lastAudio=s.audio_id;}}
async function poll(){try{const q=version===null?'':'?since_version='+version;render(await (await fetch('/device/state'+q)).json());}catch(e){}setTimeout(poll,500);}
async function loadDecision(){const d=await (await fetch('/debug/last_decision')).json();document.getElementById('decision').textContent=JSON.stringify(d,null,1);}
async function health(){try{const h=await (await fetch('/health')).json();document.getElementById('health').textContent=JSON.stringify({camera:h.camera,gestures:h.gestures,providers:h.providers},null,1);}catch(e){}setTimeout(health,3000);}
poll();health();loadDecision();
</script></body></html>"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return INDEX_HTML
