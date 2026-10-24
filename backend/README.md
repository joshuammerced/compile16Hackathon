# Cooking Helper Backend

Python backend for the hands-free cooking helper. It watches the ESP32-CAM stream. When you
show it food and give a thumbs-up (or speak to the ESP32-S3-BOX-3), it asks a vision LLM what
the food is and how long it needs, then sends a countdown timer and a spoken message to the BOX.

```
 ESP32-CAM ──MJPEG :81/stream──► StreamReader thread (OpenCV, latest frame only)
                                      │                      │
                                      ▼                      ▼
                             GestureWorker thread      get_snapshot()
                             (MediaPipe, 10 fps,             │
                              5-frame hold, 3 s cooldown)    │
                                      │ loop.call_soon_threadsafe
                                      ▼                      │
 ESP32-S3-BOX-3 ──HTTP──►   Orchestrator (asyncio) ◄─────────┘
   mic WAV  ─► /device/voice ─► STT (faster-whisper)
   buttons  ─► /device/button        │
   poll     ◄─ /device/state         ├─► LLMRouter ─► Claude (vision, JSON schema) / mock
   speaker  ◄─ /device/audio/{id}    │        └─► validation, retry once, confidence gate,
                                     │            food_safety guardrail (USDA temps, clamps)
                                     ├─► StateStore: IDLE → ANALYZING → (AWAITING_CONFIRMATION)
                                     │                → COOKING → DONE → IDLE, plus ERROR
                                     └─► TTS (edge-tts / beep) ─► 16 kHz mono 16-bit WAV cache
```

## Setup

```bash
cd backend
cp .env.example .env        # set CAM_STREAM_URL, ANTHROPIC_API_KEY
./run.sh                    # creates .venv, installs, downloads the gesture model, starts :8000
./run.sh --mock             # no hardware, no keys: webcam + mock LLM/STT + beep TTS
```

`run.sh` works on Linux/macOS and in Git Bash on Windows. Use Python 3.11 or 3.12, since MediaPipe
wheels lag behind new Python releases. To do it by hand:

```bash
python3.11 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_models.py                         # -> models/gesture_recognizer.task
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/`. The demo page shows the annotated camera stream, the timer,
and buttons for analyze, cancel, gestures and typed "voice" commands, and it plays each new clip.

### Running without hardware

| Want | Do |
|---|---|
| Fake ESP32-CAM | `python scripts/fake_camera.py --source food.mp4` (or a `.jpg`, or `0` for the webcam). Serves `/stream` and `/capture` on `:81` (`--port 8081` if 81 needs admin), then set `CAM_STREAM_URL=http://localhost:81/stream`, `CAM_CAPTURE_URL=http://localhost:81/capture` |
| Laptop webcam directly | `CAM_SOURCE=webcam` |
| A still image | `CAM_SOURCE=file:../LLM/spanish-fried-egg.webp` |
| No LLM key | `LLM_PROVIDER=mock`. Anthropic also falls back to mock automatically when `ANTHROPIC_API_KEY` is empty |
| No mic / STT | `STT_PROVIDER=mock`, and send the `X-Transcript` header to `/device/voice` |
| Offline TTS | `TTS_PROVIDER=beep` |

### Finding the camera IP

1. Flash `hardware/ESP32_CAM_COOKING_TRACKER` with your hotspot's SSID and password.
2. Open the Arduino serial monitor at 115200 baud and press the camera's RST button.
3. Under `===== NETWORK INFORMATION =====` it prints `ESP32 IP Address: <ip>` and then
   `Live Video Stream: http://<ip>:81/stream`.
4. Put that IP in `.env`: `CAM_STREAM_URL=http://<ip>:81/stream`, `CAM_CAPTURE_URL=http://<ip>/capture`.
5. Check it with `curl http://localhost:8000/health`: `camera.connected` should be `true` and `fps` above 0.

If the serial monitor isn't handy, the hotspot's connected-devices list shows it as `esp32-...`.
The laptop must be on the same hotspot. The backend binds `0.0.0.0` so the BOX can reach it at
`http://<laptop-ip>:8000`.

### Tests

```bash
cd backend && .venv/bin/python -m pytest      # Windows: .venv\Scripts\python -m pytest
```

The tests run in full mock mode and need no camera, network or keys.

## Gestures

Recognized with MediaPipe's canned gestures. A gesture fires after it is seen in
`GESTURE_HOLD_FRAMES` (5) consecutive frames with score ≥ `GESTURE_MIN_SCORE` (0.6). Firing is
then blocked for `GESTURE_COOLDOWN_S` (3 s), and the gesture must be held again before it can fire again.

| Gesture | In state | Effect |
|---|---|---|
| 👍 `Thumb_Up` | IDLE / COOKING / ERROR | **ANALYZE**: identify the food in frame and start (or replace) the timer |
| 👍 `Thumb_Up` | AWAITING_CONFIRMATION | **CONFIRM**: start the suggested timer |
| 👍 `Thumb_Up` | DONE | acknowledge the alert |
| ✋ `Open_Palm` | COOKING / AWAITING_CONFIRMATION | **CANCEL** the timer or suggestion |
| ✋ `Open_Palm` | DONE | acknowledge the alert |
| 👎 `Thumb_Down` | AWAITING_CONFIRMATION | **REJECT** the suggestion |

Gestures can be simulated with `POST /control/gesture/{Thumb_Up|Open_Palm|Thumb_Down}`.

## Device API (ESP32-S3-BOX-3)

Plain HTTP/1.1 and small JSON, with no auth (local hotspot only). Every response has a
`Content-Length` and none are chunked. Base URL: `http://<laptop-ip>:8000`.

### State object

Returned by `GET /device/state`, `POST /device/voice` and `POST /device/button`:

```json
{"version": 12, "state": "COOKING", "display_text": "Chicken breast", "remaining_s": 734,
 "total_s": 900, "audio_id": "a7f3c1", "alert": false, "paused": false, "busy": false}
```

| Field | Type | Meaning |
|---|---|---|
| `version` | int | Increases on every state change. Store it and send it back as `since_version`. |
| `state` | string | `IDLE`, `ANALYZING`, `AWAITING_CONFIRMATION`, `COOKING`, `DONE`, `ERROR` |
| `display_text` | string ≤ 24 ASCII chars | Big text for the screen (food name, `"Thinking..."`, `"DONE"`, a question) |
| `remaining_s` | int | Seconds left (rounded up). 0 unless COOKING/DONE. Computed from server time on every read. |
| `total_s` | int | Timer length including adjustments (for a progress ring) |
| `audio_id` | string or null | Id of the latest spoken clip. **When it changes, fetch and play it.** |
| `alert` | bool | `true` from the moment the timer hits zero until acknowledged: flash and ring |
| `paused` | bool | Timer paused |
| `busy` | bool | The LLM is working (show a spinner; can be true while COOKING) |

### `GET /device/state?since_version=<int>`

Poll about every 500 ms. If nothing changed since `since_version`, the server returns a tiny body
that still carries `remaining_s`, so the countdown stays in sync:

```json
{"changed": false, "version": 12, "remaining_s": 733}
```

Otherwise it returns the full state object. Leave out `since_version` to always get the full object.
The firmware can tick the display locally between polls and resync from `remaining_s`.

### `GET /device/audio/{audio_id}.wav`

Returns `audio/wav`: RIFF/WAVE, PCM, **16 000 Hz, 16-bit signed little-endian, mono**, with a
standard 44-byte header. `Content-Length` is set. The server keeps the last 10 clips, and an expired
id returns `404`. The DONE alert clip is a three-note chime followed by speech.

### `POST /device/voice`

Body: the microphone recording as `Content-Type: audio/wav` (16 kHz/16-bit/mono WAV), or headerless
16 kHz s16le PCM (`application/octet-stream`). The limit is 30 s. The server transcribes it,
routes the transcript plus the current frame through the LLM, and returns the state object
**after** the decision has been applied (so this call can take a few seconds). The new `audio_id`
holds the spoken answer.

Things you can say: "what is this?", "time this chicken", "add two minutes",
"take off 30 seconds", "cancel the timer", "how long for pasta?".

Optional header `X-Transcript: add two minutes` skips speech-to-text, which helps for testing.

```bash
curl -X POST --data-binary @hello.wav -H "Content-Type: audio/wav" http://<ip>:8000/device/voice
```

### `POST /device/button`

```json
{"button": "ack"}
```

| `button` | Effect |
|---|---|
| `ack` | Acknowledge DONE (or ERROR), then go to IDLE |
| `cancel` | Cancel the timer or pending suggestion, then go to IDLE |
| `pause` / `resume` | Pause or resume the countdown |
| `plus_1m` / `minus_1m` | Add or remove 60 s (going below zero finishes the timer) |

Returns the state object right away. A short spoken confirmation follows a moment later as a new
`audio_id`. Unknown buttons return `422`.

### `WS /ws/device` (optional)

Pushes the full state object as a text frame on every change, and once a second while
`remaining_s` ticks. Polling works fine without it.

### Suggested firmware loop

```
every 500 ms:
  GET /device/state?since_version=v
  if body.changed == false: remaining = body.remaining_s; redraw timer; continue
  v = body.version; redraw everything
  if body.audio_id != last_audio: GET /device/audio/<id>.wav → I2S; last_audio = id
  if body.alert: flash screen + loop the chime until the user taps → POST /device/button {"button":"ack"}
push-to-talk: record 16 kHz mono → POST /device/voice → handle the returned state the same way
```

## Debug / control API

| Endpoint | Purpose |
|---|---|
| `GET /` | Demo page |
| `GET /health` | Camera connected / fps / frame age, gesture worker status, provider names, state |
| `GET /frame.jpg` | Latest frame (falls back to `CAM_CAPTURE_URL`), `503` if no camera |
| `GET /debug/stream` | MJPEG with the live gesture label and state drawn on top |
| `POST /control/analyze` | `{"transcript": "..."}` optional. Same flow as a thumbs-up or voice. |
| `POST /control/gesture/{name}` | Simulate a debounced gesture |
| `POST /control/timer` | `{"action": "start", "seconds": 600, "label": "Rice"}`. Actions: `start`, `cancel`, `adjust` (`seconds` ±), `pause`, `resume` |
| `GET /debug/last_decision` | Last transcript, raw LLM output(s), validation errors, latency, final `CookingDecision` |

## How a decision is made

1. **Frame**: the latest frame if it is under 2 s old, otherwise one still from `CAM_CAPTURE_URL`.
2. **LLM** (`app/llm/`): system prompt from `prompts.py` plus the frame, transcript and current
   timer. The Claude provider uses **structured outputs** (`output_config.format` with a JSON
   schema) rather than forced tool use, because Claude Sonnet 5.5 rejects forced `tool_choice`. It also enables
   the server-side refusal fallback (`LLM_FALLBACKS`) and runs at low effort for latency.
3. **Validation**: Pydantic `CookingDecision`. If the output is invalid, the error is sent back to
   the model for **one** retry. After a second failure the router asks the user to repeat.
4. **Confidence gate**: a `start_timer` below `MIN_CONFIDENCE` (0.55) becomes `ask_clarification`.
5. **Food-safety guardrail** (`food_safety.py`, deterministic):
   - It overrides `safe_internal_temp_f` with USDA minimums: poultry 165°F, ground meat 160°F,
     whole cuts of beef/pork/lamb 145°F + 3 min rest, seafood 145°F, egg dishes 160°F, leftovers 165°F.
   - It clamps cook times to per-category bounds and logs every clamp.
   - For meat and seafood it adds a spoken "a timer is no substitute for a thermometer" reminder.
6. **Apply**: start the timer (or wait for 👍 if `AUTO_START_TIMER=false`), adjust, cancel, or just speak.
7. **TTS**: spoken feedback goes to edge-tts, is decoded with soundfile, resampled with scipy, and
   stored as a 16 kHz mono 16-bit WAV with a new `audio_id`. If TTS fails, a beep is stored instead.

The timer is server-side monotonic time (`started_at`, `duration_s`, pause bookkeeping, `ends_at`).
Remaining time is computed on read, and a 200 ms asyncio watcher moves COOKING to DONE and generates
the alert clip. Only one analysis runs at a time; gestures or voice that arrive meanwhile are
ignored. Sessions are keyed by id (`stove1` today) so more timers can be added later.

## Layout

```
app/
  main.py            FastAPI app + lifespan (starts/stops threads and tasks)
  orchestrator.py    gesture/voice/button → state transitions, TTS, timer watcher
  config.py          pydantic-settings (env / .env)
  state.py           Timer, CookingSession, StateStore (lock + version)
  schemas.py         CookingDecision, JSON schema for structured outputs, API models
  ingest/stream_reader.py
  vision/gestures.py GestureDebouncer (pure) + GestureWorker thread
  llm/               base.py (interface + router), anthropic_provider.py, mock_provider.py,
                     prompts.py, food_safety.py
  audio/             stt.py, tts.py, wav.py
  routes/            device.py, control.py, health.py
scripts/             download_models.py, fake_camera.py
tests/
```
