#!/usr/bin/env bash
# Create the venv, install deps, download models, and start the backend on 0.0.0.0:8000.
# Works on Linux/macOS and on Windows in Git Bash.
#
#   ./run.sh            # uses backend/.env if present
#   ./run.sh --mock     # no hardware, no API keys: webcam + mock LLM/STT + beep TTS
set -euo pipefail
cd "$(dirname "$0")"

# MediaPipe wheels lag new Python releases, so prefer 3.11/3.12.
pick_python() {
  for c in python3.12 python3.11 "py -3.12" "py -3.11" python3 python; do
    if $c -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
      echo "$c"; return
    fi
  done
  echo "python3"
}

if [ ! -d .venv ]; then
  PY=$(pick_python)
  echo "[run] creating venv with $PY"
  $PY -m venv .venv
fi
if [ -x .venv/bin/python ]; then VPY=.venv/bin/python; else VPY=.venv/Scripts/python; fi

echo "[run] installing requirements"
$VPY -m pip install -q --upgrade pip
$VPY -m pip install -q -r requirements.txt

echo "[run] downloading models"
$VPY scripts/download_models.py

if [ "${1:-}" = "--mock" ]; then
  export CAM_SOURCE="${CAM_SOURCE:-webcam}" LLM_PROVIDER=mock STT_PROVIDER=mock TTS_PROVIDER=beep
  echo "[run] mock mode: CAM_SOURCE=$CAM_SOURCE, mock LLM/STT, beep TTS"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
echo "[run] starting on http://$HOST:$PORT (open http://localhost:$PORT/ for the demo page)"
exec $VPY -m uvicorn app.main:app --host "$HOST" --port "$PORT"
