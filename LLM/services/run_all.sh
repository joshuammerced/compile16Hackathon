#!/usr/bin/env bash
# Start whisper, qwen, kokoro and the gateway in the background; logs and PIDs in services/logs/.
#   ./run_all.sh          start everything, wait until all are ready
#   ./stop_all.sh         stop everything
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$HOME/miniconda3/envs/cooking-vlm/bin/python}"
mkdir -p logs

start() {  # name script port
  if [[ -f logs/$1.pid ]] && kill -0 "$(cat logs/$1.pid)" 2>/dev/null; then
    echo "$1 already running (pid $(cat logs/$1.pid))"; return
  fi
  nohup "$PY" "$2" >"logs/$1.log" 2>&1 &
  echo $! >"logs/$1.pid"
  echo "started $1 (pid $!) on :$3"
}

wait_ready() {  # name port
  for _ in $(seq 1 180); do
    if curl -fs "http://127.0.0.1:$2/ready" >/dev/null; then echo "$1 ready"; return; fi
    if ! kill -0 "$(cat logs/$1.pid)" 2>/dev/null; then
      echo "$1 exited during startup, last log lines:"; tail -20 "logs/$1.log"; exit 1
    fi
    sleep 1
  done
  echo "$1 not ready after 180s, see services/logs/$1.log"; exit 1
}

start whisper whisper_service.py 8102
start qwen qwen_service.py 8101
start kokoro kokoro_service.py 8103
wait_ready whisper 8102
wait_ready qwen 8101
wait_ready kokoro 8103
start gateway gateway.py 8100
wait_ready gateway 8100
echo
echo "Gateway (ESP32 posts here): http://$(ipconfig getifaddr en0 2>/dev/null || echo 127.0.0.1):8100/v1/respond"
