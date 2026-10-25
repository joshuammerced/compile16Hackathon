#!/usr/bin/env bash
# Stop everything started by run_all.sh.
cd "$(dirname "$0")"
for name in gateway kokoro qwen whisper; do
  pidfile="logs/$name.pid"
  [[ -f $pidfile ]] || continue
  pid=$(cat "$pidfile")
  if kill "$pid" 2>/dev/null; then echo "stopped $name (pid $pid)"; fi
  rm -f "$pidfile"
done
