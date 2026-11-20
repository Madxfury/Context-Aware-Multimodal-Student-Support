#!/bin/bash
# Start Samvya for a live demo. Double-click this file in Finder, or run
#   bash scripts/start_samvya.command
# It keeps the Mac awake, restarts the app if it ever stops, and opens the page once
# it is ready. Press Ctrl+C in this window to stop Samvya.
cd "$(dirname "$0")/.." || exit 1

PORT="${PORT:-$(grep -E '^PORT=' .env 2>/dev/null | cut -d= -f2)}"
PORT="${PORT:-5002}"
export PORT
URL="http://127.0.0.1:$PORT"

if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Samvya is already running at $URL"
  [ -z "$SAMVYA_NO_OPEN" ] && open "$URL"
  exit 0
fi
if [ ! -x .venv/bin/python ]; then
  echo "The Python environment (.venv) is missing. See README.md, Quick start."
  exit 1
fi

caffeinate -dims -w $$ &            # no display or system sleep while Samvya runs

(
  for _ in $(seq 1 120); do
    if curl -s -m 1 -o /dev/null "$URL/api/health"; then
      echo "Samvya is ready at $URL"
      [ -z "$SAMVYA_NO_OPEN" ] && open "$URL"
      exit 0
    fi
    sleep 1
  done
  echo "Samvya did not start within two minutes; check the messages above."
) &

while true; do
  .venv/bin/python app.py
  status=$?
  [ "$status" -eq 130 ] && exit 0     # stopped with Ctrl+C
  echo "Samvya stopped (exit $status). Restarting in 2 seconds; press Ctrl+C to quit."
  sleep 2
done
