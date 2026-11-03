#!/usr/bin/env bash
# Start the realtime voice-link API process.
#
# This process serves the realtime WebSocket session, audio upload/transcribe
# and realtime-recording routes. It reads the shared TTS voice library (for
# snapshots) but does not run meeting/training supervision. Run the workbench
# process separately (scripts/serve/workbench.sh); both may share the same host
# and SQLite stores thanks to WAL.
set -euo pipefail

cd "$(dirname "$0")/../.."

ASTRA_REALTIME_HOST="${ASTRA_REALTIME_HOST:-0.0.0.0}"
ASTRA_REALTIME_PORT="${ASTRA_REALTIME_PORT:-8000}"

PYTHONPATH=backend ASTRA_PROFILE=realtime \
  .venv/bin/uvicorn app.main:app \
  --host "$ASTRA_REALTIME_HOST" \
  --port "$ASTRA_REALTIME_PORT" \
  "$@"
