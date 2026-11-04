#!/usr/bin/env bash
# Start the workbench API process.
#
# This process serves the meeting, speaker, notification and training routes,
# the offline training-config endpoints and the TTS voice-library management
# routes. It owns the task supervisor, job reconcile/restore and
# meeting-artifact cleanup. Run the realtime process separately
# (scripts/serve/realtime.sh); both may share the same host and SQLite stores
# thanks to WAL.
set -euo pipefail

cd "$(dirname "$0")/../.."

ASTRA_WORKBENCH_HOST="${ASTRA_WORKBENCH_HOST:-0.0.0.0}"
ASTRA_WORKBENCH_PORT="${ASTRA_WORKBENCH_PORT:-8001}"

PYTHONPATH=backend ASTRA_PROFILE=workbench \
  .venv/bin/uvicorn app.main:app \
  --host "$ASTRA_WORKBENCH_HOST" \
  --port "$ASTRA_WORKBENCH_PORT" \
  "$@"
