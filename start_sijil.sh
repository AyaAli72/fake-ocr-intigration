#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python sijil_verification_backend.py &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null || true' EXIT
python -m http.server 8000
