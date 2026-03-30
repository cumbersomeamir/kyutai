#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate
exec python tts_worker.py
