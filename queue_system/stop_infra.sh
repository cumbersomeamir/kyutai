#!/usr/bin/env bash
set -euo pipefail

if command -v brew >/dev/null 2>&1; then
  brew services stop redis || true
  brew services stop rabbitmq || true
elif command -v systemctl >/dev/null 2>&1; then
  sudo systemctl stop redis-server || true
  sudo systemctl stop rabbitmq-server || true
else
  echo "No supported service manager found for Redis/RabbitMQ shutdown." >&2
  exit 1
fi
