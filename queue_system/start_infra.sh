#!/bin/zsh
set -euo pipefail

if command -v brew >/dev/null 2>&1; then
  brew services start redis
  brew services start rabbitmq
elif command -v systemctl >/dev/null 2>&1; then
  sudo systemctl start redis-server
  sudo systemctl start rabbitmq-server
else
  echo "No supported service manager found for Redis/RabbitMQ startup." >&2
  exit 1
fi
