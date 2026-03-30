#!/bin/zsh
set -euo pipefail

brew services start redis
brew services start rabbitmq
