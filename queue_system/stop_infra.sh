#!/bin/zsh
set -euo pipefail

brew services stop redis || true
brew services stop rabbitmq || true
