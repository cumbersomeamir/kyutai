# Kyutai Pocket TTS Local

Minimal local wrapper around Kyutai's upstream `pocket-tts` package.

## What is included

- local CPU inference via `pocket-tts`
- simple browser UI at `http://127.0.0.1:8000`
- built-in voices list
- endpoint and feature display
- voice cloning from uploaded audio
- custom voice loading from `http://`, `https://`, or `hf://`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pocket-tts
```

## Run

```bash
./start.sh
```

Then open `http://127.0.0.1:8000`.

## API

- `GET /health`
- `GET /api/info`
- `POST /tts`
