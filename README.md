# Local AI Microservices

Three local microservices now live in this repo, each with its own HTML page, versioned API, queue-backed job flow, and worker process.

## Services

- Pocket TTS: `http://127.0.0.1:8773`
- Whisper Small STT: `http://127.0.0.1:8774`
- Qwen3-VL-2B-Instruct image-to-text: `http://127.0.0.1:8775`

## Queue stack

- Redis for job state
- RabbitMQ for durable job delivery
- Live polling in each HTML page for queued job status

Start infra:

```bash
./queue_system/start_infra.sh
```

Stop infra:

```bash
./queue_system/stop_infra.sh
```

## Run services

Pocket TTS:

```bash
./start.sh
./start_worker.sh
```

Whisper:

```bash
./whisper_service/start.sh
./whisper_service/start_worker.sh
```

Qwen3-VL:

```bash
./qwen_vl_service/start.sh
./qwen_vl_service/start_worker.sh
```

## Main APIs

Pocket TTS:

- `GET /api/v1/health`
- `GET /api/v1/info`
- `GET /api/v1/voices`
- `GET /api/v1/models`
- `GET /api/v1/endpoints`
- `POST /api/v1/tts`
- `POST /api/v1/jobs/tts`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/audio`
- `POST /api/v1/export-voice`

Whisper:

- `GET /api/v1/health`
- `GET /api/v1/info`
- `GET /api/v1/models`
- `GET /api/v1/endpoints`
- `POST /api/v1/transcribe-url`
- `POST /api/v1/transcribe-file`
- `POST /api/v1/jobs/transcribe-url`
- `POST /api/v1/jobs/transcribe-file`
- `GET /api/v1/jobs/{job_id}`

Qwen3-VL:

- `GET /api/v1/health`
- `GET /api/v1/info`
- `GET /api/v1/models`
- `GET /api/v1/endpoints`
- `POST /api/v1/image-to-text`
- `POST /api/v1/jobs/image-to-text`
- `GET /api/v1/jobs/{job_id}`

## Notes

- Whisper uses the existing laptop model at `/Users/rajeevkumar/Downloads/whisper-small/models/small.en.pt`.
- Qwen uses the official GGUF files under `qwen_vl_service/models/`.
- Qwen is served through local `llama-server` on internal port `8795`.
- Pocket TTS sync endpoints are unchanged; queued endpoints were added alongside them.
- On non-macOS hosts, set `WHISPER_MODEL_ROOT` to the folder that contains `small.en.pt`.
- On non-macOS hosts, install `llama-server` on `PATH` or set `LLAMA_SERVER_BINARY`.
