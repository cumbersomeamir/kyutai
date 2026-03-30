from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from common.job_queue import enqueue_job, get_job, ping_infra
from whisper_service.runtime import (
    APP_DIR,
    DEVICE,
    LOG_DIR,
    MAX_CHARS_PER_PHRASE,
    MAX_WORDS_PER_PHRASE,
    MODEL_NAME,
    MODEL_ROOT,
    TMP_DIR,
    TranscribeUrlRequest,
    ensure_ffmpeg,
    ensure_model_weights,
    get_model,
    transcribe_uploaded_media,
    transcribe_url_request,
)

INDEX_HTML = APP_DIR / "index.html"
SERVICE_NAME = "whisper-small-microservice"
SERVICE_VERSION = "1.0.0"
SERVICE_PORT = 8774

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TMP_DIR) as temp_file:
        temp_file.write(upload.file.read())
        return temp_file.name


def _save_queue_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=APP_DIR / "uploads") as temp_file:
        temp_file.write(upload.file.read())
        return temp_file.name


def _service_info() -> dict[str, Any]:
    infra = ping_infra()
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "port": SERVICE_PORT,
        "model": MODEL_NAME,
        "device": DEVICE,
        "model_root": str(MODEL_ROOT),
        "features": [
            "speech to text",
            "source URL clip transcription",
            "uploaded file transcription",
            "word timestamps",
            "phrase grouping",
            "async jobs via Redis and RabbitMQ",
        ],
        "infra": infra,
        "endpoints": {
            "GET /": "simple local UI",
            "GET /health": "legacy health alias",
            "GET /api/v1/health": "microservice health",
            "GET /api/v1/info": "service metadata",
            "GET /api/v1/models": "model and runtime defaults",
            "GET /api/v1/endpoints": "endpoint catalog",
            "POST /transcribe": "legacy JSON transcription alias",
            "POST /api/v1/transcribe-url": "sync transcription from sourceUrl clip",
            "POST /api/v1/transcribe-file": "sync transcription from uploaded media",
            "POST /api/v1/jobs/transcribe-url": "enqueue URL transcription",
            "POST /api/v1/jobs/transcribe-file": "enqueue uploaded file transcription",
            "GET /api/v1/jobs/{job_id}": "job status/result",
            "GET /docs": "interactive API docs",
        },
    }


@app.on_event("startup")
def warmup() -> None:
    ensure_ffmpeg()
    ensure_model_weights()
    get_model()


@app.get("/")
async def root():
    return FileResponse(INDEX_HTML)


@app.get("/health")
@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    ensure_ffmpeg()
    model = get_model()
    return {
        "ok": True,
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "model": MODEL_NAME,
        "device": DEVICE,
        "ffmpeg": True,
        "dims": {
            "n_audio_state": getattr(model.dims, "n_audio_state", None),
            "n_text_state": getattr(model.dims, "n_text_state", None),
        },
        "infra": ping_infra(),
    }


@app.get("/api/v1/info")
def info() -> dict[str, Any]:
    return _service_info()


@app.get("/api/v1/models")
def models() -> dict[str, Any]:
    ensure_ffmpeg()
    ensure_model_weights()
    return {
        "service": SERVICE_NAME,
        "model": MODEL_NAME,
        "device": DEVICE,
        "model_root": str(MODEL_ROOT),
        "log_dir": str(LOG_DIR),
        "defaults": {
            "max_words_per_phrase": MAX_WORDS_PER_PHRASE,
            "max_chars_per_phrase": MAX_CHARS_PER_PHRASE,
        },
    }


@app.get("/api/v1/endpoints")
def endpoints() -> dict[str, Any]:
    return {"service": SERVICE_NAME, "routes": _service_info()["endpoints"]}


@app.post("/transcribe")
@app.post("/api/v1/transcribe-url")
def transcribe_url(request: TranscribeUrlRequest) -> dict[str, Any]:
    return transcribe_url_request(request)


@app.post("/api/v1/transcribe-file")
def transcribe_file(
    language: str = Form("en"),
    max_words_per_phrase: int = Form(MAX_WORDS_PER_PHRASE),
    max_chars_per_phrase: int = Form(MAX_CHARS_PER_PHRASE),
    clip_id: str = Form(""),
    clip_label: str = Form(""),
    media: UploadFile = File(...),
) -> dict[str, Any]:
    temp_path = _save_upload(media)
    try:
        return transcribe_uploaded_media(
            Path(temp_path),
            language=language,
            max_words_per_phrase=max_words_per_phrase,
            max_chars_per_phrase=max_chars_per_phrase,
            clip_id=clip_id,
            clip_label=clip_label,
        )
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/api/v1/jobs/transcribe-url")
def enqueue_transcribe_url(request: TranscribeUrlRequest) -> dict[str, Any]:
    job = enqueue_job("whisper", "transcribe-url", request.model_dump())
    return {"job_id": job["job_id"], "status": job["status"], "service": SERVICE_NAME}


@app.post("/api/v1/jobs/transcribe-file")
def enqueue_transcribe_file(
    language: str = Form("en"),
    max_words_per_phrase: int = Form(MAX_WORDS_PER_PHRASE),
    max_chars_per_phrase: int = Form(MAX_CHARS_PER_PHRASE),
    clip_id: str = Form(""),
    clip_label: str = Form(""),
    media: UploadFile = File(...),
) -> dict[str, Any]:
    stored_path = _save_queue_upload(media)
    job = enqueue_job(
        "whisper",
        "transcribe-file",
        {
            "stored_path": stored_path,
            "language": language,
            "max_words_per_phrase": max_words_per_phrase,
            "max_chars_per_phrase": max_chars_per_phrase,
            "clip_id": clip_id,
            "clip_label": clip_label,
        },
    )
    return {"job_id": job["job_id"], "status": job["status"], "service": SERVICE_NAME}


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = get_job("whisper", job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
