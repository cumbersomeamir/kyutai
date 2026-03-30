from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from common.job_queue import enqueue_job, get_job, ping_infra
from qwen_vl_service.runtime import (
    APP_DIR,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    LLAMA_BINARY,
    LLAMA_PORT,
    MODEL_REPO,
    MODEL_FILE,
    MMPROJ_FILE,
    llama_api_base,
    model_path,
    mmproj_path,
    run_image_to_text,
    run_url_image_to_text,
)

INDEX_HTML = APP_DIR / "index.html"
SERVICE_NAME = "qwen3-vl-2b-microservice"
SERVICE_VERSION = "1.0.0"
SERVICE_PORT = 8775

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "image.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=APP_DIR / "tmp") as temp_file:
        temp_file.write(upload.file.read())
        return temp_file.name


def _save_queue_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "image.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=APP_DIR / "uploads") as temp_file:
        temp_file.write(upload.file.read())
        return temp_file.name


def _service_info() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "port": SERVICE_PORT,
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILE,
        "mmproj_file": MMPROJ_FILE,
        "llama_binary": LLAMA_BINARY,
        "llama_api_base": llama_api_base(),
        "llama_server_port": LLAMA_PORT,
        "features": [
            "image to text",
            "image upload inference",
            "image URL inference",
            "official Qwen GGUF files",
            "CPU llama.cpp execution",
            "async jobs via Redis and RabbitMQ",
        ],
        "infra": ping_infra(),
        "endpoints": {
            "GET /": "simple local UI",
            "GET /health": "legacy health alias",
            "GET /api/v1/health": "microservice health",
            "GET /api/v1/info": "service metadata",
            "GET /api/v1/models": "model and runtime defaults",
            "GET /api/v1/endpoints": "endpoint catalog",
            "POST /api/v1/image-to-text": "sync image to text",
            "POST /api/v1/jobs/image-to-text": "enqueue image to text",
            "GET /api/v1/jobs/{job_id}": "job status/result",
            "GET /docs": "interactive API docs",
        },
    }


@app.get("/")
async def root():
    return FileResponse(INDEX_HTML)


@app.get("/health")
@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "model_ready": model_path().exists() and mmproj_path().exists(),
        "model_path": str(model_path()),
        "mmproj_path": str(mmproj_path()),
        "llama_api_base": llama_api_base(),
        "infra": ping_infra(),
    }


@app.get("/api/v1/info")
def info() -> dict[str, Any]:
    return _service_info()


@app.get("/api/v1/models")
def models() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILE,
        "mmproj_file": MMPROJ_FILE,
        "llama_binary": LLAMA_BINARY,
        "llama_api_base": llama_api_base(),
        "defaults": {
            "prompt": DEFAULT_PROMPT,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
        },
    }


@app.get("/api/v1/endpoints")
def endpoints() -> dict[str, Any]:
    return {"service": SERVICE_NAME, "routes": _service_info()["endpoints"]}


@app.post("/api/v1/image-to-text")
def image_to_text(
    prompt: str = Form(DEFAULT_PROMPT),
    image_url: str | None = Form(None),
    image_path: str | None = Form(None),
    max_tokens: int = Form(DEFAULT_MAX_TOKENS),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    image_file: UploadFile | None = File(None),
) -> dict[str, Any]:
    provided = [name for name, value in [("image_url", image_url), ("image_path", image_path), ("image_file", image_file)] if value]
    if len(provided) != 1:
        raise HTTPException(status_code=400, detail="Provide exactly one of image_url, image_path, or image_file.")

    if image_file is not None:
        temp_path = _save_upload(image_file)
        try:
            return run_image_to_text(
                Path(temp_path),
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    if image_path:
        return run_image_to_text(
            Path(image_path),
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    assert image_url is not None
    return run_url_image_to_text(
        image_url,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


@app.post("/api/v1/jobs/image-to-text")
def enqueue_image_to_text(
    prompt: str = Form(DEFAULT_PROMPT),
    image_url: str | None = Form(None),
    image_path: str | None = Form(None),
    max_tokens: int = Form(DEFAULT_MAX_TOKENS),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    image_file: UploadFile | None = File(None),
) -> dict[str, Any]:
    provided = [name for name, value in [("image_url", image_url), ("image_path", image_path), ("image_file", image_file)] if value]
    if len(provided) != 1:
        raise HTTPException(status_code=400, detail="Provide exactly one of image_url, image_path, or image_file.")

    payload: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if image_file is not None:
        payload["stored_path"] = _save_queue_upload(image_file)
    elif image_path:
        payload["image_path"] = image_path
    else:
        payload["image_url"] = image_url

    job = enqueue_job("qwen_vl", "image-to-text", payload)
    return {"job_id": job["job_id"], "status": job["status"], "service": SERVICE_NAME}


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = get_job("qwen_vl", job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
