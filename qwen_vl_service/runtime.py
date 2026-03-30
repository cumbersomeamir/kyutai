from __future__ import annotations

import base64
import fcntl
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import hf_hub_download

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
TMP_DIR = APP_DIR / "tmp"
UPLOAD_DIR = APP_DIR / "uploads"
LOG_DIR = APP_DIR / "logs"
MODEL_REPO = "Qwen/Qwen3-VL-2B-Instruct-GGUF"
MODEL_FILE = "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
LLAMA_BINARY = os.getenv(
    "LLAMA_SERVER_BINARY",
    shutil.which("llama-server") or "/opt/homebrew/bin/llama-server",
)
LLAMA_HOST = os.getenv("QWEN_LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.getenv("QWEN_LLAMA_PORT", "8795"))
LLAMA_CTX_SIZE = int(os.getenv("QWEN_LLAMA_CTX_SIZE", "4096"))
LLAMA_STARTUP_TIMEOUT = int(os.getenv("QWEN_LLAMA_STARTUP_TIMEOUT", "300"))
LLAMA_REQUEST_TIMEOUT = int(os.getenv("QWEN_LLAMA_REQUEST_TIMEOUT", "600"))
DEFAULT_PROMPT = "Describe this image clearly and concisely."
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2
THREADS = min(os.cpu_count() or 4, 8)
MAX_IMAGE_DOWNLOAD_BYTES = 50 * 1024 * 1024
SERVER_LOCK_PATH = TMP_DIR / "llama-server.lock"
SERVER_LOG_PATH = LOG_DIR / "llama-server.log"
_SERVER_LOCK = threading.Lock()

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_event(event: str, payload: dict[str, Any]) -> None:
    line = json.dumps(
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            **payload,
        },
        ensure_ascii=True,
    )
    with (LOG_DIR / "service.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def model_path() -> Path:
    return MODEL_DIR / MODEL_FILE


def mmproj_path() -> Path:
    return MODEL_DIR / MMPROJ_FILE


def llama_api_base() -> str:
    return f"http://{LLAMA_HOST}:{LLAMA_PORT}"


def download_model_files(progress=None) -> None:
    if progress:
        progress(5, "Checking Qwen model files")
    for filename in (MODEL_FILE, MMPROJ_FILE):
        target = MODEL_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        if progress:
            progress(10, f"Downloading {filename}")
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            local_dir=str(MODEL_DIR),
        )


def _server_ready() -> bool:
    try:
        response = requests.get(f"{llama_api_base()}/v1/models", timeout=5)
        response.raise_for_status()
    except Exception:
        return False
    return True


def _tail_server_log(lines: int = 40) -> str:
    if not SERVER_LOG_PATH.exists():
        return ""
    text = SERVER_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def _wait_for_server_ready(timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _server_ready():
            return
        time.sleep(2)
    tail = _tail_server_log()
    detail = f"\n{tail}" if tail else ""
    raise RuntimeError(f"Qwen llama-server did not become ready within {timeout_seconds} seconds.{detail}")


def _start_llama_server() -> None:
    with SERVER_LOG_PATH.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [
                LLAMA_BINARY,
                "-t",
                str(THREADS),
                "-c",
                str(LLAMA_CTX_SIZE),
                "-m",
                str(model_path()),
                "--mmproj",
                str(mmproj_path()),
                "--host",
                LLAMA_HOST,
                "--port",
                str(LLAMA_PORT),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(1)
    if process.poll() is not None:
        tail = _tail_server_log()
        detail = f"\n{tail}" if tail else ""
        raise RuntimeError(f"Qwen llama-server exited during startup.{detail}")


def ensure_runtime() -> None:
    if not shutil.which(LLAMA_BINARY):
        raise RuntimeError(
            f"{LLAMA_BINARY} was not found. Install llama.cpp first, for example via `brew install llama.cpp`."
        )
    download_model_files()
    with _SERVER_LOCK:
        with SERVER_LOCK_PATH.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                if _server_ready():
                    return
                _start_llama_server()
                _wait_for_server_ready(LLAMA_STARTUP_TIMEOUT)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def download_image(image_url: str, destination: Path) -> None:
    with requests.get(image_url, stream=True, timeout=(20, 300)) as response:
        response.raise_for_status()
        total = 0
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_IMAGE_DOWNLOAD_BYTES:
                    raise RuntimeError("Image is too large for the local Qwen VL service.")
                handle.write(chunk)


def _data_uri_for_image(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _response_text(payload: dict[str, Any]) -> str:
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")).strip())
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def run_image_to_text(
    image_path: Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    progress=None,
) -> dict[str, Any]:
    ensure_runtime()
    if progress:
        progress(20, "Preparing multimodal request")
    payload = {
        "model": MODEL_FILE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _data_uri_for_image(image_path)}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    start = time.time()
    response = requests.post(
        f"{llama_api_base()}/v1/chat/completions",
        json=payload,
        timeout=(30, LLAMA_REQUEST_TIMEOUT),
    )
    duration = round(time.time() - start, 2)
    response.raise_for_status()
    data = response.json()

    text = _response_text(data)
    if not text:
        raise RuntimeError("Qwen returned an empty response.")

    return {
        "success": True,
        "model": "Qwen3-VL-2B-Instruct-GGUF",
        "model_file": MODEL_FILE,
        "mmproj_file": MMPROJ_FILE,
        "prompt": prompt,
        "text": text,
        "duration_seconds": duration,
        "usage": data.get("usage"),
        "timings": data.get("timings"),
    }


def run_url_image_to_text(image_url: str, *, prompt: str, max_tokens: int, temperature: float, progress=None) -> dict[str, Any]:
    request_id = uuid.uuid4().hex[:10]
    work_dir = Path(tempfile.mkdtemp(prefix=f"qwen-url-{request_id}-", dir=TMP_DIR))
    image_path = work_dir / "image"
    try:
        if progress:
            progress(10, "Downloading image")
        download_image(image_url, image_path)
        if progress:
            progress(60, "Running image understanding")
        return run_image_to_text(image_path, prompt=prompt, max_tokens=max_tokens, temperature=temperature, progress=progress)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
