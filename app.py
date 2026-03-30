from __future__ import annotations

import io
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from common.job_queue import enqueue_job, get_job, ping_infra
from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
    MAX_TOKEN_PER_CHUNK,
)
from pocket_tts.models.tts_model import TTSModel, export_model_state
from pocket_tts.utils.utils import PREDEFINED_VOICES, size_of_dict

BASE_DIR = Path(__file__).parent
INDEX_HTML = BASE_DIR / "index.html"
API_PREFIX = "/api/v1"
SERVICE_NAME = "kyutai-pocket-tts-microservice"
SERVICE_VERSION = "2.0.0"
SERVICE_PORT = 8773
DEFAULT_VOICE = "alba"
DEFAULT_CONFIG = DEFAULT_VARIANT
QUEUE_UPLOAD_DIR = BASE_DIR / "tts_uploads"
JOB_AUDIO_DIR = BASE_DIR / "tts_jobs"

QUEUE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOB_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Kyutai Pocket TTS Microservice", version=SERVICE_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass(frozen=True)
class ModelSettings:
    config: str = DEFAULT_CONFIG
    temperature: float = DEFAULT_TEMPERATURE
    lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS
    noise_clamp: float | None = DEFAULT_NOISE_CLAMP
    eos_threshold: float = DEFAULT_EOS_THRESHOLD


_cache_guard = threading.Lock()
_model_cache: dict[ModelSettings, TTSModel] = {}
_model_locks: dict[ModelSettings, threading.Lock] = {}
_default_state_cache: dict[ModelSettings, dict] = {}


class FileLikeToQueue:
    def __init__(self, target_queue: Queue):
        self.queue = target_queue

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, data: bytes) -> None:
        self.queue.put(data)

    def flush(self) -> None:
        return

    def close(self) -> None:
        self.queue.put(None)


class KeepOpenBytesIO(io.BytesIO):
    def close(self) -> None:
        self.flush()


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _get_runtime(settings: ModelSettings) -> tuple[TTSModel, threading.Lock]:
    with _cache_guard:
        model = _model_cache.get(settings)
        if model is None:
            model = TTSModel.load_model(
                settings.config,
                settings.temperature,
                settings.lsd_decode_steps,
                settings.noise_clamp,
                settings.eos_threshold,
            )
            _model_cache[settings] = model
            _model_locks[settings] = threading.Lock()
        lock = _model_locks[settings]
    return model, lock


def _get_default_state(settings: ModelSettings) -> dict:
    with _cache_guard:
        cached = _default_state_cache.get(settings)
        if cached is not None:
            return cached

    model, lock = _get_runtime(settings)
    with lock:
        state = model.get_state_for_audio_prompt(DEFAULT_VOICE)

    with _cache_guard:
        _default_state_cache[settings] = state
    return state


def _save_upload(upload: UploadFile) -> str:
    return _save_upload_to_dir(upload, None)


def _save_upload_to_dir(upload: UploadFile, directory: Path | None) -> str:
    suffix = Path(upload.filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=directory) as temp_file:
        temp_file.write(upload.file.read())
        return temp_file.name


def _raise_api_error(exc: Exception) -> None:
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise exc


def _resolve_model_state(
    *,
    settings: ModelSettings,
    voice: str | None,
    voice_url: str | None,
    voice_path: str | None,
    voice_wav: UploadFile | None,
    voice_state: UploadFile | None,
    voice_wav_path: str | None = None,
    voice_state_path: str | None = None,
    truncate_voice_prompt: bool,
) -> dict:
    voice = _normalize(voice)
    voice_url = _normalize(voice_url)
    voice_path = _normalize(voice_path)
    voice_wav_path = _normalize(voice_wav_path)
    voice_state_path = _normalize(voice_state_path)
    model, lock = _get_runtime(settings)

    sources = [
        ("voice", voice if voice and voice != DEFAULT_VOICE else None),
        ("voice_url", voice_url),
        ("voice_path", voice_path),
        ("voice_wav", voice_wav),
        ("voice_state", voice_state),
        ("voice_wav_path", voice_wav_path),
        ("voice_state_path", voice_state_path),
    ]
    supplied = [name for name, value in sources if value is not None]
    if len(supplied) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Choose exactly one voice source. Received: {', '.join(supplied)}.",
        )

    if voice_state is not None:
        temp_path = _save_upload(voice_state)
        try:
            try:
                with lock:
                    return model.get_state_for_audio_prompt(Path(temp_path), truncate=False)
            except Exception as exc:
                _raise_api_error(exc)
        finally:
            os.unlink(temp_path)

    if voice_state_path is not None:
        try:
            with lock:
                return model.get_state_for_audio_prompt(Path(voice_state_path), truncate=False)
        except Exception as exc:
            _raise_api_error(exc)

    if voice_wav is not None:
        temp_path = _save_upload(voice_wav)
        try:
            try:
                with lock:
                    return model.get_state_for_audio_prompt(
                        Path(temp_path), truncate=truncate_voice_prompt
                    )
            except Exception as exc:
                _raise_api_error(exc)
        finally:
            os.unlink(temp_path)

    if voice_wav_path is not None:
        try:
            with lock:
                return model.get_state_for_audio_prompt(
                    Path(voice_wav_path), truncate=truncate_voice_prompt
                )
        except Exception as exc:
            _raise_api_error(exc)

    if voice_url is not None:
        try:
            with lock:
                return model.get_state_for_audio_prompt(voice_url, truncate=truncate_voice_prompt)
        except Exception as exc:
            _raise_api_error(exc)

    if voice_path is not None:
        try:
            with lock:
                return model.get_state_for_audio_prompt(
                    Path(voice_path), truncate=truncate_voice_prompt
                )
        except Exception as exc:
            _raise_api_error(exc)

    if voice and voice != DEFAULT_VOICE:
        try:
            with lock:
                return model.get_state_for_audio_prompt(voice, truncate=False)
        except Exception as exc:
            _raise_api_error(exc)

    return _get_default_state(settings)


def _write_audio_to_queue(
    queue: Queue,
    *,
    settings: ModelSettings,
    text_to_generate: str,
    model_state: dict,
    max_tokens: int,
    frames_after_eos: int | None,
) -> None:
    model, lock = _get_runtime(settings)
    with lock:
        audio_chunks = model.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            max_tokens=max_tokens,
            frames_after_eos=frames_after_eos,
        )
        stream_audio_chunks(FileLikeToQueue(queue), audio_chunks, model.sample_rate)


def _stream_generated_audio(
    *,
    settings: ModelSettings,
    text_to_generate: str,
    model_state: dict,
    max_tokens: int,
    frames_after_eos: int | None,
):
    queue: Queue = Queue()
    thread = threading.Thread(
        target=_write_audio_to_queue,
        kwargs={
            "queue": queue,
            "settings": settings,
            "text_to_generate": text_to_generate,
            "model_state": model_state,
            "max_tokens": max_tokens,
            "frames_after_eos": frames_after_eos,
        },
    )
    thread.start()

    while True:
        data = queue.get()
        if data is None:
            break
        yield data

    thread.join()


def _generate_audio_bytes(
    *,
    settings: ModelSettings,
    text_to_generate: str,
    model_state: dict,
    max_tokens: int,
    frames_after_eos: int | None,
) -> bytes:
    model, lock = _get_runtime(settings)
    buffer = KeepOpenBytesIO()
    with lock:
        audio_chunks = model.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            max_tokens=max_tokens,
            frames_after_eos=frames_after_eos,
        )
        stream_audio_chunks(buffer, audio_chunks, model.sample_rate)
    return buffer.getvalue()


def _settings_to_payload(settings: ModelSettings) -> dict:
    return {
        "config": settings.config,
        "temperature": settings.temperature,
        "lsd_decode_steps": settings.lsd_decode_steps,
        "noise_clamp": settings.noise_clamp,
        "eos_threshold": settings.eos_threshold,
    }


def _settings_from_payload(payload: dict) -> ModelSettings:
    return ModelSettings(
        config=payload.get("config", DEFAULT_CONFIG),
        temperature=float(payload.get("temperature", DEFAULT_TEMPERATURE)),
        lsd_decode_steps=int(payload.get("lsd_decode_steps", DEFAULT_LSD_DECODE_STEPS)),
        noise_clamp=payload.get("noise_clamp", DEFAULT_NOISE_CLAMP),
        eos_threshold=float(payload.get("eos_threshold", DEFAULT_EOS_THRESHOLD)),
    )


def _service_info() -> dict:
    settings = ModelSettings()
    state = _get_default_state(settings)
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "port": SERVICE_PORT,
        "infra": ping_infra(),
        "default_voice": DEFAULT_VOICE,
        "default_config": DEFAULT_CONFIG,
        "built_in_voices": list(PREDEFINED_VOICES.keys()),
        "supported_voice_inputs": [
            "built-in voice id",
            "http:// voice URL",
            "https:// voice URL",
            "hf:// voice URL",
            "trusted local server path",
            "uploaded audio file for voice cloning",
            "uploaded .safetensors voice state",
        ],
        "generation_options": {
            "temperature": DEFAULT_TEMPERATURE,
            "lsd_decode_steps": DEFAULT_LSD_DECODE_STEPS,
            "noise_clamp": DEFAULT_NOISE_CLAMP,
            "eos_threshold": DEFAULT_EOS_THRESHOLD,
            "max_tokens_per_chunk": MAX_TOKEN_PER_CHUNK,
        },
        "features": [
            "CPU inference",
            "streaming WAV synthesis",
            "8 built-in voices",
            "voice cloning from uploaded audio",
            "voice state reuse via safetensors",
            "custom voices via http/https/hf URLs",
            "voice-state export for faster repeated synthesis",
            "versioned microservice endpoints",
            "async jobs via Redis and RabbitMQ",
        ],
        "default_voice_state_mb": round(size_of_dict(state) / 1_000_000, 2),
        "docs": {
            "swagger_ui": "/docs",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "GET /": "simple local UI",
            "GET /health": "legacy health alias",
            f"GET {API_PREFIX}/health": "microservice health",
            "GET /api/info": "legacy service metadata alias",
            f"GET {API_PREFIX}/info": "service metadata and defaults",
            f"GET {API_PREFIX}/voices": "built-in voices and voice references",
            f"GET {API_PREFIX}/models": "model configuration defaults",
            f"GET {API_PREFIX}/endpoints": "machine-readable endpoint catalog",
            "POST /tts": "legacy streaming synthesis alias",
            f"POST {API_PREFIX}/tts": "stream generated WAV audio",
            f"POST {API_PREFIX}/jobs/tts": "enqueue generated WAV audio",
            f"GET {API_PREFIX}/jobs/{{job_id}}": "job status and result metadata",
            f"GET {API_PREFIX}/jobs/{{job_id}}/audio": "download completed queued audio",
            f"POST {API_PREFIX}/export-voice": "export a voice state as .safetensors",
            "GET /docs": "interactive API docs",
        },
    }


@app.get("/")
async def root():
    return FileResponse(INDEX_HTML)


@app.get("/health")
@app.get(f"{API_PREFIX}/health")
async def health():
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "port": SERVICE_PORT,
        "infra": ping_infra(),
    }


@app.get("/api/info")
@app.get(f"{API_PREFIX}/info")
async def info():
    return JSONResponse(_service_info())


@app.get(f"{API_PREFIX}/voices")
async def voices():
    return {
        "default_voice": DEFAULT_VOICE,
        "voices": [
            {"id": name, "source": PREDEFINED_VOICES[name]}
            for name in PREDEFINED_VOICES
        ],
    }


@app.get(f"{API_PREFIX}/models")
async def models():
    return {
        "default_config": DEFAULT_CONFIG,
        "available_configs": [DEFAULT_CONFIG],
        "defaults": {
            "temperature": DEFAULT_TEMPERATURE,
            "lsd_decode_steps": DEFAULT_LSD_DECODE_STEPS,
            "noise_clamp": DEFAULT_NOISE_CLAMP,
            "eos_threshold": DEFAULT_EOS_THRESHOLD,
            "max_tokens": MAX_TOKEN_PER_CHUNK,
        },
    }


@app.get(f"{API_PREFIX}/endpoints")
async def endpoints():
    return {"service": SERVICE_NAME, "routes": _service_info()["endpoints"]}


@app.post("/tts")
@app.post(f"{API_PREFIX}/tts")
async def text_to_speech(
    text: str = Form(...),
    voice: str | None = Form(None),
    voice_url: str | None = Form(None),
    voice_path: str | None = Form(None),
    config: str = Form(DEFAULT_CONFIG),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    lsd_decode_steps: int = Form(DEFAULT_LSD_DECODE_STEPS),
    noise_clamp: float | None = Form(DEFAULT_NOISE_CLAMP),
    eos_threshold: float = Form(DEFAULT_EOS_THRESHOLD),
    frames_after_eos: int | None = Form(None),
    max_tokens: int = Form(MAX_TOKEN_PER_CHUNK),
    truncate_voice_prompt: bool = Form(True),
    voice_wav: UploadFile | None = File(None),
    voice_state: UploadFile | None = File(None),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    settings = ModelSettings(
        config=config,
        temperature=temperature,
        lsd_decode_steps=lsd_decode_steps,
        noise_clamp=noise_clamp,
        eos_threshold=eos_threshold,
    )
    model_state = _resolve_model_state(
        settings=settings,
        voice=voice,
        voice_url=voice_url,
        voice_path=voice_path,
        voice_wav=voice_wav,
        voice_state=voice_state,
        truncate_voice_prompt=truncate_voice_prompt,
    )
    return StreamingResponse(
        _stream_generated_audio(
            settings=settings,
            text_to_generate=text,
            model_state=model_state,
            max_tokens=max_tokens,
            frames_after_eos=frames_after_eos,
        ),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=generated_speech.wav"},
    )


@app.post(f"{API_PREFIX}/jobs/tts")
async def enqueue_text_to_speech(
    text: str = Form(...),
    voice: str | None = Form(None),
    voice_url: str | None = Form(None),
    voice_path: str | None = Form(None),
    config: str = Form(DEFAULT_CONFIG),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    lsd_decode_steps: int = Form(DEFAULT_LSD_DECODE_STEPS),
    noise_clamp: float | None = Form(DEFAULT_NOISE_CLAMP),
    eos_threshold: float = Form(DEFAULT_EOS_THRESHOLD),
    frames_after_eos: int | None = Form(None),
    max_tokens: int = Form(MAX_TOKEN_PER_CHUNK),
    truncate_voice_prompt: bool = Form(True),
    voice_wav: UploadFile | None = File(None),
    voice_state: UploadFile | None = File(None),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    settings = ModelSettings(
        config=config,
        temperature=temperature,
        lsd_decode_steps=lsd_decode_steps,
        noise_clamp=noise_clamp,
        eos_threshold=eos_threshold,
    )
    payload = {
        "text": text,
        "voice": voice,
        "voice_url": voice_url,
        "voice_path": voice_path,
        "voice_wav_path": _save_upload_to_dir(voice_wav, QUEUE_UPLOAD_DIR) if voice_wav is not None else None,
        "voice_state_path": _save_upload_to_dir(voice_state, QUEUE_UPLOAD_DIR) if voice_state is not None else None,
        "settings": _settings_to_payload(settings),
        "frames_after_eos": frames_after_eos,
        "max_tokens": max_tokens,
        "truncate_voice_prompt": truncate_voice_prompt,
    }
    job = enqueue_job("tts", "tts", payload)
    return {"job_id": job["job_id"], "status": job["status"], "service": SERVICE_NAME}


@app.get(f"{API_PREFIX}/jobs/{{job_id}}")
async def tts_job_status(job_id: str):
    job = get_job("tts", job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get(f"{API_PREFIX}/jobs/{{job_id}}/audio")
async def tts_job_audio(job_id: str):
    job = get_job("tts", job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed yet.")
    result = job.get("result") or {}
    audio_path = result.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Generated audio was not found.")
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=result.get("filename", "generated_speech.wav"),
    )


@app.post(f"{API_PREFIX}/export-voice")
async def export_voice(
    voice_url: str | None = Form(None),
    voice_path: str | None = Form(None),
    config: str = Form(DEFAULT_CONFIG),
    truncate_voice_prompt: bool = Form(True),
    voice_wav: UploadFile | None = File(None),
):
    voice_url = _normalize(voice_url)
    voice_path = _normalize(voice_path)
    supplied = [name for name, value in [("voice_url", voice_url), ("voice_path", voice_path), ("voice_wav", voice_wav)] if value is not None]
    if len(supplied) != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one source: voice_url, voice_path, or voice_wav.",
        )

    settings = ModelSettings(config=config)
    model, lock = _get_runtime(settings)

    if voice_wav is not None:
        temp_path = _save_upload(voice_wav)
        source: str | Path = Path(temp_path)
    elif voice_path is not None:
        temp_path = None
        source = Path(voice_path)
    else:
        temp_path = None
        assert voice_url is not None
        source = voice_url

    try:
        try:
            with lock:
                model_state = model.get_state_for_audio_prompt(
                    source, truncate=truncate_voice_prompt
                )
        except Exception as exc:
            _raise_api_error(exc)
        buffer = io.BytesIO()
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as temp_file:
            export_model_state(model_state, temp_file.name)
            temp_file.flush()
            with open(temp_file.name, "rb") as exported:
                buffer.write(exported.read())
        os.unlink(temp_file.name)
    finally:
        if voice_wav is not None and temp_path is not None:
            os.unlink(temp_path)

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="voice_state.safetensors"'},
    )
