from __future__ import annotations

import os
import tempfile
from pathlib import Path
from queue import Queue
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.utils import PREDEFINED_VOICES, size_of_dict

BASE_DIR = Path(__file__).parent
INDEX_HTML = BASE_DIR / "index.html"
DEFAULT_VOICE = "alba"
DEFAULT_CONFIG = "b6369a24"

app = FastAPI(title="Kyutai Pocket TTS Local UI", version="1.0.0")

tts_model: TTSModel | None = None
default_model_state: dict | None = None


def get_model() -> TTSModel:
    global tts_model, default_model_state
    if tts_model is None:
        tts_model = TTSModel.load_model(DEFAULT_CONFIG)
        default_model_state = tts_model.get_state_for_audio_prompt(DEFAULT_VOICE)
    return tts_model


def get_default_state() -> dict:
    get_model()
    assert default_model_state is not None
    return default_model_state


def write_to_queue(queue: Queue, text_to_generate: str, model_state: dict) -> None:
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

    model = get_model()
    audio_chunks = model.generate_audio_stream(
        model_state=model_state,
        text_to_generate=text_to_generate,
    )
    stream_audio_chunks(FileLikeToQueue(queue), audio_chunks, model.config.mimi.sample_rate)


def generate_data_with_state(text_to_generate: str, model_state: dict):
    queue: Queue = Queue()
    thread = threading.Thread(target=write_to_queue, args=(queue, text_to_generate, model_state))
    thread.start()

    while True:
        data = queue.get()
        if data is None:
            break
        yield data

    thread.join()


def resolve_model_state(voice: str | None, voice_wav: UploadFile | None) -> dict:
    model = get_model()

    if voice and voice_wav:
        raise HTTPException(status_code=400, detail="Choose either a built-in/custom voice or upload a file, not both.")

    if voice_wav is not None:
        suffix = Path(voice_wav.filename or "voice.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(voice_wav.file.read())
            temp_path = temp_file.name
        try:
            return model.get_state_for_audio_prompt(Path(temp_path), truncate=True)
        finally:
            os.unlink(temp_path)

    if not voice or voice == DEFAULT_VOICE:
        return get_default_state()

    return model._cached_get_state_for_audio_prompt(voice)


@app.get("/")
async def root():
    return FileResponse(INDEX_HTML)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/info")
async def info():
    state = get_default_state()
    return JSONResponse(
        {
            "default_voice": DEFAULT_VOICE,
            "config": DEFAULT_CONFIG,
            "built_in_voices": list(PREDEFINED_VOICES.keys()),
            "supported_voice_inputs": [
                "built-in voice id",
                "local audio file upload",
                "http:// URL",
                "https:// URL",
                "hf:// Hugging Face URL",
            ],
            "endpoints": {
                "GET /": "simple local UI",
                "GET /health": "service health",
                "GET /api/info": "voices, features, endpoints",
                "POST /tts": "stream a generated WAV file",
            },
            "features": [
                "CPU inference",
                "streaming WAV response",
                "8 bundled voices",
                "voice cloning from uploaded audio",
                "custom voice loading from http/https/hf URLs",
            ],
            "default_voice_state_mb": round(size_of_dict(state) / 1_000_000, 2),
        }
    )


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    voice: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    model_state = resolve_model_state(voice, voice_wav)
    return StreamingResponse(
        generate_data_with_state(text, model_state),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=generated_speech.wav"},
    )
