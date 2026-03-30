from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

import requests
import whisper
from fastapi import HTTPException
from pydantic import BaseModel, Field
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

APP_DIR = Path(__file__).resolve().parent
TMP_DIR = APP_DIR / "tmp"
UPLOAD_DIR = APP_DIR / "uploads"
LOG_DIR = APP_DIR / "logs"
MODEL_ROOT = Path(
    os.getenv("WHISPER_MODEL_ROOT", "/Users/rajeevkumar/Downloads/whisper-small/models")
)
MODEL_NAME = os.getenv("WHISPER_MODEL", "small.en")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
MAX_DOWNLOAD_SIZE_BYTES = 500 * 1024 * 1024
PHRASE_GAP_SECONDS = 0.55
MAX_WORDS_PER_PHRASE = 6
MAX_CHARS_PER_PHRASE = 28

TMP_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

_MODEL = None
_MODEL_LOCK = Lock()


class TranscribeUrlRequest(BaseModel):
    sourceUrl: str = Field(min_length=1)
    clipId: str = ""
    clipLabel: str = ""
    clipStart: float = Field(default=0, ge=0)
    clipDuration: float = Field(default=5, gt=0)
    language: str = Field(default="en", min_length=2, max_length=12)
    maxWordsPerPhrase: int = Field(default=MAX_WORDS_PER_PHRASE, ge=2, le=12)
    maxCharsPerPhrase: int = Field(default=MAX_CHARS_PER_PHRASE, ge=12, le=64)


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


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    raise RuntimeError("ffmpeg is required but was not found on PATH")


def ensure_model_weights() -> Path:
    model_url = whisper._MODELS.get(MODEL_NAME)
    if not model_url:
        raise RuntimeError(f"Unknown Whisper model: {MODEL_NAME}")

    target_path = MODEL_ROOT / model_url.rsplit("/", 1)[-1]
    if not target_path.exists():
        raise RuntimeError(
            f"Expected Whisper weights at {target_path}. "
            f"The laptop reference path was provided, but the file was not found."
        )
    return target_path


def get_model():
    global _MODEL
    if _MODEL is None:
        ensure_ffmpeg()
        ensure_model_weights()
        log_event("model.load.start", {"model": MODEL_NAME, "device": DEVICE})
        _MODEL = whisper.load_model(
            MODEL_NAME,
            device=DEVICE,
            download_root=str(MODEL_ROOT),
            in_memory=False,
        )
        log_event("model.load.done", {"model": MODEL_NAME, "device": DEVICE})
    return _MODEL


def download_source_file(source_url: str, destination: Path) -> None:
    try:
        with requests.get(source_url, stream=True, timeout=(20, 300)) as response:
            response.raise_for_status()
            total = 0
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_SIZE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="Clip is too large for the local Whisper service.",
                        )
                    handle.write(chunk)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Failed to download source media: {error}") from error


def extract_clip_audio(source_path: Path, output_path: Path, clip_start: float, clip_duration: float) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{clip_start:.3f}",
        "-t",
        f"{clip_duration:.3f}",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return
    stderr = (result.stderr or "").strip() or "ffmpeg failed"
    raise HTTPException(status_code=422, detail=f"Failed to extract clip audio: {stderr}")


def normalize_media_to_wav(source_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return
    stderr = (result.stderr or "").strip() or "ffmpeg failed"
    raise HTTPException(status_code=422, detail=f"Failed to normalize media: {stderr}")


def sanitize_word_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def build_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    segments = result.get("segments") or []
    for segment in segments:
        for word in segment.get("words") or []:
            text = sanitize_word_text(word.get("word", ""))
            start = float(word.get("start") or 0)
            end = float(word.get("end") or start)
            if not text:
                continue
            words.append(
                {
                    "id": f"word-{len(words) + 1}",
                    "text": text,
                    "start": round(start, 2),
                    "end": round(max(end, start), 2),
                }
            )
    if words:
        return words
    for segment in segments:
        text = sanitize_word_text(segment.get("text", ""))
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        if not text:
            continue
        words.append(
            {
                "id": f"word-{len(words) + 1}",
                "text": text,
                "start": round(start, 2),
                "end": round(max(end, start), 2),
            }
        )
    return words


def should_break_phrase(
    next_word: dict[str, Any],
    current_words: list[dict[str, Any]],
    max_words_per_phrase: int,
    max_chars_per_phrase: int,
) -> bool:
    if not current_words:
        return False
    previous_word = current_words[-1]
    gap = float(next_word["start"]) - float(previous_word["end"])
    if gap >= PHRASE_GAP_SECONDS:
        return True
    current_text = " ".join(word["text"] for word in current_words).strip()
    projected_text = f"{current_text} {next_word['text']}".strip()
    if len(current_words) >= max_words_per_phrase:
        return True
    if len(projected_text) > max_chars_per_phrase:
        return True
    if re.search(r"[.!?]$", previous_word["text"]):
        return True
    return False


def build_phrases(
    words: list[dict[str, Any]],
    max_words_per_phrase: int,
    max_chars_per_phrase: int,
) -> list[dict[str, Any]]:
    if not words:
        return []
    phrases: list[dict[str, Any]] = []
    current_words: list[dict[str, Any]] = []

    def flush() -> None:
        if not current_words:
            return
        start = float(current_words[0]["start"])
        end = float(current_words[-1]["end"])
        phrases.append(
            {
                "id": f"phrase-{len(phrases) + 1}",
                "text": " ".join(word["text"] for word in current_words).strip(),
                "start": round(start, 2),
                "end": round(max(end, start), 2),
                "duration": round(max(end - start, 0.2), 2),
                "words": current_words.copy(),
            }
        )
        current_words.clear()

    for word in words:
        if should_break_phrase(word, current_words, max_words_per_phrase, max_chars_per_phrase):
            flush()
        current_words.append(word)
    flush()
    return phrases


def run_transcription(
    audio_path: Path,
    *,
    clip_id: str,
    clip_label: str,
    duration: float,
    language: str,
    max_words_per_phrase: int,
    max_chars_per_phrase: int,
) -> dict[str, Any]:
    ensure_ffmpeg()
    with _MODEL_LOCK:
        model = get_model()
        result = model.transcribe(
            str(audio_path),
            task="transcribe",
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
            fp16=False,
            verbose=False,
        )

    words = build_words(result)
    phrases = build_phrases(words, max_words_per_phrase, max_chars_per_phrase)
    transcript = " ".join(phrase["text"] for phrase in phrases).strip()
    return {
        "success": True,
        "model": MODEL_NAME,
        "device": DEVICE,
        "language": result.get("language") or language,
        "clipId": clip_id,
        "clipLabel": clip_label,
        "transcript": transcript,
        "duration": round(duration, 2),
        "wordCount": len(words),
        "phraseCount": len(phrases),
        "phrases": phrases,
    }


def transcribe_url_request(request: TranscribeUrlRequest, progress=None) -> dict[str, Any]:
    request_id = uuid.uuid4().hex[:10]
    work_dir = Path(tempfile.mkdtemp(prefix=f"whisper-url-{request_id}-", dir=TMP_DIR))
    source_path = work_dir / "source-media"
    audio_path = work_dir / "clip-audio.wav"
    try:
        if progress:
            progress(10, "Downloading source media")
        log_event(
            "transcribe.start",
            {
                "requestId": request_id,
                "clipId": request.clipId,
                "clipStart": request.clipStart,
                "clipDuration": request.clipDuration,
                "language": request.language,
            },
        )
        download_source_file(request.sourceUrl, source_path)
        if progress:
            progress(35, "Extracting clip audio")
        extract_clip_audio(source_path, audio_path, request.clipStart, request.clipDuration)
        if progress:
            progress(65, "Running Whisper inference")
        payload = run_transcription(
            audio_path,
            clip_id=request.clipId,
            clip_label=request.clipLabel,
            duration=request.clipDuration,
            language=request.language,
            max_words_per_phrase=request.maxWordsPerPhrase,
            max_chars_per_phrase=request.maxCharsPerPhrase,
        )
        log_event(
            "transcribe.done",
            {
                "requestId": request_id,
                "clipId": request.clipId,
                "wordCount": payload["wordCount"],
                "phraseCount": payload["phraseCount"],
            },
        )
        return payload
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def transcribe_uploaded_media(
    source_path: Path,
    *,
    language: str = "en",
    max_words_per_phrase: int = MAX_WORDS_PER_PHRASE,
    max_chars_per_phrase: int = MAX_CHARS_PER_PHRASE,
    clip_id: str = "",
    clip_label: str = "",
    progress=None,
) -> dict[str, Any]:
    request_id = uuid.uuid4().hex[:10]
    work_dir = Path(tempfile.mkdtemp(prefix=f"whisper-file-{request_id}-", dir=TMP_DIR))
    audio_path = work_dir / "normalized.wav"
    try:
        duration = 0.0
        if progress:
            progress(20, "Preparing media")
        normalize_media_to_wav(source_path, audio_path)
        if progress:
            progress(65, "Running Whisper inference")
        payload = run_transcription(
            audio_path,
            clip_id=clip_id,
            clip_label=clip_label,
            duration=duration,
            language=language,
            max_words_per_phrase=max_words_per_phrase,
            max_chars_per_phrase=max_chars_per_phrase,
        )
        return payload
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
