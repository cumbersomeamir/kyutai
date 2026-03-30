from __future__ import annotations

import os
from pathlib import Path

from common.job_queue import run_worker
from whisper_service.runtime import TranscribeUrlRequest, transcribe_uploaded_media, transcribe_url_request


def handle_job(payload: dict, progress) -> dict:
    task = payload.get("task")
    if payload.get("sourceUrl"):
        progress(10, "Preparing URL transcription")
        return transcribe_url_request(TranscribeUrlRequest(**payload), progress=progress)

    stored_path = payload.get("stored_path")
    if stored_path:
        progress(10, "Preparing uploaded media")
        try:
            return transcribe_uploaded_media(
                Path(stored_path),
                language=payload.get("language", "en"),
                max_words_per_phrase=int(payload.get("max_words_per_phrase", 6)),
                max_chars_per_phrase=int(payload.get("max_chars_per_phrase", 28)),
                clip_id=payload.get("clip_id", ""),
                clip_label=payload.get("clip_label", ""),
                progress=progress,
            )
        finally:
            if os.path.exists(stored_path):
                os.unlink(stored_path)

    raise ValueError("Unsupported Whisper job payload.")


if __name__ == "__main__":
    run_worker("whisper", handle_job)
