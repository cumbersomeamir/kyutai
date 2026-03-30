from __future__ import annotations

import os
from pathlib import Path

from common.job_queue import run_worker
from qwen_vl_service.runtime import DEFAULT_MAX_TOKENS, DEFAULT_PROMPT, DEFAULT_TEMPERATURE, run_image_to_text, run_url_image_to_text


def handle_job(payload: dict, progress) -> dict:
    prompt = payload.get("prompt") or DEFAULT_PROMPT
    max_tokens = int(payload.get("max_tokens", DEFAULT_MAX_TOKENS))
    temperature = float(payload.get("temperature", DEFAULT_TEMPERATURE))

    if payload.get("stored_path"):
        stored_path = payload["stored_path"]
        try:
            progress(15, "Preparing uploaded image")
            return run_image_to_text(
                Path(stored_path),
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                progress=progress,
            )
        finally:
            if os.path.exists(stored_path):
                os.unlink(stored_path)

    if payload.get("image_path"):
        progress(15, "Preparing local image")
        return run_image_to_text(
            Path(payload["image_path"]),
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            progress=progress,
        )

    if payload.get("image_url"):
        progress(15, "Preparing image URL")
        return run_url_image_to_text(
            payload["image_url"],
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            progress=progress,
        )

    raise ValueError("Unsupported Qwen VL job payload.")


if __name__ == "__main__":
    run_worker("qwen_vl", handle_job)
