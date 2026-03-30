from __future__ import annotations

import os
from pathlib import Path

from common.job_queue import run_worker
from app import (
    JOB_AUDIO_DIR,
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    MAX_TOKEN_PER_CHUNK,
    _generate_audio_bytes,
    _resolve_model_state,
    _settings_from_payload,
)


def handle_job(payload: dict, progress, job: dict) -> dict:
    settings = _settings_from_payload(payload.get("settings") or {})
    progress(15, "Resolving voice state")
    voice_wav_path = payload.get("voice_wav_path")
    voice_state_path = payload.get("voice_state_path")
    try:
        model_state = _resolve_model_state(
            settings=settings,
            voice=payload.get("voice"),
            voice_url=payload.get("voice_url"),
            voice_path=payload.get("voice_path"),
            voice_wav=None,
            voice_state=None,
            voice_wav_path=voice_wav_path,
            voice_state_path=voice_state_path,
            truncate_voice_prompt=bool(payload.get("truncate_voice_prompt", True)),
        )
        progress(55, "Generating audio")
        audio_bytes = _generate_audio_bytes(
            settings=settings,
            text_to_generate=payload["text"],
            model_state=model_state,
            max_tokens=int(payload.get("max_tokens", MAX_TOKEN_PER_CHUNK)),
            frames_after_eos=payload.get("frames_after_eos"),
        )
        output_path = JOB_AUDIO_DIR / f"{job['job_id']}.wav"
        output_path.write_bytes(audio_bytes)
        progress(90, "Audio ready")
        return {
            "filename": "generated_speech.wav",
            "audio_path": str(output_path),
            "size_bytes": len(audio_bytes),
            "settings": {
                "temperature": float(payload.get("settings", {}).get("temperature", DEFAULT_TEMPERATURE)),
                "lsd_decode_steps": int(payload.get("settings", {}).get("lsd_decode_steps", DEFAULT_LSD_DECODE_STEPS)),
                "noise_clamp": payload.get("settings", {}).get("noise_clamp", DEFAULT_NOISE_CLAMP),
                "eos_threshold": float(payload.get("settings", {}).get("eos_threshold", DEFAULT_EOS_THRESHOLD)),
            },
            "download_url": f"/api/v1/jobs/{job['job_id']}/audio",
        }
    finally:
        for temp_path in (voice_wav_path, voice_state_path):
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    run_worker("tts", handle_job)
