from __future__ import annotations

import inspect
import json
import os
import time
import traceback
import uuid
from typing import Any, Callable

import pika
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@127.0.0.1:5672/%2F")
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "86400"))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _job_key(service: str, job_id: str) -> str:
    return f"job:{service}:{job_id}"


def _queue_name(service: str) -> str:
    return f"{service}.jobs"


def _redis_client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _rabbit_connection() -> pika.BlockingConnection:
    return pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))


def ping_infra() -> dict[str, Any]:
    status = {"redis": False, "rabbitmq": False}
    try:
        status["redis"] = bool(_redis_client().ping())
    except Exception:
        status["redis"] = False
    try:
        connection = _rabbit_connection()
        connection.close()
        status["rabbitmq"] = True
    except Exception:
        status["rabbitmq"] = False
    return status


def enqueue_job(service: str, task: str, payload: dict[str, Any]) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "service": service,
        "task": task,
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "payload": payload,
        "created_at": _now(),
        "updated_at": _now(),
        "result": None,
        "error": None,
    }

    redis_client = _redis_client()
    redis_client.setex(_job_key(service, job_id), JOB_TTL_SECONDS, json.dumps(job))

    connection = _rabbit_connection()
    try:
        channel = connection.channel()
        channel.queue_declare(queue=_queue_name(service), durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=_queue_name(service),
            body=json.dumps(job),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )
    finally:
        connection.close()

    return job


def get_job(service: str, job_id: str) -> dict[str, Any] | None:
    raw = _redis_client().get(_job_key(service, job_id))
    if raw is None:
        return None
    return json.loads(raw)


def update_job(
    service: str,
    job_id: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    result: Any = None,
    error: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    job = get_job(service, job_id) or {
        "job_id": job_id,
        "service": service,
        "created_at": _now(),
    }
    if status is not None:
        job["status"] = status
    if progress is not None:
        job["progress"] = max(0, min(100, progress))
    if message is not None:
        job["message"] = message
    if result is not None:
        job["result"] = result
    if error is not None:
        job["error"] = error
    if extra:
        job.update(extra)
    if status in {"completed", "failed"} and "completed_at" not in job:
        job["completed_at"] = _now()
    job["updated_at"] = _now()
    _redis_client().setex(_job_key(service, job_id), JOB_TTL_SECONDS, json.dumps(job))
    return job


def make_progress_updater(service: str, job_id: str) -> Callable[[int, str], None]:
    def _update(progress: int, message: str, **extra: Any) -> None:
        update_job(service, job_id, progress=progress, message=message, extra=extra)

    return _update


def run_worker(
    service: str,
    handler: Callable[[dict[str, Any], Callable[[int, str], None]], Any],
) -> None:
    handler_params = inspect.signature(handler).parameters
    pass_job = len(handler_params) >= 3
    connection = _rabbit_connection()
    channel = connection.channel()
    queue_name = _queue_name(service)
    channel.queue_declare(queue=queue_name, durable=True)
    channel.basic_qos(prefetch_count=1)

    def _on_message(ch, method, properties, body) -> None:
        raw = json.loads(body)
        job_id = raw["job_id"]
        payload = raw["payload"]
        progress = make_progress_updater(service, job_id)
        update_job(service, job_id, status="running", progress=5, message="Worker started")
        try:
            if pass_job:
                result = handler(payload, progress, raw)
            else:
                result = handler(payload, progress)
            update_job(
                service,
                job_id,
                status="completed",
                progress=100,
                message="Completed",
                result=result,
                error=None,
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as exc:
            update_job(
                service,
                job_id,
                status="failed",
                progress=100,
                message="Failed",
                error={
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                },
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=queue_name, on_message_callback=_on_message)
    print(f"[worker] listening on {queue_name}")
    channel.start_consuming()
