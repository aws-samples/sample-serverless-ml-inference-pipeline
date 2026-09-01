"""Stateful serverless ML inference handler backed by Amazon S3 Files.

The ONNX model is read directly from the S3 Files NFS mount (default
``/mnt/models``) using memory-mapped file I/O. ONNX Runtime memory-maps the
model file when it is loaded from a path on the mounted volume, so the model is
never copied to ``/tmp`` and there is no per-invocation download.

The ``InferenceSession`` is stored in a module-level variable so it persists
across warm Lambda invocations; the model is therefore loaded at most once per
execution environment.

Optional inference-result caching uses the shared S3 Files mount, so a result
computed by one invocation can be reused by any other concurrent or subsequent
invocation (cross-invocation shared state).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit

# --------------------------------------------------------------------------- #
# Configuration (read once at import time)
# --------------------------------------------------------------------------- #
MODEL_MOUNT_PATH = os.environ.get("MODEL_MOUNT_PATH", "/mnt/models")
MODEL_FILE = os.environ.get("MODEL_FILE", "model.onnx")
MODEL_NAME = os.environ.get("MODEL_NAME", "default")
MODEL_PATH = os.path.join(MODEL_MOUNT_PATH, MODEL_FILE)

CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(MODEL_MOUNT_PATH, "cache"))
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))

MODEL_LOAD_MAX_ATTEMPTS = int(os.environ.get("MODEL_LOAD_MAX_ATTEMPTS", "3"))
MODEL_LOAD_BASE_DELAY = float(os.environ.get("MODEL_LOAD_BASE_DELAY_SECONDS", "0.5"))

logger = Logger()
tracer = Tracer()
metrics = Metrics()

# --------------------------------------------------------------------------- #
# Module-level session cache (persists across warm invocations)
# --------------------------------------------------------------------------- #
_SESSION: Any = None
_INPUT_NAME: Optional[str] = None
_MODEL_LOAD_MS: float = 0.0


class ValidationError(ValueError):
    """Raised when the incoming request payload is invalid."""


@tracer.capture_method
def _load_session() -> Tuple[Any, str, float]:
    """Load the ONNX InferenceSession from the S3 Files mount.

    ONNX Runtime memory-maps the model file when loading from a path, avoiding a
    copy into ``/tmp``. Transient NFS mount errors (which occur almost
    exclusively on the first invocation in a new execution environment) are
    retried with exponential back-off.
    """
    import onnxruntime as ort  # imported lazily so unit tests can inject a session

    last_error: Optional[Exception] = None
    for attempt in range(1, MODEL_LOAD_MAX_ATTEMPTS + 1):
        try:
            start = time.perf_counter()
            session_options = ort.SessionOptions()
            # Read the weights via mmap instead of loading them all into RAM.
            session_options.enable_mem_pattern = True
            session = ort.InferenceSession(
                MODEL_PATH,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            load_ms = (time.perf_counter() - start) * 1000.0
            input_name = session.get_inputs()[0].name
            logger.info(
                "Loaded model",
                extra={
                    "model_path": MODEL_PATH,
                    "model_load_ms": round(load_ms, 2),
                    "attempt": attempt,
                    "input_name": input_name,
                },
            )
            return session, input_name, load_ms
        except FileNotFoundError:
            # Missing model is not transient - fail fast with a clear message.
            raise
        except OSError as exc:  # transient NFS mount / IO error
            last_error = exc
            if attempt == MODEL_LOAD_MAX_ATTEMPTS:
                break
            delay = MODEL_LOAD_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Transient error loading model; retrying",
                extra={"attempt": attempt, "delay_seconds": delay, "error": str(exc)},
            )
            time.sleep(delay)

    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {last_error}")


def get_session() -> Tuple[Any, str]:
    """Return the cached InferenceSession, loading it on the first call."""
    global _SESSION, _INPUT_NAME, _MODEL_LOAD_MS
    if _SESSION is None:
        _SESSION, _INPUT_NAME, _MODEL_LOAD_MS = _load_session()
    return _SESSION, _INPUT_NAME  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Result caching on the shared S3 Files mount
# --------------------------------------------------------------------------- #
def _cache_key(input_data: Any) -> str:
    """Deterministic SHA-256 key over the model identity and the input."""
    payload = json.dumps(
        {"model": MODEL_NAME, "file": MODEL_FILE, "input": input_data},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def read_cache(key: str) -> Optional[Dict[str, Any]]:
    """Return a cached result if present and within its TTL, else ``None``."""
    if not CACHE_ENABLED:
        return None
    path = _cache_path(key)
    try:
        age = time.time() - os.path.getmtime(path)
        if age > CACHE_TTL_SECONDS:
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cache read failed", extra={"error": str(exc), "key": key})
        return None


def write_cache(key: str, result: Dict[str, Any]) -> None:
    """Best-effort write of a result to the shared mount.

    The mount may be read-only or the execution role may lack
    ``s3files:ClientWrite``; in that case caching is skipped, not fatal.
    """
    if not CACHE_ENABLED:
        return
    path = _cache_path(key)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        os.replace(tmp, path)  # atomic within the mount
    except OSError as exc:
        logger.warning(
            "Cache write skipped (mount read-only or missing s3files:ClientWrite)",
            extra={"error": str(exc), "key": key},
        )


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@tracer.capture_method
def run_inference(input_data: List[Any]) -> Dict[str, Any]:
    """Run the model over ``input_data`` and return predictions."""
    session, input_name = get_session()

    array = np.asarray(input_data, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)

    outputs = session.run(None, {input_name: array})
    primary = outputs[0]
    return {
        "predictions": np.asarray(primary).tolist(),
        "raw_outputs": [np.asarray(o).tolist() for o in outputs],
        "input_shape": list(array.shape),
    }


# --------------------------------------------------------------------------- #
# Event handling
# --------------------------------------------------------------------------- #
def _is_api_gateway_event(event: Dict[str, Any]) -> bool:
    return isinstance(event, dict) and (
        "requestContext" in event or "httpMethod" in event or "routeKey" in event
    )


def _parse_payload(event: Dict[str, Any], is_api: bool) -> Dict[str, Any]:
    """Normalize API Gateway / direct-invoke / Step Functions payloads."""
    if is_api:
        body = event.get("body")
        if body is None:
            return {}
        if event.get("isBase64Encoded"):
            import base64

            body = base64.b64decode(body).decode("utf-8")
        try:
            return json.loads(body) if isinstance(body, str) else dict(body)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Request body is not valid JSON: {exc}")

    if not isinstance(event, dict):
        return {}

    # When invoked from the API Gateway -> Step Functions integration, the HTTP
    # request body is wrapped under a "body" key (alongside "path"/"querystring")
    # rather than being the payload directly. Unwrap it unless the event is
    # already in the direct {"input_data": ...} shape (plain invoke / SFN task
    # that passed the body through).
    if "input_data" not in event and "body" in event:
        body = event["body"]
        if isinstance(body, str):
            try:
                return json.loads(body)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"Request body is not valid JSON: {exc}")
        if isinstance(body, dict):
            return body

    return dict(event)


def _validate(payload: Dict[str, Any]) -> List[Any]:
    if "input_data" not in payload:
        raise ValidationError("Missing required field 'input_data'.")
    input_data = payload["input_data"]
    if not isinstance(input_data, list) or len(input_data) == 0:
        raise ValidationError("'input_data' must be a non-empty array.")
    return input_data


def _api_response(status: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


@metrics.log_metrics(capture_cold_start_metric=True)
@logger.inject_lambda_context
@tracer.capture_lambda_handler
def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    is_api = _is_api_gateway_event(event)
    metrics.add_dimension(name="model", value=MODEL_NAME)
    request_start = time.perf_counter()

    try:
        payload = _parse_payload(event, is_api)
        input_data = _validate(payload)

        key = _cache_key(input_data)
        cached = read_cache(key)
        if cached is not None:
            metrics.add_metric(name="CacheHit", unit=MetricUnit.Count, value=1)
            latency_ms = (time.perf_counter() - request_start) * 1000.0
            metrics.add_metric(
                name="InferenceLatencyMs", unit=MetricUnit.Milliseconds, value=latency_ms
            )
            result = {
                **cached,
                "model": MODEL_NAME,
                "cache_hit": True,
                "latency_ms": round(latency_ms, 3),
                "request_id": getattr(context, "aws_request_id", None),
            }
            return _api_response(200, result) if is_api else result

        metrics.add_metric(name="CacheHit", unit=MetricUnit.Count, value=0)

        # Cold model load time is recorded once per execution environment.
        was_cold = _SESSION is None
        prediction = run_inference(input_data)
        if was_cold:
            metrics.add_metric(
                name="ModelLoadTimeMs",
                unit=MetricUnit.Milliseconds,
                value=_MODEL_LOAD_MS,
            )

        latency_ms = (time.perf_counter() - request_start) * 1000.0
        metrics.add_metric(
            name="InferenceLatencyMs", unit=MetricUnit.Milliseconds, value=latency_ms
        )

        result = {
            **prediction,
            "model": MODEL_NAME,
            "cache_hit": False,
            "latency_ms": round(latency_ms, 3),
            "request_id": getattr(context, "aws_request_id", None),
        }
        write_cache(key, {k: result[k] for k in ("predictions", "raw_outputs", "input_shape")})
        return _api_response(200, result) if is_api else result

    except ValidationError as exc:
        logger.warning("Validation error", extra={"error": str(exc)})
        metrics.add_metric(name="ValidationError", unit=MetricUnit.Count, value=1)
        if is_api:
            return _api_response(400, {"error": str(exc)})
        raise
    except Exception as exc:  # noqa: BLE001 - surface as 5xx / SFN failure
        logger.exception("Inference failed")
        metrics.add_metric(name="InferenceError", unit=MetricUnit.Count, value=1)
        if is_api:
            return _api_response(500, {"error": "Internal inference error."})
        raise
