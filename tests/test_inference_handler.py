"""Unit tests for the inference handler (ONNX session is faked)."""
import json

import numpy as np
import pytest


class FakeSession:
    """Minimal stand-in for onnxruntime.InferenceSession."""

    def __init__(self, output):
        self._output = output

    def run(self, output_names, feed):
        return [np.asarray(self._output, dtype=np.float32)]

    def get_inputs(self):
        return [types_input()]


def types_input():
    obj = type("Inp", (), {})()
    obj.name = "input"
    return obj


def _inject(module, output=None):
    module._SESSION = FakeSession(output or [[0.1, 0.2, 0.7]])
    module._INPUT_NAME = "input"


def test_direct_invoke_returns_predictions(load_inference, lambda_context):
    mod = load_inference(CACHE_ENABLED="false", MODEL_NAME="predict")
    _inject(mod)
    result = mod.handler({"input_data": [5.1, 3.5, 1.4, 0.2]}, lambda_context)
    assert result["model"] == "predict"
    assert result["cache_hit"] is False
    # ONNX float32 outputs widen when converted to Python floats, so compare
    # with tolerance rather than exact equality.
    assert result["predictions"][0] == pytest.approx([0.1, 0.2, 0.7])
    assert result["input_shape"] == [1, 4]
    assert "latency_ms" in result


def test_apigw_event_returns_proxy_response(load_inference, lambda_context):
    mod = load_inference(CACHE_ENABLED="false")
    _inject(mod)
    event = {"requestContext": {}, "body": json.dumps({"input_data": [1, 2, 3, 4]})}
    resp = mod.handler(event, lambda_context)
    assert resp["statusCode"] == 200
    assert resp["headers"]["Content-Type"] == "application/json"
    body = json.loads(resp["body"])
    assert body["predictions"][0] == pytest.approx([0.1, 0.2, 0.7])


def test_missing_input_data_direct_raises(load_inference, lambda_context):
    mod = load_inference(CACHE_ENABLED="false")
    _inject(mod)
    with pytest.raises(mod.ValidationError):
        mod.handler({"foo": "bar"}, lambda_context)


def test_missing_input_data_apigw_returns_400(load_inference, lambda_context):
    mod = load_inference(CACHE_ENABLED="false")
    _inject(mod)
    resp = mod.handler({"requestContext": {}, "body": json.dumps({})}, lambda_context)
    assert resp["statusCode"] == 400
    assert "error" in json.loads(resp["body"])


def test_invalid_input_type_apigw_returns_400(load_inference, lambda_context):
    mod = load_inference(CACHE_ENABLED="false")
    _inject(mod)
    resp = mod.handler(
        {"requestContext": {}, "body": json.dumps({"input_data": "not-a-list"})},
        lambda_context,
    )
    assert resp["statusCode"] == 400


def test_cache_miss_then_hit(load_inference, lambda_context, tmp_path):
    cache_dir = tmp_path / "cache"
    mod = load_inference(
        CACHE_ENABLED="true",
        CACHE_DIR=str(cache_dir),
        CACHE_TTL_SECONDS="3600",
        MODEL_MOUNT_PATH=str(tmp_path),
        MODEL_NAME="predict",
    )
    _inject(mod)
    first = mod.handler({"input_data": [1, 2, 3, 4]}, lambda_context)
    assert first["cache_hit"] is False
    second = mod.handler({"input_data": [1, 2, 3, 4]}, lambda_context)
    assert second["cache_hit"] is True
    assert second["predictions"] == first["predictions"]


def test_stepfunctions_wrapped_body_object(load_inference, lambda_context):
    """API Gateway -> Step Functions wraps the body under a 'body' key (object)."""
    mod = load_inference(CACHE_ENABLED="false")
    _inject(mod)
    event = {"body": {"input_data": [1, 2, 3, 4]}, "path": {}, "querystring": {}}
    result = mod.handler(event, lambda_context)
    assert result["cache_hit"] is False
    assert result["predictions"][0] == pytest.approx([0.1, 0.2, 0.7])


def test_stepfunctions_wrapped_body_string(load_inference, lambda_context):
    """The wrapped 'body' may also arrive as a JSON string."""
    mod = load_inference(CACHE_ENABLED="false")
    _inject(mod)
    event = {"body": json.dumps({"input_data": [1, 2, 3, 4]})}
    result = mod.handler(event, lambda_context)
    assert result["predictions"][0] == pytest.approx([0.1, 0.2, 0.7])
