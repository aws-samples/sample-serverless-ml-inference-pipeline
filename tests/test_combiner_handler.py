"""Unit tests for the ensemble combiner handler."""
import pytest


def _member(model, probs, latency):
    return {
        "prediction": {
            "model": model,
            "predictions": [probs],
            "latency_ms": latency,
            "cache_hit": False,
        }
    }


def test_average_strategy(load_combiner, lambda_context):
    mod = load_combiner()
    event = {
        "strategy": "average",
        "models": [
            _member("model_a", [0.1, 0.2, 0.7], 10.0),
            _member("model_b", [0.3, 0.3, 0.4], 20.0),
        ],
    }
    result = mod.handler(event, lambda_context)
    assert result["ensemble_strategy"] == "average"
    assert result["model_count"] == 2
    assert result["prediction"]["label"] == 2
    assert result["prediction"]["probabilities"][2] == round((0.7 + 0.4) / 2, 6)


def test_majority_strategy(load_combiner, lambda_context):
    mod = load_combiner()
    event = {
        "strategy": "majority",
        "models": [
            _member("model_a", [0.1, 0.2, 0.7], 5.0),  # argmax 2
            _member("model_b", [0.8, 0.1, 0.1], 5.0),  # argmax 0
            _member("model_c", [0.2, 0.3, 0.5], 5.0),  # argmax 2
        ],
    }
    result = mod.handler(event, lambda_context)
    assert result["prediction"]["label"] == 2


def test_inverse_latency_strategy(load_combiner, lambda_context):
    mod = load_combiner()
    event = {
        "strategy": "inverse_latency",
        "models": [
            _member("fast", [0.9, 0.05, 0.05], 1.0),  # low latency -> high weight
            _member("slow", [0.0, 0.0, 1.0], 100.0),  # high latency -> low weight
        ],
    }
    result = mod.handler(event, lambda_context)
    assert result["prediction"]["label"] == 0


def test_empty_members_raises(load_combiner, lambda_context):
    mod = load_combiner()
    with pytest.raises(ValueError):
        mod.handler({"models": []}, lambda_context)
