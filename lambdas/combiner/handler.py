"""Ensemble combiner for the multi-model inference workflow.

Invoked by the Step Functions ``CombineResults`` state with the outputs of the
parallel model branches. Supports three ensemble strategies:

* ``average``          - element-wise mean of the class-probability vectors.
* ``majority``         - majority vote over each member's argmax label.
* ``inverse_latency``  - probability average weighted by ``1 / latency_ms`` so
                         faster (typically warm / cached) models count for more.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List

import numpy as np
from aws_lambda_powertools import Logger, Metrics
from aws_lambda_powertools.metrics import MetricUnit

logger = Logger()
metrics = Metrics()

DEFAULT_STRATEGY = os.environ.get("ENSEMBLE_STRATEGY", "average")


def _probability_vector(prediction: Dict[str, Any]) -> np.ndarray:
    """Extract a 1-D probability vector from a model member's output."""
    preds = prediction.get("predictions")
    if preds is None:
        raise ValueError("Member is missing 'predictions'.")
    array = np.asarray(preds, dtype=np.float64)
    if array.ndim == 2:  # e.g. [[p0, p1, p2]] -> [p0, p1, p2]
        array = array[0]
    return array.reshape(-1)


def _members(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Unwrap the Parallel-state output into a list of model predictions."""
    raw = event.get("models", [])
    members = []
    for item in raw:
        # ResultSelector wraps each branch result as {"prediction": <payload>}.
        members.append(item.get("prediction", item) if isinstance(item, dict) else item)
    return members


def _average(vectors: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(vectors), axis=0)


def _inverse_latency(vectors: List[np.ndarray], latencies: List[float]) -> np.ndarray:
    weights = np.array([1.0 / max(lat, 1e-6) for lat in latencies], dtype=np.float64)
    weights = weights / weights.sum()
    return np.average(np.vstack(vectors), axis=0, weights=weights)


@metrics.log_metrics
@logger.inject_lambda_context
def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    strategy = event.get("strategy", DEFAULT_STRATEGY)
    members = _members(event)
    if not members:
        raise ValueError("No model predictions provided to combiner.")

    vectors = [_probability_vector(m) for m in members]
    latencies = [float(m.get("latency_ms", 1.0)) for m in members]

    if strategy == "majority":
        labels = [int(np.argmax(v)) for v in vectors]
        label = Counter(labels).most_common(1)[0][0]
        probabilities = _average(vectors)  # reported for reference
    elif strategy == "inverse_latency":
        probabilities = _inverse_latency(vectors, latencies)
        label = int(np.argmax(probabilities))
    else:  # default: average
        strategy = "average"
        probabilities = _average(vectors)
        label = int(np.argmax(probabilities))

    metrics.add_metric(name="EnsembleInvocation", unit=MetricUnit.Count, value=1)
    metrics.add_metric(
        name="EnsembleMemberCount", unit=MetricUnit.Count, value=len(members)
    )

    result = {
        "ensemble_strategy": strategy,
        "model_count": len(members),
        "prediction": {
            "label": label,
            "probabilities": [round(float(p), 6) for p in probabilities.tolist()],
        },
        "members": [
            {
                "model": m.get("model"),
                "label": int(np.argmax(v)),
                "latency_ms": lat,
                "cache_hit": m.get("cache_hit"),
            }
            for m, v, lat in zip(members, vectors, latencies)
        ],
    }
    logger.info("Ensemble complete", extra={"strategy": strategy, "label": label})
    return result
