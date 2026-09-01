"""Shared pytest fixtures.

The two Lambda handlers are both named ``handler.py``; they are loaded from their
file paths under unique module names so they can be imported side by side and
re-imported with different environment configuration per test.
"""
import importlib.util
import os
import sys
import types

import pytest

# Make Powertools safe/quiet outside of a real Lambda environment.
os.environ.setdefault("POWERTOOLS_TRACE_DISABLED", "true")
os.environ.setdefault("POWERTOOLS_METRICS_NAMESPACE", "ServerlessMLInferenceTest")
os.environ.setdefault("POWERTOOLS_SERVICE_NAME", "test")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_HANDLER = os.path.join(ROOT, "lambdas", "inference", "handler.py")
COMBINER_HANDLER = os.path.join(ROOT, "lambdas", "combiner", "handler.py")


def _load(module_name: str, path: str, env: dict):
    for key, value in env.items():
        os.environ[key] = str(value)
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load_inference():
    def _factory(**env):
        return _load("inference_handler_under_test", INFERENCE_HANDLER, env)

    return _factory


@pytest.fixture
def load_combiner():
    def _factory(**env):
        return _load("combiner_handler_under_test", COMBINER_HANDLER, env)

    return _factory


@pytest.fixture
def lambda_context():
    ctx = types.SimpleNamespace(
        function_name="test-fn",
        function_version="$LATEST",
        invoked_function_arn="arn:aws:lambda:us-east-1:123456789012:function:test-fn",
        memory_limit_in_mb=2048,
        aws_request_id="test-request-id",
        log_group_name="/aws/lambda/test-fn",
        log_stream_name="2026/01/01/[$LATEST]abcdef",
    )
    ctx.get_remaining_time_in_millis = lambda: 30000
    return ctx
