"""Production lockdown: docs toggle and /metrics protection."""

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.config import settings
from app.security.production import docs_kwargs


def test_docs_disabled_in_production():
    assert docs_kwargs("production") == {"docs_url": None, "redoc_url": None, "openapi_url": None}
    assert docs_kwargs("prototype")["openapi_url"] == "/openapi.json"


def _metrics(token=None):
    headers = {"X-Metrics-Token": token} if token else {}
    return TestClient(app).get("/metrics", headers=headers).status_code


def test_metrics_open_in_dev_without_token(monkeypatch):
    monkeypatch.setattr(settings, "APP_ENV", "prototype")
    monkeypatch.setattr(settings, "METRICS_TOKEN", "")
    assert _metrics() == 200


def test_metrics_hidden_in_production_without_configured_token(monkeypatch):
    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "METRICS_TOKEN", "")
    assert _metrics() == 404
    assert _metrics("anything") == 404


@pytest.mark.parametrize("env", ["production", "prototype"])
def test_metrics_token_enforced_when_configured(monkeypatch, env):
    monkeypatch.setattr(settings, "APP_ENV", env)
    monkeypatch.setattr(settings, "METRICS_TOKEN", "s3cret-token")
    assert _metrics() == 404
    assert _metrics("wrong") == 404
    assert _metrics("s3cret-token") == 200