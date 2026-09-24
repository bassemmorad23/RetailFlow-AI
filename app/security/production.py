"""Production lockdown: API docs off in production, /metrics behind a token."""

import hmac

from fastapi import Header, HTTPException

from app.config import settings


def is_production() -> bool:
    return settings.APP_ENV == "production"


def docs_kwargs(app_env: str) -> dict:
    """FastAPI docs settings: disabled in production."""
    if app_env == "production":
        return {"docs_url": None, "redoc_url": None, "openapi_url": None}
    return {"docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json"}


def require_metrics_access(x_metrics_token: str | None = Header(default=None)) -> None:
    """404 (not 401) on failure so the endpoint's existence isn't revealed."""
    expected = getattr(settings, "METRICS_TOKEN", "") or ""
    if not expected:
        if is_production():
            raise HTTPException(status_code=404, detail="Not Found")
        return  # development convenience
    if not x_metrics_token or not hmac.compare_digest(x_metrics_token, expected):
        raise HTTPException(status_code=404, detail="Not Found")