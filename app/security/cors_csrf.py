"""
CORS + CSRF protection.

Two CORS policies:
- /widget/*  : any origin (embedded on merchants' websites), NO credentials
               (the widget authenticates with its own bearer token)
- everything else (dashboard API): only DASHBOARD_ORIGINS, WITH credentials

CSRF (defense in depth on top of SameSite=Lax cookies):
- State-changing requests (POST/PUT/PATCH/DELETE) carrying an Origin that
  is not allowed are rejected with 403 — including /auth/login.
- Requests without an Origin header (scripts, server-to-server) pass:
  browsers always send Origin on cross-site state-changing requests.
- Widget and Meta webhook paths are exempt (no cookies; webhooks are
  signature-verified).
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from app.config import settings

_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_WEBHOOK_PATHS = {"/instagram/webhook", "/messenger/webhook", "/whatsapp/webhook"}
_WIDGET_PREFIX = "/widget/"
_MAX_AGE = "600"


def dashboard_origins() -> set[str]:
    raw = getattr(settings, "DASHBOARD_ORIGINS", "") or ""
    return {o.strip().rstrip("/") for o in raw.split(",") if o.strip()}


def _widget_headers(preflight: bool) -> dict[str, str]:
    headers = {"Access-Control-Allow-Origin": "*"}
    if preflight:
        headers.update({
            "Access-Control-Allow-Methods": "GET, POST",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": _MAX_AGE,
        })
    return headers


def _dashboard_headers(origin: str, preflight: bool) -> dict[str, str]:
    headers = {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Credentials": "true",
        "Vary": "Origin",
    }
    if preflight:
        headers.update({
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE",
            "Access-Control-Allow-Headers": "Content-Type, Last-Event-ID",
            "Access-Control-Max-Age": _MAX_AGE,
        })
    return headers


class CorsCsrfMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        origin = request.headers.get("origin")
        allowed = dashboard_origins()
        is_widget = path.startswith(_WIDGET_PREFIX)
        is_webhook = path in _WEBHOOK_PATHS

        # CORS preflight
        if request.method == "OPTIONS" and "access-control-request-method" in request.headers:
            if is_widget:
                return Response(status_code=204, headers=_widget_headers(preflight=True))
            if origin in allowed:
                return Response(status_code=204, headers=_dashboard_headers(origin, preflight=True))
            return PlainTextResponse("CORS origin not allowed", status_code=403)

        # CSRF: a browser on an unknown site may not change state here
        if (
            request.method in _UNSAFE_METHODS
            and not is_widget
            and not is_webhook
            and origin is not None
            and origin not in allowed
        ):
            return PlainTextResponse("Cross-site request blocked", status_code=403)

        response = await call_next(request)
        if is_widget:
            response.headers.update(_widget_headers(preflight=False))
        elif origin in allowed:
            response.headers.update(_dashboard_headers(origin, preflight=False))
        return response