"""
FastAPI application — HTTP entrypoint for the agent.

WHY THIS IS A SEPARATE FILE FROM main.py:
main.py is the CLI entrypoint (a dev/testing tool). This is the HTTP
entrypoint. Both are thin wrappers around the SAME orchestrator.

WHY SYNC ENDPOINTS (def, not async def):
The pipeline is synchronous (pymongo, local models, blocking HTTP).
FastAPI runs plain `def` endpoints in a threadpool, so blocking code
doesn't freeze the server. Webhooks are the exception: they are `async`
only to read the body, then hand the (blocking) processing to a
BackgroundTask so Meta gets an immediate 200 and the event loop is
never blocked.

AUTH MODEL:
- Merchant endpoints (/stores/{store_id}/..., OAuth install) require a
  logged-in session AND store membership (require_store_member).
- OAuth callbacks verify the logged-in user owns the store in `state`.
- Webhooks, /chat, /health are public by design (called by Meta,
  customers, and monitors).

CORS:
Still wide open for the widget during development. Will be tightened
together with the CSRF origin check before the dashboard goes live.
"""

import logging
import time
import uuid

import sentry_sdk
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse

from app.auth.dependencies import get_current_user_id, require_store_member
from app.auth.repository import is_store_member
from app.auth.routes import router as auth_router
from app.channels.instagram_channel import send_dm
from app.channels.messenger_channel import send_message
from app.channels.whatsapp_channel import send_text
from app.config import settings
from app.core.orchestrator import handle_message
from app.ingestion.csv_adapter import CSVAdapter
from app.ingestion.ingestion_service import IngestionResult, ingest_products
from app.ingestion.shopify_adapter import ShopifyAdapter
from app.ingestion.woocommerce_adapter import WooCommerceAdapter
from app.log_context import clear_request_context, set_request_context
from app.logging_config import setup_logging
from app.metrics import get_metrics, record_request
from app.oauth.instagram_oauth import (
    build_authorize_url as ig_build_authorize_url,
    exchange_code_for_token as ig_exchange_code,
    exchange_for_long_lived_token as ig_exchange_long,
    get_account_info as ig_get_account_info,
)
from app.oauth.messenger_oauth import (
    build_authorize_url as msg_build_authorize_url,
    exchange_code_for_user_token as msg_exchange_code,
    get_user_pages as msg_get_pages,
)
from app.oauth.shopify_oauth import (
    build_authorize_url,
    exchange_code_for_token,
    verify_hmac,
    verify_state,
)
from app.rate_limiter import rate_limit
from app.schemas.models import AgentReply, CustomerMessage, WooCommerceCredentialsRequest
from app.settings.store_credentials import (
    find_store_by_fb_page,
    find_store_by_ig_account,
    get_credentials,
    set_credentials,
)
from app.settings.store_settings import get_industry
from app.stores.routes import router as stores_router


if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.APP_ENV,
        send_default_pii=False,
        traces_sample_rate=0.1 if settings.APP_ENV == "production" else 1.0,
    )

setup_logging()
logger = logging.getLogger(__name__)

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB catalog file limit

app = FastAPI(
    title="StoreFlow AI",
    description="Multi-tenant AI sales agent for retail stores.",
    version="0.1.0",
)

app.include_router(auth_router)
app.include_router(stores_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Assign a request_id per request; clear context on the way out."""
    request_id = str(uuid.uuid4())
    set_request_context(request_id=request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        clear_request_context()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_state_owner(request: Request, store_id: str) -> None:
    """
    OAuth callbacks: the logged-in user must own the store named in `state`.
    Prevents attaching an attacker's account to someone else's store.
    """
    user_id = get_current_user_id(request)  # 401 if not logged in
    if not is_store_member(user_id, store_id):
        raise HTTPException(status_code=404, detail="Store not found")


def _require_industry(store_id: str) -> str:
    industry_id = get_industry(store_id)
    if industry_id is None:
        raise HTTPException(
            status_code=400,
            detail="This store has no industry set. Configure the industry first.",
        )
    return industry_id


async def _read_json(request: Request) -> dict:
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    return payload if isinstance(payload, dict) else {}


# ---------------------------------------------------------------------------
# Public: root, health, metrics, chat
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    """Liveness check. Does not touch MongoDB or models."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    """Operational counts/latencies only. TODO: lock down before production."""
    return get_metrics()


@app.post("/chat", response_model=AgentReply)
def chat(request: Request, message: CustomerMessage, _rate_limit: None = Depends(rate_limit)) -> AgentReply:
    """Customer-facing chat (widget). Public by design, rate limited."""
    set_request_context(store_id=message.store_id, conversation_id=message.conversation_id)
    logger.info("Received message", extra={"channel": message.channel})

    started = time.monotonic()
    status_code = 200
    try:
        return handle_message(message)
    except Exception:
        status_code = 500
        raise
    finally:
        latency_ms = (time.monotonic() - started) * 1000
        record_request(status_code=status_code, store_id=message.store_id, latency_ms=latency_ms)


# ---------------------------------------------------------------------------
# Merchant: product ingestion (owner only)
# ---------------------------------------------------------------------------

@app.post(
    "/stores/{store_id}/products/upload",
    response_model=IngestionResult,
    dependencies=[Depends(require_store_member)],
)
def upload_products(store_id: str, file: UploadFile = File(...)) -> IngestionResult:
    """Upload a CSV/Excel catalog. Upserts by product_id."""
    set_request_context(store_id=store_id)
    logger.info("Product upload received", extra={"upload_filename": file.filename})

    industry_id = _require_industry(store_id)

    filename = (file.filename or "").lower()
    if not filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="File must be .csv, .xlsx, or .xls")

    file_bytes = file.file.read(_MAX_UPLOAD_BYTES + 1)
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

    try:
        result = ingest_products(CSVAdapter(file_bytes, filename), store_id, industry_id)
    except Exception:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail="Ingestion failed. Please check the file and try again.")

    logger.info("Upload complete", extra={
        "upload_created": result.created,
        "upload_updated": result.updated,
        "upload_failed": result.failed,
    })
    return result


@app.post("/stores/{store_id}/credentials/woocommerce", dependencies=[Depends(require_store_member)])
def save_woocommerce_credentials(store_id: str, creds: WooCommerceCredentialsRequest) -> dict:
    """Save WooCommerce credentials (overwrites existing)."""
    set_request_context(store_id=store_id)
    logger.info("WC credentials save requested")

    try:
        set_credentials(store_id, "woocommerce", creds.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Failed to save WC credentials")
        raise HTTPException(status_code=500, detail="Failed to save credentials")

    logger.info("WC credentials saved successfully")
    return {"status": "saved", "store_id": store_id, "source": "woocommerce"}


@app.post(
    "/stores/{store_id}/sync/woocommerce",
    response_model=IngestionResult,
    dependencies=[Depends(require_store_member)],
)
def sync_woocommerce(store_id: str) -> IngestionResult:
    """Trigger a WooCommerce sync."""
    set_request_context(store_id=store_id)
    logger.info("WC sync triggered")

    industry_id = _require_industry(store_id)
    if get_credentials(store_id, "woocommerce") is None:
        raise HTTPException(status_code=400, detail="WooCommerce is not connected for this store.")

    try:
        result = ingest_products(WooCommerceAdapter(), store_id, industry_id)
    except Exception:
        logger.exception("WC sync failed")
        raise HTTPException(status_code=502, detail="WooCommerce sync failed. Check the store connection and try again.")

    logger.info("WC sync complete", extra={
        "sync_created": result.created,
        "sync_updated": result.updated,
        "sync_failed": result.failed,
    })
    return result


# ---------------------------------------------------------------------------
# Shopify OAuth + sync
# ---------------------------------------------------------------------------

@app.get("/oauth/shopify/install", dependencies=[Depends(require_store_member)])
def shopify_install(shop: str, store_id: str) -> RedirectResponse:
    """Start Shopify OAuth (owner only)."""
    set_request_context(store_id=store_id)
    logger.info("Shopify install initiated", extra={"shop": shop})

    if not shop.endswith(".myshopify.com"):
        raise HTTPException(status_code=400, detail="Invalid shop domain")

    return RedirectResponse(url=build_authorize_url(shop, store_id))


@app.get("/oauth/shopify/callback")
def shopify_callback(request: Request) -> dict:
    """
    Shopify redirects here. Protected by server-stored random state + HMAC.
    (Install is owner-only, so a valid state always belongs to the owner.)
    """
    params = dict(request.query_params)
    code = params.get("code")
    shop = params.get("shop")
    state = params.get("state")
    hmac_value = params.pop("hmac", "")

    if not (code and shop and state and hmac_value):
        raise HTTPException(status_code=400, detail="Missing required params")

    state_data = verify_state(state)
    if state_data is None:
        raise HTTPException(status_code=400, detail="Invalid or expired state")

    store_id = state_data["store_id"]
    set_request_context(store_id=store_id)

    if state_data["shop"] != shop:
        raise HTTPException(status_code=400, detail="Shop mismatch")
    if not verify_hmac(params, hmac_value):
        raise HTTPException(status_code=400, detail="Invalid HMAC")

    try:
        access_token = exchange_code_for_token(shop, code)
    except Exception:
        logger.exception("Shopify token exchange failed")
        raise HTTPException(status_code=502, detail="Could not complete Shopify connection. Please try again.")

    set_credentials(store_id, "shopify", {"shop": shop, "access_token": access_token})
    logger.info("Shopify OAuth completed", extra={"oauth_shop": shop})
    return {"status": "installed", "shop": shop, "store_id": store_id}


@app.post(
    "/stores/{store_id}/sync/shopify",
    response_model=IngestionResult,
    dependencies=[Depends(require_store_member)],
)
def sync_shopify(store_id: str) -> IngestionResult:
    """Trigger a Shopify sync."""
    set_request_context(store_id=store_id)
    logger.info("Shopify sync triggered")

    industry_id = _require_industry(store_id)
    if get_credentials(store_id, "shopify") is None:
        raise HTTPException(status_code=400, detail="Shopify is not connected for this store.")

    try:
        result = ingest_products(ShopifyAdapter(), store_id, industry_id)
    except Exception:
        logger.exception("Shopify sync failed")
        raise HTTPException(status_code=502, detail="Shopify sync failed. Check the store connection and try again.")

    logger.info("Shopify sync complete", extra={
        "sync_created": result.created,
        "sync_updated": result.updated,
        "sync_failed": result.failed,
    })
    return result


# ---------------------------------------------------------------------------
# Instagram
# ---------------------------------------------------------------------------

@app.get("/instagram/install", dependencies=[Depends(require_store_member)])
def instagram_install(store_id: str) -> RedirectResponse:
    """Start Instagram OAuth (owner only)."""
    set_request_context(store_id=store_id)
    logger.info("Instagram install initiated")
    return RedirectResponse(url=ig_build_authorize_url(store_id))


@app.get("/instagram/callback")
def instagram_callback(request: Request, code: str = "", state: str = "", error: str = "") -> dict:
    """Meta redirects here after approval. Logged-in owner of `state` store only."""
    if error:
        raise HTTPException(status_code=400, detail=f"Instagram OAuth error: {error}")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state")

    store_id = state
    set_request_context(store_id=store_id)
    _require_state_owner(request, store_id)

    try:
        short_token = ig_exchange_code(code)["access_token"]
        long_token = ig_exchange_long(short_token)["access_token"]
        info = ig_get_account_info(long_token)
    except Exception:
        logger.exception("Instagram OAuth failed")
        raise HTTPException(status_code=502, detail="Could not complete Instagram connection. Please try again.")

    set_credentials(store_id, "instagram", {
        "access_token": long_token,
        "instagram_business_account_id": info["id"],
        "username": info.get("username", ""),
    })

    logger.info("Instagram OAuth completed", extra={"ig_username": info.get("username")})
    return {"status": "installed", "store_id": store_id, "instagram_username": info.get("username")}


@app.get("/instagram/webhook")
def instagram_webhook_verify(request: Request) -> PlainTextResponse:
    """Meta webhook verification handshake."""
    params = dict(request.query_params)
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == settings.META_WEBHOOK_VERIFY_TOKEN
    ):
        logger.info("Instagram webhook verified")
        return PlainTextResponse(content=params.get("hub.challenge", ""), status_code=200)
    logger.warning("Instagram webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/instagram/webhook")
async def instagram_webhook_receive(request: Request, background_tasks: BackgroundTasks) -> dict:
    """Acknowledge immediately; process in the background (never blocks the event loop)."""
    payload = await _read_json(request)
    entries = payload.get("entry", [])
    logger.info("Instagram webhook event received", extra={"webhook_entries": len(entries)})
    background_tasks.add_task(_process_instagram_entries, entries)
    return {"status": "ok"}


def _process_instagram_entries(entries: list) -> None:
    for entry in entries:
        try:
            _process_instagram_entry(entry)
        except Exception:
            logger.exception("Failed to process IG entry")


def _process_instagram_entry(entry: dict) -> None:
    """Route inbound DMs through the orchestrator and send the reply."""
    for event in entry.get("messaging", []):
        message = event.get("message")
        if not message or message.get("is_echo"):
            continue

        text = message.get("text")
        sender_id = event.get("sender", {}).get("id")
        recipient_id = event.get("recipient", {}).get("id")
        if not (text and sender_id and recipient_id):
            continue

        store_id = find_store_by_ig_account(recipient_id)
        if store_id is None:
            logger.warning("No store for IG account", extra={"ig_recipient": recipient_id})
            continue

        reply = handle_message(CustomerMessage(
            store_id=store_id,
            conversation_id=f"ig_{sender_id}",
            text=text,
        ))
        send_dm(store_id, sender_id, reply.reply_text)


# ---------------------------------------------------------------------------
# Messenger
# ---------------------------------------------------------------------------

@app.get("/messenger/install", dependencies=[Depends(require_store_member)])
def messenger_install(store_id: str) -> RedirectResponse:
    """Start Messenger OAuth (owner only)."""
    set_request_context(store_id=store_id)
    return RedirectResponse(url=msg_build_authorize_url(store_id))


@app.get("/messenger/callback")
def messenger_callback(request: Request, code: str = "", state: str = "", error: str = "") -> dict:
    """Meta redirects here after approval. Logged-in owner of `state` store only."""
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code/state")

    store_id = state
    set_request_context(store_id=store_id)
    _require_state_owner(request, store_id)

    try:
        user_token = msg_exchange_code(code)
        pages = msg_get_pages(user_token)
    except Exception:
        logger.exception("Messenger OAuth failed")
        raise HTTPException(status_code=502, detail="Could not complete Messenger connection. Please try again.")

    if not pages:
        raise HTTPException(status_code=400, detail="No Facebook Pages found for this account")

    page = pages[0]  # TODO: let merchant pick when they manage several Pages
    set_credentials(store_id, "messenger", {
        "page_id": page["id"],
        "page_name": page["name"],
        "page_access_token": page["access_token"],
    })

    logger.info("Messenger OAuth completed", extra={"page_name": page["name"]})
    return {"status": "installed", "store_id": store_id, "page_name": page["name"]}


@app.get("/messenger/webhook")
def messenger_webhook_verify(request: Request) -> PlainTextResponse:
    """Meta webhook verification handshake."""
    params = dict(request.query_params)
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == settings.META_WEBHOOK_VERIFY_TOKEN
    ):
        logger.info("Messenger webhook verified")
        return PlainTextResponse(content=params.get("hub.challenge", ""), status_code=200)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/messenger/webhook")
async def messenger_webhook_receive(request: Request, background_tasks: BackgroundTasks) -> dict:
    """Acknowledge immediately; process in the background."""
    payload = await _read_json(request)
    entries = payload.get("entry", [])
    logger.info("Messenger webhook event received", extra={"webhook_entries": len(entries)})
    background_tasks.add_task(_process_messenger_entries, entries)
    return {"status": "ok"}


def _process_messenger_entries(entries: list) -> None:
    for entry in entries:
        try:
            _process_messenger_entry(entry)
        except Exception:
            logger.exception("Failed to process Messenger entry")


def _process_messenger_entry(entry: dict) -> None:
    page_id = entry.get("id")
    store_id = find_store_by_fb_page(page_id) if page_id else None
    if not store_id:
        logger.warning("No store for Page", extra={"fb_page": page_id})
        return

    for event in entry.get("messaging", []):
        message = event.get("message")
        if not message or message.get("is_echo"):
            continue
        text = message.get("text")
        sender_id = event.get("sender", {}).get("id")
        if not (text and sender_id):
            continue

        reply = handle_message(CustomerMessage(
            store_id=store_id,
            conversation_id=f"fb_{sender_id}",
            text=text,
        ))
        send_message(store_id, sender_id, reply.reply_text)


# ---------------------------------------------------------------------------
# WhatsApp
# ---------------------------------------------------------------------------

@app.get("/whatsapp/webhook")
def whatsapp_webhook_verify(request: Request) -> PlainTextResponse:
    """Meta webhook verification handshake."""
    params = dict(request.query_params)
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == settings.META_WEBHOOK_VERIFY_TOKEN
    ):
        logger.info("WhatsApp webhook verified")
        return PlainTextResponse(content=params.get("hub.challenge", ""), status_code=200)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/whatsapp/webhook")
async def whatsapp_webhook_receive(request: Request, background_tasks: BackgroundTasks) -> dict:
    """Acknowledge immediately; process in the background."""
    payload = await _read_json(request)
    entries = payload.get("entry", [])
    logger.info("WhatsApp webhook event received", extra={"webhook_entries": len(entries)})
    background_tasks.add_task(_process_whatsapp_entries, entries)
    return {"status": "ok"}


def _process_whatsapp_entries(entries: list) -> None:
    for entry in entries:
        try:
            _process_whatsapp_entry(entry)
        except Exception:
            logger.exception("Failed to process WhatsApp entry")


def _process_whatsapp_entry(entry: dict) -> None:
    for change in entry.get("changes", []):
        value = change.get("value", {})
        for msg in value.get("messages", []):
            if msg.get("type") != "text":
                continue
            sender_phone = msg.get("from")
            text = msg.get("text", {}).get("body", "")
            if not (sender_phone and text):
                continue

            # TODO: multi-tenant via phone_number_id lookup (single test store for now)
            store_id = "store_wa_test"

            reply = handle_message(CustomerMessage(
                store_id=store_id,
                conversation_id=f"wa_{sender_phone}",
                text=text,
            ))
            send_text(sender_phone, reply.reply_text)