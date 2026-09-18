"""
FastAPI application — HTTP entrypoint for the agent.

WHY THIS IS A SEPARATE FILE FROM main.py:
main.py is the CLI entrypoint (a dev/testing tool). This is the HTTP
entrypoint. Both are thin wrappers around the SAME orchestrator —
handle_message() is called identically by both. Keeping them in separate
files means running the API never depends on CLI code and vice versa.

WHY SYNC ENDPOINTS (def, not async def):
The pipeline underneath is synchronous (pymongo, local transformer models,
blocking HTTP to OpenRouter). FastAPI runs plain `def` endpoints in a
threadpool automatically, so blocking code doesn't freeze the server.
This lets us expose the fully-tested sync pipeline over HTTP with ZERO
changes to any module below this layer. Converting to true async is a
later step, justified by real traffic — not now.

WHY THE SCHEMAS ARE REUSED AS-IS:
CustomerMessage and AgentReply are already Pydantic models. FastAPI uses
Pydantic natively for request and response bodies, so they plug straight
in — request validation (including all the input-validation rules we added)
happens automatically, and responses are serialized automatically.

CORS:
The widget will be loaded from many different store websites, each on
its own domain. Browsers block cross-origin requests unless the API
explicitly allows them. `allow_origins=["*"]` is fine for development;
tighten this to the real store domains before onboarding real customers.
"""




import logging
from fastapi import FastAPI , Request , Depends , UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.ingestion.csv_adapter import CSVAdapter
from app.ingestion.ingestion_service import ingest_products, IngestionResult
from app.settings.store_settings import get_industry
from app.core.orchestrator import handle_message
from app.logging_config import setup_logging
from app.schemas.models import AgentReply, CustomerMessage
from app.rate_limiter import rate_limit
import uuid
from app.log_context import set_request_context, clear_request_context
import sentry_sdk
from app.config import settings
import time
from fastapi.responses import JSONResponse
from app.metrics import record_request, get_metrics
from app.log_context import set_request_context, clear_request_context, get_request_context
from fastapi.responses import RedirectResponse
from app.settings.store_credentials import set_credentials
from app.schemas.models import WooCommerceCredentialsRequest
from app.ingestion.woocommerce_adapter import WooCommerceAdapter
from app.settings.store_credentials import get_credentials
from fastapi.responses import RedirectResponse
from app.oauth.shopify_oauth import (
    build_authorize_url,
    verify_state,
    verify_hmac,
    exchange_code_for_token,
)
from app.ingestion.shopify_adapter import ShopifyAdapter




if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.APP_ENV,
        send_default_pii=False,
        traces_sample_rate=0.1 if settings.APP_ENV == "production" else 1.0,
    )


# Configure logging once, at import time, before any request is served.
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="StoreFlow AI",
    description="Multi-tenant AI sales agent for retail stores.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect the bare domain to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """
    Per-request setup:
      - Assign a unique request_id (visible in logs and returned via header)
      - Clear context on the way out so nothing leaks to the next request

    Note: Metrics are recorded inside the /chat endpoint itself, not here,
    because store_id is only known once the request body is parsed by
    FastAPI — and contextvars set inside sync endpoints don't propagate
    back to this async middleware.
    """
    request_id = str(uuid.uuid4())
    set_request_context(request_id=request_id)

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        clear_request_context()



@app.get("/health")
def health() -> dict:
    """
    Liveness check. Returns 200 if the service is up.
    Used by load balancers, uptime monitors, and deployment health checks.
    Deliberately does NOT touch MongoDB or any model — it only answers
    "is the web server responding?", which is what a liveness probe needs.
    """
    return {"status": "ok"}

@app.get("/metrics")
def metrics() -> dict:
    """
    Basic operational metrics: request count, latency, error rate.
    Public (no auth) — same pattern as Prometheus. Contains only counts
    and latencies, never customer data.
    """
    return get_metrics()


@app.post("/chat", response_model=AgentReply)
def chat(request: Request, message: CustomerMessage, _rate_limit: None = Depends(rate_limit)) -> AgentReply:
    """
    Main endpoint: receive a customer message, return the agent's reply.
    """
    set_request_context(
        store_id=message.store_id,
        conversation_id=message.conversation_id,
    )
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



        
@app.post("/stores/{store_id}/products/upload", response_model=IngestionResult)
def upload_products(store_id: str, file: UploadFile = File(...)) -> IngestionResult:
    """
    Upload a CSV or Excel product catalog for a store.
    Upserts by product_id. Returns summary with created/updated/failed
    counts and any unmapped columns.
    """
    set_request_context(store_id=store_id)
    logger.info("Product upload received", extra={"upload_filename": file.filename})

    # Store must have an industry configured before ingestion.
    industry_id = get_industry(store_id)
    if industry_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Store '{store_id}' has no industry set. "
                f"Configure the industry before uploading products."
            ),
        )

    # Validate file type
    filename = (file.filename or "").lower()
    if not filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(
            status_code=400,
            detail="File must be .csv, .xlsx, or .xls",
        )

    # Read bytes into memory (small-file assumption for MVP)
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        adapter = CSVAdapter(file_bytes, filename)
        result = ingest_products(adapter, store_id, industry_id)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print("=== INGESTION ERROR ===")
        print(tb)
        print("=== END ===")

        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    logger.info(
    "Upload complete",
    extra={
        "upload_created": result.created,
        "upload_updated": result.updated,
        "upload_failed": result.failed,
    },
    )
    return result



@app.post("/stores/{store_id}/credentials/woocommerce")
def save_woocommerce_credentials(
    store_id: str,
    creds: WooCommerceCredentialsRequest,
) -> dict:
    """
    Save WooCommerce credentials for a store.
    Validates HTTPS enforcement and required fields.
    Overwrites any existing WC credentials for this store.
    """
    set_request_context(store_id=store_id)
    logger.info("WC credentials save requested")

    try:
        set_credentials(
            store_id,
            "woocommerce",
            creds.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Failed to save WC credentials")
        raise HTTPException(status_code=500, detail="Failed to save credentials")

    logger.info("WC credentials saved successfully")
    return {"status": "saved", "store_id": store_id, "source": "woocommerce"}



@app.post("/stores/{store_id}/sync/woocommerce", response_model=IngestionResult)
def sync_woocommerce(store_id: str) -> IngestionResult:
    """
    Trigger a WooCommerce sync for this store.
    Requires:
      - Store has an industry configured (POST /stores/... first)
      - WC credentials saved (POST /stores/.../credentials/woocommerce)
    Returns ingestion summary (created / updated / failed).
    """
    set_request_context(store_id=store_id)
    logger.info("WC sync triggered")

    # Industry required
    industry_id = get_industry(store_id)
    if industry_id is None:
        raise HTTPException(
            status_code=400,
            detail=f"Store '{store_id}' has no industry set. Configure the industry first.",
        )

    # Credentials required
    creds = get_credentials(store_id, "woocommerce")
    if creds is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No WooCommerce credentials for store '{store_id}'. "
                f"POST them to /stores/{store_id}/credentials/woocommerce first."
            ),
        )

    try:
        adapter = WooCommerceAdapter()
        result = ingest_products(adapter, store_id, industry_id)
    except Exception as exc:
        logger.exception("WC sync failed")
        raise HTTPException(status_code=500, detail=f"WC sync failed: {type(exc).__name__}: {exc}")

    logger.info(
        "WC sync complete",
        extra={
            "sync_created": result.created,
            "sync_updated": result.updated,
            "sync_failed": result.failed,
        },
    )
    return result


@app.get("/oauth/shopify/install")
def shopify_install(shop: str, store_id: str) -> RedirectResponse:
    """
    Start Shopify OAuth. Merchant clicks 'Connect Shopify' on our site,
    passing their shop domain. We redirect them to Shopify authorize.
    """
    set_request_context(store_id=store_id)
    logger.info("Shopify install initiated", extra={"shop": shop})

    if not shop.endswith(".myshopify.com"):
        raise HTTPException(status_code=400, detail="Invalid shop domain")

    url = build_authorize_url(shop, store_id)
    return RedirectResponse(url=url)




@app.get("/oauth/shopify/callback")
def shopify_callback(request: Request) -> dict:
    """..."""
    # Extract ALL query params (Shopify signs the complete set)
    params = dict(request.query_params)

    code = params.get("code")
    shop = params.get("shop")
    state = params.get("state")
    hmac_value = params.pop("hmac", "")  # remove hmac from dict for verification

    if not (code and shop and state and hmac_value):
        raise HTTPException(status_code=400, detail="Missing required params")

    # 1. Verify state (CSRF protection)
    state_data = verify_state(state)
    if state_data is None:
        raise HTTPException(status_code=400, detail="Invalid or expired state")

    store_id = state_data["store_id"]
    set_request_context(store_id=store_id)

    if state_data["shop"] != shop:
        raise HTTPException(status_code=400, detail="Shop mismatch")

    # 2. Verify HMAC against ALL params (except hmac itself, already popped)
    if not verify_hmac(params, hmac_value):
        raise HTTPException(status_code=400, detail="Invalid HMAC")

    # 3. Exchange code for permanent access token
    try:
        access_token = exchange_code_for_token(shop, code)
    except Exception as exc:
        logger.exception("Token exchange failed")
        raise HTTPException(status_code=500, detail=f"Token exchange failed: {exc}")

    # 4. Save credentials
    set_credentials(store_id, "shopify", {
        "shop": shop,
        "access_token": access_token,
    })

    logger.info("Shopify OAuth completed", extra={"oauth_shop": shop})
    return {"status": "installed", "shop": shop, "store_id": store_id}


@app.post("/stores/{store_id}/sync/shopify", response_model=IngestionResult)
def sync_shopify(store_id: str) -> IngestionResult:
    """
    Trigger Shopify sync for a store.
    Requires:
      - Store has industry set
      - Shopify OAuth completed (credentials in store_credentials)
    """
    set_request_context(store_id=store_id)
    logger.info("Shopify sync triggered")

    industry_id = get_industry(store_id)
    if industry_id is None:
        raise HTTPException(
            status_code=400,
            detail=f"Store '{store_id}' has no industry set.",
        )

    creds = get_credentials(store_id, "shopify")
    if creds is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No Shopify credentials for '{store_id}'. "
                f"Complete OAuth first via /oauth/shopify/install."
            ),
        )

    try:
        adapter = ShopifyAdapter()
        result = ingest_products(adapter, store_id, industry_id)
    except Exception as exc:
        logger.exception("Shopify sync failed")
        raise HTTPException(
            status_code=500,
            detail=f"Shopify sync failed: {type(exc).__name__}: {exc}",
        )

    logger.info(
        "Shopify sync complete",
        extra={
            "sync_created": result.created,
            "sync_updated": result.updated,
            "sync_failed": result.failed,
        },
    )
    return result