"""
Commerce Intelligence API (owner only). Every endpoint accepts:
  period=today|7d|30d|custom, from=YYYY-MM-DD, to=YYYY-MM-DD (custom, max 90 days)
and returns {"period": {...}, "currency": "...", ...section}.
"""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from app.analytics import metrics
from app.analytics.period import Period, PeriodError, resolve_period
from app.auth.dependencies import require_store_member
from app.settings.store_settings import get_settings

router = APIRouter(prefix="/stores/{store_id}/analytics", tags=["analytics"])


class _Ctx:
    def __init__(self, store_id: str, period: Period, currency: str):
        self.store_id, self.period, self.currency = store_id, period, currency


def _ctx(
    store_id: str = Depends(require_store_member),
    period: str = Query(default="7d", pattern="^(today|7d|30d|custom)$"),
    from_: date | None = Query(default=None, alias="from"),
    to: date | None = Query(default=None),
) -> _Ctx:
    s = get_settings(store_id)
    try:
        p = resolve_period(period, from_, to, s.country)
    except PeriodError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _Ctx(store_id, p, s.currency or "EGP")


def _wrap(ctx: _Ctx, section: dict) -> dict:
    return {"period": ctx.period.describe(), "currency": ctx.currency, **section}


@router.get("/overview")
def overview(ctx: _Ctx = Depends(_ctx)) -> dict:
    return _wrap(ctx, metrics.overview(ctx.store_id, ctx.period))


@router.get("/sales")
def sales(ctx: _Ctx = Depends(_ctx)) -> dict:
    return _wrap(ctx, metrics.sales(ctx.store_id, ctx.period))


@router.get("/ai")
def ai(ctx: _Ctx = Depends(_ctx)) -> dict:
    return _wrap(ctx, metrics.ai(ctx.store_id, ctx.period))


@router.get("/channels")
def channels(ctx: _Ctx = Depends(_ctx)) -> dict:
    return _wrap(ctx, metrics.channels(ctx.store_id, ctx.period))


@router.get("/products")
def products(ctx: _Ctx = Depends(_ctx)) -> dict:
    return _wrap(ctx, metrics.products(ctx.store_id, ctx.period))


@router.get("/customers")
def customers(ctx: _Ctx = Depends(_ctx)) -> dict:
    return _wrap(ctx, metrics.customers(ctx.store_id, ctx.period))


@router.get("/support")
def support(ctx: _Ctx = Depends(_ctx)) -> dict:
    return _wrap(ctx, metrics.support(ctx.store_id, ctx.period))


@router.get("/orders")
def orders(ctx: _Ctx = Depends(_ctx)) -> dict:
    return _wrap(ctx, metrics.orders_ops(ctx.store_id, ctx.period))