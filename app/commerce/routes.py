"""Merchant orders API (owner only; identical 404 for missing and other stores' orders)."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.auth.dependencies import get_current_user_id, require_store_member
from app.commerce import order_actions as actions
from app.commerce import repository as repo
from app.commerce.order_status import refresh_platform_orders
from app.commerce.models import CaseStatus, Order, OrderStatus, RejectReason, SupportCase
from app.inbox import repository as inbox_repo

router = APIRouter(prefix="/stores/{store_id}/orders", tags=["orders"])
cases_router = APIRouter(prefix="/stores/{store_id}/cases", tags=["support cases"])


class OrderList(BaseModel):
    orders: list[Order]


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shipping_fee: float | None = Field(default=None, ge=0, description="Required when shipping is set by you.")


class RejectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: RejectReason
    message: str = Field(default="", max_length=500, description="Optional message sent to the customer.")


def _run(fn, *args, **kwargs) -> dict:
    try:
        return fn(*args, **kwargs)
    except actions.OrderNotFound:
        raise HTTPException(status_code=404, detail="Order not found")
    except actions.OrderActionError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": exc.message})


@router.get("", response_model=OrderList)
def list_orders(store_id: str = Depends(require_store_member),
                status: OrderStatus | None = None,
                limit: int = Query(default=50, ge=1, le=100)) -> dict:
    return {"orders": repo.list_orders(store_id, status=status, limit=limit)}


class RefreshResult(BaseModel):
    checked: int
    changed: int
    failed: int


@router.post("/refresh-platform", response_model=RefreshResult)
def refresh_platform(store_id: str = Depends(require_store_member)) -> dict:
    """Refresh fulfilment status from Shopify/WooCommerce for open orders (max 50 per call)."""
    return refresh_platform_orders(store_id)


@router.get("/{order_id}", response_model=Order)
def get_order(order_id: str, store_id: str = Depends(require_store_member)) -> dict:
    order = repo.get_order(store_id, order_id)
    if order is None:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@router.post("/{order_id}/approve", response_model=Order)
def approve(order_id: str, body: ApproveRequest, store_id: str = Depends(require_store_member),
            user_id: str = Depends(get_current_user_id)) -> dict:
    return _run(actions.approve_order, store_id, order_id, user_id=user_id, shipping_fee=body.shipping_fee)


@router.post("/{order_id}/reject", response_model=Order)
def reject(order_id: str, body: RejectRequest, store_id: str = Depends(require_store_member),
           user_id: str = Depends(get_current_user_id)) -> dict:
    return _run(actions.reject_order, store_id, order_id, user_id=user_id, reason=body.reason, message=body.message)


@router.post("/{order_id}/retry-push", response_model=Order)
def retry_push(order_id: str, store_id: str = Depends(require_store_member)) -> dict:
    return _run(actions.retry_push, store_id, order_id)


@router.post("/{order_id}/ship", response_model=Order)
def ship(order_id: str, store_id: str = Depends(require_store_member),
         user_id: str = Depends(get_current_user_id)) -> dict:
    return _run(actions.mark_shipped, store_id, order_id, user_id=user_id)


@router.post("/{order_id}/deliver", response_model=Order)
def deliver(order_id: str, store_id: str = Depends(require_store_member),
            user_id: str = Depends(get_current_user_id)) -> dict:
    return _run(actions.mark_delivered, store_id, order_id, user_id=user_id)


# ---------------------------------------------------------------- support cases

class CaseList(BaseModel):
    cases: list[SupportCase]


class CaseStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: CaseStatus


@cases_router.get("", response_model=CaseList)
def list_cases(store_id: str = Depends(require_store_member), status: CaseStatus | None = None) -> dict:
    return {"cases": repo.list_cases(store_id, status=status)}


@cases_router.get("/{case_id}", response_model=SupportCase)
def get_case(case_id: str, store_id: str = Depends(require_store_member)) -> dict:
    case = repo.get_case(store_id, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@cases_router.post("/{case_id}/status", response_model=SupportCase)
def set_case_status(case_id: str, body: CaseStatusRequest, store_id: str = Depends(require_store_member)) -> dict:
    if not repo.set_case_status(store_id, case_id, body.status):
        raise HTTPException(status_code=404, detail="Case not found")
    case = repo.get_case(store_id, case_id)
    cid = case.get("conversation_id")
    if cid and body.status in ("resolved", "closed") and repo.find_open_case(store_id, cid) is None:
        inbox_repo.set_needs_attention(store_id, cid, False)
    return case