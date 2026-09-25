"""Merchant orders API (owner only; identical 404 for missing and other stores' orders)."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.auth.dependencies import get_current_user_id, require_store_member
from app.commerce import order_actions as actions
from app.commerce import repository as repo
from app.commerce.models import Order, OrderStatus, RejectReason

router = APIRouter(prefix="/stores/{store_id}/orders", tags=["orders"])


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