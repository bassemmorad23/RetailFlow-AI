"""
Merchant actions on orders.

approve : (manual shipping -> fee required) -> live stock re-check ->
          approved -> platform stores: create the order in Shopify/WooCommerce;
          the customer is told "confirmed" only after that succeeds.
          CSV stores: confirmed immediately.
reject  : reason (internal) + optional message -> polite customer message.
retry   : re-push an approved order whose platform push failed.
ship / deliver : CSV stores only (platform stores track this in the platform).
"""

import logging

from app.commerce import order_notify as notify
from app.commerce import repository as repo
from app.commerce.platform_orders import push_order
from app.commerce.stock import check_stock, store_platform

logger = logging.getLogger(__name__)


class OrderNotFound(Exception):
    pass


class OrderActionError(Exception):
    """Business rule violation; `code` is stable for the UI."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _get(store_id: str, order_id: str) -> dict:
    order = repo.get_order(store_id, order_id)
    if order is None:
        raise OrderNotFound(order_id)
    return order


def _transition(store_id: str, order_id: str, new_status: str, user_id: str) -> dict:
    try:
        return repo.change_order_status(store_id, order_id, new_status=new_status, by=user_id)
    except repo.InvalidTransition:
        raise OrderActionError("invalid_status", f"This order can't be moved to '{new_status}' from its current status.")


# ---------------------------------------------------------------- approve

def approve_order(store_id: str, order_id: str, *, user_id: str, shipping_fee: float | None = None) -> dict:
    order = _get(store_id, order_id)
    if order["status"] != "pending_approval":
        raise OrderActionError("invalid_status", "Only orders awaiting approval can be approved.")

    if order["shipping_status"] == "pending_merchant":
        if shipping_fee is None:
            raise OrderActionError("shipping_fee_required", "Enter the shipping fee to approve this order.")
        order = repo.set_shipping_fee(store_id, order_id, shipping_fee) or _get(store_id, order_id)

    checks = check_stock(store_id, [(i["product_id"], i["variant_sku"]) for i in order["items"]])
    sold_out = [i["name"] for i, c in zip(order["items"], checks) if c.level == "out_of_stock"]
    if sold_out:
        raise OrderActionError("out_of_stock", "Out of stock now: " + ", ".join(sold_out) +
                               ". Reject the order or contact the customer.")

    order = _transition(store_id, order_id, "approved", user_id)
    platform = store_platform(store_id)
    if platform is None:
        repo.set_push_state(store_id, order_id, "not_needed")
        notify.notify_customer(store_id, order, notify.approved_text(order))
        return _get(store_id, order_id)
    return _push(store_id, order, platform)


def retry_push(store_id: str, order_id: str) -> dict:
    order = _get(store_id, order_id)
    if order["status"] != "approved" or (order.get("push") or {}).get("status") != "failed":
        raise OrderActionError("nothing_to_retry", "Only approved orders with a failed platform push can be retried.")
    platform = store_platform(store_id)
    if platform is None:
        raise OrderActionError("no_platform", "This store has no connected platform.")
    return _push(store_id, order, platform)


def _push(store_id: str, order: dict, platform: str) -> dict:
    repo.set_push_state(store_id, order["id"], "pending", count_attempt=True)
    try:
        result = push_order(store_id, platform, order)
    except Exception as exc:  # never leave the order stuck in "pending"
        logger.exception("Platform push crashed")
        result = None
        error = type(exc).__name__
    else:
        error = result.error

    if result is not None and result.ok:
        repo.set_platform_ref(store_id, order["id"], platform=platform,
                              platform_order_id=result.platform_order_id,
                              platform_order_number=result.platform_order_number)
        repo.set_push_state(store_id, order["id"], "succeeded")
        fresh = _get(store_id, order["id"])
        notify.notify_customer(store_id, fresh, notify.approved_text(fresh))
        return fresh

    repo.set_push_state(store_id, order["id"], "failed", error=error or "unknown")
    logger.warning("Platform push failed", extra={"order_number": order["number"], "push_error": error})
    return _get(store_id, order["id"])


# ---------------------------------------------------------------- reject / ship / deliver

def reject_order(store_id: str, order_id: str, *, user_id: str, reason: str, message: str = "") -> dict:
    _get(store_id, order_id)
    order = _transition(store_id, order_id, "rejected", user_id)
    repo.set_rejection(store_id, order_id, reason, message)
    notify.notify_customer(store_id, order, notify.rejected_text(order, message))
    return _get(store_id, order_id)


def _csv_only(store_id: str) -> None:
    if store_platform(store_id) is not None:
        raise OrderActionError("managed_by_platform",
                               "Shipping and delivery are tracked in your Shopify/WooCommerce admin.")


def mark_shipped(store_id: str, order_id: str, *, user_id: str) -> dict:
    _get(store_id, order_id)
    _csv_only(store_id)
    order = _transition(store_id, order_id, "shipped", user_id)
    notify.notify_customer(store_id, order, notify.shipped_text(order))
    return _get(store_id, order_id)


def mark_delivered(store_id: str, order_id: str, *, user_id: str) -> dict:
    _get(store_id, order_id)
    _csv_only(store_id)
    _transition(store_id, order_id, "delivered", user_id)
    return _get(store_id, order_id)