"""
Create an approved order in the store's platform (Shopify / WooCommerce).

Part 2b fills in the real platform calls. Until then every push fails
cleanly with "not_implemented", so approvals on platform stores are
visible as failed pushes (with Retry) instead of silently succeeding.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PushResult:
    ok: bool
    platform_order_id: str | None = None
    platform_order_number: str | None = None
    error: str | None = None


def push_order(store_id: str, platform: str, order: dict) -> PushResult:
    return PushResult(ok=False, error="not_implemented")