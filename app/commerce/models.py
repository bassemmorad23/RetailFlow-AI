"""
Commerce models: orders, support cases, and the in-conversation order draft.

Rules baked in:
- Totals are always computed by the server from item prices, never taken
  from the AI or the client.
- COD only (pilot).
- Max 10 order lines.
- Money is stored in the store's currency, rounded to 2 decimals.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.models import Channel

OrderStatus = Literal["pending_approval", "approved", "rejected", "cancelled", "shipped", "delivered"]
PaymentMethod = Literal["cod"]
CaseType = Literal["return", "exchange", "refund", "complaint", "delivery_issue", "other"]
CaseStatus = Literal["open", "in_progress", "resolved", "closed"]
DraftStep = Literal["collecting_items", "collecting_details", "awaiting_confirmation"]

MAX_ORDER_LINES = 10
MAX_QUANTITY_PER_LINE = 20

# Allowed status transitions (anything else is rejected).
ORDER_TRANSITIONS: dict[str, set[str]] = {
    "pending_approval": {"approved", "rejected", "cancelled"},
    "approved": {"shipped", "cancelled"},
    "shipped": {"delivered"},
    "rejected": set(),
    "cancelled": set(),
    "delivered": set(),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Mongo docs carry internal fields


class OrderItem(_Strict):
    product_id: str = Field(min_length=1, max_length=200)
    variant_sku: str | None = Field(default=None, max_length=200)
    name: str = Field(min_length=1, max_length=300)
    variant_attrs: dict[str, str] = Field(default_factory=dict)
    quantity: int = Field(ge=1, le=MAX_QUANTITY_PER_LINE)
    unit_price: float = Field(ge=0)


class CustomerDetails(_Strict):
    name: str = Field(min_length=1, max_length=100)
    phone: str = Field(min_length=5, max_length=20)
    governorate: str = Field(min_length=1, max_length=50)
    city: str = Field(min_length=1, max_length=80)
    address_line: str = Field(min_length=1, max_length=300)
    notes: str = Field(default="", max_length=500)


class DraftCustomer(_Strict):
    """Partially collected details during the conversation."""
    name: str | None = None
    phone: str | None = None
    governorate: str | None = None
    city: str | None = None
    address_line: str | None = None
    notes: str | None = None


class OrderDraft(_Strict):
    step: DraftStep = "collecting_items"
    items: list[OrderItem] = Field(default_factory=list, max_length=MAX_ORDER_LINES)
    customer: DraftCustomer = Field(default_factory=DraftCustomer)
    confirmation_hash: str | None = None  # hash of the exact summary the customer was shown


class PlatformRef(_Loose):
    type: Literal["shopify", "woocommerce"]
    order_id: str
    order_number: str | None = None


class StatusChange(_Loose):
    status: OrderStatus
    at: datetime
    by: str  # user_id, "customer" or "system"


class Order(_Loose):
    id: str
    store_id: str
    number: str
    conversation_id: str | None = None
    channel: Channel
    customer_external_id: str
    customer: CustomerDetails
    items: list[OrderItem]
    currency: str
    subtotal: float
    shipping_fee: float
    total: float
    payment_method: PaymentMethod = "cod"
    status: OrderStatus
    status_history: list[StatusChange]
    platform: PlatformRef | None = None
    created_at: datetime
    updated_at: datetime


class SupportCase(_Loose):
    id: str
    store_id: str
    number: str
    conversation_id: str | None = None
    order_id: str | None = None
    type: CaseType
    description: str
    status: CaseStatus
    created_at: datetime
    updated_at: datetime