"""
Order draft flow — runs once per customer message.

Deterministic: the LLM only extracts what the customer said. Items, prices,
stock, shipping and totals come from the system. The AI receives an
"order state" block describing exactly what to say / ask next.

- No draft: starts only on ready_to_buy, and only if shipping is configured.
- Active draft: always continues, whatever the intent classifier says.
- Saves are version-checked and retried on a concurrent change.
- Complete draft -> summary + confirmation hash (confirmation itself: C4).
"""

import hashlib
import json
import logging
from dataclasses import dataclass

from app.commerce import workflow
from app.commerce.customer_fields import apply_customer_details, missing_fields
from app.commerce.item_resolver import resolve_item
from app.commerce.models import MAX_ORDER_LINES, OrderDraft, OrderItem
from app.commerce.order_extractor import OrderExtraction, extract_order_details
from app.commerce.shipping import ShippingSettings, quote_shipping
from app.schemas.models import IntentLabel, IntentResult, ProductRecommendation
from app.settings.store_settings import get_settings, get_shipping

logger = logging.getLogger(__name__)

START_CONFIDENCE = 0.5
_SAVE_ATTEMPTS = 3
_FIELD_LABELS = {"name": "full name", "phone": "phone number", "region_code": "region / governorate",
                 "city": "city or area", "address_line": "delivery address"}

UNAVAILABLE_TEXT = (
    "ORDERING NOT AVAILABLE YET: the customer wants to order, but online ordering is not set up "
    "for this store. Tell them politely the store will contact them to complete the order. "
    "Do not collect order details and do not promise prices for shipping."
)
CANCELLED_TEXT = "ORDER CANCELLED: the customer cancelled the order in progress. Confirm briefly and offer further help."


@dataclass
class OrderTurn:
    event: str            # unavailable | cancelled | collecting_items | collecting_details | awaiting_confirmation
    state_text: str
    draft: OrderDraft | None = None


# ---------------------------------------------------------------- public

def process_order_turn(
    *,
    store_id: str,
    conversation_id: str,
    channel: str,
    customer_external_id: str,
    text: str,
    intent: IntentResult,
    recommendations: list[ProductRecommendation],
) -> OrderTurn | None:
    try:
        state = workflow.load_draft(store_id, conversation_id)
    except LookupError:
        return None  # not an inbox conversation (e.g. CLI)

    starting = state.draft is None
    if starting:
        if intent.label != IntentLabel.READY_TO_BUY or intent.confidence < START_CONFIDENCE:
            return None
        if get_shipping(store_id) is None:
            return OrderTurn("unavailable", UNAVAILABLE_TEXT)

    pending = state.draft.customer.region_suggestions if state.draft else []
    extraction = extract_order_details(text, pending_region_suggestions=pending)

    if extraction.cancel and not starting:
        workflow.clear_draft(store_id, conversation_id, expected_version=state.version)
        return OrderTurn("cancelled", CANCELLED_TEXT)

    settings = get_settings(store_id)
    shipping = get_shipping(store_id)
    country = settings.country or "EG"
    currency = settings.currency or "EGP"

    notes: list[str] = []
    for attempt in range(_SAVE_ATTEMPTS):
        if attempt:
            state = workflow.load_draft(store_id, conversation_id)
        draft = state.draft.model_copy(deep=True) if state.draft else OrderDraft()
        notes = _apply(draft, extraction, store_id=store_id, channel=channel, sender_id=customer_external_id,
                       country=country, recommendations=recommendations)
        summary = _finalize(draft, shipping, currency)
        if workflow.save_draft(store_id, conversation_id, draft, expected_version=state.version):
            return OrderTurn(draft.step, _state_text(draft, summary, notes, currency), draft)
        logger.info("Order draft changed concurrently; retrying", extra={"inbox_conversation": conversation_id})

    logger.warning("Order draft save failed after retries", extra={"inbox_conversation": conversation_id})
    return None


# ---------------------------------------------------------------- apply one message

def _apply(draft: OrderDraft, ex: OrderExtraction, *, store_id, channel, sender_id, country, recommendations) -> list[str]:
    notes: list[str] = []

    for request in ex.items:
        res = resolve_item(store_id, request, recommendations)
        name = res.product_name or request.get("product") or "that item"
        if res.status == "ok":
            if not _merge_item(draft, res.item):
                notes.append(f"The order already has {MAX_ORDER_LINES} lines; tell the customer the maximum.")
            elif res.stock == "unknown":
                notes.append(f"Availability of {name} will be confirmed by the store — say so.")
            elif res.stock == "low_stock":
                notes.append(f"Only a few {name} left — you may mention it.")
        elif res.status == "needs_variant":
            opts = "; ".join(f"{k}: {', '.join(v)}" for k, v in res.options.items())
            notes.append(f"Ask which option of {name} they want ({opts}). Offer only these options.")
        elif res.status == "out_of_stock":
            notes.append(f"{name} in the requested option is OUT OF STOCK — say so and offer alternatives. It was not added.")
        else:
            notes.append(f"Could not find '{request.get('product')}' in the catalog — ask the customer to clarify. Never invent products.")

    update = apply_customer_details(draft.customer, ex.customer, store_country=country,
                                    channel=channel, sender_id=sender_id)
    draft.customer = update.customer
    for problem in update.problems:
        if problem == "region_suggest":
            notes.append(f"Ask: did you mean {' or '.join(draft.customer.region_suggestions)}? Do not assume.")
        elif problem == "region_unknown":
            notes.append("The region was not recognised — ask the customer to write it again. Never guess it.")
        elif problem == "phone_invalid":
            notes.append("The phone number is not valid — ask for it again (with country code if abroad).")
        elif problem == "country_unknown":
            notes.append("The country was not recognised — ask again.")
    return notes


def _merge_item(draft: OrderDraft, item: OrderItem) -> bool:
    """Same product+variant again replaces that line (customer restating). False if the line limit is hit."""
    for i, line in enumerate(draft.items):
        if line.product_id == item.product_id and line.variant_sku == item.variant_sku:
            draft.items[i] = item
            return True
    if len(draft.items) >= MAX_ORDER_LINES:
        return False
    draft.items.append(item)
    return True


# ---------------------------------------------------------------- step + summary

def _finalize(draft: OrderDraft, shipping: ShippingSettings | None, currency: str) -> dict | None:
    if not draft.items:
        draft.step, draft.confirmation_hash = "collecting_items", None
        return None
    if missing_fields(draft.customer):
        draft.step, draft.confirmation_hash = "collecting_details", None
        return None

    subtotal = round(sum(i.unit_price * i.quantity for i in draft.items), 2)
    quote = quote_shipping(shipping, country=draft.customer.country, region=draft.customer.region_code,
                           subtotal=subtotal)  # platform rate plugs in with C4
    total = round(subtotal + quote.fee, 2) if quote.fee is not None else None
    summary = {
        "items": [i.model_dump() for i in draft.items],
        "customer": draft.customer.model_dump(exclude={"region_suggestions", "phone_source"}),
        "currency": currency, "subtotal": subtotal,
        "shipping_fee": quote.fee, "shipping_status": quote.status, "total": total, "payment": "cod",
    }
    draft.step = "awaiting_confirmation"
    draft.confirmation_hash = hashlib.sha256(json.dumps(summary, sort_keys=True, default=str).encode()).hexdigest()
    return summary


def _money(value: float, currency: str) -> str:
    return f"{value:,.2f} {currency}"


def _item_line(i: OrderItem, currency: str) -> str:
    attrs = ", ".join(f"{k}: {v}" for k, v in i.variant_attrs.items())
    return f"- {i.quantity} × {i.name}{f' ({attrs})' if attrs else ''} — {_money(i.unit_price, currency)} each"


def _state_text(draft: OrderDraft, summary: dict | None, notes: list[str], currency: str) -> str:
    lines = ["ORDER IN PROGRESS — the system manages this order. Use ONLY these facts; never invent prices, "
             "stock, shipping fees or totals. Payment is cash on delivery."]
    lines.append("Items:" if draft.items else "Items: none yet.")
    lines += [_item_line(i, currency) for i in draft.items]

    c = draft.customer
    if draft.step == "awaiting_confirmation" and summary:
        lines.append(f"Subtotal: {_money(summary['subtotal'], currency)}")
        if summary["shipping_fee"] is not None:
            lines.append(f"Shipping: {_money(summary['shipping_fee'], currency)}")
            lines.append(f"Total: {_money(summary['total'], currency)}")
        else:
            lines.append("Shipping: the store will confirm the shipping fee and the final total. Do NOT state a total.")
        lines.append(f"Deliver to: {c.name}, {c.phone}, {c.address_line}, {c.city}, {c.region_name}, {c.country}"
                     + (f" (notes: {c.notes})" if c.notes else ""))
        lines.append("NEXT: show this summary clearly and ask the customer to confirm. The order is NOT placed "
                     "until they confirm — do not say it is placed.")
    else:
        collected = [f"{_FIELD_LABELS[f]}: {getattr(c, f) if f != 'region_code' else c.region_name}"
                     for f in _FIELD_LABELS if getattr(c, f)]
        if collected:
            lines.append("Collected: " + "; ".join(collected))
        if draft.step == "collecting_items":
            lines.append("NEXT: ask which product (and size/colour if relevant) and how many they want.")
        else:
            missing = [_FIELD_LABELS[f] for f in missing_fields(c)]
            lines.append("Missing: " + ", ".join(missing))
            lines.append("NEXT: ask for the missing details (one or two at a time).")
    if c.phone and c.phone_source == "channel":
        lines.append(f"The phone {c.phone} comes from their WhatsApp — ask them to confirm it's the right delivery number.")
    if notes:
        lines.append("Notes for this reply:")
        lines += [f"- {n}" for n in notes]
    return "\n".join(lines)