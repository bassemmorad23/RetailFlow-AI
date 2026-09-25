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
import re
from dataclasses import dataclass

from app.commerce import workflow
from app.commerce.customer_fields import apply_customer_details, missing_fields, to_customer_details
from app.commerce.repository import create_order
from app.commerce.platform_shipping import shopify_shipping_rates
from app.commerce.stock import check_stock, store_platform
from app.inbox import events
from app.inbox import repository as inbox_repo
from app.products.product_store import fetch_products
from app.commerce.item_resolver import resolve_item
from app.commerce.models import MAX_ORDER_LINES, OrderDraft, OrderItem
from app.commerce.order_extractor import OrderExtraction, extract_order_details
from app.commerce.shipping import ShippingSettings, quote_shipping
from app.schemas.models import IntentLabel, IntentResult, ProductRecommendation
from app.settings.store_settings import get_settings, get_shipping

logger = logging.getLogger(__name__)

START_CONFIDENCE = 0.5

_BUY_PHRASES = re.compile(
    r"\b(buy|purchase|i'?ll take|want to order|like to order|place an order|order (it|this|one|them|\d+)|checkout)\b"
    r"|اشتري|أشتري|هشتري|هاشتري|اطلب|أطلب|هاخد|هاخده|عايز اخد|\b(ashtery|ashteri|a4tery|hakhod|a5od)\b",
    re.IGNORECASE,
)
_NEVER_START = {IntentLabel.ORDER_STATUS, IntentLabel.COMPLAINT}
_SAVE_ATTEMPTS = 3
_FIELD_LABELS = {"name": "full name", "phone": "phone number", "region_code": "region / governorate",
                 "city": "city or area", "address_line": "delivery address"}

UNAVAILABLE_TEXT = (
    "ORDERING NOT AVAILABLE YET: the customer wants to order, but online ordering is not set up "
    "for this store. Tell them politely the store will contact them to complete the order. "
    "Do not collect order details and do not promise prices for shipping."
)
CANCELLED_TEXT = "ORDER CANCELLED: the customer cancelled the order in progress. Confirm briefly and offer further help."


_CONFIRM_WORDS = {
    "yes", "y", "yeah", "yep", "ok", "okay", "sure", "confirm", "confirmed", "confirm order", "go ahead", "done",
    "تمام", "اكد", "أكد", "اكيد", "أكيد", "موافق", "ايوه", "أيوه", "اه", "نعم", "تم", "ماشي", "اوكي",
    "tamam", "akid", "aywa", "mashy",
}


@dataclass
class OrderTurn:
    event: str            # unavailable | cancelled | collecting_items | collecting_details | awaiting_confirmation | order_created
    state_text: str
    draft: OrderDraft | None = None
    order: dict | None = None


# ---------------------------------------------------------------- public

def wants_to_start(intent: IntentResult, text: str) -> bool:
    if intent.label in _NEVER_START:
        return False
    if intent.label == IntentLabel.READY_TO_BUY and intent.confidence >= START_CONFIDENCE:
        return True
    return bool(_BUY_PHRASES.search(text or ""))


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
        if not wants_to_start(intent, text):
            return None
        if get_shipping(store_id) is None:
            return OrderTurn("unavailable", UNAVAILABLE_TEXT)

    pending = state.draft.customer.region_suggestions if state.draft else []
    offered = ([f"{o.title} ({o.price:.2f})" for o in state.draft.shipping_options]
               if state.draft and state.draft.step == "choosing_shipping" else None)
    extraction = extract_order_details(text, pending_region_suggestions=pending, shipping_options=offered)

    if extraction.cancel and not starting:
        workflow.clear_draft(store_id, conversation_id, expected_version=state.version)
        return OrderTurn("cancelled", CANCELLED_TEXT)

    settings = get_settings(store_id)
    shipping = get_shipping(store_id)
    country = settings.country or "EG"
    currency = settings.currency or "EGP"
    rates = None
    if shipping is not None and shipping.method == "platform" and store_platform(store_id) == "shopify":
        def rates(d):
            return shopify_shipping_rates(store_id, d.items, d.customer, currency)

    if state.draft and state.draft.step == "awaiting_confirmation" and _is_confirmation(extraction, text):
        return _confirm(store_id, conversation_id, state, channel=channel,
                        customer_external_id=customer_external_id, shipping=shipping, currency=currency, rates=rates)

    notes: list[str] = []
    for attempt in range(_SAVE_ATTEMPTS):
        if attempt:
            state = workflow.load_draft(store_id, conversation_id)
        draft = state.draft.model_copy(deep=True) if state.draft else OrderDraft()
        notes = _apply(draft, extraction, store_id=store_id, channel=channel, sender_id=customer_external_id,
                       country=country, recommendations=recommendations)
        summary = _finalize(draft, shipping, currency, rates)
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

    if ex.shipping_choice and draft.shipping_options:
        handle = _match_choice(ex.shipping_choice, draft.shipping_options)
        if handle:
            draft.shipping_choice = handle
        else:
            notes.append("It wasn't clear which shipping option they chose — ask again, listing the options.")

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

def _match_choice(choice: str, options: list) -> str | None:
    c = choice.strip().lower()

    if c.isdigit() and 1 <= int(c) <= len(options):
        return options[int(c) - 1].title

    for o in options:
        if o.title.strip().lower() == c:
            return o.title

    for o in options:
        if c in o.title.lower() or o.title.lower() in c:
            return o.title

    return None


def _finalize(draft: OrderDraft, shipping: ShippingSettings | None, currency: str, rates=None) -> dict | None:
    if not draft.items:
        draft.step, draft.confirmation_hash = "collecting_items", None
        return None
    if missing_fields(draft.customer):
        draft.step, draft.confirmation_hash = "collecting_details", None
        return None

    subtotal = round(sum(i.unit_price * i.quantity for i in draft.items), 2)
    fee, status, title = None, None, None

    options = rates(draft) if rates else None
    if options:
        draft.shipping_options = options
        if len(options) == 1:
            draft.shipping_choice = options[0].title
        chosen = next((o for o in options if o.title == draft.shipping_choice), None)
        if chosen is None:  # several options (or the chosen one disappeared) -> customer picks
            draft.shipping_choice, draft.step, draft.confirmation_hash = None, "choosing_shipping", None
            return None
        fee, status, title = chosen.price, "quoted", chosen.title
    else:
        draft.shipping_options, draft.shipping_choice = [], None
        quote = quote_shipping(shipping, country=draft.customer.country, region=draft.customer.region_code,
                               subtotal=subtotal)  # platform without rates -> configured fallback
        fee, status = quote.fee, quote.status

    total = round(subtotal + fee, 2) if fee is not None else None
    summary = {
        "items": [i.model_dump() for i in draft.items],
        "customer": draft.customer.model_dump(exclude={"region_suggestions", "phone_source"}),
        "currency": currency, "subtotal": subtotal,
        "shipping_fee": fee, "shipping_status": status, "shipping_title": title,
        "total": total, "payment": "cod",
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
            label = f"Shipping ({summary['shipping_title']})" if summary.get("shipping_title") else "Shipping"
            lines.append(f"{label}: {_money(summary['shipping_fee'], currency)}")
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
        elif draft.step == "choosing_shipping":
            lines.append("Shipping options:")
            lines += [f"{n}. {o.title} — {_money(o.price, currency)}" for n, o in enumerate(draft.shipping_options, 1)]
            lines.append("NEXT: list these shipping options with prices and ask which one they prefer.")
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


# ---------------------------------------------------------------- confirmation -> order

def _is_confirmation(ex: OrderExtraction, text: str) -> bool:
    """A pure 'yes'. Any new item or detail in the same message is a change, not a confirmation."""
    if ex.items or ex.customer or ex.cancel:
        return False
    if ex.confirm:
        return True
    words = " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())
    return words in _CONFIRM_WORDS


def _revalidate(store_id: str, draft: OrderDraft, currency: str) -> list[str]:
    """Current DB prices + live stock at the moment of confirmation."""
    notes: list[str] = []
    products = {p.product_id: p for p in fetch_products(store_id, list({i.product_id for i in draft.items}))}
    checks = check_stock(store_id, [(i.product_id, i.variant_sku) for i in draft.items])
    kept: list[OrderItem] = []
    for item, check in zip(draft.items, checks):
        product = products.get(item.product_id)
        variant = (next((v for v in product.variants if v.sku == item.variant_sku), None)
                   if product and item.variant_sku else None)
        if product is None or (item.variant_sku and variant is None):
            notes.append(f"{item.name} is no longer sold by the store — it was removed.")
            continue
        if check.level == "out_of_stock":
            notes.append(f"{item.name} just went out of stock — it was removed.")
            continue
        price = variant.price if variant else product.price
        if round(price, 2) != round(item.unit_price, 2):
            notes.append(f"The price of {item.name} changed to {_money(price, currency)}.")
            item = item.model_copy(update={"unit_price": price})
        kept.append(item)
    draft.items = kept
    return notes


def _confirm(store_id, conversation_id, state, *, channel, customer_external_id, shipping, currency,
             rates=None) -> OrderTurn | None:
    draft = state.draft.model_copy(deep=True)
    confirmed_hash = draft.confirmation_hash
    notes = _revalidate(store_id, draft, currency)
    summary = _finalize(draft, shipping, currency, rates)

    if summary is None or draft.confirmation_hash != confirmed_hash:
        notes.insert(0, "Something changed since the summary (price or availability). Explain it, show the "
                        "updated order, and ask the customer to confirm again. Nothing was ordered yet.")
        if not workflow.save_draft(store_id, conversation_id, draft, expected_version=state.version):
            return None
        return OrderTurn(draft.step, _state_text(draft, summary, notes, currency), draft)

    order = create_order(
        store_id, channel=channel, customer_external_id=customer_external_id,
        customer=to_customer_details(draft.customer), items=draft.items, currency=currency,
        shipping_fee=summary["shipping_fee"], conversation_id=conversation_id,
        idempotency_key=f"{conversation_id}:{confirmed_hash}",
        shipping_title=summary.get("shipping_title"),
    )
    workflow.clear_draft(store_id, conversation_id, expected_version=state.version)
    _notify_merchant(store_id, conversation_id, order)
    logger.info("Order created", extra={"inbox_conversation": conversation_id, "order_number": order["number"]})
    return OrderTurn("order_created", _placed_text(order), draft, order=order)


def _notify_merchant(store_id: str, conversation_id: str, order: dict) -> None:
    total = (_money(order["total"], order["currency"]) if order["total"] is not None
             else "shipping fee to be set by you")
    try:
        note = inbox_repo.add_message(store_id, conversation_id, "system",
                                      f"New order {order['number']} ({total}) — awaiting your approval",
                                      delivery_status="internal")
        if note:
            events.emit_message_created(store_id, note)
    except LookupError:
        pass
    events.emit(store_id, "order.created", conversation_id, {"order": {
        k: order.get(k) for k in ("id", "number", "status", "total", "currency", "shipping_status")}})


def _placed_text(order: dict) -> str:
    cur = order["currency"]
    lines = [f"ORDER PLACED: order number {order['number']}. Status: awaiting the store's approval. "
             "Payment: cash on delivery.", "Items:"]
    lines += [_item_line(OrderItem(**i), cur) for i in order["items"]]
    if order["total"] is not None:
        lines.append(f"Total: {_money(order['total'], cur)} (including {_money(order['shipping_fee'], cur)} shipping)")
    else:
        lines.append(f"Subtotal: {_money(order['subtotal'], cur)}. The store will confirm the shipping fee and final total.")
    lines.append("NEXT: thank the customer, give them the order number, and say the store will confirm the order. "
                 "Do not promise a delivery date.")
    return "\n".join(lines)