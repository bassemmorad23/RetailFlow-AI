"""
Customer notifications about their order, sent through the conversation's
own channel. Deterministic bilingual templates (Arabic + English): no AI
wording on money matters. Respects reply windows; if a message can't be
sent, the merchant sees an internal note instead.
"""

from app.channels.dispatcher import send
from app.inbox import events
from app.inbox import repository as inbox_repo
from app.inbox.windows import reply_window


def _money(value: float | None, currency: str) -> str:
    return f"{value:,.2f} {currency}" if value is not None else ""


def approved_text(order: dict) -> str:
    total = _money(order["total"], order["currency"])
    return (f"تم تأكيد طلبك رقم {order['number']} ✅ الإجمالي: {total} (الدفع عند الاستلام). هنتواصل معاك بخصوص التوصيل.\n"
            f"Your order {order['number']} is confirmed ✅ Total: {total} (cash on delivery). "
            f"We'll contact you about delivery.")


def rejected_text(order: dict, message: str = "") -> str:
    extra = f"\n{message.strip()}" if message and message.strip() else ""
    return (f"نعتذر، لا يمكننا تنفيذ طلبك رقم {order['number']} حالياً.\n"
            f"Sorry, we can't fulfil your order {order['number']} right now.{extra}")


def shipped_text(order: dict) -> str:
    return (f"تم شحن طلبك رقم {order['number']} 🚚\n"
            f"Your order {order['number']} has been shipped 🚚")


def notify_customer(store_id: str, order: dict, text: str) -> bool:
    """True if delivered. Otherwise the merchant gets an internal note explaining why."""
    cid = order.get("conversation_id")
    conv = inbox_repo.get_conversation(store_id, cid) if cid else None
    if conv is None:
        return False

    is_open, _ = reply_window(conv)
    if not is_open:
        _internal_note(store_id, cid, f"Customer not notified about {order['number']}: "
                                      f"the channel's reply window has closed.")
        return False

    msg = inbox_repo.add_message(store_id, cid, "system", text, delivery_status="pending")
    if msg is None:
        return False
    events.emit_message_created(store_id, msg)

    result = send(conv["channel"], store_id, conv["customer"]["external_id"], text)
    status = "sent" if result.ok else "failed"
    inbox_repo.set_delivery(store_id, msg["id"], status,
                            external_message_ids=result.external_message_ids, error=result.error_code)
    msg.update(delivery_status=status, delivery_error=result.error_code)
    events.emit_message_updated(store_id, msg)
    events.emit_conversation_updated(store_id, cid)
    return result.ok


def _internal_note(store_id: str, cid: str, text: str) -> None:
    note = inbox_repo.add_message(store_id, cid, "system", text, delivery_status="internal")
    if note:
        events.emit_message_created(store_id, note)