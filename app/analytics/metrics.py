"""
Commerce Intelligence metrics — computed on request from existing collections.
Every query is filtered by store_id. No invented numbers: anything not
measurable returns {"available": false, "reason": ...}.

Definitions (approved):
- Orders: created through StoreFlow in the period, excluding rejected/cancelled.
- Revenue placed: those orders' totals (subtotal while shipping is pending).
  Revenue confirmed: only approved/shipped/delivered.
- Conversation: >=1 customer message in the period (one per customer per channel,
  so conversations == customers served).
- Handled by AI: AI replied, no merchant reply, no support case in the period.
- Escalation rate: conversations with a support case opened in the period.
- Human takeover rate: conversations with a merchant reply in the period.
"""

import statistics
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache

from pymongo import MongoClient

from app.analytics.period import Period
from app.billing.usage import check_usage
from app.config import settings
from app.core.orchestrator import PIPELINE_ERROR_REPLY
from app.response.response_generator import FALLBACK_REPLY
from app.settings.store_settings import get_settings

CONFIRMED = {"approved", "shipped", "delivered"}
EXCLUDED = {"rejected", "cancelled"}
LOW_STOCK_MAX = 3
TOP_N = 10
_FALLBACKS = {FALLBACK_REPLY, PIPELINE_ERROR_REPLY}
_MISSING_POLICY_NOTE = "Policy for this question is not set"
_NOT_NOTIFIED_NOTE = "not notified"


@lru_cache(maxsize=1)
def _db():
    return MongoClient(settings.MONGO_URI, tz_aware=True)[settings.MONGO_DB]


# ---------------------------------------------------------------- small helpers

def _pct(part: float, whole: float) -> float | None:
    return round(100.0 * part / whole, 1) if whole else None


def metric(value, previous=None) -> dict:
    change = None
    if isinstance(value, (int, float)) and isinstance(previous, (int, float)) and previous:
        change = round(100.0 * (value - previous) / abs(previous), 1)
    return {"value": value, "previous": previous, "change_pct": change}


def unavailable(reason: str = "requires_instrumentation") -> dict:
    return {"value": None, "available": False, "reason": reason}


def _money(value: float) -> float:
    return round(value, 2)


def _order_value(o: dict) -> float:
    return o["total"] if o.get("total") is not None else o.get("subtotal", 0.0)


def _window(field: str, start: datetime, end: datetime) -> dict:
    return {field: {"$gte": start, "$lt": end}}


def _orders(store_id, start, end) -> list[dict]:
    return list(_db()["orders"].find(
        {"store_id": store_id, **_window("created_at", start, end)},
        {"_id": 0, "id": 1, "status": 1, "total": 1, "subtotal": 1, "channel": 1, "conversation_id": 1,
         "items": 1, "created_at": 1}))


def _messages(store_id, start, end) -> list[dict]:
    return list(_db()["inbox_messages"].find(
        {"store_id": store_id, **_window("created_at", start, end)},
        {"_id": 0, "conversation_id": 1, "sender_type": 1, "created_at": 1, "text": 1,
         "delivery_status": 1, "ai_meta": 1}))


def _cases(store_id, start, end) -> list[dict]:
    return list(_db()["support_cases"].find(
        {"store_id": store_id, **_window("created_at", start, end)},
        {"_id": 0, "conversation_id": 1, "type": 1, "created_at": 1}))


def _conversations(store_id, ids) -> dict[str, dict]:
    docs = _db()["inbox_conversations"].find({"store_id": store_id, "id": {"$in": list(ids)}},
                                             {"_id": 0, "id": 1, "channel": 1, "created_at": 1})
    return {d["id"]: d for d in docs}


# ---------------------------------------------------------------- core aggregates for one window

def _sales(orders: list[dict]) -> dict:
    placed = [o for o in orders if o["status"] not in EXCLUDED]
    revenue_placed = sum(_order_value(o) for o in placed)
    return {
        "orders": len(placed),
        "revenue_placed": _money(revenue_placed),
        "revenue_confirmed": _money(sum(_order_value(o) for o in placed if o["status"] in CONFIRMED)),
        "average_order_value": _money(revenue_placed / len(placed)) if placed else None,
    }


def _conv_stats(store_id: str, start: datetime, end: datetime) -> dict:
    msgs = _messages(store_id, start, end)
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for m in msgs:
        by_conv[m["conversation_id"]].append(m)

    served = {cid for cid, ms in by_conv.items() if any(m["sender_type"] == "customer" for m in ms)}
    ai = {cid for cid, ms in by_conv.items() if any(m["sender_type"] == "ai" for m in ms)}
    human = {cid for cid, ms in by_conv.items() if any(m["sender_type"] == "human" for m in ms)}
    escalated = {c["conversation_id"] for c in _cases(store_id, start, end) if c.get("conversation_id")} & served
    handled = (served & ai) - human - escalated

    first_responses = []
    for cid in served:
        ms = sorted(by_conv[cid], key=lambda m: m["created_at"])
        first_customer = next((m for m in ms if m["sender_type"] == "customer"), None)
        reply = next((m for m in ms if m["sender_type"] in ("ai", "human")
                      and m["created_at"] >= first_customer["created_at"]), None)
        if reply:
            first_responses.append((reply["created_at"] - first_customer["created_at"]).total_seconds())

    return {"messages": msgs, "served": served, "ai": ai, "human": human, "escalated": escalated,
            "handled": handled, "first_responses": first_responses,
            "ai_replies": sum(1 for m in msgs if m["sender_type"] == "ai"),
            "merchant_replies": sum(1 for m in msgs if m["sender_type"] == "human")}


def _rates(stats: dict, orders: int) -> dict:
    n = len(stats["served"])
    return {"conversations": n,
            "conversion_rate": _pct(orders, n),
            "handled_by_ai_rate": _pct(len(stats["handled"]), n),
            "human_takeover_rate": _pct(len(stats["human"] & stats["served"]), n),
            "escalation_rate": _pct(len(stats["escalated"]), n)}


def _daily(p: Period, items: list[dict], field: str = "created_at") -> dict:
    buckets: dict = defaultdict(list)
    for it in items:
        buckets[p.local_day(it[field])].append(it)
    return buckets


# ---------------------------------------------------------------- "now" state (not period-bound)

def _now_counts(store_id: str) -> dict:
    db = _db()
    orders = db["orders"]
    usage = check_usage(store_id)
    return {
        "awaiting_approval": orders.count_documents({"store_id": store_id, "status": "pending_approval"}),
        "shipping_fee_missing": orders.count_documents({"store_id": store_id, "status": "pending_approval",
                                                        "shipping_status": "pending_merchant"}),
        "push_failed": orders.count_documents({"store_id": store_id, "status": "approved", "push.status": "failed"}),
        "customers_waiting": db["inbox_conversations"].count_documents({"store_id": store_id, "needs_attention": True}),
        "paused_unanswered": db["inbox_conversations"].count_documents(
            {"store_id": store_id, "ai_mode": "paused", "unread_count": {"$gt": 0}}),
        "open_cases": db["support_cases"].count_documents({"store_id": store_id, "status": {"$in": ["open", "in_progress"]}}),
        "out_of_stock_products": db["products"].count_documents({"store_id": store_id, "stock_status": "out_of_stock"}),
        "usage": usage,
    }


def usage_block(store_id: str) -> dict:
    u = check_usage(store_id)
    return {"plan": get_settings(store_id).plan, "used": u.monthly_used, "limit": u.monthly_limit,
            "remaining": max(u.monthly_limit - u.monthly_used, 0), "used_pct": _pct(u.monthly_used, u.monthly_limit),
            "daily_used": u.daily_used, "daily_limit": u.daily_cap, "period_start": u.period_start}


def _latest_failed_syncs(store_id: str) -> list[dict]:
    """Sources whose most recent sync failed."""
    latest: dict[str, dict] = {}
    for run in _db()["sync_runs"].find({"store_id": store_id}, {"_id": 0}).sort("started_at", -1).limit(50):
        latest.setdefault(run["source"], run)
    return [r for r in latest.values() if r["status"] == "failed"]


def needs_attention(store_id: str) -> list[dict]:
    now = _now_counts(store_id)
    u = now["usage"]
    items = [
        ("orders_awaiting_approval", "high", now["awaiting_approval"], "{n} order(s) waiting for your approval", "/orders?status=pending_approval"),
        ("shipping_fee_missing", "high", now["shipping_fee_missing"], "{n} order(s) need a shipping fee before approval", "/orders?status=pending_approval"),
        ("platform_push_failed", "high", now["push_failed"], "{n} approved order(s) failed to reach your store platform", "/orders?status=approved"),
        ("customers_waiting", "high", now["customers_waiting"], "{n} customer conversation(s) need your attention", "/inbox?needs_attention=true"),
        ("paused_unanswered", "medium", now["paused_unanswered"], "{n} paused conversation(s) have unread customer messages", "/inbox?ai_mode=paused"),
        ("open_support_cases", "medium", now["open_cases"], "{n} support case(s) are open", "/cases?status=open"),
        ("out_of_stock_products", "low", now["out_of_stock_products"], "{n} product(s) are out of stock", "/products?stock=out_of_stock"),
    ]
    out = [{"type": t, "severity": sev, "count": n, "message": msg.format(n=n), "link": link}
           for t, sev, n, msg, link in items if n]

    failed = _latest_failed_syncs(store_id)
    if failed:
        out.append({"type": "failed_syncs", "severity": "medium", "count": len(failed),
                    "message": f"Last product sync failed for: {', '.join(r['source'] for r in failed)}",
                    "link": "/settings/sync"})
    used_pct = _pct(u.monthly_used, u.monthly_limit) or 0
    if used_pct >= 100 or u.daily_used >= u.daily_cap:
        out.append({"type": "usage_limit_reached", "severity": "high", "count": 1,
                    "message": "AI message limit reached — customers get a fallback reply", "link": "/billing"})
    elif used_pct >= 80:
        out.append({"type": "usage_high", "severity": "medium", "count": 1,
                    "message": f"{used_pct:.0f}% of this month's AI messages used", "link": "/billing"})
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(out, key=lambda x: order[x["severity"]])


# ---------------------------------------------------------------- sections

def overview(store_id: str, p: Period) -> dict:
    cur_orders, prev_orders = _orders(store_id, p.start, p.end), _orders(store_id, p.prev_start, p.prev_end)
    cur_sales, prev_sales = _sales(cur_orders), _sales(prev_orders)
    cur_stats, prev_stats = _conv_stats(store_id, p.start, p.end), _conv_stats(store_id, p.prev_start, p.prev_end)
    cur_rates, prev_rates = _rates(cur_stats, cur_sales["orders"]), _rates(prev_stats, prev_sales["orders"])
    now = _now_counts(store_id)
    kpis = {k: metric(cur_sales[k], prev_sales[k]) for k in cur_sales}
    kpis.update({k: metric(cur_rates[k], prev_rates[k]) for k in cur_rates})
    kpis["customers_served"] = kpis["conversations"]
    kpis["pending_orders"] = {"value": now["awaiting_approval"]}
    kpis["orders_needing_attention"] = {"value": now["awaiting_approval"] + now["push_failed"]}
    return {"kpis": kpis, "usage": usage_block(store_id), "needs_attention": needs_attention(store_id)}


def sales(store_id: str, p: Period) -> dict:
    cur_orders, prev_orders = _orders(store_id, p.start, p.end), _orders(store_id, p.prev_start, p.prev_end)
    cur, prev = _sales(cur_orders), _sales(prev_orders)
    cur_stats, prev_stats = _conv_stats(store_id, p.start, p.end), _conv_stats(store_id, p.prev_start, p.prev_end)
    totals = {k: metric(cur[k], prev[k]) for k in cur}
    totals["conversion_rate"] = metric(_pct(cur["orders"], len(cur_stats["served"])),
                                       _pct(prev["orders"], len(prev_stats["served"])))

    orders_by_day = _daily(p, cur_orders)
    conv_days: dict = defaultdict(set)
    for m in cur_stats["messages"]:
        if m["sender_type"] == "customer":
            conv_days[p.local_day(m["created_at"])].add(m["conversation_id"])
    daily = []
    for day in p.days():
        s = _sales(orders_by_day.get(day, []))
        convs = len(conv_days.get(day, set()))
        daily.append({"date": day.isoformat(), "revenue_confirmed": s["revenue_confirmed"],
                      "revenue_placed": s["revenue_placed"], "orders": s["orders"],
                      "average_order_value": s["average_order_value"], "conversations": convs,
                      "conversion_rate": _pct(s["orders"], convs)})
    return {"totals": totals, "daily": daily, "by_status": dict(Counter(o["status"] for o in cur_orders))}


def ai(store_id: str, p: Period) -> dict:
    cur, prev = _conv_stats(store_id, p.start, p.end), _conv_stats(store_id, p.prev_start, p.prev_end)
    cur_orders, prev_orders = _orders(store_id, p.start, p.end), _orders(store_id, p.prev_start, p.prev_end)

    def from_conv(orders):
        return sum(1 for o in orders if o.get("conversation_id") and o["status"] not in EXCLUDED)

    def median(xs):
        return round(statistics.median(xs)) if xs else None

    days = {d: {"conversations": set(), "ai_replies": 0, "merchant_replies": 0} for d in p.days()}
    for m in cur["messages"]:
        d = days.get(p.local_day(m["created_at"]))
        if d is None:
            continue
        if m["sender_type"] == "customer":
            d["conversations"].add(m["conversation_id"])
        elif m["sender_type"] == "ai":
            d["ai_replies"] += 1
        elif m["sender_type"] == "human":
            d["merchant_replies"] += 1
    escalated_by_day = Counter(p.local_day(c["created_at"]) for c in _cases(store_id, p.start, p.end))

    return {
        "conversations": metric(len(cur["served"]), len(prev["served"])),
        "ai_replies": metric(cur["ai_replies"], prev["ai_replies"]),
        "merchant_replies": metric(cur["merchant_replies"], prev["merchant_replies"]),
        "handled_by_ai": metric(len(cur["handled"]), len(prev["handled"])),
        "handled_by_ai_rate": metric(_rates(cur, 0)["handled_by_ai_rate"], _rates(prev, 0)["handled_by_ai_rate"]),
        "escalated": metric(len(cur["escalated"]), len(prev["escalated"])),
        "escalation_rate": metric(_rates(cur, 0)["escalation_rate"], _rates(prev, 0)["escalation_rate"]),
        "human_takeover_rate": metric(_rates(cur, 0)["human_takeover_rate"], _rates(prev, 0)["human_takeover_rate"]),
        "first_response_seconds_median": metric(median(cur["first_responses"]), median(prev["first_responses"])),
        "resolution_time": unavailable("no_resolved_conversation_definition"),
        "orders_from_ai_conversations": metric(from_conv(cur_orders), from_conv(prev_orders)),
        "needs_attention_now": {"value": _db()["inbox_conversations"].count_documents(
            {"store_id": store_id, "needs_attention": True})},
        "failed_deliveries": metric(sum(1 for m in cur["messages"] if m["delivery_status"] == "failed"),
                                    sum(1 for m in prev["messages"] if m["delivery_status"] == "failed")),
        "daily": [{"date": d.isoformat(), "conversations": len(v["conversations"]), "ai_replies": v["ai_replies"],
                   "merchant_replies": v["merchant_replies"], "escalated": escalated_by_day.get(d, 0)}
                  for d, v in days.items()],
    }


def channels(store_id: str, p: Period) -> dict:
    stats = _conv_stats(store_id, p.start, p.end)
    convs = _conversations(store_id, stats["served"])
    orders = [o for o in _orders(store_id, p.start, p.end) if o["status"] not in EXCLUDED]
    names = sorted({c["channel"] for c in convs.values()} | {o["channel"] for o in orders})
    out = []
    for ch in names:
        served = {cid for cid, c in convs.items() if c["channel"] == ch}
        ch_orders = [o for o in orders if o["channel"] == ch]
        out.append({
            "channel": ch, "conversations": len(served), "orders": len(ch_orders),
            "revenue_confirmed": _money(sum(_order_value(o) for o in ch_orders if o["status"] in CONFIRMED)),
            "revenue_placed": _money(sum(_order_value(o) for o in ch_orders)),
            "conversion_rate": _pct(len(ch_orders), len(served)),
            "handled_by_ai_rate": _pct(len(stats["handled"] & served), len(served)),
            "escalation_rate": _pct(len(stats["escalated"] & served), len(served)),
        })
    return {"channels": out}


def _product_names(store_id: str, ids) -> dict[str, str]:
    return {d["product_id"]: d["name"] for d in _db()["products"].find(
        {"store_id": store_id, "product_id": {"$in": list(ids)}}, {"_id": 0, "product_id": 1, "name": 1})}


def products(store_id: str, p: Period) -> dict:
    orders = [o for o in _orders(store_id, p.start, p.end) if o["status"] not in EXCLUDED]
    qty, n_orders, revenue, names = Counter(), Counter(), Counter(), {}
    ordered_in_conv: dict[str, set] = defaultdict(set)
    for o in orders:
        for i in o["items"]:
            qty[i["product_id"]] += i["quantity"]
            revenue[i["product_id"]] += i["unit_price"] * i["quantity"]
            names[i["product_id"]] = i["name"]
            if o.get("conversation_id"):
                ordered_in_conv[i["product_id"]].add(o["conversation_id"])
        for pid in {i["product_id"] for i in o["items"]}:
            n_orders[pid] += 1
    most_ordered = [{"product_id": pid, "name": names[pid], "quantity": q, "orders": n_orders[pid],
                     "revenue": _money(revenue[pid])} for pid, q in qty.most_common(TOP_N)]

    recommended_in_conv: dict[str, set] = defaultdict(set)
    for m in _messages(store_id, p.start, p.end):
        if m["sender_type"] == "ai":
            for pid in (m.get("ai_meta") or {}).get("product_ids") or []:
                recommended_in_conv[pid].add(m["conversation_id"])
    rec_names = _product_names(store_id, recommended_in_conv)
    ranked = sorted(recommended_in_conv.items(), key=lambda kv: -len(kv[1]))[:TOP_N]
    most_recommended = [{"product_id": pid, "name": rec_names.get(pid, pid), "conversations": len(convs),
                         "ordered_in_those_conversations": len(convs & ordered_in_conv.get(pid, set())),
                         "conversion_rate": _pct(len(convs & ordered_in_conv.get(pid, set())), len(convs))}
                        for pid, convs in ranked]
    all_rec = sum(len(c) for c in recommended_in_conv.values())
    all_conv = sum(len(c & ordered_in_conv.get(pid, set())) for pid, c in recommended_in_conv.items())

    low, out = [], []
    for d in _db()["products"].find({"store_id": store_id}, {"_id": 0, "product_id": 1, "name": 1, "stock_status": 1,
                                                            "stock_quantity": 1, "variants": 1}):
        units = d.get("variants") or [d]
        for u in units:
            variant = u.get("attributes") if d.get("variants") else None
            entry = {"product_id": d["product_id"], "name": d["name"], "variant": variant or None}
            if u.get("stock_status") == "out_of_stock":
                out.append(entry)
            elif u.get("stock_status") == "in_stock" and u.get("stock_quantity") is not None \
                    and u["stock_quantity"] <= LOW_STOCK_MAX:
                low.append({**entry, "level": "low_stock"})
    return {"most_ordered": most_ordered, "most_recommended": most_recommended,
            "recommendation_to_order": {"recommended_conversations": all_rec, "ordered": all_conv,
                                        "conversion_rate": _pct(all_conv, all_rec)},
            "low_stock": low[:50], "out_of_stock": out[:50], "stock_as_of": "last product sync"}


def customers(store_id: str, p: Period) -> dict:
    def new_vs_returning(start, end):
        stats = _conv_stats(store_id, start, end)
        convs = _conversations(store_id, stats["served"])
        new = sum(1 for c in convs.values() if start <= c["created_at"] < end)
        return new, len(convs) - new, stats

    cur_new, cur_ret, cur = new_vs_returning(p.start, p.end)
    prev_new, prev_ret, _ = new_vs_returning(p.prev_start, p.prev_end)
    intents = Counter((m.get("ai_meta") or {}).get("intent") for m in cur["messages"] if m["sender_type"] == "ai")
    intents.pop(None, None)
    case_types = Counter(c["type"] for c in _cases(store_id, p.start, p.end))
    notes = [m["text"] for m in cur["messages"] if m["delivery_status"] == "internal"]
    missing_policy = sum(1 for t in notes if _MISSING_POLICY_NOTE in t)
    fallbacks = sum(1 for m in cur["messages"] if m["sender_type"] == "system" and m["text"] in _FALLBACKS)
    reasons = [{"reason": t, "count": n} for t, n in case_types.most_common()]
    if missing_policy:
        reasons.append({"reason": "missing_policy", "count": missing_policy})
    return {
        "new_conversations": metric(cur_new, prev_new),
        "returning_conversations": metric(cur_ret, prev_ret),
        "top_intents": [{"intent": k, "count": v} for k, v in intents.most_common(TOP_N)],
        "top_product_questions": unavailable("requires_question_grouping"),
        "top_support_issues": [{"type": k, "count": v} for k, v in case_types.most_common()],
        "top_escalation_reasons": sorted(reasons, key=lambda r: -r["count"]),
        "unresolved_by_ai": {"fallback_replies": metric(fallbacks), "missing_policy_escalations": metric(missing_policy)},
    }


def support(store_id: str, p: Period) -> dict:
    cases = _db()["support_cases"]

    def window(start, end):
        opened = list(cases.find({"store_id": store_id, **_window("created_at", start, end)},
                                 {"_id": 0, "type": 1, "created_at": 1}))
        resolved = list(cases.find({"store_id": store_id, **_window("resolved_at", start, end)},
                                   {"_id": 0, "created_at": 1, "resolved_at": 1}))
        outcomes = Counter((m.get("ai_meta") or {}).get("aftersales")
                           for m in _db()["inbox_messages"].find(
                               {"store_id": store_id, "sender_type": "ai", "ai_meta.aftersales": {"$ne": None},
                                **_window("created_at", start, end)}, {"_id": 0, "ai_meta.aftersales": 1}))
        hours = [(c["resolved_at"] - c["created_at"]).total_seconds() / 3600 for c in resolved]
        return opened, resolved, outcomes, hours

    o, r, out, hrs = window(p.start, p.end)
    po, pr, pout, phrs = window(p.prev_start, p.prev_end)

    def ai_rate(outcomes):
        return _pct(outcomes["resolved"], outcomes["resolved"] + outcomes["escalated"])

    def median(xs):
        return round(statistics.median(xs), 1) if xs else None

    opened_by_day = Counter(p.local_day(c["created_at"]) for c in o)
    resolved_by_day = Counter(p.local_day(c["resolved_at"]) for c in r)
    return {
        "opened": metric(len(o), len(po)), "resolved": metric(len(r), len(pr)),
        "pending_now": {"value": cases.count_documents({"store_id": store_id, "status": {"$in": ["open", "in_progress"]}})},
        "escalated": metric(len(o), len(po)),
        "ai_resolved_rate": metric(ai_rate(out), ai_rate(pout)),
        "resolution_time_hours_median": metric(median(hrs), median(phrs)),
        "by_type": [{"type": k, "count": v} for k, v in Counter(c["type"] for c in o).most_common()],
        "daily": [{"date": d.isoformat(), "opened": opened_by_day.get(d, 0), "resolved": resolved_by_day.get(d, 0)}
                  for d in p.days()],
    }


def orders_ops(store_id: str, p: Period) -> dict:
    db = _db()
    by_status = {d["_id"]: d["n"] for d in db["orders"].aggregate(
        [{"$match": {"store_id": store_id}}, {"$group": {"_id": "$status", "n": {"$sum": 1}}}])}
    fulfilment = {d["_id"]: d["n"] for d in db["orders"].aggregate([
        {"$match": {"store_id": store_id, "platform.order_id": {"$exists": True, "$ne": None}}},
        {"$group": {"_id": {"$ifNull": ["$fulfilment.state", "not_checked"]}, "n": {"$sum": 1}}}])}
    last_checked = db["orders"].find_one({"store_id": store_id, "fulfilment.checked_at": {"$exists": True}},
                                         {"_id": 0, "fulfilment.checked_at": 1}, sort=[("fulfilment.checked_at", -1)])
    now = _now_counts(store_id)
    not_notified = db["inbox_messages"].count_documents(
        {"store_id": store_id, "delivery_status": "internal", "text": {"$regex": _NOT_NOTIFIED_NOTE},
         **_window("created_at", p.start, p.end)})

    def approval_minutes(start, end):
        mins = []
        for o in db["orders"].find({"store_id": store_id, "status_history": {"$elemMatch": {
                "status": "approved", "at": {"$gte": start, "$lt": end}}}}, {"_id": 0, "status_history": 1}):
            hist = {h["status"]: h["at"] for h in o["status_history"]}
            if "pending_approval" in hist and "approved" in hist:
                mins.append((hist["approved"] - hist["pending_approval"]).total_seconds() / 60)
        return round(statistics.median(mins), 1) if mins else None

    failed_runs = list(db["sync_runs"].find({"store_id": store_id, "status": "failed",
                                             **_window("started_at", p.start, p.end)},
                                            {"_id": 0, "source": 1, "error": 1, "started_at": 1}).sort("started_at", -1))
    return {
        "by_status_now": by_status,
        "platform_fulfilment": {"by_state": fulfilment,
                                "last_checked_at": (last_checked or {}).get("fulfilment", {}).get("checked_at"),
                                "note": "as of last check"},
        "requiring_action": [
            {"reason": "awaiting_approval", "count": now["awaiting_approval"]},
            {"reason": "shipping_fee_missing", "count": now["shipping_fee_missing"]},
            {"reason": "platform_push_failed", "count": now["push_failed"]},
            {"reason": "customer_not_notified", "count": not_notified},
        ],
        "failed_syncs": {"count": len(failed_runs),
                         "last_error": failed_runs[0]["error"] if failed_runs else None,
                         "last_failed_at": failed_runs[0]["started_at"] if failed_runs else None},
        "average_approval_minutes": metric(approval_minutes(p.start, p.end),
                                           approval_minutes(p.prev_start, p.prev_end)),
    }