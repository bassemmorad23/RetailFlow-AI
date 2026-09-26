"""Store settings API, setup checklist, brand style prompt, business hours note, store-wide AI switch."""

import types
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.commerce import repository as orders
from app.commerce.models import CustomerDetails, OrderItem
from app.settings.store_credentials import _get_collection as creds_col
from app.stores.settings_api import hours_note, style_prompt

HOURS = {"days": {"sun": {"open": "10:00", "close": "18:00"}, "mon": {"open": "10:00", "close": "18:00"},
                  "fri": None}}


def _merchant():
    c = TestClient(app)
    c.post("/auth/register", json={"email": f"st_{uuid.uuid4().hex[:8]}@example.com", "password": "Strong-pass-123"})
    sid = c.post("/stores", json={"display_name": "Shop", "industry": "fashion", "country": "EG"}).json()["store_id"]
    return c, sid


# ---------------------------------------------------------------- settings

def test_defaults_and_update():
    c, sid = _merchant()
    s = c.get(f"/stores/{sid}/settings").json()
    assert (s["display_name"], s["country"], s["currency"], s["ai_enabled"], s["reply_language"]) == \
           ("Shop", "EG", "EGP", True, "auto")
    r = c.patch(f"/stores/{sid}/settings", json={"brand_tone": "friendly", "brand_instructions": "  Use emojis sparingly ",
                                                   "reply_language": "ar", "business_hours": HOURS, "display_name": "Shop 2"})
    s = r.json()
    assert (s["brand_tone"], s["brand_instructions"], s["reply_language"], s["display_name"]) == \
           ("friendly", "Use emojis sparingly", "ar", "Shop 2")
    assert s["business_hours"]["days"]["sun"] == {"open": "10:00", "close": "18:00"}
    assert c.patch(f"/stores/{sid}/settings", json={"brand_tone": None}).json()["brand_tone"] is None


@pytest.mark.parametrize("body", [
    {"brand_instructions": "x" * 501},
    {"business_hours": {"days": {"sun": {"open": "18:00", "close": "10:00"}}}},
    {"business_hours": {"days": {"sun": {"open": "9am", "close": "18:00"}}}},
    {"business_hours": {"days": {"someday": None}}},
    {"reply_language": "fr"},
    {"industry": "electronics"},            # fixed after creation
    {"country": "FR"},
    {"currency": "EUR"},
    {"ai_enabled": None},
])
def test_invalid_updates_rejected(body):
    c, sid = _merchant()
    assert c.patch(f"/stores/{sid}/settings", json=body).status_code == 422


def test_country_change_defaults_currency_then_locks_after_first_order():
    c, sid = _merchant()
    assert c.patch(f"/stores/{sid}/settings", json={"country": "SA"}).json()["currency"] == "SAR"
    assert c.patch(f"/stores/{sid}/settings", json={"currency": "USD"}).json()["currency"] == "USD"
    orders.create_order(sid, channel="web", customer_external_id="v",
                        customer=CustomerDetails(name="A", phone="+966551234567", country="SA", region_code="SA-01",
                                                 region_name="Riyadh", city="Riyadh", address_line="x"),
                        items=[OrderItem(product_id="p", name="P", quantity=1, unit_price=1)],
                        currency="USD", shipping_fee=0, idempotency_key=uuid.uuid4().hex)
    r = c.patch(f"/stores/{sid}/settings", json={"country": "EG"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "currency_locked"
    assert c.get(f"/stores/{sid}/settings").json()["currency_locked"] is True


def test_settings_isolation():
    c, sid = _merchant()
    other, _ = _merchant()
    assert other.get(f"/stores/{sid}/settings").status_code == 404
    assert other.patch(f"/stores/{sid}/settings", json={"ai_enabled": False}).status_code == 404
    assert other.get(f"/stores/{sid}/setup").status_code == 404


# ---------------------------------------------------------------- setup checklist

def test_setup_checklist_progression():
    c, sid = _merchant()
    s = c.get(f"/stores/{sid}/setup").json()
    assert s["ready_to_take_orders"] is False and s["ready_for_customers"] is False and s["completion_pct"] == 0

    from app.products.product_store import _get_collection as products_col
    from app.schemas.models import Product
    products_col().insert_one(Product(product_id="p", store_id=sid, name="P", price=1).model_dump())
    c.put(f"/stores/{sid}/shipping", json={"method": "fixed", "fixed": {"fee": 50}})
    s = c.get(f"/stores/{sid}/setup").json()
    assert s["ready_to_take_orders"] is True and s["ready_for_customers"] is False

    c.put(f"/stores/{sid}/policies", json={"returns": "14 days", "delivery": "2-4 days"})
    creds_col().insert_one({"store_id": sid, "source": "instagram", "credentials": {"access_token": "x", "username": "s"}})
    s = c.get(f"/stores/{sid}/setup").json()
    assert s["ready_for_customers"] is True
    done = {i["key"]: i["done"] for i in s["checklist"]}
    assert done["brand_voice"] is False and done["business_hours"] is False


# ---------------------------------------------------------------- AI behaviour helpers

def _s(**kw):
    base = {"country": "EG", "reply_language": "auto", "brand_tone": None, "brand_instructions": "", "business_hours": None}
    return types.SimpleNamespace(**{**base, **kw})


def test_style_prompt_is_style_only():
    text = style_prompt(_s(reply_language="ar", brand_tone="playful", brand_instructions="Call customers 'ya basha'"))
    assert "never override the rules" in text and "Always reply in Arabic." in text
    assert "light and playful" in text and "ya basha" in text
    assert "customer's language" in style_prompt(_s())


@pytest.mark.parametrize("utc_now,expected", [
    (datetime(2026, 9, 27, 9, 0, tzinfo=timezone.utc), "available now"),            # Sun 12:00 Cairo
    (datetime(2026, 9, 27, 5, 0, tzinfo=timezone.utc), "back today at 10:00"),      # Sun 08:00
    (datetime(2026, 9, 27, 17, 0, tzinfo=timezone.utc), "back tomorrow at 10:00"),  # Sun 20:00 -> Mon
    (datetime(2026, 9, 29, 9, 0, tzinfo=timezone.utc), "back Sunday at 10:00"),     # Tue (closed) -> Sun
])
def test_hours_note(utc_now, expected):
    assert expected in hours_note(_s(business_hours=HOURS), utc_now)


def test_hours_note_without_hours():
    assert hours_note(_s()) is None