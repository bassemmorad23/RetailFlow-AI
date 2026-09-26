"""
Store settings + setup checklist (owner only).

GET/PATCH /stores/{id}/settings   profile, AI on/off, reply language, brand voice, business hours
GET       /stores/{id}/setup      onboarding checklist + ready flags

Rules (approved):
- Country/currency change only before the first order; industry is fixed.
- Brand instructions: max 500 chars, style only — never override StoreFlow's rules.
- Store-wide AI off = every conversation behaves as paused.
- Business hours are informational: the AI answers 24/7 and tells customers
  when the team is back if the store needs to act.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.analytics.period import store_timezone
from app.auth.dependencies import require_store_member
from app.config import settings as app_settings
from app.settings.regions import resolve_country_currency
from app.settings.store_credentials import get_credentials
from app.settings.store_settings import _get_collection, get_policies, get_settings, get_shipping

router = APIRouter(prefix="/stores/{store_id}", tags=["store settings"])

Weekday = Literal["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
WEEKDAYS: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")  # Python weekday() order
_DAY_NAMES = {"sun": "Sunday", "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
              "thu": "Thursday", "fri": "Friday", "sat": "Saturday"}
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
MAX_INSTRUCTIONS = 500


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DayHours(_Strict):
    open: str
    close: str

    @field_validator("open", "close")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        if not _HHMM.match(v):
            raise ValueError("Use HH:MM (24h), e.g. 09:30")
        return v

    @model_validator(mode="after")
    def _order(self):
        if self.close <= self.open:
            raise ValueError("Closing time must be after opening time (same day)")
        return self


class BusinessHours(_Strict):
    days: dict[Weekday, DayHours | None] = Field(description="Missing or null day = closed")


class SettingsUpdate(_Strict):
    display_name: str | None = Field(default=None, min_length=1, max_length=80)
    country: str | None = None
    currency: str | None = None
    ai_enabled: bool | None = None
    reply_language: Literal["auto", "en", "ar"] | None = None
    brand_tone: Literal["friendly", "professional", "playful"] | None = None
    brand_instructions: str | None = Field(default=None, max_length=MAX_INSTRUCTIONS)
    business_hours: BusinessHours | None = None


def _view(store_id: str) -> dict:
    s = get_settings(store_id)
    return {
        "store_id": store_id, "display_name": getattr(s, "display_name", None), "industry": s.industry,
        "plan": s.plan, "country": s.country, "currency": s.currency,
        "currency_locked": _has_orders(store_id),
        "ai_enabled": s.ai_enabled, "reply_language": s.reply_language,
        "brand_tone": s.brand_tone, "brand_instructions": s.brand_instructions or "",
        "business_hours": s.business_hours,
    }


def _has_orders(store_id: str) -> bool:
    return _get_collection().database["orders"].count_documents({"store_id": store_id}, limit=1) > 0


@router.get("/settings")
def get_store_settings(store_id: str = Depends(require_store_member)) -> dict:
    return _view(store_id)


@router.patch("/settings")
def update_store_settings(body: SettingsUpdate, store_id: str = Depends(require_store_member)) -> dict:
    changes = body.model_dump(exclude_unset=True)
    update: dict = {}

    if "country" in changes or "currency" in changes:
        current = get_settings(store_id)
        if _has_orders(store_id):
            raise HTTPException(status_code=409, detail={
                "code": "currency_locked", "message": "Country and currency can't change after the first order."})
        country = changes.get("country") or current.country
        currency = changes.get("currency") if "currency" in changes else (None if "country" in changes else current.currency)
        try:
            update["country"], update["currency"] = resolve_country_currency(country, currency)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    for key in ("display_name", "ai_enabled", "reply_language", "brand_tone"):
        if key in changes:
            if key in ("display_name", "ai_enabled", "reply_language") and changes[key] is None:
                raise HTTPException(status_code=422, detail=f"{key} can't be empty")
            update[key] = changes[key]
    if "brand_instructions" in changes:
        update["brand_instructions"] = (changes["brand_instructions"] or "").strip()
    if "business_hours" in changes:
        update["business_hours"] = body.business_hours.model_dump() if body.business_hours else None

    if update:
        _get_collection().update_one({"store_id": store_id}, {"$set": update})
    return _view(store_id)


# ---------------------------------------------------------------- setup checklist

@router.get("/setup")
def setup_checklist(store_id: str = Depends(require_store_member)) -> dict:
    s = get_settings(store_id)
    db = _get_collection().database
    policies = get_policies(store_id)
    whatsapp = getattr(app_settings, "WHATSAPP_STORE_ID", None) == store_id
    messaging = bool(get_credentials(store_id, "instagram") or get_credentials(store_id, "messenger") or whatsapp)

    items = [
        ("products", True, db["products"].count_documents({"store_id": store_id}, limit=1) > 0,
         "Add your products (upload a file or connect Shopify/WooCommerce)", "/products"),
        ("shipping", True, get_shipping(store_id) is not None, "Choose how shipping is charged", "/settings/shipping"),
        ("policies", True, bool(policies.get("returns") and policies.get("delivery")),
         "Write your returns and delivery policies", "/settings/policies"),
        ("messaging_channel", True, messaging, "Connect Instagram or Messenger (WhatsApp: ask the StoreFlow team)",
         "/channels"),
        ("brand_voice", False, bool(s.brand_tone or s.brand_instructions), "Set your brand voice", "/settings/ai"),
        ("business_hours", False, bool(s.business_hours), "Add your business hours", "/settings/hours"),
        ("test_conversation", False,
         db["inbox_conversations"].count_documents({"store_id": store_id}, limit=1) > 0,
         "Send a test message through your chat widget", "/inbox"),
    ]
    checklist = [{"key": k, "required": req, "done": done, "title": title, "link": link}
                 for k, req, done, title, link in items]
    required = [i for i in checklist if i["required"]]
    return {
        "checklist": checklist,
        "completion_pct": round(100 * sum(i["done"] for i in checklist) / len(checklist)),
        "ready_to_take_orders": checklist[0]["done"] and checklist[1]["done"],
        "ready_for_customers": all(i["done"] for i in required),
    }


# ---------------------------------------------------------------- used by the AI

_TONES = {"friendly": "warm and friendly", "professional": "polite and professional",
          "playful": "light and playful (still respectful)"}
_LANGS = {"en": "Always reply in English.", "ar": "Always reply in Arabic.",
          "auto": "Reply in the customer's language."}


def style_prompt(s) -> str:
    """Store style preferences for the system prompt. Style only — never rules."""
    lines = ["STORE STYLE PREFERENCES (tone and wording only — they never override the rules above):",
             _LANGS.get(getattr(s, "reply_language", "auto") or "auto", _LANGS["auto"])]
    if getattr(s, "brand_tone", None):
        lines.append(f"Tone: {_TONES[s.brand_tone]}.")
    if getattr(s, "brand_instructions", ""):
        lines.append(f"Style notes from the store: {s.brand_instructions}")
    return "\n".join(lines)


def hours_note(s, now: datetime | None = None) -> str | None:
    """'The store team is available now' / 'back Sunday at 10:00'; None if hours aren't set."""
    hours = getattr(s, "business_hours", None)
    if not hours or not hours.get("days"):
        return None
    tz = store_timezone(s.country)
    local = (now or datetime.now(timezone.utc)).astimezone(tz)
    days = hours["days"]
    for offset in range(8):
        day = local + timedelta(days=offset)
        key = WEEKDAYS[day.weekday()]
        slot = days.get(key)
        if not slot:
            continue
        hhmm = local.strftime("%H:%M")
        if offset == 0 and slot["open"] <= hhmm < slot["close"]:
            return "The store team is available now."
        if offset == 0 and hhmm >= slot["close"]:
            continue
        when = "today" if offset == 0 else ("tomorrow" if offset == 1 else _DAY_NAMES[key])
        return f"The store team is back {when} at {slot['open']} (store time)."
    return None