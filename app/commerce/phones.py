"""
International phone numbers, stored as E.164 (+201012345678).

- Parsed with Google's libphonenumber (`phonenumbers`), using the
  customer's/store's country for local formats ("01012345678" in Egypt).
- Arabic-Indic / Persian digits are converted first.
- WhatsApp sender ids are digits without "+"; parse_whatsapp_sender adds it.
"""

from dataclasses import dataclass

import phonenumbers

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


@dataclass(frozen=True)
class PhoneResult:
    valid: bool
    e164: str | None = None
    reason: str | None = None  # empty | unparseable | invalid


def normalize_phone(raw: str | None, default_country: str | None) -> PhoneResult:
    text = (raw or "").strip().translate(_DIGITS)
    if not text:
        return PhoneResult(False, reason="empty")
    if text.startswith("00"):
        text = "+" + text[2:]
    try:
        number = phonenumbers.parse(text, (default_country or "").upper() or None)
    except phonenumbers.NumberParseException:
        return PhoneResult(False, reason="unparseable")
    if not phonenumbers.is_valid_number(number):
        return PhoneResult(False, reason="invalid")
    return PhoneResult(True, phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164))


def parse_whatsapp_sender(sender_id: str) -> PhoneResult:
    """WhatsApp gives '201012345678' (international, no '+')."""
    digits = "".join(ch for ch in (sender_id or "") if ch.isdigit())
    return normalize_phone("+" + digits if digits else "", None)