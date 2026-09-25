"""
Turn any merchant/platform stock value into an honest (status, quantity).

status: "in_stock" | "out_of_stock" | "unknown"
quantity: int or None (never shown to customers)

Missing or unrecognised values are ALWAYS "unknown" — never "in stock".
"""

import re
from typing import Any

_IN = {
    "yes", "y", "true", "in stock", "instock", "in_stock", "available",
    "backorder", "onbackorder", "on backorder",
    "متوفر", "متاح", "موجود", "نعم", "اه", "ايوه",
}
_OUT = {
    "no", "n", "false", "out of stock", "outofstock", "out_of_stock",
    "sold out", "soldout", "unavailable", "not available",
    "غير متوفر", "غير متاح", "نفذ", "نفذت", "نفدت", "خلصان", "خلص", "لا",
}
_NUMBER = re.compile(r"^\d+(?:\.0+)?$")  # \d includes Arabic-Indic digits


def parse_stock(value: Any) -> tuple[str, int | None]:
    if value is None:
        return "unknown", None
    if isinstance(value, bool):
        return ("in_stock" if value else "out_of_stock"), None
    if isinstance(value, (int, float)):
        qty = max(int(value), 0)
        return ("in_stock" if qty > 0 else "out_of_stock"), qty

    text = str(value).strip().lower()
    if not text:
        return "unknown", None
    if _NUMBER.match(text):
        qty = int(float(text))
        return ("in_stock" if qty > 0 else "out_of_stock"), qty
    if text in _IN:
        return "in_stock", None
    if text in _OUT:
        return "out_of_stock", None
    return "unknown", None


def aggregate_status(statuses: list[str]) -> str:
    """Parent of variants: in stock if any variant is, out if all are, else unknown."""
    if any(s == "in_stock" for s in statuses):
        return "in_stock"
    if statuses and all(s == "out_of_stock" for s in statuses):
        return "out_of_stock"
    return "unknown"