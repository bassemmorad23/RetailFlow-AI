"""
Apply customer details from one message to the draft, validating each.

Never guesses anything that affects shipping:
- country: recognised names/codes only
- region: exact match, or a "Did you mean ...?" suggestion, or ask again
- phone: valid for the country, stored as E.164
On WhatsApp the sender's number is offered automatically.
"""

from dataclasses import dataclass, field

import pycountry

from app.commerce.locations import normalize, resolve_region
from app.commerce.models import CustomerDetails, DraftCustomer
from app.commerce.phones import normalize_phone, parse_whatsapp_sender

REQUIRED = ("name", "phone", "region_code", "city", "address_line")

_COUNTRY_ALIASES = {
    "مصر": "EG", "egypt": "EG", "masr": "EG",
    "السعوديه": "SA", "saudi": "SA", "saudi arabia": "SA", "ksa": "SA",
    "الامارات": "AE", "uae": "AE", "emirates": "AE",
    "الكويت": "KW", "kuwait": "KW", "قطر": "QA", "qatar": "QA",
    "البحرين": "BH", "bahrain": "BH", "عمان": "OM", "oman": "OM",
    "الاردن": "JO", "jordan": "JO",
}


def resolve_country(text: str) -> str | None:
    key = normalize(text)
    for alias, code in _COUNTRY_ALIASES.items():
        if normalize(alias) == key:
            return code
    try:
        return pycountry.countries.lookup(text.strip()).alpha_2
    except LookupError:
        return None


@dataclass
class DetailsUpdate:
    customer: DraftCustomer
    problems: list[str] = field(default_factory=list)  # country_unknown | phone_invalid | region_suggest | region_unknown


def apply_customer_details(
    current: DraftCustomer,
    stated: dict,
    *,
    store_country: str,
    channel: str,
    sender_id: str,
) -> DetailsUpdate:
    c = current.model_copy(deep=True)
    problems: list[str] = []

    if stated.get("country"):
        code = resolve_country(stated["country"])
        if code:
            if code != c.country:
                c.region_code = c.region_name = None  # region belonged to another country
            c.country = code
        else:
            problems.append("country_unknown")
    if not c.country:
        c.country = store_country

    for key in ("name", "city", "address_line", "notes"):
        if stated.get(key):
            setattr(c, key, stated[key])

    if stated.get("phone"):
        result = normalize_phone(stated["phone"], c.country)
        if result.valid:
            c.phone, c.phone_source = result.e164, "customer"
        else:
            problems.append("phone_invalid")
    if not c.phone and channel == "whatsapp":
        result = parse_whatsapp_sender(sender_id)
        if result.valid:
            c.phone, c.phone_source = result.e164, "channel"

    if stated.get("region"):
        match = resolve_region(c.country, stated["region"])
        if match.status == "matched":
            c.region_code, c.region_name, c.region_suggestions = match.code, match.name, []
        elif match.status == "suggest":
            c.region_suggestions = [name for _, name in match.suggestions]
            problems.append("region_suggest")
        else:
            c.region_suggestions = []
            problems.append("region_unknown")

    return DetailsUpdate(customer=c, problems=problems)


def missing_fields(c: DraftCustomer) -> list[str]:
    return [f for f in REQUIRED if not getattr(c, f)]


def to_customer_details(c: DraftCustomer) -> CustomerDetails:
    """Only call when missing_fields(c) is empty."""
    return CustomerDetails(
        name=c.name, phone=c.phone, country=c.country, region_code=c.region_code,
        region_name=c.region_name, city=c.city, address_line=c.address_line, notes=c.notes or "",
    )