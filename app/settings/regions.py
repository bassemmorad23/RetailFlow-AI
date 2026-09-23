"""
Supported countries and currencies. Edit here to add markets;
no other code changes needed.
"""

# country ISO code -> default currency ISO code
SUPPORTED_COUNTRIES: dict[str, str] = {
    "EG": "EGP",  # Egypt
    "SA": "SAR",  # Saudi Arabia
    "AE": "AED",  # UAE
    "KW": "KWD",  # Kuwait
    "QA": "QAR",  # Qatar
    "BH": "BHD",  # Bahrain
    "OM": "OMR",  # Oman
    "JO": "JOD",  # Jordan
}

# Currencies a store may use (a store may sell in a currency other than its country's, e.g. USD)
SUPPORTED_CURRENCIES: set[str] = set(SUPPORTED_COUNTRIES.values()) | {"USD"}


def resolve_country_currency(country: str, currency: str | None) -> tuple[str, str]:
    """Validate country, default currency from country if not given. Raises ValueError."""
    country = country.strip().upper()
    if country not in SUPPORTED_COUNTRIES:
        raise ValueError(f"Unsupported country '{country}'. Supported: {sorted(SUPPORTED_COUNTRIES)}")
    currency = (currency or SUPPORTED_COUNTRIES[country]).strip().upper()
    if currency not in SUPPORTED_CURRENCIES:
        raise ValueError(f"Unsupported currency '{currency}'. Supported: {sorted(SUPPORTED_CURRENCIES)}")
    return country, currency