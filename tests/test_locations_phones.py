"""Region recognition (never guesses) and international phone normalization (E.164)."""

import pytest

from app.commerce.locations import resolve_region
from app.commerce.phones import normalize_phone, parse_whatsapp_sender


# ---------------------------------------------------------------- regions

@pytest.mark.parametrize("country,text,code", [
    ("EG", "Giza", "EG-GZ"), ("EG", "الجيزة", "EG-GZ"), ("EG", "الجيزه", "EG-GZ"),
    ("EG", "el giza", "EG-GZ"), ("EG", "geeza", "EG-GZ"),
    ("EG", "alex", "EG-ALX"), ("EG", "الاسكندرية", "EG-ALX"),
    ("EG", "Cairo Governorate", "EG-C"), ("EG", "محافظة القاهرة", "EG-C"),
    ("EG", "Sharkia", "EG-SHR"), ("EG", "بور سعيد", "EG-PTS"),
    ("AE", "دبي", "AE-DU"), ("AE", "Abu Dhabi", "AE-AZ"),
    ("SA", "الرياض", "SA-01"), ("SA", "الشرقية", "SA-04"),
    ("KW", "Hawalli", "KW-HA"),
])
def test_exact_matches(country, text, code):
    m = resolve_region(country, text)
    assert (m.status, m.code) == ("matched", code)


def test_same_name_resolves_per_country():
    assert resolve_region("EG", "الشرقية").code == "EG-SHR"
    assert resolve_region("SA", "الشرقية").code == "SA-04"


def test_near_miss_is_only_suggested_never_filled():
    m = resolve_region("EG", "Gizza")
    assert m.status == "suggest" and m.code is None
    assert ("EG-GZ", "Giza") in m.suggestions


@pytest.mark.parametrize("country,text", [("EG", "Narnia"), ("EG", ""), ("EG", "Dubai"), ("ZZ", "Giza")])
def test_unknown_asks_again(country, text):
    assert resolve_region(country, text).status in ("unknown", "suggest")
    assert resolve_region(country, text).code is None


def test_display_names_have_no_accents():
    assert resolve_region("KW", "Hawalli").name == "Hawalli"


# ---------------------------------------------------------------- phones

@pytest.mark.parametrize("raw,country,e164", [
    ("01012345678", "EG", "+201012345678"),
    ("010 1234 5678", "EG", "+201012345678"),
    ("٠١٠١٢٣٤٥٦٧٨", "EG", "+201012345678"),
    ("+201012345678", None, "+201012345678"),
    ("00201012345678", "SA", "+201012345678"),
    ("0501234567", "AE", "+971501234567"),
    ("+971 50 123 4567", "EG", "+971501234567"),
    ("0551234567", "SA", "+966551234567"),
])
def test_valid_phones_normalized(raw, country, e164):
    r = normalize_phone(raw, country)
    assert (r.valid, r.e164) == (True, e164)


@pytest.mark.parametrize("raw,country,reason", [
    ("", "EG", "empty"), ("call me", "EG", "unparseable"), ("0101234", "EG", "invalid"),
    ("01012345678", None, "unparseable"),
])
def test_invalid_phones(raw, country, reason):
    r = normalize_phone(raw, country)
    assert (r.valid, r.reason) == (False, reason)


def test_whatsapp_sender():
    assert parse_whatsapp_sender("201012345678").e164 == "+201012345678"
    assert not parse_whatsapp_sender("").valid