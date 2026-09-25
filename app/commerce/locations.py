"""
Region recognition for shipping. Never guesses.

- matched : exact match (after normalization) to an official ISO 3166-2
            name or a known alias (English, Arabic, Franco)
- suggest : close but not exact -> the AI asks "Did you mean ...?"
- unknown : ask the customer again

Aliases: full for EG, AE, SA. Other countries use official names.
"""

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import pycountry

_ALIASES: dict[str, list[str]] = {
    # ---- Egypt (27 governorates)
    "EG-C": ["Cairo", "القاهرة", "qahera", "kahera", "el qahera"],
    "EG-GZ": ["Giza", "الجيزة", "gizah", "geeza", "el giza"],
    "EG-ALX": ["Alexandria", "الإسكندرية", "alex", "eskendereya", "iskandariya", "eskenderia"],
    "EG-ASN": ["Aswan", "أسوان"],
    "EG-AST": ["Asyut", "أسيوط", "assiut", "asiut"],
    "EG-BA": ["Red Sea", "البحر الأحمر", "bahr el ahmar"],
    "EG-BH": ["Beheira", "البحيرة", "behera", "buhayrah"],
    "EG-BNS": ["Beni Suef", "بني سويف", "bani sweif", "beni sweif"],
    "EG-DK": ["Dakahlia", "الدقهلية", "daqahliya", "dakahleya"],
    "EG-DT": ["Damietta", "دمياط", "domyat", "dumyat"],
    "EG-FYM": ["Faiyum", "الفيوم", "fayoum", "fayum"],
    "EG-GH": ["Gharbia", "الغربية", "gharbiya", "gharbeya"],
    "EG-IS": ["Ismailia", "الإسماعيلية", "ismailiya", "esmaelia"],
    "EG-JS": ["South Sinai", "جنوب سيناء", "ganoub sina"],
    "EG-KB": ["Qalyubia", "القليوبية", "kalyoubia", "qaliubiya", "qalyoubeya"],
    "EG-KFS": ["Kafr El Sheikh", "كفر الشيخ", "kafr elsheikh", "kafr el sheikh"],
    "EG-KN": ["Qena", "قنا", "kena"],
    "EG-LX": ["Luxor", "الأقصر", "el uqsor", "loxor"],
    "EG-MN": ["Minya", "المنيا", "menya", "el minya"],
    "EG-MNF": ["Monufia", "المنوفية", "menoufia", "minufiya", "menofeya"],
    "EG-MT": ["Matrouh", "مطروح", "marsa matrouh"],
    "EG-PTS": ["Port Said", "بورسعيد", "بور سعيد", "bor said", "borsaid"],
    "EG-SHG": ["Sohag", "سوهاج", "suhag"],
    "EG-SHR": ["Sharqia", "الشرقية", "sharkia", "sharqiya", "sharkeya"],
    "EG-SIN": ["North Sinai", "شمال سيناء", "shamal sina"],
    "EG-SUZ": ["Suez", "السويس", "el suez"],
    "EG-WAD": ["New Valley", "الوادي الجديد", "wadi gedid", "el wadi el gedid"],
    # ---- UAE (7 emirates)
    "AE-AZ": ["Abu Dhabi", "أبوظبي", "أبو ظبي", "abudhabi"],
    "AE-DU": ["Dubai", "دبي", "dubay"],
    "AE-SH": ["Sharjah", "الشارقة", "sharja"],
    "AE-AJ": ["Ajman", "عجمان"],
    "AE-UQ": ["Umm Al Quwain", "أم القيوين", "umm al qaiwain"],
    "AE-RK": ["Ras Al Khaimah", "رأس الخيمة", "rak"],
    "AE-FU": ["Fujairah", "الفجيرة", "fujeira"],
    # ---- Saudi Arabia (13 regions)
    "SA-01": ["Riyadh", "الرياض", "riyad"],
    "SA-02": ["Makkah", "مكة", "مكة المكرمة", "mecca", "makka"],
    "SA-03": ["Madinah", "المدينة", "المدينة المنورة", "medina"],
    "SA-04": ["Eastern Province", "الشرقية", "المنطقة الشرقية", "sharqiya"],
    "SA-05": ["Qassim", "القصيم", "al qassim"],
    "SA-06": ["Hail", "حائل", "ha'il"],
    "SA-07": ["Tabuk", "تبوك"],
    "SA-08": ["Northern Borders", "الحدود الشمالية"],
    "SA-09": ["Jazan", "جازان", "jizan"],
    "SA-10": ["Najran", "نجران"],
    "SA-11": ["Al Bahah", "الباحة", "baha"],
    "SA-12": ["Al Jawf", "الجوف", "jouf"],
    "SA-14": ["Asir", "عسير", "aseer"],
}

# Applied AFTER Arabic letter normalization, so the Arabic words are in normalized form.
_STRIP_WORDS = re.compile(
    r"\b(governorate|province|region|emirate|state|muhafazat|mintaqat)\b|محافظه|منطقه|اماره"
)
_AR_FIXES = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # accents + Arabic diacritics
    s = s.lower().translate(_AR_FIXES)
    s = _STRIP_WORDS.sub(" ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(al|el)\s+", "", s)            # "el giza" -> "giza"
    s = re.sub(r"(^|\s)ال", r"\1", s)              # "الجيزه" -> "جيزه"
    return re.sub(r"\s+", " ", s).strip()


def _plain(name: str) -> str:
    """Display name without accents: 'Ḩawallī' -> 'Hawalli'."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", name) if not unicodedata.combining(ch))


@dataclass(frozen=True)
class RegionMatch:
    status: Literal["matched", "suggest", "unknown"]
    code: str | None = None
    name: str | None = None
    suggestions: list[tuple[str, str]] = field(default_factory=list)  # (code, display name)


@lru_cache(maxsize=32)
def _index(country: str) -> tuple[dict[str, str], dict[str, str]]:
    """({normalized name -> code}, {code -> display name}) for one country."""
    lookup: dict[str, str] = {}
    display: dict[str, str] = {}
    for sub in pycountry.subdivisions.get(country_code=country) or []:
        aliases = _ALIASES.get(sub.code, [])
        display[sub.code] = aliases[0] if aliases else _plain(sub.name)
        for name in [sub.name, *aliases]:
            key = normalize(name)
            if key:
                lookup.setdefault(key, sub.code)
    return lookup, display


def resolve_region(country: str, text: str) -> RegionMatch:
    country = (country or "").upper()
    lookup, display = _index(country)
    key = normalize(text)
    if not key or not lookup:
        return RegionMatch("unknown")

    code = lookup.get(key)
    if code:
        return RegionMatch("matched", code, display[code])

    close = difflib.get_close_matches(key, list(lookup), n=3, cutoff=0.75)
    codes = list(dict.fromkeys(lookup[c] for c in close))
    if codes:
        return RegionMatch("suggest", suggestions=[(c, display[c]) for c in codes])
    return RegionMatch("unknown")