"""
Shipping configuration (chosen by the merchant at setup) and quoting.

Methods:
- platform : Shopify calculates the rate; if unavailable -> configured fallback
- rules    : fee per country / region, optional default fee, optional free-over
- fixed    : one fee, optional free-over
- manual   : merchant sets the fee when approving each order

The AI never asks anyone about shipping; it applies this configuration.
No configuration -> the AI cannot take orders.
"""

from dataclasses import dataclass
from typing import Callable, Literal

import pycountry
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ShippingMethod = Literal["platform", "rules", "fixed", "manual"]
FallbackMethod = Literal["rules", "fixed", "manual"]
MAX_RULES = 500


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _valid_country(code: str) -> str:
    code = code.strip().upper()
    if pycountry.countries.get(alpha_2=code) is None:
        raise ValueError(f"Unknown country code '{code}'")
    return code


class ShippingRule(_Strict):
    country: str = Field(min_length=2, max_length=2)
    region: str | None = Field(default=None, description="ISO 3166-2 code, e.g. EG-GZ. Omit for the whole country.")
    fee: float = Field(ge=0)

    @field_validator("country")
    @classmethod
    def _country(cls, v: str) -> str:
        return _valid_country(v)

    @model_validator(mode="after")
    def _region(self):
        if self.region is not None:
            code = self.region.strip().upper()
            sub = pycountry.subdivisions.get(code=code)
            if sub is None or sub.country_code != self.country:
                raise ValueError(f"Unknown region '{code}' for {self.country}")
            self.region = code
        return self


class RulesConfig(_Strict):
    rules: list[ShippingRule] = Field(default_factory=list, max_length=MAX_RULES)
    default_fee: float | None = Field(default=None, ge=0)
    free_over: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _no_duplicates(self):
        keys = [(r.country, r.region) for r in self.rules]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate shipping rule for the same country/region")
        if not self.rules and self.default_fee is None:
            raise ValueError("Rules need at least one rule or a default fee")
        return self


class FixedConfig(_Strict):
    fee: float = Field(ge=0)
    free_over: float | None = Field(default=None, gt=0)


class ShippingSettings(_Strict):
    method: ShippingMethod
    fallback: FallbackMethod | None = None
    rules: RulesConfig | None = None
    fixed: FixedConfig | None = None

    @model_validator(mode="after")
    def _consistent(self):
        if self.method == "platform" and self.fallback is None:
            raise ValueError("Platform shipping needs a fallback method (rules, fixed or manual)")
        if self.method != "platform" and self.fallback is not None:
            raise ValueError("A fallback is only used with platform shipping")
        effective = {self.method, self.fallback}
        if "rules" in effective and self.rules is None:
            raise ValueError("Shipping rules are required for the rules method")
        if "fixed" in effective and self.fixed is None:
            raise ValueError("A fixed fee is required for the fixed method")
        return self


@dataclass(frozen=True)
class ShippingQuote:
    fee: float | None                 # None = merchant sets it at approval
    status: Literal["quoted", "pending_merchant", "not_configured"]
    method_used: str | None
    reason: str | None = None         # region_not_covered | platform_unavailable | manual | not_configured


def _free(fee: float, free_over: float | None, subtotal: float) -> float:
    return 0.0 if free_over is not None and subtotal >= free_over else round(fee, 2)


def _quote_rules(cfg: RulesConfig, country: str, region: str | None, subtotal: float) -> ShippingQuote:
    country = (country or "").upper()
    region = (region or "").upper() or None
    by_key = {(r.country, r.region): r.fee for r in cfg.rules}
    fee = by_key.get((country, region)) if region else None
    if fee is None:
        fee = by_key.get((country, None))
    if fee is None:
        fee = cfg.default_fee
    if fee is None:
        return ShippingQuote(None, "pending_merchant", "rules", "region_not_covered")
    return ShippingQuote(_free(fee, cfg.free_over, subtotal), "quoted", "rules")


def _quote_method(settings: ShippingSettings, method: str, country, region, subtotal) -> ShippingQuote:
    if method == "rules":
        return _quote_rules(settings.rules, country, region, subtotal)
    if method == "fixed":
        return ShippingQuote(_free(settings.fixed.fee, settings.fixed.free_over, subtotal), "quoted", "fixed")
    return ShippingQuote(None, "pending_merchant", "manual", "manual")


def quote_shipping(
    settings: ShippingSettings | None,
    *,
    country: str,
    region: str | None,
    subtotal: float,
    platform_quote: Callable[[], float | None] | None = None,
) -> ShippingQuote:
    """Apply the store's configuration. platform_quote is called only for platform shipping."""
    if settings is None:
        return ShippingQuote(None, "not_configured", None, "not_configured")

    if settings.method == "platform":
        fee = None
        if platform_quote is not None:
            try:
                fee = platform_quote()
            except Exception:
                fee = None
        if fee is not None:
            return ShippingQuote(round(float(fee), 2), "quoted", "platform")
        fallback = _quote_method(settings, settings.fallback, country, region, subtotal)
        reason = fallback.reason or "platform_unavailable"
        return ShippingQuote(fallback.fee, fallback.status, fallback.method_used, reason)

    return _quote_method(settings, settings.method, country, region, subtotal)