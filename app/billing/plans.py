"""
Plan configuration for StoreFlow AI.

Plans define monthly AI-message limits per store. Enterprise limits
are overridden per-store in StoreSettings.plan_overrides.

CONFIGURABLE: change limits here without touching business logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    """A subscription plan with its monthly AI message limit."""
    name: str
    display_name: str
    price_egp: int  # 0 for enterprise (custom)
    monthly_message_limit: int  # 0 = unlimited (only for custom enterprise)


# ---- Plan catalog ----
# Change these values to update pricing/limits without touching enforcement logic.

PLANS: dict[str, Plan] = {
    "starter": Plan(
        name="starter",
        display_name="Starter",
        price_egp=7000,
        monthly_message_limit=16000,
    ),
    "pro": Plan(
        name="pro",
        display_name="Pro",
        price_egp=9600,
        monthly_message_limit=25000,
    ),
    "enterprise": Plan(
        name="enterprise",
        display_name="Enterprise",
        price_egp=0,  # custom, set per store
        monthly_message_limit=0,  # 0 = read from StoreSettings.enterprise_limit
    ),
}

# Reply sent when a store hits its daily cap or monthly limit.
LIMIT_REACHED_REPLY = (
    "Our assistant is temporarily unavailable. "
    "Our team will get back to you as soon as possible."
)


# ---- Daily safety cap ----
# Hard stop regardless of plan, prevents runaway loops.
# Change when moving to paid models + real $ tracking.

DAILY_MESSAGE_CAP_PER_STORE = 300


def get_plan(plan_name: str) -> Plan:
    """Return plan by name; falls back to starter if unknown."""
    return PLANS.get(plan_name, PLANS["starter"])


def get_monthly_limit(plan_name: str, enterprise_limit: int | None = None) -> int:
    """
    Get monthly message limit for a store.
    Enterprise stores use their custom limit from StoreSettings.
    Returns 0 for truly unlimited (only enterprise with no limit set).
    """
    plan = get_plan(plan_name)
    if plan.name == "enterprise":
        return enterprise_limit if enterprise_limit is not None else 0
    return plan.monthly_message_limit