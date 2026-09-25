"""
Store policies written by the merchant (plain text per topic).
The AI answers policy questions ONLY from these; an empty topic means
"the store will confirm" — never an invented policy.
"""

from pydantic import BaseModel, ConfigDict, Field

POLICY_TOPICS = ("returns", "exchanges", "refunds", "delivery", "warranty", "other")
MAX_POLICY_CHARS = 2000


class StorePolicies(BaseModel):
    model_config = ConfigDict(extra="forbid")
    returns: str = Field(default="", max_length=MAX_POLICY_CHARS)
    exchanges: str = Field(default="", max_length=MAX_POLICY_CHARS)
    refunds: str = Field(default="", max_length=MAX_POLICY_CHARS)
    delivery: str = Field(default="", max_length=MAX_POLICY_CHARS)
    warranty: str = Field(default="", max_length=MAX_POLICY_CHARS)
    other: str = Field(default="", max_length=MAX_POLICY_CHARS)

    def get(self, topic: str) -> str:
        return (getattr(self, topic, "") or "").strip()