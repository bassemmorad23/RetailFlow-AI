from enum import Enum
from pydantic import BaseModel ,Field , field_validator
import unicodedata
from typing import Literal



# ---------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------

MAX_MESSAGE_LENGTH = 2000




class CustomerMessage(BaseModel):
    conversation_id :str
    customer_id : str
    text: str
    channel : str
    
    
    
    @field_validator("conversation_id", "customer_id", "channel")
    @classmethod
    def _reject_blank_identifiers(cls, value: str) -> str:
        """
        Memory documents and analytics records are keyed on these fields.
        A blank key doesn't fail loudly — it silently writes to the wrong
        place and corrupts data, which is far worse than an error here.
        """
        value = value.strip()
        if not value:
            raise ValueError("must not be empty or whitespace")
        return value

    @field_validator("text")
    @classmethod
    def _clean_and_bound_text(cls, value: str) -> str:
        """
        Strip control characters, reject empty, truncate over-long.

        Control chars (null bytes, ANSI escapes) can break tokenizers and
        corrupt the JSONL analytics file — one bad line makes the whole
        file harder to parse later. Newlines and tabs are kept: customers
        legitimately use them.
        """
        cleaned = "".join(
            ch for ch in value
            if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
        )
        cleaned = cleaned.strip()

        if not cleaned:
            raise ValueError("message text must not be empty or whitespace")

        if len(cleaned) > MAX_MESSAGE_LENGTH:
            cleaned = cleaned[:MAX_MESSAGE_LENGTH]

        return cleaned
    
    
    
# ---------------------------------------------------------------------------
# EMOTION
# ---------------------------------------------------------------------------



class EmotionLabel(str,Enum):
    HAPPY = "happy"
    SADNESS = "frustrated"
    EXCITED = "excited"
    CONFUSED = "confused"
    NEUTRAL = "neutral"
    ANGRY = "angry"
    
    
    
class EmotionResult(BaseModel):
    label:EmotionLabel
    confidence:float
    scores: dict[str, float]
    
    
    
# ---------------------------------------------------------------------------
# INTENT
# ---------------------------------------------------------------------------


class IntentLabel(str, Enum):
    BROWSING = "browsing"
    ASKING_PRICE = "asking_price"
    ASKING_AVAILABILITY = "asking_availability"
    WANTS_RECOMMENDATION = "wants_recommendation"
    ASKING_DETAILS = "asking_details"
    COMPLAINT = "complaint" 
    READY_TO_BUY = "ready_to_buy"
    OTHER = "other"
    
    
    
class IntentResult(BaseModel):
    label: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    
    
    


#---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------



class ConversationTurn(BaseModel):
    role: Literal["customer", "agent"]
    text: str
    
    
class KnownFacts(BaseModel):
    preferred_size: str | None = None
    preferred_color: str | None = None
    budget_max: float | None = None
    mentioned_products: list[str] = Field(default_factory=list)
    
    
class MemoryState(BaseModel):
    conversation_id : str
    history :list[ConversationTurn]
    known_facts :KnownFacts
    
    
# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    content: str
    source: str
    score: float
    name: str | None = None
    price: float | None = None
    
    
# ---------------------------------------------------------------------------
# RECOMMENDATION
# ---------------------------------------------------------------------------

class ProductRecommendation(BaseModel):
    product_id: str
    name: str
    price: float
    reason: str
    
    
# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

class AgentReply(BaseModel):
    conversation_id: str
    reply_text: str
    emotion: EmotionResult
    intent: IntentResult
    recommendations: list[ProductRecommendation]
    retrieved_context: list[RetrievedChunk]
    
 
