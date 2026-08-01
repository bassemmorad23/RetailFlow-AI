from app.schemas.models import IntentLabel,IntentResult ,KnownFacts ,MemoryState ,ProductRecommendation,RetrievedChunk 



_RECOMMENDATION_INTENTS = {
    IntentLabel.WANTS_RECOMMENDATION,
    IntentLabel.BROWSING,
    IntentLabel.ASKING_DETAILS,
    IntentLabel.ASKING_AVAILABILITY,
    IntentLabel.READY_TO_BUY,
}

_MIN_SCORE_THRESHOLD = 0.4
_PRODUCT_SOURCE_PREFIX = "product_"





def _source_to_name(source:str):
    
    without_prefix=source.removeprefix(_PRODUCT_SOURCE_PREFIX)
    
    return without_prefix.replace("_", " ").title()



def _build_reason(chunk: RetrievedChunk, known_facts: KnownFacts) -> str:
    base = f"Matches your query well (score: {chunk.score:.2f})."

    if known_facts.preferred_size:
        base += f" Available in your preferred size ({known_facts.preferred_size})."

    if known_facts.preferred_color:
        base += f" Comes in {known_facts.preferred_color}."

    return base








def recommend_products(intent:IntentResult,memory:MemoryState,retrieved_context:list[RetrievedChunk]):
    
    if intent.label not in _RECOMMENDATION_INTENTS:
        return []
    
    if not retrieved_context:
        return []
    
    known_facts: KnownFacts = memory.known_facts
    
    recommendations: list[ProductRecommendation] = []
    
    for chunk in retrieved_context :
        if not chunk.source.startswith(_PRODUCT_SOURCE_PREFIX):
            continue
        
        if chunk.score < _MIN_SCORE_THRESHOLD:
            continue
        
        reason=_build_reason(chunk,known_facts)
        display_name = _source_to_name(chunk.source)
        
        recommendations.append(
            ProductRecommendation(
                product_id=chunk.source,
                name=display_name,
                price=0.0,
                reason=reason,
            )
        )

    return recommendations
        