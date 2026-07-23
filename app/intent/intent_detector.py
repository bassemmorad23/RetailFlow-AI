from transformers import pipeline
from app.schemas.models import IntentLabel ,IntentResult
from functools import lru_cache

_CANDIDATE_LABEL_TO_ENUM = {
    "browsing or window shopping": IntentLabel.BROWSING,
    "asking about the price of a product": IntentLabel.ASKING_PRICE,
    "asking if a product is in stock or available": IntentLabel.ASKING_AVAILABILITY,
    "asking for a product recommendation": IntentLabel.WANTS_RECOMMENDATION,
    "asking for details about a product": IntentLabel.ASKING_DETAILS,
    "complaining about a problem or issue": IntentLabel.COMPLAINT,
    "ready to purchase or buy now": IntentLabel.READY_TO_BUY,
    "something else not related to shopping": IntentLabel.OTHER
}


@lru_cache(maxsize=1)
def get_pipeline():
    return pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0
    )
    
    
def detect_intent(text:str):
    if not text or not text.strip():
        return IntentResult(
            Label=IntentLabel.OTHER,
            confidence=0.0
        )
        
    model=get_pipeline()
    candidate_labels = list(_CANDIDATE_LABEL_TO_ENUM.keys())
    
    results=model(text,candidate_labels=candidate_labels)
    
    
    
    top_label=results['labels'][0]
    top_score =results['scores'][0]
    
    mapped_label =_CANDIDATE_LABEL_TO_ENUM.get(top_label,IntentLabel.OTHER)
    
    return IntentResult(
        Label=mapped_label,
        confidence=top_score 
    )
    

if __name__ == "__main__":
    print(detect_intent('How much does this red dress cost?'))
    print(detect_intent('Do you have this in size large?'))
    print(detect_intent('I want to buy this now'))
    print(detect_intent(''))

    
    

    
    