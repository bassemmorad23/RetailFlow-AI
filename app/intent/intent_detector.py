from transformers import pipeline
from app.schemas.models import IntentLabel ,IntentResult
from functools import lru_cache

_CANDIDATE_LABEL_TO_ENUM = {
    "The customer is browsing products without asking for a specific product or action.": IntentLabel.BROWSING,

    "The customer is asking about the price or cost of a product.": IntentLabel.ASKING_PRICE,

    "The customer is asking whether a product, size, color, or variant is available.": IntentLabel.ASKING_AVAILABILITY,

    "The customer wants help choosing or recommending a product or outfit.": IntentLabel.WANTS_RECOMMENDATION,

    "The customer is asking for information or details about a product, such as material, fit, or care.": IntentLabel.ASKING_DETAILS,

    "The customer is reporting a problem, dissatisfaction, delay, damage, or other complaint.": IntentLabel.COMPLAINT,

    "The customer has decided to purchase a product or wants to place or confirm an order.": IntentLabel.READY_TO_BUY,
    
    "The customer wants to compare two or more specific products to see the differences between them.": IntentLabel.COMPARE_PRODUCTS,

    "The customer is asking something that does not fit the other shopping intents.": IntentLabel.OTHER,
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
            label=IntentLabel.OTHER,
            confidence=0.0
        )
        
    model=get_pipeline()
    candidate_labels = list(_CANDIDATE_LABEL_TO_ENUM.keys())
    
    results=model(text,candidate_labels=candidate_labels)
    
    
    
    top_label=results['labels'][0]
    top_score =results['scores'][0]
    
    mapped_label =_CANDIDATE_LABEL_TO_ENUM.get(top_label,IntentLabel.OTHER)
    
    return IntentResult(
        label=mapped_label,
        confidence=top_score 
    )
    

if __name__ == "__main__":
    print(detect_intent('How much does this red dress cost?'))
    print(detect_intent('Do you have this in size large?'))
    print(detect_intent('I want to buy this now'))
    print(detect_intent(''))

    
    

    
    