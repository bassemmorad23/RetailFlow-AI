from transformers import pipeline
from app.schemas.models import EmotionLabel ,EmotionResult
from functools import lru_cache

_MODEL_LABEL_TO_ENUM={
   "joy": EmotionLabel.HAPPY,
    "anger": EmotionLabel.ANGRY,
    "sadness": EmotionLabel.SADNESS,
    "fear": EmotionLabel.CONFUSED,
    "surprise": EmotionLabel.EXCITED,
    "disgust": EmotionLabel.ANGRY,
    "neutral": EmotionLabel.NEUTRAL,
 
}


@lru_cache(maxsize=1)
def get_pipeline():
    return pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0  # CPU explicit
        )
    
    
    
    
def detect_emotion(text:str):
    
    if not text or not text.strip():
        
        return EmotionResult(
            
            label=EmotionLabel.NEUTRAL,
            confidence=0.0,
            scores={}   
        )
        
    
    model= get_pipeline()
    raw_results=model(text)[0]
        
    scores={x['label']:x['score'] for x in raw_results}
    top=max(raw_results, key=lambda x:x['score'])
        
    label=_MODEL_LABEL_TO_ENUM.get(top['label'],EmotionLabel.NEUTRAL)
        
    return EmotionResult(
            
        label=label,
        confidence=top['score'],
        scores=scores
           
    )
    
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
