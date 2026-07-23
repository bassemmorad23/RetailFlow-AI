from functools import lru_cache
from pymongo import MongoClient
from pymongo.collection import Collection






from app.config import settings
from app.schemas.models import ConversationTurn ,KnownFacts ,MemoryState




_MAX_HISTORY_TURNS = 20




@lru_cache(maxsize=1)
def _get_collection():
    
    client=MongoClient(settings.MONGO_URI)
    db=client[settings.MONGO_DB]
    return db["conversations"]





def _doc_to_memory(doc:dict):
    
    return MemoryState(
        conversation_id=doc["conversation_id"],
        history=[ConversationTurn(role=t["role"],text=t["text"]) for t in doc.get("history",[])],
        known_facts=KnownFacts(**doc.get("known_facts",{}))
        
    )
    
    
    
    
    
def get_memory(conversation_id: str):
    
    col=_get_collection()
    doc=col.find_one({"conversation_id":conversation_id})
    
    if doc is None:
        fresh=MemoryState(conversation_id=conversation_id,history=[],known_facts=KnownFacts())
        col.insert_one(fresh.model_dump())
        return fresh
    
    return _doc_to_memory(doc)





def add_turn(conversation_id: str, role: str, text: str):
    
    memory=get_memory(conversation_id)
    memory.history.append(ConversationTurn(role=role,text=text))
    memory.history=memory.history[-_MAX_HISTORY_TURNS:]
    
    col = _get_collection()
    col.update_one(
        {"conversation_id": conversation_id},
        {"$set": {"history": [t.model_dump() for t in memory.history]}},
    )
    return memory




def update_known_facts(conversation_id: str, **facts) -> MemoryState:
    """
    Update one or more known facts for a conversation without touching
    other fields. Only explicitly passed kwargs are overwritten.
    """
    memory = get_memory(conversation_id)
    updated_facts = memory.known_facts.model_copy(update=facts)
    memory.known_facts = updated_facts

    col = _get_collection()
    col.update_one(
        {"conversation_id": conversation_id},
        {"$set": {"known_facts": updated_facts.model_dump()}},
    )
    return memory


    
  
    







