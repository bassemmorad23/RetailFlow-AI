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



def get_memory(store_id: str, conversation_id: str) -> MemoryState:
    col = _get_collection()
    doc = col.find_one({"store_id": store_id, "conversation_id": conversation_id})
    
    if doc is None:
        fresh = MemoryState(
            conversation_id=conversation_id,
            history=[],
            known_facts=KnownFacts(),
        )
        doc_to_save = fresh.model_dump()
        doc_to_save["store_id"] = store_id
        col.insert_one(doc_to_save)
        return fresh

    return _doc_to_memory(doc)








def _doc_to_memory(doc:dict):
    
    return MemoryState(
        conversation_id=doc["conversation_id"],
        history=[ConversationTurn(role=t["role"],text=t["text"]) for t in doc.get("history",[])],
        known_facts=KnownFacts(**doc.get("known_facts",{}))
        
    )
    
    
    




def add_turn(store_id: str, conversation_id: str, role: str, text: str) -> MemoryState:
    memory = get_memory(store_id, conversation_id)
    memory.history.append(ConversationTurn(role=role, text=text))
    memory.history = memory.history[-_MAX_HISTORY_TURNS:]

    col = _get_collection()
    col.update_one(
        {"store_id": store_id, "conversation_id": conversation_id},
        {"$set": {"history": [t.model_dump() for t in memory.history]}},
    )
    return memory






def update_known_facts(store_id: str, conversation_id: str, **facts) -> MemoryState:
    memory = get_memory(store_id, conversation_id)
    updated_facts = memory.known_facts.model_copy(update=facts)
    memory.known_facts = updated_facts

    col = _get_collection()
    col.update_one(
        {"store_id": store_id, "conversation_id": conversation_id},
        {"$set": {"known_facts": updated_facts.model_dump()}},
    )
    return memory 
  
    







