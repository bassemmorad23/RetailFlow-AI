from functools import lru_cache
from pymongo import MongoClient , ReturnDocument
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError
from app.config import settings
from app.schemas.models import ConversationTurn ,KnownFacts ,MemoryState


_MAX_HISTORY_TURNS = 20

@lru_cache(maxsize=1)
def _get_collection():
    client = MongoClient(settings.MONGO_URI)
    col = client[settings.MONGO_DB]["conversations"]
    col.create_index([("store_id", 1), ("conversation_id", 1)], unique=True)
    return col



def get_memory(store_id: str, conversation_id: str) -> MemoryState:
    """Load memory, creating it atomically if missing (safe under concurrent first messages)."""
    col = _get_collection()
    flt = {"store_id": store_id, "conversation_id": conversation_id}
    try:
        doc = col.find_one_and_update(
            flt,
            {"$setOnInsert": {"history": [], "known_facts": KnownFacts().model_dump()}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError:
        doc = col.find_one(flt)
    return _doc_to_memory(doc)


def _doc_to_memory(doc:dict):
    
    return MemoryState(
        conversation_id=doc["conversation_id"],
        history=[ConversationTurn(role=t["role"],text=t["text"]) for t in doc.get("history",[])],
        known_facts=KnownFacts(**doc.get("known_facts",{}))
        
    )
    

def add_turn(store_id: str, conversation_id: str, role: str, text: str) -> MemoryState:
    """Append a turn atomically, keeping only the last _MAX_HISTORY_TURNS."""
    col = _get_collection()
    col.update_one(
        {"store_id": store_id, "conversation_id": conversation_id},
        {"$push": {"history": {
            "$each": [ConversationTurn(role=role, text=text).model_dump()],
            "$slice": -_MAX_HISTORY_TURNS,
        }}},
        upsert=True,
    )
    return get_memory(store_id, conversation_id)



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
  
    







