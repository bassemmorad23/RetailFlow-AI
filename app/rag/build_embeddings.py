import json
import pickle
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


_RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
_OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "embeddings" / "store_embeddings.pkl"

_MODEL_NAME = "all-MiniLM-L6-v2"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




def load_products():
    
    path=_RAW_DIR / "products.json"
    with open(path,"r",encoding="utf-8") as f:
       products = json.load(f)

    
    docs=[]
    for p in products:
        
        sizes=",".join(p.get("sizes",[]))
        
        content=(
            f"{p['title']} by {p.get('brand','Unknown')} "
            f"{p['description']} "
            f"Available sizes: {sizes}. "
            f"Category: {p.get('category', '')}. "
            f"Color: {p.get('color', '')}. "
            f"Price: {p['price']} {p.get('currency', 'EGP')}. "
            f"Stock: {p.get('stock', 0)} units."
            
        )
        
        docs.append({"source": p["id"], "content": content, "name": p["title"], "price": p["price"]})
        
    return docs




def _load_policies():
    
    path=_RAW_DIR / "policies.json"
    with open(path,"r",encoding="utf-8") as f:
        policies = json.load(f)
    
    docs=[]
    
    for p in policies:
        
        content=(f"{p['title']}. {p['content']}")
        docs.append({"source": p["id"], "content": content})

    return docs




def _load_faqs():
    
    path=_RAW_DIR / "faq.json"
    with open(path,"r",encoding="utf-8") as f:
        faqs = json.load(f)
    
    docs=[]
    for item in faqs:
        content = f"Q: {item['question']} A: {item['answer']}"
        docs.append({"source": item["id"], "content": content})
    
    return docs 



def _load_store_info() -> list[dict]:
    path = _RAW_DIR / "store_info.json"
    with open(path, encoding="utf-8") as f:
        info = json.load(f)

    docs = []

    if "contact" in info:
        c = info["contact"]
        content = (
            f"Store contact information: "
            f"Phone: {c.get('phone', '')}. "
            f"Email: {c.get('email', '')}. "
            f"Instagram: {c.get('instagram', '')}. "
            f"Facebook: {c.get('facebook', '')}."
        )
        docs.append({"source": "store_contact", "content": content})

    for key, value in info.items():
        if key == "contact":
            continue
        if isinstance(value, str):
            docs.append({"source": f"store_{key}", "content": f"{key}: {value}"})
        elif isinstance(value, dict):
            content = f"{key}: " + ", ".join(f"{k}: {v}" for k, v in value.items())
            docs.append({"source": f"store_{key}", "content": content})

    return docs



def build_and_save_embeddings():
    
    print(f"Loading source files from {_RAW_DIR}/...")
    
    all_docs = []
    loaders = [
        ("products.json",   load_products),
        ("policies.json",   _load_policies),
        ("faq.json",       _load_faqs),
        ("store_info.json", _load_store_info),
        
    ]
    
    for filename, loader in loaders:
        path = _RAW_DIR / filename
        
        if not path.exists():
            print(f"Warning: {filename} not found in {_RAW_DIR}. Skipping.")
            
            continue
        
        docs = loader()
        print(f"  {filename}: {len(docs)} chunks")
        all_docs.extend(docs)
        
        
        
    if not all_docs:
        print("No documents loaded. Aborting.")
        return

    print(f"\nTotal chunks to embed: {len(all_docs)}")
    print(f"Device: {_DEVICE}")
    print("Loading embedding model...")
    
    
    
    model=SentenceTransformer(_MODEL_NAME, device=_DEVICE)
    texts = [doc["content"] for doc in all_docs]
    print("Computing embeddings...")

    embeddings= model.encode(texts,convert_to_tensor=True,show_progress_bar=True,normalize_embeddings=True,)
   
    print(embeddings.device)
    
    payload={
        "model_name": _MODEL_NAME,
        "device": _DEVICE,
        "docs": all_docs,
        "embeddings": embeddings
    } 
    
    
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH,"wb") as f:
        pickle.dump(payload,f)
        
        print(f"\nSaved {len(all_docs)} embeddings to {_OUTPUT_PATH}")


if __name__ == "__main__":
    build_and_save_embeddings()  





        