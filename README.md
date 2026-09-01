# StoreFlow AI

A multi-tenant AI sales agent for clothing stores that understands customer messages, retrieves each store's private knowledge, recommends products, maintains per-store conversation memory, and generates personalized replies in Arabic and English. One deployment can serve multiple independent stores, each with fully isolated data.

This project demonstrates the practical engineering work required to take an AI product idea to a deployable multi-tenant system — not just calling an LLM, but designing for failure, testing the logic, isolating tenants, and packaging it for reproducible deployment. It covers NLP inference, retrieval-augmented generation (RAG) over a cloud vector database, product recommendation, LLM integration, persistent conversation memory, fault tolerance, automated testing, a FastAPI service layer, and Docker-based deployment.

```text
Prototype → Production-hardened core → FastAPI API → Containerized deployment → Multi-tenant architecture (current) → Store onboarding & channel integrations (next)
```

The current system supports multiple independent stores in a single deployment. Each store's product catalog, policies, FAQ, store info, conversations, and known customer facts are fully isolated by `store_id`. Verified with two live stores (a clothing store and a bookstore) sharing one deployment with zero cross-tenant data leaks.

---

## Demo

The following is real, unedited output captured during local development against a store loaded from MongoDB and embedded in Qdrant Cloud.

**Turn 1 — request (`POST /chat`):**

```json
{
  "conversation_id": "test5",
  "customer_id": "9",
  "store_id": "store_001",
  "text": "I want a red dress in size M",
  "channel": "web"
}
```

**Response:**

```json
{
  "reply_text": "Great! We have the Red Summer Dress by Zara in size M — it's a lightweight floral cotton dress, perfect for warm weather, and we have 12 in stock at 1,499 EGP. Would you like to see it or try it on?",
  "emotion": {
    "label": "neutral",
    "confidence": 0.59
  },
  "intent": {
    "label": "asking_details",
    "confidence": 0.57
  },
  "recommendations": [
    {
      "product_id": "prod_001",
      "name": "Red Summer Dress",
      "price": 1499,
      "reason": "Matches your query well (score: 0.61)."
    }
  ]
}
```

**Turn 2 — same `conversation_id`, testing memory recall:**

```json
{
  "store_id": "store_001",
  "text": "what did I just ask for?"
}
```

**Response:**

```json
{
  "reply_text": "You asked for a red dress in size M — and I showed you the Red Summer Dress by Zara in that exact size and color. Would you like to see it or try it on?",
  "recommendations": [],
  "retrieved_context": []
}
```

Memory persisted across two separate HTTP requests, scoped to `store_001`.

**Multi-tenant isolation** — two live stores sharing one deployment:

```text
Query: "do you have a red dress?"
  → store_001 (clothing store) → finds prod_001 (Red Summer Dress)
  → store_002 (bookstore)      → returns nothing (correctly)

Query: "do you have a book about Python?"
  → store_001 (clothing store) → returns nothing (correctly)
  → store_002 (bookstore)      → finds book_002 (Modern Python Programming)
```

**Bilingual fact extraction** — verified separately:

```text
Input:  "عايز جاكيت أسود مقاس لارج"
Output: { "preferred_size": "L", "preferred_color": "black", "mentioned_products": ["jacket"] }
```

---

## What it does

For each customer message, the pipeline:

1. **Detects emotion** — local HuggingFace DistilRoBERTa classifier with six application-level emotion labels
2. **Detects intent** — local zero-shot BART classifier against a fixed intent taxonomy
3. **Loads conversation memory** — per-store, per-conversation history and known facts, persisted in MongoDB
4. **Retrieves store knowledge** — semantic search over the current store's private collection in Qdrant Cloud (products, policies, FAQ, store info)
5. **Recommends products** — intent-aware filtering of retrieved items, with real names and prices
6. **Generates a reply** — via OpenRouter, with a multi-model retry-and-failover chain
7. **Updates memory** — appends the turn and extracts known facts such as size, color, and budget
8. **Logs the turn** — appended to a JSONL analytics file

Every operation is scoped by `store_id` — a customer conversation with Store A never touches Store B's data.

---

## Key Engineering Decisions

- **Multi-tenant by design, not by patching.** Every request carries a `store_id`; memory queries filter on `(store_id, conversation_id)`; Qdrant has one collection per store; recommendations only see the current store's retrieved documents. Isolation is enforced at the data layer, not through defensive code sprinkled everywhere.

- **Stable module boundaries.** Storage was swapped from an in-memory dict to MongoDB, the LLM provider from HuggingFace to OpenRouter, the entrypoint from a CLI to a FastAPI HTTP API, the vector store from a local pickle file to Qdrant Cloud, and the system evolved from single-tenant to multi-tenant. `core/orchestrator.py`, the pipeline's coordinator, only ever needed to pass `store_id` through — no rewiring of the pipeline itself.

- **Independent failure handling per pipeline step.** Emotion detection, intent detection, memory access, RAG retrieval, and recommendation each catch their own exceptions and fall back to a safe default rather than propagating the failure. Response generation, the customer-facing step, falls back to a fixed polite message if every model in its chain fails.

- **Retry plus multi-model failover for response generation.** Transient errors such as rate limits and server errors are retried on the current model with exponential backoff before the pipeline moves to the next model in a configured chain.

- **LLM-based fact extraction for bilingual input.** Customer messages arrive in Arabic and English with unpredictable phrasing. An LLM-based extractor handles both from a single prompt. The output parser is defensive against failure modes observed during testing, including prose instead of JSON, JSON wrapped in code fences, and invented keys.

- **Migration script, not hand-edited data.** Store data is loaded via an idempotent migration script that reads from JSON source files and writes to MongoDB with proper indexes, then a build script pushes embeddings to Qdrant. Rebuilds are always safe — collections are dropped and recreated fresh, no stale data ever survives.

- **Tests that found real defects, not just coverage.** The unit test suite has caught a Pydantic `Literal` regression, a schema field silently dropped between refactors, and multiple crash paths from malformed LLM responses — all before they reached production.

---

## Architecture

```text
HTTP request (POST /chat, carries store_id)
      │
      ▼
api.py  ── FastAPI transport layer: request/response (de)serialization
      │
      ▼
core/orchestrator.py  ── passes store_id through the whole pipeline
      │
      ├── emotion/emotion_detector.py       → EmotionResult      (local model)
      ├── intent/intent_detector.py         → IntentResult       (local model)
      ├── memory/conversation_memory.py     → MemoryState        (MongoDB, scoped by store_id + conversation_id)
      ├── rag/retriever.py                  → list[RetrievedChunk] (Qdrant Cloud, per-store collection)
      ├── recommendation/product_recommender.py → list[ProductRecommendation]
      ├── response/response_generator.py    → reply text         (OpenRouter + retry/failover)
      ├── memory/fact_extractor.py          → known facts        (OpenRouter)
      └── analytics/conversation_logger.py  → JSONL log
```

**Data isolation by store:**

```text
MongoDB
  ├── conversations   (documents include store_id, conversation_id)
  ├── products        (documents include store_id, product_id)
  ├── policies        (documents include store_id, policy_id)
  ├── faq             (documents include store_id, faq_id)
  └── store_info      (one document per store_id, includes name/currency/address/hours/contact)

Qdrant Cloud
  ├── store_001       (one collection per store, isolated by design)
  ├── store_002
  └── ...             (each collection holds only its own store's products+policies+FAQ+store_info embeddings)
```

FastAPI's responsibility is transport. All business logic — including operation order, failure-handling policy, and store scoping — lives in the orchestrator. Each downstream module has a single isolated responsibility and exposes a typed interface.

The split between local and external inference is deliberate: emotion detection, intent detection, and query embedding run locally, while response generation and fact extraction are delegated to an external LLM through OpenRouter, and vector search runs against Qdrant Cloud.

---

## Engineering Trade-offs

| Decision                                    | Why                                                                                          | Trade-off                                                                                 |
| ------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Multi-tenant with one Qdrant collection per store | Physical separation makes cross-store leaks impossible by design                       | Adds a collection-creation step when a new store onboards                                 |
| MongoDB as source of truth for catalog data | Standard multi-tenant pattern (shared collections filtered by store_id); persists across restarts | Requires a manual embedding rebuild when a store's catalog changes (auto-sync deferred) |
| Qdrant Cloud (hosted) rather than self-hosted | Removes operational burden of running a vector DB                                          | Adds an external dependency and a network hop per query                                   |
| Local emotion/intent models                 | Avoids external latency and per-request cost for classification tasks                        | Lower accuracy ceiling than a larger hosted or fine-tuned model                           |
| Zero-shot intent classification (BART-MNLI) | No labeled conversation data exists yet; allows intent detection without a training pipeline | 70.0% accuracy and 0.67 macro F1 on n=40 (English-only starter set)                     |
| LLM-based fact extraction                   | Handles Arabic and English phrasing without separate rule sets                               | Adds one model call per turn and may occasionally fail to produce structured output       |
| Multi-model failover chain                  | Keeps conversations working when individual free-tier models rate-limit                      | Free-tier models can still rate-limit under sustained concurrent traffic                  |
| Central orchestrator                        | Keeps operation order, failure-handling policy, and store scoping in one place               | Requires stable typed interfaces between modules                                          |

---

## Evaluation

The project has deterministic unit test coverage, manual end-to-end verification, and dedicated evaluation scripts for every ML-dependent component: intent, emotion, fact extraction, RAG retrieval, and recommendation.

All datasets are small, hand-crafted starter sets, not real customer traffic. They are honest baselines, not production benchmarks.

### Intent Classification (n=40)

- Accuracy: **70.0%**, Macro F1: **0.67**
- Model: `facebook/bart-large-mnli`, zero-shot classification
- Weakest areas: `browsing`, `wants_recommendation`, `ready_to_buy`, `other`

### Emotion Classification (n=48)

- Accuracy: **75.0%**, Macro F1: **0.72**
- Model: `j-hartmann/emotion-english-distilroberta-base`
- Weakest class: `confused` (0.12 recall — frequently misread as `neutral` or `excited`)

### Fact Extraction (n=24)

- `preferred_size`: precision 1.00, recall 1.00
- `preferred_color`: precision 1.00, recall 1.00
- `budget_max`: precision 1.00, recall 1.00
- `mentioned_products`: precision 1.00, recall 0.92
- **Hallucinations: 0/24** — the extractor never invented a fact that wasn't stated in the message

### RAG Retrieval (n=22 positive, 3 negative)

- Recall@1: **90.91%**
- Recall@3: **95.45%**
- False positives on out-of-catalog queries: 0/3
- Known miss: short factual entries (e.g. store address) fall below the similarity threshold with the current small embedding model on sparse factual text.

### Recommendation (n=9 positive, 5 negative)

- Correct recommendation rate: **100%**
- Unwanted recommendation rate: 20% (1/5) — semantic similarity between "jackets" and a hoodie
- Safety check passed: a complaint referencing a real product correctly produced **no** recommendation

### Multi-Tenant Isolation (end-to-end)

Verified with two live stores sharing one deployment (store_001: clothing, store_002: bookstore):

| Query | store_001 (clothing) | store_002 (bookstore) |
| --- | --- | --- |
| "do you have a red dress?" | Finds `prod_001` (0.552) | Returns nothing |
| "do you have a book about python?" | Returns nothing | Finds `book_002` (0.661) |

Verified across memory (conversations), RAG (Qdrant collections), and recommendations. Zero cross-store leaks in either direction.

### Evaluation-Driven Fixes

Running these evaluations surfaced and fixed several real defects before they reached production: a missing retry/failover path in fact extraction, an unhandled crash on malformed API responses (in two files), a prompt-injection gap caused by a duplicated unguarded message, and a schema regression (a silently dropped field) that broke recommendation output.

### Not yet formally evaluated

Response-generation quality (groundedness, human review), API latency (p50/p95), reliability under load, and cost per conversation.

---

## Project Structure

```text
StoreFlowAI/
├── app/
│   ├── main.py                        # CLI entrypoint (dev tool)
│   ├── api.py                         # FastAPI entrypoint
│   ├── config.py                      # central settings (MongoDB + Qdrant + OpenRouter)
│   ├── logging_config.py              # centralized logging
│   ├── core/orchestrator.py           # pipeline flow + error handling (passes store_id)
│   ├── schemas/models.py              # shared Pydantic contracts (CustomerMessage requires store_id)
│   ├── emotion/emotion_detector.py
│   ├── intent/intent_detector.py
│   ├── memory/
│   │   ├── conversation_memory.py     # MongoDB-backed memory, scoped by store_id
│   │   └── fact_extractor.py          # LLM-based fact extraction
│   ├── rag/
│   │   ├── build_embeddings.py        # reads MongoDB, pushes to Qdrant per store
│   │   └── retriever.py               # queries Qdrant Cloud per store
│   ├── recommendation/
│   │   └── product_recommender.py
│   ├── response/
│   │   └── response_generator.py      # OpenRouter + retry/failover
│   └── analytics/
│       └── conversation_logger.py
├── scripts/
│   └── migrate_json_to_mongodb.py     # one-time migration of a store's JSON catalog into MongoDB
├── data/
│   ├── embeddings/                    # (gitignored, legacy — Qdrant Cloud is the current store)
│   └── analytics/                     # conversation logs (gitignored)
├── eval/
│   ├── data/                          # labeled evaluation datasets
│   ├── results/                       # timestamped evaluation results (gitignored)
│   ├── evaluate_intent.py
│   ├── evaluate_emotion.py
│   ├── evaluate_fact_extraction.py
│   ├── evaluate_rag.py
│   ├── evaluate_recommendation.py
│   └── dump_index.py                  # (legacy — for inspecting the old local pickle index)
├── tests/                             # pytest unit tests
│   └── store_002/                     # fixture data for multi-tenant isolation testing
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech Stack

| Layer                           | Technology                                          |
| ------------------------------- | --------------------------------------------------- |
| API                             | FastAPI, Uvicorn                                    |
| Data validation / configuration | Pydantic, pydantic-settings                         |
| Emotion detection               | `j-hartmann/emotion-english-distilroberta-base`     |
| Intent detection                | `facebook/bart-large-mnli` zero-shot classification |
| Query embeddings                | Sentence-Transformers (`all-MiniLM-L6-v2`)          |
| Vector store                    | Qdrant Cloud (per-store collections)                |
| Response generation             | OpenRouter, OpenAI-compatible API                   |
| Fact extraction                 | OpenRouter                                          |
| Retry logic                     | Tenacity                                            |
| Conversation memory & catalog   | MongoDB                                             |
| Testing                         | Pytest                                              |
| Evaluation                      | scikit-learn                                        |
| Containerization                | Docker, Docker Compose                              |

---

## Running with Docker

```bash
cp .env.example .env
# fill in OPENROUTER_API_KEY, MONGO_USER, MONGO_PASSWORD,
#         QDRANT_URL, QDRANT_API_KEY

docker-compose up -d --build
```

Then, for each store you want to serve, run these one-time setup steps:

```bash
# Load the store's catalog into MongoDB
python -m scripts.migrate_json_to_mongodb --store-id store_001 --currency EGP --source-dir path/to/store_001/json

# Build the store's embeddings in Qdrant Cloud
python -m app.rag.build_embeddings --store-id store_001
```

Verify:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

---

## Running Locally

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
docker-compose up -d mongo
```

Load a store's data and build its embeddings:

```powershell
python -m scripts.migrate_json_to_mongodb --store-id store_001 --currency EGP --source-dir path/to/store_001/json
python -m app.rag.build_embeddings --store-id store_001
```

Start the API:

```powershell
uvicorn app.api:app --reload
```

---

## Model Evaluation

Each script is standalone and prints per-class metrics plus a timestamped JSON result under `eval/results/`.

```powershell
python -m eval.evaluate_intent
python -m eval.evaluate_emotion
python -m eval.evaluate_fact_extraction
python -m eval.evaluate_rag
python -m eval.evaluate_recommendation
```

These are intentionally separate from pytest because they measure model behavior rather than deterministic application logic.

---

## API

### `GET /health`

Liveness check. Returns `{"status": "ok"}`. Does not require MongoDB or model inference.

### `POST /chat`

**Request:**

```json
{
  "conversation_id": "c1",
  "customer_id": "u1",
  "store_id": "store_001",
  "text": "Do you have a red dress in size M?",
  "channel": "web"
}
```

**Response:**

```json
{
  "conversation_id": "c1",
  "reply_text": "...",
  "emotion": {
    "label": "happy",
    "confidence": 0.92,
    "scores": {}
  },
  "intent": {
    "label": "wants_recommendation",
    "confidence": 0.85
  },
  "recommendations": [
    {
      "product_id": "prod_001",
      "name": "Red Summer Dress",
      "price": 1499,
      "reason": "..."
    }
  ],
  "retrieved_context": [
    {
      "content": "...",
      "source": "prod_001",
      "score": 0.61
    }
  ]
}
```

Full interactive API documentation is automatically generated at `/docs`.

---

## Testing

```powershell
python -m pytest -v
```

The deterministic unit test suite runs without models, network access, or a database. Current coverage:

- **`test_schemas.py`** — input validation, `store_id` requirement, control-character stripping
- **`test_recommender.py`** — recommendation filtering: intent gating, similarity threshold, source-type filtering
- **`test_fact_extractor.py`** — robust JSON parsing against observed LLM failure modes

Model-dependent behavior is evaluated separately through the dedicated scripts in `eval/`.

---

## Design Principles

- **One responsibility per module.**
- **Stable interfaces over convenience.**
- **Fail gracefully.**
- **Measure before optimizing.**
- **Explicit trade-offs.**
- **Real failures are documented rather than hidden.**
- **Build for the current phase without premature infrastructure.**
- **Multi-tenant isolation is enforced at the data layer, never through convention alone.**

---

## Roadmap

### Completed

- End-to-end AI sales-agent pipeline
- Emotion detection
- Intent detection
- RAG retrieval
- Product recommendation
- Conversation memory
- LLM-based fact extraction
- Response generation
- Per-step production hardening
- Retry and multi-model failover
- Input validation
- Structured logging
- Unit tests
- FastAPI service layer
- Docker deployment
- MongoDB-backed persistent memory
- Migration from local JSON to MongoDB (per-store catalog)
- Migration from local pickle vector index to Qdrant Cloud (per-store collections)
- Multi-tenant architecture (verified with two live stores, zero cross-tenant leaks)
- Intent classification evaluation
- Emotion classification evaluation
- Fact extraction evaluation
- RAG retrieval evaluation
- Recommendation evaluation

### Planned

- Store onboarding flow (self-service store creation)
- Auto-sync of catalog changes into Qdrant (currently a manual rebuild)
- Website chat widget
- WhatsApp integration
- Instagram integration
- Structured tool-calling for inventory and aggregate queries
- Expand evaluation with real labeled customer conversations
- Arabic-specific intent and emotion evaluation
- API latency and reliability benchmarks

---

## License

Proprietary — all rights reserved.

## 👨‍💻 Author

Basem Morad
AI/ML Engineer | Computer Vision Specialist
B.Sc. Mechatronics Engineering