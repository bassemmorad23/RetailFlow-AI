# StoreFlow AI

![API Status](https://img.shields.io/website?url=https%3A%2F%2Fretailflow-ai-production.up.railway.app%2Fhealth&label=live%20api&up_message=up&down_message=down)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/fastapi-0.115-teal)
![Deployment](https://img.shields.io/badge/deployed-Railway-8B5CF6)
![License](https://img.shields.io/badge/license-Proprietary-red)

A multi-tenant AI sales agent for retail stores, built from prototype to live production deployment.

StoreFlow AI helps retail stores answer customer questions, search their own catalog and store knowledge, recommend relevant products, remember customer information across conversations, and serve multiple independent stores from one backend.

**Live API:** [retailflow-ai-production.up.railway.app](https://retailflow-ai-production.up.railway.app) → interactive Swagger UI
**Author:** [Basem Morad](https://github.com/bassemmorad23) — AI/ML Engineer

> **Current status:** The backend is live in production. The website chat widget and self-service onboarding are still in progress, and catalog-to-Qdrant synchronization is currently a manual rebuild. This file is honest about what's built and what isn't.

One deployment serves multiple independent stores. Each store's data (catalog, policies, conversations, extracted customer facts) is fully isolated by `store_id` across MongoDB Atlas and Qdrant Cloud. Verified with two live stores sharing one deployment — zero cross-tenant data leaks.

---

## What it does

For each customer message, the pipeline:

1. Detects emotion (local DistilRoBERTa)
2. Detects intent (local zero-shot BART)
3. Loads per-store conversation memory (MongoDB Atlas)
4. Retrieves store knowledge (Qdrant Cloud, per-store collections)
5. Recommends products (intent-filtered)
6. Generates a reply (OpenRouter with multi-model failover)
7. Extracts and persists customer facts (LLM-based, Arabic + English)
8. Records structured JSON logs and operational metrics

Every step is `store_id`-scoped. A conversation with Store A never touches Store B's data.

---

## Live demo

Real output captured from the production API against a store loaded in MongoDB Atlas and embedded in Qdrant Cloud.

**Request:**

```json
POST /chat
{
  "conversation_id": "demo1",
  "customer_id": "u1",
  "store_id": "store_001",
  "text": "I want a red dress in size M",
  "channel": "web"
}
```

**Response:**

```json
{
  "reply_text": "Great! We have the Red Summer Dress by Zara in size M — lightweight floral cotton, 12 in stock at 1,499 EGP. Would you like to see it or try it on?",
  "emotion":  { "label": "neutral", "confidence": 0.59 },
  "intent":   { "label": "asking_details", "confidence": 0.57 },
  "recommendations": [
    { "product_id": "prod_001", "name": "Red Summer Dress", "price": 1499 }
  ]
}
```

**Multi-tenant isolation, end-to-end:**

```
"do you have a red dress?"           → store_001 finds prod_001 (score 0.552)
"do you have a red dress?"           → store_002 returns nothing (correctly)
"do you have a book about python?"   → store_001 returns nothing (correctly)
"do you have a book about python?"   → store_002 finds book_002 (score 0.661)
```

**Bilingual fact extraction** (Arabic input, structured output):

```
Input:  "عايز جاكيت أسود مقاس لارج"
Output: { "preferred_size": "L", "preferred_color": "black", "mentioned_products": ["jacket"] }
```

---

## Engineering highlights

- **Multi-tenant by design, not by patching.** Every request carries `store_id`; memory queries filter on `(store_id, conversation_id)`; each store gets its own Qdrant collection; recommendations only see the current store's retrieved documents. Isolation is enforced at the data layer, not through defensive code sprinkled around the app.

- **Stable module boundaries.** Storage swapped in-memory → MongoDB, LLM provider HuggingFace → OpenRouter, entrypoint CLI → FastAPI, vector store local pickle → Qdrant Cloud, single-tenant → multi-tenant. The orchestrator only ever needed to pass `store_id` through — no rewiring of the pipeline itself.

- **Independent failure handling per pipeline step.** Every step catches its own exceptions and returns a safe fallback rather than propagating the failure. Response generation, the customer-facing step, falls back to a fixed polite message only after every model in the chain fails.

- **Retry + multi-model failover.** Transient errors are retried with exponential backoff on the same model; persistent failures move to the next model in the configured chain.

- **LLM-based fact extraction for bilingual input.** Handles Arabic and English from one prompt. The output parser is defensive against real observed failure modes: prose instead of JSON, JSON wrapped in code fences, invented keys.

- **Evaluation-driven, not evaluation-decorative.** The dedicated eval suite found and fixed real defects before they reached production, including a missing retry/failover path, a `NoneType` crash on empty LLM responses (in two files), a prompt-injection gap from a duplicated unguarded message, and a schema regression that silently broke recommendations.

- **Documented trade-offs.** Every non-trivial technical decision — with the alternative considered and the honest trade-off — is written down in [`docs/decisions.md`](docs/decisions.md).

---

## Production stack (live)

| Layer | Service | Purpose |
| --- | --- | --- |
| API host | Railway | FastAPI backend, public HTTPS URL |
| Catalog + memory DB | MongoDB Atlas | Products, policies, FAQ, store info, conversations |
| Vector search | Qdrant Cloud | Per-store collections, semantic retrieval |
| LLM | OpenRouter | Response generation + fact extraction, multi-model chain |
| Error monitoring | Sentry | Auto-capture of exceptions with request context |
| Uptime monitoring | UptimeRobot | 5-minute health checks + alerts |

---

## Evaluation

> **Important:** These are small, hand-crafted starter datasets intended as engineering baselines — not production benchmarks. Real customer data will replace them as it becomes available.

| Component | Result |
| --- | --- |
| Intent classification (n=40) | 70.0% accuracy, 0.67 macro F1 |
| Emotion classification (n=48) | 75.0% accuracy, 0.72 macro F1 |
| Fact extraction (n=24) | 100% precision on every field, 0 hallucinations on n=24 |
| RAG retrieval (n=25) | 90.9% recall@1, 95.5% recall@3, 0 false positives on out-of-catalog queries |
| Recommendation (n=14) | 100% correct on positive cases; safety gate blocks recommendations during complaints |
| Multi-tenant isolation | Verified across memory, RAG, and recommendation — zero cross-store leaks |

Details: [`docs/evaluation.md`](docs/evaluation.md)

---

## Observability

- **Structured JSON logs** with per-request context propagated across every module (`request_id`, `store_id`, `conversation_id`).
- **Fallback events** logged with the failing model, failure reason (`timeout`, `rate_limit`, `empty_response`, etc.), and latency.
- **Per-step pipeline timing** — every stage of the pipeline records its own latency, so slow requests can be traced to the actual bottleneck.
- **Rotating file logs** at `data/logs/app.jsonl` (10MB per file, 5 backups) for cross-restart debugging.
- **Sentry** for automatic error alerting with request context.
- **`GET /metrics`** endpoint exposes request count, per-status breakdown, per-store breakdown, average and p95 latency, and error rate.
- **Rate limiting** on `/chat` (20 req/min per IP) and **30s timeouts** on external LLM calls to protect against runaway costs.

---

## Architecture

```
Customer
   │
   ▼
POST /chat
   │
   ▼
FastAPI
(api.py — transport, CORS, rate limit, request context, metrics)
   │
   ▼
Core Orchestrator
(times each step, isolates failures, passes store_id)
   │
   ├── Emotion Detector ───────────────► local model
   ├── Intent Detector ────────────────► local model
   ├── Conversation Memory ────────────► MongoDB Atlas
   ├── RAG Retriever ──────────────────► Qdrant Cloud
   ├── Product Recommender
   ├── Response Generator ─────────────► OpenRouter
   ├── Fact Extractor ─────────────────► OpenRouter
   └── Conversation Logger ────────────► JSONL
```

**Data isolation:**

```
MongoDB Atlas                     Qdrant Cloud
  conversations                     store_001  (only store_001 data)
  products                          store_002  (only store_002 data)
  policies                          ...
  faq
  store_info
  → every document tagged with store_id
```

Full detail: [`docs/architecture.md`](docs/architecture.md)

---

## Tech stack

| Layer | Technology |
| --- | --- |
| API | FastAPI, Uvicorn |
| Data validation | Pydantic v2, pydantic-settings |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` |
| Intent | `facebook/bart-large-mnli` (zero-shot) |
| Query embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector store | Qdrant Cloud |
| Response + fact extraction | OpenRouter (OpenAI-compatible API) |
| Retry | Tenacity |
| Memory + catalog | MongoDB Atlas |
| Container | Docker |
| Deployment | Railway |
| Error monitoring | Sentry |
| Uptime monitoring | UptimeRobot |
| Testing | Pytest |

---

## Run locally

```bash
# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
cp .env.example .env               # then fill in credentials

# Load one store's data + build its embeddings
python -m scripts.migrate_json_to_mongodb --store-id store_001 --currency EGP --source-dir path/to/store_001/json
python -m app.rag.build_embeddings --store-id store_001

# Run
uvicorn app.api:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) — redirects to interactive API docs.

Full setup guide: [`docs/deployment.md`](docs/deployment.md)

---

## Roadmap

**Completed**
- Multi-tenant architecture (data + memory + RAG + recommendation, all `store_id`-scoped, verified with 2 live stores)
- Production deployment (Railway + MongoDB Atlas + Qdrant Cloud)
- Observability (structured JSON logs, request context, per-step timing, Sentry, `/metrics`, UptimeRobot)
- Rate limiting, external-call timeouts, request validation
- Evaluation suite for intent, emotion, fact extraction, RAG, and recommendation

**Next**
- Website chat widget (frontend integration in progress)
- Store onboarding flow (self-service store creation)
- Auto-sync of catalog changes into Qdrant (currently a manual rebuild)
- WhatsApp + Instagram channel adapters
- Structured tool-calling for inventory and aggregate queries
- Expand evaluation with real labeled customer conversations
- Arabic-specific intent and emotion evaluation

---

## Author

**Basem Morad** — AI/ML Engineer | Computer Vision | B.Sc. Mechatronics Engineering

- GitHub: [bassemmorad23](https://github.com/bassemmorad23)
- Live API: [retailflow-ai-production.up.railway.app](https://retailflow-ai-production.up.railway.app)

---

## License

Proprietary — all rights reserved.