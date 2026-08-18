# StoreFlow AI

An end-to-end AI sales agent for clothing stores that understands customer messages, retrieves relevant store knowledge, recommends products, maintains conversation memory, and generates personalized replies in Arabic and English.

Built to demonstrate production-oriented AI engineering across the full pipeline: NLP inference, RAG, product recommendation, LLM integration, persistent memory, fault tolerance, testing, FastAPI, and Docker.


```
Prototype ✅  →  Production-hardened core ✅  →  FastAPI ✅  →  Docker deployment ✅  →  Multi-tenant SaaS (next)
```

---

## Engineering highlights

A few decisions worth a technical reviewer's attention:

- **Zero-rewrite architecture.** Storage was swapped from an in-memory dict to MongoDB, the LLM provider from HuggingFace to OpenRouter, and the entrypoint from a CLI to a FastAPI HTTP API — `core/orchestrator.py`, the pipeline's central coordinator, was never modified across any of these changes. Every module is called through a fixed function signature; nothing depends on another module's internals.

- **Every pipeline step degrades independently.** Emotion detection, intent detection, memory, RAG, and recommendations each fail closed to a safe default (e.g. neutral emotion, empty recommendation list) if their underlying call throws. Response generation — the one customer-facing step — falls back to a polite message rather than a raw error. No single component failure can break a conversation.

- **Multi-model failover with two-layer retry.** Response generation retries transient errors (rate limits, server errors) on the current model with exponential backoff, then fails over to the next model in a configured chain if retries are exhausted. Verified live: a rate-limited primary model was automatically bypassed in favor of a working backup, with no customer-visible failure.

- **LLM-based fact extraction, chosen for a specific reason.** Customer messages arrive in both Arabic and English with unpredictable phrasing. A regex-based extractor would need a hand-written rule set per language and still miss most real phrasing; an LLM extractor handles both languages with one prompt. The parser is defensively written against the extractor's real observed failure modes (models returning prose instead of JSON, wrapping JSON in code fences, inventing keys) — verified with a dedicated test suite built directly from failures seen in live testing.

- **Self-contained, reproducible deployment.** The Docker image pre-downloads all local ML models and pre-builds the RAG vector index at *build time*, not first request — the container never depends on an external service being reachable at cold start. Verified end-to-end: conversation memory correctly persisted across two separate HTTP requests through the containerized MongoDB instance, confirmed via Docker's internal service-name networking (a common misconfiguration point that was caught and fixed during testing).

- **Tests catch real bugs, not just coverage.** The unit test suite found a live type-safety regression (a `Literal` type constraint that had silently reverted to an unrestricted `str`) and was subsequently extended to directly reproduce and guard against a bug seen in production output (a case-sensitive attribute typo silently breaking analytics logging on every request until caught).

---

## What it does

For each customer message, the pipeline:

1. **Detects emotion** — local HuggingFace DistilRoBERTa classifier
2. **Detects intent** — local zero-shot BART classifier against a fixed intent taxonomy
3. **Loads conversation memory** — per-conversation history and known facts, persisted in MongoDB
4. **Retrieves store knowledge** — semantic search (RAG) over products, policies, FAQ, and store info
5. **Recommends products** — intent-aware filtering of retrieved items, with real names and prices
6. **Generates a reply** — via OpenRouter, with a multi-model failover chain
7. **Updates memory** — appends the turn, and extracts known facts (size, color, budget) from the message
8. **Logs the turn** — appended to a JSONL analytics file for future fine-tuning and reporting

---

## Architecture

```
HTTP request (POST /chat)
      │
      ▼
api.py  ── thin FastAPI wrapper, no business logic
      │
      ▼
core/orchestrator.py  ── owns the pipeline flow; never modified across any module swap
      │
      ├── emotion/emotion_detector.py            → EmotionResult
      ├── intent/intent_detector.py              → IntentResult
      ├── memory/conversation_memory.py          → MemoryState (MongoDB)
      ├── rag/retriever.py                       → list[RetrievedChunk]
      ├── recommendation/product_recommender.py  → list[ProductRecommendation]
      ├── response/response_generator.py         → reply text (OpenRouter, retry + failover)
      ├── memory/fact_extractor.py               → known facts (LLM extraction)
      └── analytics/conversation_logger.py       → JSONL log
```

Modules never call each other directly — only the orchestrator knows the flow, and each module exposes a single typed function. This is the boundary that made every major swap so far a one-file change.

---

## Project structure

```
StoreFlowAI/
├── app/
│   ├── main.py                        # CLI entrypoint (dev tool)
│   ├── api.py                         # FastAPI entrypoint
│   ├── config.py                      # central settings (pydantic-settings)
│   ├── logging_config.py              # centralized logging setup
│   ├── core/orchestrator.py           # pipeline flow, per-step error handling
│   ├── schemas/models.py              # shared Pydantic data contracts + validation
│   ├── emotion/emotion_detector.py
│   ├── intent/intent_detector.py
│   ├── memory/
│   │   ├── conversation_memory.py     # MongoDB-backed memory
│   │   └── fact_extractor.py          # LLM-based fact extraction
│   ├── rag/
│   │   ├── build_embeddings.py        # offline: build the vector index
│   │   └── retriever.py               # runtime: semantic search
│   ├── recommendation/product_recommender.py
│   ├── response/response_generator.py # OpenRouter + retry + model failover chain
│   └── analytics/conversation_logger.py
├── data/
│   ├── raw/                           # products.json, policies.json, faq.json, store_info.json
│   ├── embeddings/                    # generated vector index (gitignored)
│   └── analytics/                     # conversation logs (gitignored)
├── tests/                             # pytest unit tests (30+)
├── Dockerfile                         # self-contained image, models pre-baked
├── docker-compose.yml                 # app + MongoDB
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI, uvicorn |
| Data validation / config | Pydantic, pydantic-settings |
| Emotion & intent detection | HuggingFace Transformers (local inference) |
| RAG embeddings | Sentence-Transformers (local inference) |
| Response generation | OpenRouter (OpenAI-compatible API), multi-model failover |
| Retry logic | tenacity |
| Conversation memory | MongoDB |
| Testing | pytest |
| Containerization | Docker, Docker Compose |

---

## Running with Docker

The image is self-contained — models and the RAG index are built in, so no manual setup is needed after `docker-compose up`.

```bash
cp .env.example .env
# fill in OPENROUTER_API_KEY, MONGO_USER, MONGO_PASSWORD

docker-compose up -d --build
```

Verify:
```
http://localhost:8000/health
http://localhost:8000/docs
```

## Running locally (development)

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate         # macOS/Linux

pip install -r requirements.txt
cp .env.example .env               # fill in OPENROUTER_API_KEY, MONGO_URI

docker-compose up -d mongo         # MongoDB only
python -m app.rag.build_embeddings # build the RAG index once

python -m app.main                 # CLI
# or
uvicorn app.api:app --reload       # API with auto-reload
```

---

## API

### `GET /health`
Liveness check. Returns `{"status": "ok"}` without touching MongoDB or any model.

### `POST /chat`

**Request:**
```json
{
  "conversation_id": "c1",
  "customer_id": "u1",
  "text": "Do you have a red dress in size M?",
  "channel": "web"
}
```

**Response:**
```json
{
  "conversation_id": "c1",
  "reply_text": "...",
  "emotion": { "label": "happy", "confidence": 0.92, "scores": {} },
  "intent": { "label": "wants_recommendation", "confidence": 0.85 },
  "recommendations": [
    { "product_id": "prod_001", "name": "Red Summer Dress", "price": 1499, "reason": "..." }
  ],
  "retrieved_context": [ { "content": "...", "source": "prod_001", "score": 0.61 } ]
}
```

Full interactive documentation is auto-generated at `/docs`.

---

## Testing

```bash
python -m pytest -v
```

30+ deterministic unit tests, running in under a second — no models, no network, no database required:

- **`test_schemas.py`** — input validation: blank/empty rejection, over-length truncation, control-character stripping
- **`test_recommender.py`** — recommendation filtering: intent gating, similarity threshold, source-type filtering
- **`test_fact_extractor.py`** — JSON parsing robustness against real observed LLM failure modes

---

## Design principles

- **One responsibility per module.**
- **Evolve without rewriting** — every dependency swap so far has been isolated to a single module.
- **Fail gracefully** — every pipeline step is independently fault-tolerant.
- **Explicit trade-offs** — limitations are documented, not hidden.
- **Build for the current phase** — no infrastructure added before it's needed.

---

## Known limitations

- **RAG cannot answer aggregate questions** (e.g. "how many products total?"). Semantic search finds similar items, not counts — this requires structured tool-calling, planned for a later phase.
- **Free-tier OpenRouter models rate-limit under load.** The failover chain cushions this; a production deployment at real volume should place a paid model first.
- **Fact extraction occasionally misses on free models** that return prose instead of structured JSON — handled safely (no facts extracted that turn), never corrupts existing memory.
- **Zero-shot intent classification has a real error rate** — no training data exists yet to fine-tune against.
- **Single-tenant** — no `tenant_id` isolation yet; one deployment serves one store.

---

## Roadmap

**Shipped:** prototype pipeline · production hardening (error handling, retry + failover, input validation, LLM fact extraction, structured logging, tests) · FastAPI API · Docker deployment, verified end-to-end.

**Next:** onboard a first real store · structured tool-calling for inventory queries · WhatsApp/Instagram integrations · multi-tenant architecture.

---

## License

Proprietary — all rights reserved.