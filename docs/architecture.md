# Architecture

Deep dive into how StoreFlow AI is structured, why each piece exists, and how multi-tenant isolation is enforced end-to-end.

For a high-level overview, see the [main README](../README.md).

---

## Request lifecycle

```
HTTP request (POST /chat, carries store_id)
      │
      ▼
api.py  ── FastAPI transport layer
      │      • CORS middleware (allows the widget from any store's domain)
      │      • Rate limit (per-IP, in-memory sliding window)
      │      • Request context middleware (generates request_id, sets store_id / conversation_id)
      │      • Metrics recording (status code, latency, store_id)
      │
      ▼
core/orchestrator.py  ── the single place that owns pipeline shape
      │      • Times each step
      │      • Isolates failures per step
      │      • Passes store_id through the whole pipeline
      │
      ├── emotion/emotion_detector.py       → EmotionResult      (local DistilRoBERTa)
      ├── intent/intent_detector.py         → IntentResult       (local BART zero-shot)
      ├── memory/conversation_memory.py     → MemoryState        (MongoDB Atlas)
      ├── rag/retriever.py                  → RetrievedChunk[]   (Qdrant Cloud)
      ├── recommendation/product_recommender.py → ProductRecommendation[]
      ├── response/response_generator.py    → reply text         (OpenRouter + retry/failover)
      ├── memory/fact_extractor.py          → known facts        (OpenRouter)
      └── analytics/conversation_logger.py  → JSONL log
      │
      ▼
AgentReply (JSON response back to caller)
```

---

## Module responsibilities

Every module has one job and one typed interface. This is what allows storage backends, LLM providers, and even the deployment topology to change without rewiring the pipeline.

| Module | Input | Output | External dependency |
|---|---|---|---|
| `api.py` | HTTP request | HTTP response | FastAPI |
| `core/orchestrator.py` | `CustomerMessage` | `AgentReply` | — (coordinator only) |
| `emotion/emotion_detector.py` | `str` (text) | `EmotionResult` | local HF model |
| `intent/intent_detector.py` | `str` (text) | `IntentResult` | local HF model |
| `memory/conversation_memory.py` | `store_id`, `conversation_id` | `MemoryState` | MongoDB Atlas |
| `rag/retriever.py` | `store_id`, `str` (query) | `RetrievedChunk[]` | Qdrant Cloud |
| `recommendation/product_recommender.py` | `intent`, `memory`, `retrieved_context` | `ProductRecommendation[]` | — (pure logic) |
| `response/response_generator.py` | full pipeline context | `str` (reply) | OpenRouter |
| `memory/fact_extractor.py` | `str` (text) | `dict` (extracted facts) | OpenRouter |
| `analytics/conversation_logger.py` | `CustomerMessage`, `AgentReply` | — (writes JSONL) | local filesystem |
| `rate_limiter.py` | request IP | raises 429 or returns None | — (in-memory) |
| `metrics.py` | request stats | metrics dict | — (in-memory) |
| `log_context.py` | context fields | context fields | — (Python `contextvars`) |

---

## Data isolation — multi-tenant by design

### MongoDB Atlas

Every collection includes `store_id` on every document. Queries always filter on `store_id` first.

```
MongoDB Atlas
  conversations   { store_id, conversation_id, history, known_facts }
  products        { store_id, product_id, name, price, ... }
  policies        { store_id, policy_id, title, body }
  faq             { store_id, faq_id, question, answer }
  store_info      { store_id, name, currency, address, ... }
```

Compound indexes on `(store_id, <local_id>)` for fast per-store queries and uniqueness enforcement.

### Qdrant Cloud

Each store gets its own Qdrant collection, named after `store_id`. Retrieval only queries the current store's collection — cross-store leaks are impossible by design, not by "always remember to filter."

```
Qdrant Cloud
  store_001   (only store_001's products + policies + FAQ + store_info embeddings)
  store_002   (only store_002's ...)
  ...
```

### Verified isolation

Tested with two live stores sharing one deployment (`store_001`: clothing, `store_002`: bookstore):

| Query | store_001 result | store_002 result |
|---|---|---|
| "do you have a red dress?" | Finds `prod_001` (score 0.552) | Returns nothing |
| "do you have a book about python?" | Returns nothing | Finds `book_002` (score 0.661) |

Isolation verified across three layers: conversation memory, RAG retrieval, and recommendation output.

---

## Why the orchestrator owns pipeline shape

`core/orchestrator.py` is the only file that knows:
- The order steps run in
- How results connect between steps
- What to do when one step fails
- How to pass `store_id` down the chain

That means schemas can change, storage can be swapped (in-memory dict → MongoDB → Atlas), LLM providers can change (HuggingFace → OpenRouter), and the vector store can migrate (local pickle → Qdrant Cloud → Qdrant Cloud with per-store collections) without any of that forcing changes in more than one file at a time.

The single-tenant → multi-tenant migration was itself an example: the orchestrator only needed to start passing `store_id` through. No downstream module needed rewiring.

---

## Independent failure handling per step

Every pipeline step runs inside its own timing + fallback wrapper. If any step throws, it returns a safe default and the pipeline continues.

Why this matters: a single try/except around the whole pipeline would mean any single failure short-circuits everything else and returns a generic error. Real pipelines aren't that fragile — if emotion detection fails, we can still recommend a product; if RAG fails, we can still generate a reply from history alone.

| Step | Fallback on failure |
|---|---|
| Emotion detection | `EmotionResult(label=NEUTRAL, confidence=0.0)` |
| Intent detection | `IntentResult(label=OTHER, confidence=0.0)` |
| Memory load | Empty `MemoryState` (no history, no known facts) |
| RAG retrieval | `[]` (no context) |
| Recommendation | `[]` (no recommendations) |
| Response generation | Fixed polite fallback message |
| Memory writes (add turn, extract facts) | Logged as warning, no user impact |
| Analytics log | Logged as warning, no user impact |

Writes happen last, so nothing is persisted for a failed request.

---

## Retry + multi-model failover

Response generation and fact extraction both use a two-layer resilience model:

1. **Same-model retries** (Tenacity, exponential backoff) — handle transient errors like rate limits, connection blips, and 5xx.
2. **Model chain failover** — if all retries on model A fail, move to model B. If all models fail, return a fixed fallback reply (never leak an exception to the customer).

Every fallback event is logged with the failing model, the reason (`timeout`, `rate_limit`, `empty_response`, etc.), and the latency of the failed attempt. This makes "which model is failing most often?" a queryable question.

---

## Split between local and external inference

Deliberate design choice:

**Local (runs inside the API process):**
- Emotion detection (small, fast, no per-request cost)
- Intent detection (larger, but avoids external latency)
- Query embedding (small model, sub-100ms)

**External:**
- Response generation (OpenRouter — quality matters, latency less critical)
- Fact extraction (OpenRouter — needs LLM reasoning to handle Arabic + English + free-form input)
- Vector search (Qdrant Cloud — persistent, per-store isolation, scales past what fits in RAM)

---

## Observability layers

- **Structured JSON logs** with per-request context (`request_id`, `store_id`, `conversation_id`) automatically injected into every log line across every module.
- **Per-step pipeline timing** — every stage records its own latency, so slow requests can be traced to the actual bottleneck.
- **Fallback events** — logged separately from step success, with `failed_model`, `failure_reason`, `latency_ms`.
- **Rotating file logs** at `data/logs/app.jsonl` (10MB per file, 5 backups) for cross-restart debugging.
- **Sentry** for automatic exception capture with request context.
- **`GET /metrics`** endpoint exposing request count, per-status breakdown, per-store breakdown, average and p95 latency, and error rate.
- **UptimeRobot** external ping on `/health` every 5 minutes with email alerts.

---

## Protection layers

- **Rate limiting** on `/chat` (20 requests per minute per IP, in-memory sliding window).
- **30-second timeouts** on external LLM calls, so a stalled provider can't tie up server resources or rack up cost on hung requests.
- **Input validation** via Pydantic on every request — invalid input returns 422 before touching the pipeline.
- **CORS** configured to allow cross-origin requests (widget will be embedded on many store domains).