# StoreFlow AI

An end-to-end AI sales agent for clothing stores that understands customer messages, retrieves relevant store knowledge, recommends products, maintains conversation memory, and generates personalized replies in Arabic and English.

This project demonstrates the practical engineering work required to take an AI product idea to a deployable system — not just calling an LLM, but designing for failure, testing the logic, and packaging it for reproducible deployment. It covers NLP inference, retrieval-augmented generation (RAG), product recommendation, LLM integration, persistent conversation memory, fault tolerance, automated testing, a FastAPI service layer, and Docker-based deployment.

```text
Prototype → Production-hardened core → FastAPI API → Containerized deployment (current) → Multi-tenant SaaS (planned)
```

The current system is a production-hardened core with a working API and a verified Docker deployment, built and tested against a single store's data. Multi-tenant SaaS support is the next planned phase, not yet implemented.

---

## Demo

The following is real, unedited output captured during local development and Docker-based verification testing — not a synthetic or illustrative example.

**Turn 1 — request (`POST /chat`):**

```json
{
  "conversation_id": "test5",
  "customer_id": "9",
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
    },
    {
      "product_id": "prod_002",
      "name": "Black Oversized Hoodie",
      "price": 1299,
      "reason": "Matches your query well (score: 0.42)."
    }
  ]
}
```

**Known-facts extraction for the same message** (logged to conversation memory, not part of the API response):

```json
{
  "preferred_size": "M",
  "preferred_color": "red",
  "mentioned_products": ["dress"]
}
```

**Turn 2 — same `conversation_id`, testing memory recall:**

```json
{
  "text": "what did I just ask for?"
}
```

**Response:**

```json
{
  "reply_text": "You asked for a red dress in size M — and I showed you the Red Summer Dress by Zara in that exact size and color. Would you like to see it or try it on?",
  "emotion": {
    "label": "excited",
    "confidence": 0.61
  },
  "intent": {
    "label": "complaint",
    "confidence": 0.57
  },
  "recommendations": [],
  "retrieved_context": []
}
```

The reply correctly recalls the prior turn, confirming that conversation memory persisted through two separate HTTP requests. The intent classifier incorrectly labeled the neutral follow-up as `complaint`. This is consistent with the measured intent-classification errors reported in the Evaluation section and is intentionally left unedited to show an actual system failure rather than a curated demo.

**Bilingual fact extraction** — verified separately against the fact-extraction module:

```text
Input:  "عايز جاكيت أسود مقاس لارج"
Output: { "preferred_size": "L", "preferred_color": "black", "mentioned_products": ["jacket"] }
```

---

## What it does

For each customer message, the pipeline:

1. **Detects emotion** — local HuggingFace DistilRoBERTa classifier with six application-level emotion labels
2. **Detects intent** — local zero-shot BART classifier against a fixed intent taxonomy
3. **Loads conversation memory** — per-conversation history and known facts, persisted in MongoDB
4. **Retrieves store knowledge** — semantic search (RAG) over products, policies, FAQ, and store information
5. **Recommends products** — intent-aware filtering of retrieved items, with real names and prices
6. **Generates a reply** — via OpenRouter, with a multi-model retry-and-failover chain
7. **Updates memory** — appends the turn and extracts known facts such as size, color, and budget
8. **Logs the turn** — appended to a JSONL analytics file, intended as a future source for evaluation and fine-tuning data

---

## Key Engineering Decisions

- **Stable module boundaries.** Storage was swapped from an in-memory dict to MongoDB, the LLM provider from HuggingFace to OpenRouter, and the entrypoint from a CLI to a FastAPI HTTP API. `core/orchestrator.py`, the pipeline's coordinator, required no changes across these swaps. Modules communicate through typed interfaces, while business-flow coordination remains centralized in the orchestrator.

- **Independent failure handling per pipeline step.** Emotion detection, intent detection, memory access, RAG retrieval, and recommendation each catch their own exceptions and fall back to a safe default rather than propagating the failure. Response generation, the customer-facing step, falls back to a fixed polite message if every model in its chain fails.

- **Retry plus multi-model failover for response generation.** Transient errors such as rate limits and server errors are retried on the current model with exponential backoff before the pipeline moves to the next model in a configured chain. This was verified during live testing: a rate-limited primary model was automatically bypassed in favor of a working backup without a customer-visible failure.

- **LLM-based fact extraction for bilingual input.** Customer messages arrive in Arabic and English with unpredictable phrasing. An LLM-based extractor handles both from a single prompt. The output parser is defensive against failure modes observed during testing, including prose instead of JSON, JSON wrapped in code fences, and invented keys. These cases are covered by dedicated unit tests.

- **Self-contained Docker deployment.** The image pre-downloads local ML models and pre-builds the RAG vector index at build time rather than on first request, so the container does not depend on an external service being reachable at cold start. The deployment was verified end-to-end, including MongoDB-backed memory across separate HTTP requests.

- **Tests that found real defects, not just coverage.** The unit test suite caught a live type-safety regression — a `Literal` role constraint that had silently reverted to an unrestricted `str` — before deployment. It was later extended to reproduce a bug observed in production output involving a case-sensitive attribute typo that silently broke analytics logging.

---

## Architecture

```text
HTTP request (POST /chat)
      │
      ▼
api.py  ── thin FastAPI transport layer: request/response (de)serialization only
      │
      ▼
core/orchestrator.py  ── owns business-flow coordination
      │
      ├── emotion/emotion_detector.py
      │       → EmotionResult
      │       → local model
      │
      ├── intent/intent_detector.py
      │       → IntentResult
      │       → local model
      │
      ├── memory/conversation_memory.py
      │       → MemoryState
      │       → MongoDB
      │
      ├── rag/retriever.py
      │       → list[RetrievedChunk]
      │       → local embeddings
      │
      ├── recommendation/product_recommender.py
      │       → list[ProductRecommendation]
      │
      ├── response/response_generator.py
      │       → reply text
      │       → external LLM + retry/failover
      │
      ├── memory/fact_extractor.py
      │       → known facts
      │       → external LLM
      │
      └── analytics/conversation_logger.py
              → JSONL log
```

FastAPI's responsibility is transport: parsing the incoming request into a validated `CustomerMessage` and serializing the returned `AgentReply`.

All business logic — including operation order and failure-handling policy — lives in the orchestrator. Each downstream module has a single isolated responsibility and exposes a typed interface.

The split between local and external inference is deliberate: emotion detection, intent detection, and RAG retrieval run locally, while response generation and fact extraction are delegated to an external LLM through OpenRouter.

MongoDB is the persistent store for per-conversation history and extracted known facts.

---

## Engineering Trade-offs

| Decision                                    | Why                                                                                          | Trade-off                                                                                 |
| ------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Local emotion/intent models                 | Avoids external latency and per-request cost for classification tasks                        | Lower accuracy ceiling than a larger hosted or fine-tuned model                           |
| Zero-shot intent classification (BART-MNLI) | No labeled conversation data exists yet; allows intent detection without a training pipeline | Current English-only starter evaluation achieved 70.0% accuracy and 0.67 macro F1 on n=40 |
| Semantic RAG over structured store data     | Handles natural-language queries without hand-written matching rules                         | Cannot answer aggregate questions such as "how many products do you have?"                |
| MongoDB for conversation memory             | Document model matches conversation state and persists across restarts                       | Adds an operational dependency                                                            |
| LLM-based fact extraction                   | Handles Arabic and English phrasing without separate rule sets                               | Adds one model call per turn and may occasionally fail to produce structured output       |
| Multi-model failover chain                  | Keeps conversations working when individual free-tier models rate-limit                      | Free-tier models can still rate-limit under sustained concurrent traffic                  |
| Central orchestrator                        | Keeps operation order and failure-handling policy in one place                               | Requires stable typed interfaces between modules                                          |

---

## Evaluation

The project has deterministic unit test coverage, manual end-to-end verification, and dedicated evaluation scripts for every ML-dependent component: intent, emotion, fact extraction, RAG retrieval, and recommendation.

All datasets are small, hand-crafted starter sets, not real customer traffic. They are honest baselines, not production benchmarks. As real conversation data becomes available, they should be expanded or replaced.

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

### RAG Retrieval (n=22 positive, 3 negative, real 27-document catalog)

- Recall@1: **90.91%**
- Recall@3: **95.45%**
- False positives on out-of-catalog queries: 0/3
- Known miss: short factual entries (e.g. store address) fall below the similarity threshold. Ten phrasings were tested; the best (0.33) still fell short of the 0.35 cutoff. This reflects a limitation of the small embedding model on sparse factual text, not a retrieval logic defect.

### Recommendation (n=9 positive, 5 negative)

- Correct recommendation rate: **100%**
- Unwanted recommendation rate: **20%** (1/5) — a query for "jackets" (not in the catalog) recommended a hoodie due to semantic similarity
- Safety check passed: a complaint referencing a real product correctly produced **no** recommendation

### Evaluation-Driven Fixes

Running these evaluations surfaced and fixed four real defects before they reached production: a missing retry/failover path in fact extraction, an unhandled crash on malformed API responses (present in two files), a prompt-injection gap caused by a duplicated unguarded message, and a schema regression (a silently dropped field) that broke recommendation output.

### Evaluation Scope

Not yet formally evaluated: response-generation quality (groundedness, human review), API latency (p50/p95), reliability under load, and cost per conversation. All numbers in this README are tied to a stated dataset and sample size; future numbers will follow the same standard.

---

## Project Structure

```text
StoreFlowAI/
├── app/
│   ├── main.py                        # CLI entrypoint (dev tool)
│   ├── api.py                         # FastAPI entrypoint
│   ├── config.py                      # central settings
│   ├── logging_config.py              # centralized logging
│   ├── core/orchestrator.py           # pipeline flow + error handling
│   ├── schemas/models.py              # shared Pydantic contracts
│   ├── emotion/emotion_detector.py
│   ├── intent/intent_detector.py
│   ├── memory/
│   │   ├── conversation_memory.py     # MongoDB-backed memory
│   │   └── fact_extractor.py          # LLM-based fact extraction
│   ├── rag/
│   │   ├── build_embeddings.py        # offline vector-index builder
│   │   └── retriever.py               # runtime semantic search
│   ├── recommendation/
│   │   └── product_recommender.py
│   ├── response/
│   │   └── response_generator.py      # OpenRouter + retry/failover
│   └── analytics/
│       └── conversation_logger.py
├── data/
│   ├── raw/                           # products, policies, FAQ, store info
│   ├── embeddings/                    # generated vector index (gitignored)
│   └── analytics/                     # conversation logs (gitignored)
├── eval/
│   ├── data/                          # labeled evaluation datasets
│   ├── results/                       # timestamped evaluation results
│   ├── evaluate_intent.py
│   ├── evaluate_emotion.py
│   ├── evaluate_fact_extraction.py
│   ├── evaluate_rag.py
│   └── evaluate_recommendation.py
├── tests/                             # pytest unit tests
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
| RAG embeddings                  | Sentence-Transformers                               |
| Response generation             | OpenRouter, OpenAI-compatible API                   |
| Fact extraction                 | OpenRouter                                          |
| Retry logic                     | Tenacity                                            |
| Conversation memory             | MongoDB                                             |
| Testing                         | Pytest                                              |
| Evaluation                      | scikit-learn                                        |
| Containerization                | Docker, Docker Compose                              |

---

## Running with Docker

The image is self-contained — local models and the RAG index are prepared at image build time.

```bash
cp .env.example .env
# fill in OPENROUTER_API_KEY, MONGO_USER, MONGO_PASSWORD

docker-compose up -d --build
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
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Configure environment variables:

```powershell
cp .env.example .env
```

Start MongoDB:

```powershell
docker-compose up -d mongo
```

Build the RAG index once:

```powershell
python -m app.rag.build_embeddings
```

Run the CLI:

```powershell
python -m app.main
```

Or run the FastAPI API:

```powershell
uvicorn app.api:app --reload
```

---

## Model Evaluation

### Intent

Run:

```powershell
python -m eval.evaluate_intent
```

The script:

- loads the labeled intent dataset
- runs the actual `detect_intent()` implementation
- reports overall accuracy
- reports per-class precision/recall/F1
- prints a confusion matrix
- saves a timestamped JSON result under `eval/results/`

### Emotion

Run:

```powershell
python -m eval.evaluate_emotion
```

The script follows the same evaluation pattern for the emotion classifier.

These evaluation scripts are intentionally separate from pytest because they measure model behavior rather than deterministic application logic.

---

## API

### `GET /health`

Liveness check. Returns:

```json
{
  "status": "ok"
}
```

The endpoint does not require MongoDB or model inference.

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

Run:

```powershell
python -m pytest -v
```

The deterministic unit test suite contains 30+ tests and runs without models, network access, or a database.

Current coverage includes:

- **`test_schemas.py`** — input validation: blank/empty rejection, over-length handling, control-character stripping
- **`test_recommender.py`** — recommendation filtering: intent gating, similarity threshold, and source-type filtering
- **`test_fact_extractor.py`** — robust JSON parsing against observed LLM failure modes

The pytest suite covers deterministic application logic.

Model-dependent behavior is evaluated separately through the dedicated intent and emotion evaluation scripts.

RAG retrieval quality and response-generation quality are still primarily verified through manual testing.

---

## Design Principles

- **One responsibility per module.**
- **Stable interfaces over convenience.**
- **Fail gracefully.**
- **Measure before optimizing.**
- **Explicit trade-offs.**
- **Real failures are documented rather than hidden.**
- **Build for the current phase without premature infrastructure.**

---

## Known Limitations

- **RAG cannot answer aggregate questions** such as "how many products do you have?" Semantic retrieval finds relevant documents rather than performing database aggregation. Structured tool-calling would be required for this.

- **Zero-shot intent classification has a measurable error rate.** The current English-only starter evaluation achieved **70.0% accuracy and 0.67 macro F1 on n=40**. The dataset is small and hand-crafted, so this is a baseline rather than a production benchmark.

- **Emotion classification has a measurable error rate.** The current English-only starter evaluation achieved **75.0% accuracy and 0.72 macro F1 on n=48**. The `confused` class is particularly weak, with **0.12 recall** in the current evaluation.

- **Free-tier OpenRouter models rate-limit under load.** The failover chain mitigates this for moderate use. Production traffic should use a reliable paid model as the primary provider, with free-tier models as secondary fallbacks at most.

- **Fact extraction occasionally fails on free-tier models.** Some models return prose instead of structured JSON. The parser handles this safely by rejecting invalid output without overwriting existing memory, but the fact may be missed for that turn.

- **English-first model evaluation.** The current formal evaluation datasets are English-only. Arabic support is implemented in parts of the pipeline and has been manually verified for fact extraction, but Arabic intent and emotion performance have not yet been formally measured.

- **Single-tenant.** There is currently no `tenant_id` isolation. The deployment serves one store's data per instance. Multi-tenant architecture is planned but not implemented.

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
- End-to-end Docker verification
- Initial quantitative evaluation for intent classification
- Initial quantitative evaluation for emotion classification
- RAG retrieval evaluation
- Recommendation evaluation

### Planned

- Expand evaluation with real labeled customer conversations
- Arabic-specific intent and emotion evaluation
- Improve intent classification using better label definitions and/or supervised training once sufficient labeled data exists
- Fact-extraction evaluation
- Response-quality evaluation
- API latency and reliability benchmarks
- Structured tool-calling for inventory and aggregate queries
- WhatsApp integration
- Instagram integration
- Multi-tenant architecture

---

## License

Proprietary — all rights reserved.

## 👨‍💻 Author

Basem Morad
AI/ML Engineer | Computer Vision Specialist
B.Sc. Mechatronics Engineering
