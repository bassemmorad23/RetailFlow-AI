StoreFlow AI

An AI-powered sales agent for clothing stores. It understands customer emotion and intent, searches store knowledge with RAG, recommends products, remembers conversation context, and generates personalized replies in Arabic and English.

Built as a modular pipeline designed to evolve from prototype to a multi-tenant SaaS platform without rewrites.

Status

Phase: Production-hardened core. The full conversation pipeline works end-to-end and is fault-tolerant. It is not yet exposed as an HTTP API — it currently runs as a CLI. FastAPI, deployment, and channel integrations (WhatsApp, Instagram) are the next phases.

What it does

For each customer message, the pipeline:

Detects emotion — local HuggingFace DistilRoBERTa classifier
Detects intent — local zero-shot BART classifier against a fixed intent set
Loads conversation memory — per-conversation history and known facts, stored in MongoDB
Retrieves store knowledge — semantic search (RAG) over products, policies, FAQ, and store info
Recommends products — intent-aware filtering of retrieved items
Generates a reply — via OpenRouter, with a multi-model failover chain
Updates memory — appends the turn, and extracts known facts (size, color, budget) from the message
Logs the turn — appended to a JSONL analytics file

Each step degrades gracefully — one failing component never crashes the conversation.

Architecture
Customer message
      |
      v
core/orchestrator.py  -- the pipeline "brain"; owns the flow, calls every module
      |
      |-- emotion/emotion_detector.py            -> EmotionResult
      |-- intent/intent_detector.py              -> IntentResult
      |-- memory/conversation_memory.py          -> MemoryState (MongoDB)
      |-- rag/retriever.py                       -> list[RetrievedChunk]
      |-- recommendation/product_recommender.py  -> list[ProductRecommendation]
      |-- response/response_generator.py         -> reply text (OpenRouter + failover)
      |-- memory/fact_extractor.py               -> known facts (LLM extraction)
      |-- analytics/conversation_logger.py       -> JSONL log

Modules never call each other directly — only the orchestrator knows the flow. This is what lets any single module be swapped without touching the rest.

Project structure
StoreFlowAI/
├── app/
│   ├── main.py                        # CLI entrypoint
│   ├── config.py                      # central settings (pydantic-settings)
│   ├── logging_config.py              # centralized logging setup
│   ├── core/
│   │   └── orchestrator.py            # pipeline flow
│   ├── schemas/
│   │   └── models.py                  # shared Pydantic data contracts
│   ├── emotion/
│   │   └── emotion_detector.py
│   ├── intent/
│   │   └── intent_detector.py
│   ├── memory/
│   │   ├── conversation_memory.py     # MongoDB-backed memory
│   │   └── fact_extractor.py          # LLM-based fact extraction
│   ├── rag/
│   │   ├── build_embeddings.py        # offline: build the vector index
│   │   └── retriever.py               # runtime: semantic search
│   ├── recommendation/
│   │   └── product_recommender.py
│   ├── response/
│   │   └── response_generator.py      # OpenRouter + failover chain
│   └── analytics/
│       └── conversation_logger.py
├── data/
│   ├── raw/                           # products.json, policies.json, faq.json, store_info.json
│   └── embeddings/                    # generated vector index (gitignored)
├── tests/
├── docker-compose.yml                 # local MongoDB
├── requirements.txt
├── .env.example
└── README.md
Tech stack

Current

Python
Pydantic / pydantic-settings — data contracts and config
HuggingFace Transformers — emotion and intent detection (local)
Sentence-Transformers — RAG embeddings (local)
MongoDB — conversation memory
OpenRouter — response generation (OpenAI-compatible API)
tenacity — retry logic
Docker Compose — local MongoDB

Planned

FastAPI — HTTP API
WhatsApp and Instagram integrations
Multi-tenant architecture
Setup
1. Prerequisites
Python 3.11+
Docker (for MongoDB)
An OpenRouter API key — https://openrouter.ai/keys
2. Install dependencies
bash
pip install -r requirements.txt
3. Configure environment

Copy the template and fill in your values:

bash
cp .env.example .env

Set at minimum:

OPENROUTER_API_KEY=sk-or-v1-your-key-here
MONGO_URI=mongodb://storeflow:changeme@localhost:27017/storeflow?authSource=admin
4. Start MongoDB
bash
docker-compose up -d
5. Add store data

Place your data files in data/raw/:

products.json
policies.json
faq.json
store_info.json
6. Build the RAG index

Run once, and again whenever store data changes:

bash
python -m app.rag.build_embeddings
7. Run the agent
bash
python -m app.main
Configuration

All settings live in app/config.py, overridable via .env.

Setting	Purpose	Default
openrouter_api_key	OpenRouter auth (required)	—
response_model_chain	Failover list of models, tried in order	free-tier chain
mongo_uri	MongoDB connection string	—
app_env	prototype / development / production (controls log level)	prototype

The model chain lives in config.py, not .env — it is configuration, not a secret. Only the API key belongs in .env.

Design principles
One responsibility per module — emotion detects emotion, memory stores memory, nothing overlaps.
Evolve without rewriting — swapping storage (dict to MongoDB) or the LLM provider (HF to OpenRouter) touched only the relevant module; the orchestrator never changed.
Fail gracefully — every pipeline step is independently fault-tolerant; the customer always gets a reply.
Explicit trade-offs — limitations are documented in the code, not hidden.
Build for the current phase — no multi-tenant or production infrastructure added before it is needed.
Known limitations

These are deliberate, documented, and scoped to the current phase:

No HTTP API yet — runs as a CLI. FastAPI is the next phase.
RAG cannot answer aggregate questions — semantic search finds similar items ("do you have hoodies?") but cannot answer "how many products total?" or "list everything". Those are database queries, not similarity search. Structured tool-calling will address this in production.
Free-tier models rate-limit — the OpenRouter free models return 429 under load; the failover chain cushions this, but a production setup should put a paid model first.
Fact extraction can miss on free models — free models sometimes return prose instead of JSON; the parser handles this safely (returns no facts) but occasionally misses a fact it should have caught.
mentioned_products replaces rather than appends — only the most recent message's products are kept, not accumulated across the conversation.
Roadmap
Prototype -> Production-hardened core -> MVP (FastAPI) -> Deployment -> SaaS

Next:

Test suite (per-module coverage)
FastAPI HTTP layer
Dockerized deployment
WhatsApp and Instagram webhooks
Multi-tenant architecture
Structured tool-calling for inventory and aggregate queries
License

Proprietary — all rights reserved. (Update this line if you intend to open-source.