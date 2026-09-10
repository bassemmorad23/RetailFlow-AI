# Engineering Decisions

Every non-trivial technical choice made in StoreFlow AI, with honest reasoning and trade-offs.

For a summary table, see the [main README](../README.md).

The format for each decision: **what was chosen, why, alternative considered, honest trade-off**.

---

## Architecture-level

### Multi-tenant with one Qdrant collection per store

**Why:** Physical separation makes cross-store leaks impossible by design, not by "always remember to filter." A query against `store_002`'s collection cannot return `store_001`'s data — the collection doesn't contain it.

**Alternative considered:** One shared Qdrant collection with a `store_id` filter on every query. Simpler infra, but one forgotten `.filter(store_id=...)` anywhere leaks data.

**Trade-off:** Adds a collection-creation step when a new store onboards. Solved with `build_embeddings.py --store-id <new>`. Acceptable one-time cost per store.

---

### MongoDB Atlas as source of truth for catalog data

**Why:** Standard multi-tenant pattern (shared collections filtered by `store_id`); persists across restarts; free tier is enough for many small stores; Qdrant is derived data (rebuilt from Mongo), so Mongo has to be the source of truth.

**Alternative considered:** Local JSON files per store (what the prototype had). Fine for a single dev, breaks the moment more than one store exists or the container restarts.

**Trade-off:** Requires a manual embedding rebuild when a store's catalog changes (auto-sync deferred). Every store owner change to their catalog = one `build_embeddings.py --store-id X` command. Acceptable at v1.

---

### Qdrant Cloud (hosted) rather than self-hosted

**Why:** Removes operational burden of running a vector DB. Free tier handles our current scale. Same-region hosting available (Frankfurt for Egypt-based deployment).

**Alternative considered:** Self-hosted Qdrant in a Docker container next to the app. Would work but adds a service to babysit. Not worth it at v1.

**Trade-off:** External dependency and a network hop per query. Mitigated by choosing the same region as Mongo Atlas, and by the small size of embedding queries (sub-100ms typical).

---

### Central orchestrator owns pipeline shape

**Why:** Keeps operation order, failure-handling policy, and `store_id` scoping in one place. Every downstream module can be swapped, refactored, or reimplemented without touching the pipeline flow.

**Alternative considered:** Each module calls the next directly. Fragile — a schema change ripples through every file. Debugging is harder because pipeline order is implicit.

**Trade-off:** Requires stable typed interfaces between modules. Enforced through Pydantic contracts (`app/schemas/models.py`). Worth the discipline.

---

## Model-level

### Local emotion + intent models

**Why:** Avoids external latency and per-request cost for classification tasks that don't need frontier-model reasoning. Emotion + intent are stable enough with off-the-shelf models.

**Alternative considered:** Use OpenRouter for these too. Costs money, adds latency, doesn't meaningfully improve accuracy.

**Trade-off:** Lower accuracy ceiling than a larger hosted or fine-tuned model. Currently 70% intent accuracy, 75% emotion — good enough for v1, improvable later with real conversation data.

---

### Zero-shot intent classification (BART-MNLI)

**Why:** No labeled conversation data existed when the project started. Zero-shot lets us define intent labels in a config file and get useful classification immediately, without a training pipeline.

**Alternative considered:** Train a small classifier on synthetic data. Would probably do worse than zero-shot BART, and the synthetic data itself would need writing.

**Trade-off:** 70% accuracy on n=40 (English-only starter set). Will improve as real conversation data replaces the starter dataset.

---

### LLM-based fact extraction

**Why:** Customer messages arrive in Arabic and English with unpredictable phrasing. Rule-based extraction (regex, keyword) would need separate hand-tuned rules per language, and would still miss most natural phrasing.

**Alternative considered:** Two rule-based extractors, one per language. Would be brittle, high-maintenance, and still worse than a small LLM prompt.

**Trade-off:** Adds one model call per turn and may occasionally fail to produce structured output. Handled by defensive parsing (strips code fences, ignores prose, returns `{}` on failure) — extraction failures never break the customer conversation.

---

### Multi-model failover chain for LLM calls

**Why:** Free-tier models rate-limit unpredictably. A single-model setup goes down the moment that model 429s. Chain lets us fail over to a backup, then a backup-of-backup, before giving up.

**Alternative considered:** Single model. Cheaper mental model, but every rate-limit means a customer sees a fallback message.

**Trade-off:** Free-tier models can still rate-limit together at peak times, since they share provider capacity. Long-term fix is putting a cheap paid model first in the chain — deferred until OpenRouter credits are added.

---

### `all-MiniLM-L6-v2` for query embedding

**Why:** Small (~90MB), fast (sub-100ms per query on CPU), good enough recall (95% recall@3) for a starter setup. Zero external calls for embedding.

**Alternative considered:**
- `bge-m3` — better multilingual quality, but 1024 dimensions (Qdrant collection rebuild) and 2GB model size (bigger Docker image, slower cold starts).
- Paid API embedding (OpenAI, Cohere) — best quality, but every query costs money and adds a network hop.

**Trade-off:** Known miss on short factual entries (store address falls below the similarity threshold on sparse factual text). Documented in evaluation. Upgrade path: swap to `intfloat/multilingual-e5-small` when Arabic support becomes a real need.

---

## Infrastructure-level

### Railway for API hosting

**Why:** Fastest path from `git push` to a live HTTPS URL. Free trial + $5/month afterwards. Auto-deploys from GitHub. Enough for v1 traffic.

**Alternative considered:**
- **AWS App Runner / ECS** — more control, but 4-6 hours of setup vs. 30 min on Railway, and 5-10x the cost at our scale.
- **Vercel / Netlify** — designed for frontend, not long-running Python APIs.
- **Fly.io** — comparable to Railway.

**Trade-off:** Single-region, single-instance by default. Adequate for v1. Migration path to AWS is straightforward when scale justifies it — the app is a plain Docker container, portable anywhere.

---

### FastAPI with sync endpoints

**Why:** The pipeline underneath is synchronous (pymongo, local transformer models, blocking HTTP to OpenRouter). FastAPI runs plain `def` endpoints in a threadpool automatically, so blocking code doesn't freeze the server.

**Alternative considered:** Convert everything to `async def`. Would require replacing pymongo with motor, adopting async LLM clients, and making the transformer inference async-safe. Big refactor for zero user-facing benefit at v1 traffic.

**Trade-off:** Concurrency is bounded by the size of the threadpool (default ~40 workers). Fine at v1. When request rate justifies it, convert step by step — not all at once.

---

### In-memory rate limiter (not slowapi, not Redis)

**Why:** Rate limiting logic is 30 lines of Python. We tried `slowapi` — it silently no-ops with the current FastAPI + Pydantic v2 combination (verified with debug logs). Hand-rolled version works, is easy to read, and has zero mystery bugs.

**Alternative considered:**
- **slowapi** — abandoned after confirming the silent-failure behavior.
- **Redis + slowapi/fastapi-limiter** — the proper multi-instance solution. Overkill at single-instance v1 (~$5/month extra for Redis + real complexity).

**Trade-off:** In-memory counters work correctly only when there's one process serving requests. Multi-instance = each has its own counter, so a "20/min" cap becomes "20/min per instance." When scaling justifies it, swap in Redis — the interface stays the same.

---

### File-based logging with rotation (not database, not Sentry-only)

**Why:** Every log line goes to disk (`data/logs/app.jsonl`, 10MB per file, 5 backups). File I/O is fast, unlimited, works when MongoDB is down.

**Alternative considered:**
- **MongoDB for logs** — every log line = one Mongo write (~460ms Cairo→Frankfurt latency). 12 pipeline steps × 460ms = 5.5 seconds added per request. Also pollutes the operational database and burns Atlas free-tier storage.
- **Sentry for everything** — Sentry's free tier is ~10k events/month. Full-stream logging would burn through in a day. Sentry is for important events (errors), not stream logging.

**Trade-off:** Files are ephemeral on Railway (lost per container restart). Mitigated by Sentry for important events and `/metrics` for aggregate stats. If cross-restart raw logs matter later, stream to a log-ingest service (Datadog, Grafana Loki, etc.).

---

### Sentry for error monitoring

**Why:** Auto-captures exceptions with request context. Free tier is generous. Standard tool.

**Alternative considered:**
- **CloudWatch** — designed for AWS-hosted apps. We're on Railway.
- **Rollbar / Bugsnag** — comparable to Sentry, no meaningful difference at v1.
- **Nothing** — errors would only show up in file logs, no alerting.

**Trade-off:** Free tier caps at 5k errors + 10k transactions per month. `send_default_pii=False` and `traces_sample_rate=0.1` in production keep us well inside the cap. Upgrade path when volume grows: paid plan or self-hosted GlitchTip (Sentry-compatible).

---

### UptimeRobot for uptime monitoring

**Why:** External pinger — sees the same thing a real customer would. If Railway crashes and takes down our own monitoring, UptimeRobot still knows. Free tier: 50 monitors, 5-minute interval, email alerts.

**Alternative considered:**
- **Internal monitoring** — pointless if it dies when the app dies.
- **Better Uptime / Pingdom** — comparable, comparable free tier.

**Trade-off:** 5-minute check interval means you could be down for up to 5 minutes before knowing. Acceptable at v1.

---

### `/metrics` endpoint (in-memory JSON, not Prometheus)

**Why:** One endpoint, ~40 lines of code, tells you request count, per-status breakdown, per-store breakdown, average and p95 latency, error rate. Enough for basic operational health.

**Alternative considered:**
- **Prometheus + Grafana Cloud** — proper production observability. ~1 hour of setup, real dashboards, real alerting. Deferred until real users make it worthwhile.

**Trade-off:** Counters reset on container restart. Fine for basic health checks. When real traffic + retention matter, swap in Prometheus — same shape of data, different storage.

---

### Structured JSON logs with per-request context

**Why:** Machine-readable logs are filterable by any field (`request_id`, `store_id`, `conversation_id`). One `request_id` follows a request through every module. Turns "the system feels slow" into "step X for request Y took 12 seconds."

**Alternative considered:** Plain-text logs. Readable by humans, useless at scale.

**Trade-off:** Slightly harder to eyeball raw. Solved by piping through `jq` for local reads, and by structured search when logs go to Sentry / a log service.

---

## Development-process-level

### Tests that isolate app logic from ML behavior

**Why:** Unit tests (`pytest`) run without models, network, or a database. They cover Pydantic validation, recommendation filtering, and JSON parsing — logic that must be right regardless of which LLM answers. ML behavior is evaluated separately in `eval/`.

**Alternative considered:** End-to-end tests that call real models. Slow, flaky, and don't isolate the failing layer when they break.

**Trade-off:** Model behavior isn't in the `pytest` gate — a regression in the LLM chain won't fail CI. Mitigated by running eval scripts before releases.

---

### Migration script, not hand-edited data

**Why:** Store data is loaded via an idempotent migration script that reads from JSON source files and writes to MongoDB with proper indexes. Rebuilds are always safe — collections are dropped and recreated fresh, no stale data ever survives.

**Alternative considered:** Manually inserting documents into Mongo. Fine for one dev, breaks the moment anyone else needs to onboard a store or the DB gets wiped.

**Trade-off:** Every store change requires re-running the script. Deliberate — makes catalog updates explicit and reproducible.

---

### Evaluation-driven fixes

**Why:** Writing an evaluation forces us to think about failure modes we hadn't considered. Every eval script has surfaced at least one real bug — see [`docs/evaluation.md`](evaluation.md#evaluation-driven-fixes) for the full list.

**Trade-off:** Building eval datasets takes time. Worth it — several of the bugs found would have hit production and been much more expensive to diagnose from customer reports.

---

## Deferred (explicitly not built yet)

Things we thought about, decided against, and know why:

| Deferred | Why not now | When we'd add it |
|---|---|---|
| Redis for rate limiting | Single-instance deployment | When scaling to multiple Railway instances |
| Prometheus + Grafana | `/metrics` endpoint is enough | Real traffic + a need for dashboards |
| CloudWatch / full APM | We're not on AWS | Only if we move hosting to AWS |
| Auto-sync catalog → Qdrant | Manual rebuild works at 1-2 stores | When onboarding gets frequent |
| WhatsApp / Instagram adapters | No pilot store using them yet | After first pilot goes live |
| Multi-region deployment | One region is fine at v1 | Real users complain about latency |
| Self-service store onboarding UI | One store at a time is manual | When we have 5+ pilot stores |
| Paid embedding API | Local embedding is good enough | When Arabic quality becomes a real complaint |
| `gpt-4o-mini` as primary LLM | Free-tier works for testing | Immediately when OpenRouter credits are added — this is the top upgrade priority |
| Sentry paid tier | Free tier covers current volume | If real traffic pushes past 5k errors/month |

The list is deliberate. "Deferred" here means "we know this exists, we know why we're not building it now, and we know what would make it worth building."