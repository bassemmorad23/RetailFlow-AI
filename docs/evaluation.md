# Evaluation

Full results for every ML-dependent component in StoreFlow AI.

For a summary table, see the [main README](../README.md).

All datasets are small, hand-crafted starter sets, not real customer traffic. They are honest baselines, not production benchmarks. As real conversation data becomes available, they will be expanded or replaced.

Each evaluation script lives in [`eval/`](../eval/) and can be re-run independently:

```bash
python -m eval.evaluate_intent
python -m eval.evaluate_emotion
python -m eval.evaluate_fact_extraction
python -m eval.evaluate_rag
python -m eval.evaluate_recommendation
```

These are intentionally separate from `pytest` because they measure model behavior rather than deterministic application logic.

---

## Intent classification (n=40)

- **Accuracy: 70.0%, Macro F1: 0.67**
- Model: `facebook/bart-large-mnli`, zero-shot classification
- Dataset: 5 examples per label across 8 intent labels, English-only

### Weakest areas

- `browsing` — often misread as `wants_recommendation`
- `wants_recommendation` — often misread as `asking_details`
- `ready_to_buy` — sample size too small to be conclusive
- `other` — the catch-all is inherently noisy

### Practical impact

The `wants_recommendation` ↔ `asking_details` confusion has **low practical impact**, because both intents are in `_RECOMMENDATION_INTENTS`. The recommender still fires either way.

---

## Emotion classification (n=48)

- **Accuracy: 75.0%, Macro F1: 0.72**
- Model: `j-hartmann/emotion-english-distilroberta-base`
- Dataset: 8 examples per label across 6 application-level labels, English-only

### Weakest class

- `confused` — 0.12 recall. Frequently misread as `neutral` or `excited`.

### Important note

The underlying model has 7 native labels; our app collapses them into 6. Test datasets must use the 6-label enum space, not the model's native labels.

---

## Fact extraction (n=24)

Real API calls, not deterministic — 24 sequential requests to OpenRouter with real fallback behavior.

| Field | Precision | Recall |
|---|---|---|
| `preferred_size` | 1.00 | 1.00 |
| `preferred_color` | 1.00 | 1.00 |
| `budget_max` | 1.00 | 1.00 |
| `mentioned_products` | 1.00 | 0.92 |

**Hallucinations: 0 / 24** — the extractor never invented a fact that wasn't stated in the message.

### Failure modes observed and handled

- **Prose instead of JSON** — parser returns `{}` safely
- **Truncated JSON** (e.g. `'{"budget_max": 2'`) — parser returns `{}` safely
- **Model returns empty `response.choices`** — raised as `ValueError`, caught by failover chain
- **JSON wrapped in code fences** — parser strips fences before parsing

### Key insight

Failover **cannot** rescue content-quality failures (prose, truncation). Those return HTTP 200, and Tenacity's retry logic never fires. Handling them requires the parser to be defensive, which it is.

---

## RAG retrieval (n=22 positive, 3 negative)

- **Recall@1: 90.91%**
- **Recall@3: 95.45%**
- **False positives on out-of-catalog queries: 0 / 3**

Tested against the real 27-document catalog dumped from Qdrant Cloud via [`eval/dump_index.py`](../eval/dump_index.py).

### The one persistent miss

Short factual entries (e.g. store address: `"address: Our store address is..."`) fall below the similarity threshold with the current small embedding model on sparse factual text. Ten phrasings were tested; the best scored 0.33, still short of the 0.35 cutoff.

This is a limitation of `all-MiniLM-L6-v2` on short factual content, not a retrieval logic defect. Documented and deferred until a multilingual/larger embedding model is adopted.

### Negative-example testing

3 queries about products the store doesn't sell (e.g. "do you sell shoes?"). Retrieval correctly returned nothing for all 3, confirming the similarity threshold gates confidence appropriately.

---

## Recommendation (n=9 positive, 5 negative)

- **Correct recommendation rate: 100%** on positive cases
- **Unwanted recommendation rate: 20%** (1 / 5) — a query for "jackets" (not in the catalog) recommended a hoodie due to semantic similarity between the two terms
- **Safety check passed:** a complaint referencing a real product (e.g. "this red dress arrived damaged") correctly produced **no** recommendation, because the intent-gating logic blocked the sales pitch

### What this eval isolates

The intent for each test case is **manually assigned**, deliberately bypassing the real intent classifier. This isolates `recommend_products()` in itself, without conflating recommendation failures with intent-classification errors (already measured separately).

The retrieval side uses the real `retrieve_context()` output.

---

## Multi-tenant isolation (end-to-end)

Verified with two live stores sharing one deployment:

- `store_001` — clothing store (Red Summer Dress, Black Oversized Hoodie, Blue Slim Fit Jeans, etc.)
- `store_002` — bookstore (Modern Python Programming, The Silent Library, etc.)

| Query | store_001 (clothing) | store_002 (bookstore) |
| --- | --- | --- |
| "do you have a red dress?" | Finds `prod_001` (score 0.552) | Returns nothing |
| "do you have a book about python?" | Returns nothing | Finds `book_002` (score 0.661) |

Isolation verified across three layers:

1. **Conversation memory** — `store_001`'s history and known facts never surface in `store_002`'s conversations
2. **RAG retrieval** — each store's Qdrant collection is queried in isolation
3. **Recommendation** — `store_002` asking about a product `store_001` sells returns no recommendation

Zero cross-store leaks in either direction.

---

## Evaluation-driven fixes

Running these evaluations surfaced and fixed several real defects **before they reached production**:

- **Missing retry/failover in `fact_extractor.py`** — was calling only the first model in the chain once. Fixed by adopting the same retry + failover loop as `response_generator.py`.
- **`NoneType` crash on empty LLM responses** — some free-tier models return no `choices`, causing `response.choices[0]` to fail. Fixed with an explicit `if not response.choices: raise ValueError(...)` guard in both `fact_extractor.py` and `response_generator.py`.
- **Prompt-injection gap from a duplicated unguarded message** — the response prompt was wrapping the customer message in `<customer_message>` tags AND appending it again unguarded right before "Your reply:". Removed the duplicate.
- **Schema regression** — `price` field silently dropped from `RetrievedChunk` between refactors, breaking recommendation output. Only surfaced when the recommendation eval was written.
- **`store_id` validation gap** — empty/missing `store_id` was passing validation and then failing silently downstream with a fake reply. Fixed by adding `store_id` to the blank-check validator.
- **CORS not applied** — a duplicated `app = FastAPI(...)` in `api.py` was wiping out the CORS middleware. Deduplicated, verified from a real cross-origin browser request.

---

## Not yet formally evaluated

- Response-generation quality (groundedness, human review)
- API latency percentiles (p50/p95) under real load
- Reliability under sustained concurrent traffic
- Cost per conversation (needs paid-model swap first for stable numbers)
- Arabic-specific intent and emotion accuracy