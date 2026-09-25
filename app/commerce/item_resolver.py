"""
Turn "the black shirt, size M, 2 pieces" into a real OrderItem.

- Product: the product just recommended in this conversation if it matches,
  otherwise the store catalog via RAG (strict threshold). Never invented.
- Variant: must be uniquely identified by the stated attributes; otherwise
  we return the available options so the AI can ask.
- Price: always from the database (variant price when applicable).
- Stock: checked through the stock service; out of stock is never added.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from app.commerce.models import MAX_QUANTITY_PER_LINE, OrderItem
from app.commerce.stock import check_stock
from app.products.product_store import fetch_products
from app.rag.retriever import retrieve_context
from app.schemas.models import Product, ProductRecommendation

MATCH_THRESHOLD = 0.5


@dataclass
class ResolvedItem:
    status: Literal["ok", "needs_variant", "out_of_stock", "not_found"]
    item: OrderItem | None = None
    product_name: str | None = None
    options: dict[str, list[str]] = field(default_factory=dict)  # for needs_variant
    stock: str = "unknown"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


def _pick_from_recommendations(ref: str, recs: list[ProductRecommendation]) -> str | None:
    if not recs:
        return None
    ref_n = _norm(ref)
    if not ref_n or ref_n in {"it", "this", "that", "this one", "that one", "ده", "دي", "هذا", "هذه"}:
        return recs[0].product_id if len(recs) == 1 else None
    ref_words = set(ref_n.split())
    best, best_overlap = None, 0
    for r in recs:
        overlap = len(ref_words & set(_norm(r.name).split()))
        if overlap > best_overlap:
            best, best_overlap = r.product_id, overlap
    return best


def _find_product(store_id: str, ref: str, recs: list[ProductRecommendation]) -> Product | None:
    pid = _pick_from_recommendations(ref, recs)
    if pid is None and ref.strip():
        chunks = retrieve_context(store_id, ref, top_k=1)
        if chunks and chunks[0].score >= MATCH_THRESHOLD:
            pid = chunks[0].source
    if pid is None:
        return None
    found = fetch_products(store_id, [pid])
    return found[0] if found else None


def _values(product: Product, variant) -> dict[str, str]:
    merged = {**product.attributes, **product.specifications, **variant.attributes, **variant.specifications}
    return {str(k).lower(): str(v).strip().lower() for k, v in merged.items() if v is not None}


def _options(product: Product, variants: list) -> dict[str, list[str]]:
    keys = sorted({k for v in variants for k in v.attributes})
    return {k: sorted({str(v.attributes[k]) for v in variants if k in v.attributes}) for k in keys}


def resolve_item(store_id: str, request: dict, recommendations: list[ProductRecommendation]) -> ResolvedItem:
    product = _find_product(store_id, request.get("product", ""), recommendations)
    if product is None:
        return ResolvedItem("not_found")

    variant = None
    if product.variants:
        wanted = {k: str(v).strip().lower() for k, v in (request.get("attributes") or {}).items()}
        matches = [v for v in product.variants
                   if all(_values(product, v).get(k) == val for k, val in wanted.items())]
        if len(matches) != 1:
            pool = matches or product.variants
            return ResolvedItem("needs_variant", product_name=product.name, options=_options(product, pool))
        variant = matches[0]

    level = check_stock(store_id, [(product.product_id, variant.sku if variant else None)])[0].level
    if level == "out_of_stock":
        return ResolvedItem("out_of_stock", product_name=product.name, stock=level)

    quantity = min(max(request.get("quantity") or 1, 1), MAX_QUANTITY_PER_LINE)
    item = OrderItem(
        product_id=product.product_id,
        variant_sku=variant.sku if variant else None,
        name=product.name,
        variant_attrs={k: str(v) for k, v in (variant.attributes if variant else {}).items()},
        quantity=quantity,
        unit_price=variant.price if variant else product.price,
    )
    return ResolvedItem("ok", item=item, product_name=product.name, stock=level)