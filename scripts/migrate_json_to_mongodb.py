"""
One-time migration script: JSON files -> MongoDB (multi-tenant).

WHY THIS EXISTS:
Moves the existing single-store JSON catalog into MongoDB with proper
per-store isolation via a `store_id` field on every document. Same store's
data can be re-migrated safely — the script deletes existing store data
before re-inserting to keep it idempotent.

FIELD TRANSFORMS (JSON -> MongoDB):
  products.json    id -> product_id, title -> name, currency -> dropped
  policies.json    id -> policy_id, content -> body
  faq.json         id -> faq_id
  store_info.json  store_name -> name, contact.* fields flattened,
                   currency added from --currency CLI arg

HOW TO RUN:
    python -m scripts.migrate_json_to_mongodb --store-id store_001 --currency EGP

    Optional: --source-dir <path>   (defaults to data/raw)
    Optional: --yes                  (skip confirmation prompt)
"""

import argparse
import json
import sys
from pathlib import Path

from pymongo import MongoClient

from app.config import settings

_DEFAULT_SOURCE_DIR = Path(__file__).parent.parent / "data" / "raw"


def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _transform_product(raw: dict, store_id: str) -> dict:
    return {
        "store_id": store_id,
        "product_id": raw["id"],
        "name": raw["title"],
        "description": raw.get("description"),
        "category": raw.get("category"),
        "brand": raw.get("brand"),
        "color": raw.get("color"),
        "sizes": raw.get("sizes", []),
        "price": raw["price"],
        "stock": raw.get("stock", 0),
    }


def _transform_policy(raw: dict, store_id: str) -> dict:
    return {
        "store_id": store_id,
        "policy_id": raw["id"],
        "title": raw["title"],
        "body": raw["content"],
    }


def _transform_faq(raw: dict, store_id: str) -> dict:
    return {
        "store_id": store_id,
        "faq_id": raw["id"],
        "question": raw["question"],
        "answer": raw["answer"],
    }


def _transform_store_info(raw: dict, store_id: str, currency: str) -> dict:
    """Flatten nested 'contact' block; move currency in from CLI."""
    contact = raw.get("contact", {}) or {}
    return {
        "store_id": store_id,
        "name": raw.get("store_name"),
        "description": raw.get("description"),
        "currency": currency,
        "phone": contact.get("phone"),
        "email": contact.get("email"),
        "instagram": contact.get("instagram"),
        "facebook": contact.get("facebook"),
        "business_hours": raw.get("business_hours", {}),
        "shipping": raw.get("shipping", {}),
        "payment_methods": raw.get("payment_methods", []),
        "address": raw.get("address"),
    }


def _wipe_existing(db, store_id: str) -> None:
    for collection_name in ("products", "policies", "faq", "store_info"):
        result = db[collection_name].delete_many({"store_id": store_id})
        print(f"  cleared {result.deleted_count} existing {collection_name}")


def _ensure_indexes(db) -> None:
    """Compound indexes for fast per-store queries + uniqueness where relevant."""
    db["products"].create_index([("store_id", 1), ("product_id", 1)], unique=True)
    db["policies"].create_index([("store_id", 1), ("policy_id", 1)], unique=True)
    db["faq"].create_index([("store_id", 1), ("faq_id", 1)], unique=True)
    db["store_info"].create_index([("store_id", 1)], unique=True)


def migrate(store_id: str, currency: str, source_dir: Path, skip_confirm: bool) -> None:
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]

    if not skip_confirm:
        answer = input(
            f"About to delete any existing data for store '{store_id}' "
            f"in DB '{settings.MONGO_DB}' and re-migrate. Continue? [y/N] "
        )
        if answer.strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

    print(f"\nMigrating from {source_dir} -> MongoDB (store_id={store_id}, currency={currency})")

    _ensure_indexes(db)

    print("\nWiping existing store data...")
    _wipe_existing(db, store_id)

    products_raw = _load_json(source_dir / "products.json")
    products_docs = [_transform_product(p, store_id) for p in products_raw]
    db["products"].insert_many(products_docs)
    print(f"  inserted {len(products_docs)} products")

    policies_raw = _load_json(source_dir / "policies.json")
    policies_docs = [_transform_policy(p, store_id) for p in policies_raw]
    db["policies"].insert_many(policies_docs)
    print(f"  inserted {len(policies_docs)} policies")

    faq_raw = _load_json(source_dir / "faq.json")
    faq_docs = [_transform_faq(f, store_id) for f in faq_raw]
    db["faq"].insert_many(faq_docs)
    print(f"  inserted {len(faq_docs)} faq entries")

    store_info_raw = _load_json(source_dir / "store_info.json")
    store_info_doc = _transform_store_info(store_info_raw, store_id, currency)
    db["store_info"].insert_one(store_info_doc)
    print("  inserted 1 store_info document")

    print("\nMigration complete.")


def main():
    parser = argparse.ArgumentParser(description="Migrate store JSON files into MongoDB.")
    parser.add_argument("--store-id", required=True, help="Store identifier, e.g. store_001")
    parser.add_argument("--currency", required=True, help="Store default currency, e.g. EGP")
    parser.add_argument("--source-dir", default=str(_DEFAULT_SOURCE_DIR),
                        help=f"Directory containing the 4 JSON files (default: {_DEFAULT_SOURCE_DIR})")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    migrate(
        store_id=args.store_id,
        currency=args.currency,
        source_dir=source_dir,
        skip_confirm=args.yes,
    )


if __name__ == "__main__":
    main()