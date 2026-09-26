"""Product write layer: single-product upsert/remove, reconciliation guards, CSV products never removed."""

import uuid

import pytest

from app.ingestion import platform_updates as pu
from app.products.product_store import _get_collection, fetch_products
from app.schemas.models import Product


@pytest.fixture
def env(monkeypatch):
    st = {"upserted": [], "deleted": []}
    monkeypatch.setattr(pu, "get_industry", lambda sid: "fashion")
    monkeypatch.setattr(pu, "upsert_products_to_qdrant",
                        lambda sid, products: st["upserted"].append([p.product_id for p in products]) or len(products))
    monkeypatch.setattr(pu, "delete_products_from_qdrant", lambda sid, ids: st["deleted"].append(sorted(ids)) or len(ids))
    return st


def _rows(handle, price="300", sizes=("M", "L"), parent=None):
    parent = parent or f"gid://P/{handle}"
    return [{"product_id": handle, "name": handle.title(), "price": price, "sku": f"{handle}-{s}", "stock": 5,
             "Size": s, "external_id": f"gid://V/{handle}-{s}", "external_parent_id": parent} for s in sizes]


def _store():
    return f"store_pu_{uuid.uuid4().hex[:8]}"


def _csv_product(sid, pid="csv-only"):
    _get_collection().insert_one(Product(product_id=pid, store_id=sid, name="CSV", price=10).model_dump())


def test_upsert_single_product_only_touches_its_vectors(env):
    sid = _store()
    assert pu.upsert_platform_products(sid, _rows("tee")) == ["tee"]
    p = fetch_products(sid, ["tee"])[0]
    assert {v.attributes["size"] for v in p.variants} == {"M", "L"} and p.external_parent_id == "gid://P/tee"
    assert env["upserted"] == [["tee"]] and env["deleted"] == []   # no snapshot, nothing else wiped


def test_repeated_update_is_idempotent_and_applies_changes(env):
    sid = _store()
    pu.upsert_platform_products(sid, _rows("tee", price="300"))
    pu.upsert_platform_products(sid, _rows("tee", price="350"))
    pu.upsert_platform_products(sid, _rows("tee", price="350"))  # duplicate delivery
    assert _get_collection().count_documents({"store_id": sid}) == 1
    assert {v.price for v in fetch_products(sid, ["tee"])[0].variants} == {350.0}


def test_remove_by_platform_id_never_touches_csv(env):
    sid = _store()
    pu.upsert_platform_products(sid, _rows("tee"))
    pu.upsert_platform_products(sid, _rows("cap", sizes=("One",)))
    _csv_product(sid)
    assert pu.remove_platform_product(sid, "gid://P/tee") == ["tee"]
    assert env["deleted"] == [["tee"]]
    assert sorted(_get_collection().distinct("product_id", {"store_id": sid})) == ["cap", "csv-only"]
    assert pu.remove_platform_product(sid, "gid://P/unknown") == []
    assert pu.remove_platform_product(sid, "csv-only") == []           # a product_id is not a platform id


def test_remove_by_simple_product_id(env):
    sid = _store()
    pu.upsert_platform_products(sid, [{"product_id": "mug", "name": "Mug", "price": "50", "stock": 3,
                                       "external_id": "77", "external_parent_id": None}])
    assert pu.remove_platform_product(sid, "77") == ["mug"]


def test_reconciliation_and_guards(env):
    sid = _store()
    for h in ("a", "b", "c", "d"):
        pu.upsert_platform_products(sid, _rows(h, sizes=("M",)))
    _csv_product(sid)

    assert pu.remove_missing_platform_products(sid, set()) == []                     # platform returned nothing
    assert pu.remove_missing_platform_products(sid, {"a"}) == []                     # would remove 3/4 > 50%
    assert pu.remove_missing_platform_products(sid, {"a", "b", "c"}) == ["d"]        # normal
    assert sorted(_get_collection().distinct("product_id", {"store_id": sid})) == ["a", "b", "c", "csv-only"]


def test_all_store_products_includes_csv(env):
    sid = _store()
    pu.upsert_platform_products(sid, _rows("tee"))
    _csv_product(sid)
    assert sorted(p.product_id for p in pu.all_store_products(sid)) == ["csv-only", "tee"]