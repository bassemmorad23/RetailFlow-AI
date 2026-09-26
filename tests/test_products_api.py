"""Products API: list/search/filter/paging, detail without internals, CSV-only delete, isolation."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.products import routes as product_routes
from app.products.product_store import _get_collection
from app.schemas.models import Product, Variant


@pytest.fixture
def env(monkeypatch):
    st = {"platform": None, "synced": []}
    monkeypatch.setattr(product_routes, "store_platform", lambda sid: st["platform"])
    monkeypatch.setattr(product_routes, "sync_products_to_qdrant",
                        lambda sid, products: st["synced"].append(sorted(p.product_id for p in products)))
    return st


def _merchant():
    c = TestClient(app)
    c.post("/auth/register", json={"email": f"pr_{uuid.uuid4().hex[:8]}@example.com", "password": "Strong-pass-123"})
    sid = c.post("/stores", json={"display_name": "S", "industry": "fashion", "country": "EG"}).json()["store_id"]
    return c, sid


def _seed(sid):
    products = [
        Product(product_id="tee", store_id=sid, name="Cotton T-Shirt", price=300, category="tops",
                stock_status="in_stock", external_parent_id="P1",
                variants=[Variant(sku="tee-M", price=300, attributes={"size": "M"}, external_id="V1"),
                          Variant(sku="tee-L", price=320, attributes={"size": "L"}, external_id="V2")]),
        Product(product_id="cap", store_id=sid, name="Baseball Cap", price=150, category="accessories",
                stock_status="out_of_stock"),
        Product(product_id="scarf", store_id=sid, name="Silk Scarf", price=500, category="accessories"),
    ]
    _get_collection().insert_many([p.model_dump() for p in products])


def test_list_search_filters_and_paging(env):
    c, sid = _merchant()
    _seed(sid)
    body = c.get(f"/stores/{sid}/products").json()
    assert body["total"] == 3 and [p["name"] for p in body["products"]] == ["Baseball Cap", "Cotton T-Shirt", "Silk Scarf"]
    tee = next(p for p in body["products"] if p["product_id"] == "tee")
    assert (tee["variants_count"], tee["currency"], tee["stock_status"]) == (2, "EGP", "in_stock")

    assert [p["product_id"] for p in c.get(f"/stores/{sid}/products", params={"q": "shirt"}).json()["products"]] == ["tee"]
    assert c.get(f"/stores/{sid}/products", params={"q": ".*"}).json()["total"] == 0  # regex is escaped
    assert c.get(f"/stores/{sid}/products", params={"category": "accessories"}).json()["total"] == 2
    assert [p["product_id"] for p in c.get(f"/stores/{sid}/products", params={"stock": "unknown"}).json()["products"]] == ["scarf"]
    page2 = c.get(f"/stores/{sid}/products", params={"page": 2, "limit": 2}).json()
    assert [p["product_id"] for p in page2["products"]] == ["scarf"] and page2["total"] == 3


def test_detail_hides_platform_ids(env):
    c, sid = _merchant()
    _seed(sid)
    d = c.get(f"/stores/{sid}/products/tee").json()
    assert d["name"] == "Cotton T-Shirt" and len(d["variants"]) == 2 and d["currency"] == "EGP"
    assert "external_parent_id" not in d and "store_id" not in d
    assert all("external_id" not in v for v in d["variants"])
    assert c.get(f"/stores/{sid}/products/nope").status_code == 404


def test_delete_csv_product_refreshes_index(env):
    c, sid = _merchant()
    _seed(sid)
    assert c.delete(f"/stores/{sid}/products/cap").status_code == 204
    assert c.get(f"/stores/{sid}/products/cap").status_code == 404
    assert env["synced"] == [["scarf", "tee"]]
    assert c.delete(f"/stores/{sid}/products/cap").status_code == 404


def test_delete_blocked_for_platform_stores(env):
    env["platform"] = "shopify"
    c, sid = _merchant()
    _seed(sid)
    r = c.delete(f"/stores/{sid}/products/tee")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "managed_by_platform"
    assert c.get(f"/stores/{sid}/products/tee").status_code == 200


def test_isolation(env):
    c, sid = _merchant()
    other, other_sid = _merchant()
    _seed(sid)
    assert other.get(f"/stores/{sid}/products").status_code == 404
    assert other.get(f"/stores/{other_sid}/products/tee").status_code == 404
    assert other.delete(f"/stores/{other_sid}/products/tee").status_code == 404
    assert c.get(f"/stores/{sid}/products").json()["total"] == 3
    assert TestClient(app).get(f"/stores/{sid}/products").status_code == 401