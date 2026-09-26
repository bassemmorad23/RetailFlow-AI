"""
Test setup. Env vars MUST be set before any `app` import.
Each pytest-xdist worker gets its own database (storeflow_test_gw0, ...),
so parallel workers never touch each other's data.
"""

import os

_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "main")
TEST_DB = f"storeflow_test_{_WORKER}"

os.environ["MONGO_DB"] = TEST_DB
# Tests use a local MongoDB (fast, offline). Override with TEST_MONGO_URI if needed.
#os.environ["MONGO_URI"] = os.environ.get("TEST_MONGO_URI", "mongodb://localhost:27017")

os.environ["SENTRY_DSN"] = ""
os.environ["SESSION_COOKIE_SECURE"] = "false"

import pytest
from pymongo import MongoClient


def _wipe(client: MongoClient) -> None:
    db = client[TEST_DB]
    for name in db.list_collection_names():
        db.drop_collection(name)


@pytest.fixture(scope="session", autouse=True)
def _isolated_test_db():
    from app.config import settings
    # Safety: never touch anything but a test database.
    assert settings.MONGO_DB == TEST_DB and TEST_DB.startswith("storeflow_test_"), \
        f"Refusing to run: MONGO_DB={settings.MONGO_DB}"
    client = MongoClient(settings.MONGO_URI)
    _wipe(client)
    yield
    _wipe(client)