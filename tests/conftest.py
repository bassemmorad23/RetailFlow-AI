"""
Test setup. Env vars MUST be set before any `app` import.
All tests run against a separate database; its collections are
dropped before and after the run.
"""

import os

os.environ["MONGO_DB"] = "storeflow_test"
os.environ["SENTRY_DSN"] = ""
os.environ["SESSION_COOKIE_SECURE"] = "false"

import pytest
from pymongo import MongoClient

TEST_DB = "storeflow_test"


def _wipe(client: MongoClient) -> None:
    db = client[TEST_DB]
    for name in db.list_collection_names():
        db.drop_collection(name)


@pytest.fixture(scope="session", autouse=True)
def _isolated_test_db():
    from app.config import settings
    # Safety: never touch anything but the test database.
    assert settings.MONGO_DB == TEST_DB, f"Refusing to run: MONGO_DB={settings.MONGO_DB}"
    client = MongoClient(settings.MONGO_URI)
    _wipe(client)
    yield
    _wipe(client)