"""Store management endpoints for authenticated merchants."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.auth.dependencies import get_current_user_id, require_store_member
from app.auth.repository import add_store_member
from app.settings.store_settings import create_store, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stores", tags=["stores"])


class CreateStoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(min_length=1, max_length=100)
    industry: str = Field(min_length=1, max_length=50)
    country: str = Field(min_length=2, max_length=2)
    currency: str | None = Field(default=None, min_length=3, max_length=3)


def _public_store(s) -> dict:
    return {
        "store_id": s.store_id,
        "display_name": s.display_name,
        "industry": s.industry,
        "plan": s.plan,
        "billing_anchor_day": s.billing_anchor_day,
        "country": s.country,
        "currency": s.currency,
    }


@router.post("", status_code=201)
def create_store_endpoint(body: CreateStoreRequest, user_id: str = Depends(get_current_user_id)) -> dict:
    store_id = f"store_{uuid.uuid4().hex[:16]}"
    try:
            store = create_store(store_id, body.display_name, body.industry, body.country, body.currency)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    add_store_member(user_id, store_id, role="owner")
    logger.info("Store created", extra={"auth_user_id": user_id, "new_store_id": store_id})
    return _public_store(store)


@router.get("/{store_id}")
def get_store_endpoint(store_id: str = Depends(require_store_member)) -> dict:
    return _public_store(get_settings(store_id))