"""FastAPI dependencies for authentication and tenant isolation."""

from fastapi import Depends, HTTPException, Request

from app.auth.repository import is_store_member, resolve_session
from app.config import settings


def get_current_user_id(request: Request) -> str:
    raw = request.cookies.get(settings.SESSION_COOKIE_NAME)
    user_id = resolve_session(raw) if raw else None
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_id


def require_store_member(store_id: str, user_id: str = Depends(get_current_user_id)) -> str:
    """
    Guard for /stores/{store_id}/... routes.
    Returns 404 (not 403) so outsiders can't learn which store_ids exist.
    """
    if not is_store_member(user_id, store_id):
        raise HTTPException(status_code=404, detail="Store not found")
    return store_id