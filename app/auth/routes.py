"""Auth endpoints: register, login, logout, me."""

import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.auth import repository as repo
from app.auth.dependencies import get_current_user_id
from app.auth.passwords import hash_password, needs_rehash, verify_password
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Verified against when the email doesn't exist, so login timing doesn't reveal accounts.
_DUMMY_HASH = hash_password("dummy-password-for-timing")


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(max_length=254)
    password: str = Field(min_length=10, max_length=128)
    full_name: str = Field(default="", max_length=100)

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        v = v.strip()
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email address")
        return v


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


def _public_user(user: dict) -> dict:
    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "full_name": user.get("full_name", ""),
    }


def _set_session_cookie(response: Response, raw_token: str) -> None:
    response.set_cookie(
        key=settings.SESSION_COOKIE_NAME,
        value=raw_token,
        max_age=int(repo.SESSION_TTL.total_seconds()),
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


@router.post("/register", status_code=201)
def register(body: RegisterRequest, response: Response) -> dict:
    try:
        user_id = repo.create_user(body.email, hash_password(body.password), body.full_name)
    except repo.EmailAlreadyRegistered:
        raise HTTPException(status_code=409, detail="Email already registered")
    _set_session_cookie(response, repo.create_session(user_id))
    logger.info("User registered", extra={"auth_user_id": user_id})
    return {"user": _public_user(repo.get_user_by_id(user_id)), "stores": []}


@router.post("/login")
def login(body: LoginRequest, response: Response) -> dict:
    user = repo.get_user_by_email(body.email)
    stored_hash = user["password_hash"] if user else _DUMMY_HASH
    if not verify_password(stored_hash, body.password) or user is None:
        logger.info("Login failed")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if needs_rehash(user["password_hash"]):
        repo.update_password_hash(user["user_id"], hash_password(body.password))

    _set_session_cookie(response, repo.create_session(user["user_id"]))
    logger.info("Login succeeded", extra={"auth_user_id": user["user_id"]})
    return {"user": _public_user(user), "stores": repo.list_user_stores(user["user_id"])}


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response) -> None:
    repo.delete_session(request.cookies.get(settings.SESSION_COOKIE_NAME, ""))
    response.delete_cookie(settings.SESSION_COOKIE_NAME, path="/")


@router.get("/me")
def me(user_id: str = Depends(get_current_user_id)) -> dict:
    user = repo.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user": _public_user(user), "stores": repo.list_user_stores(user_id)}