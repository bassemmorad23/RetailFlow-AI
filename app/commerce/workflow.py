"""
In-conversation workflow state (the order draft).

Stored on the inbox conversation document (like the summary), so no extra
collection. Every save is version-checked (compare-and-set): a save based
on an older version fails instead of overwriting newer state. Drafts expire
after DRAFT_TTL of inactivity.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.commerce.models import OrderDraft
from app.inbox import repository as inbox_repo

DRAFT_TTL = timedelta(hours=24)


@dataclass(frozen=True)
class WorkflowState:
    draft: OrderDraft | None   # None = no active draft (or it expired)
    version: int               # pass back to save_draft / clear_draft


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _col():
    return inbox_repo._db()["inbox_conversations"]


def _version_filter(store_id: str, conversation_id: str, version: int) -> dict:
    flt: dict = {"store_id": store_id, "id": conversation_id}
    if version == 0:
        flt["$or"] = [{"workflow_version": 0}, {"workflow_version": {"$exists": False}}]
    else:
        flt["workflow_version"] = version
    return flt


def load_draft(store_id: str, conversation_id: str) -> WorkflowState:
    doc = _col().find_one({"store_id": store_id, "id": conversation_id},
                          {"_id": 0, "workflow": 1, "workflow_version": 1})
    if doc is None:
        raise LookupError(conversation_id)

    version = int(doc.get("workflow_version") or 0)
    wf = doc.get("workflow")
    if not wf or wf.get("type") != "order":
        return WorkflowState(draft=None, version=version)

    expires = wf.get("expires_at")
    if expires is not None and expires <= _now():
        return WorkflowState(draft=None, version=version)
    return WorkflowState(draft=OrderDraft(**wf["draft"]), version=version)


def save_draft(store_id: str, conversation_id: str, draft: OrderDraft, *, expected_version: int) -> bool:
    """True if saved; False if the draft changed since expected_version (reload and retry)."""
    now = _now()
    result = _col().update_one(
        _version_filter(store_id, conversation_id, expected_version),
        {"$set": {
            "workflow": {"type": "order", "draft": draft.model_dump(),
                         "updated_at": now, "expires_at": now + DRAFT_TTL},
            "workflow_version": expected_version + 1,
        }},
    )
    return result.modified_count == 1


def clear_draft(store_id: str, conversation_id: str, *, expected_version: int) -> bool:
    result = _col().update_one(
        _version_filter(store_id, conversation_id, expected_version),
        {"$set": {"workflow": None, "workflow_version": expected_version + 1}},
    )
    return result.modified_count == 1