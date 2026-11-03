"""
Application persistence service.

- create a tracked application for a student + scholarship
- update its status / notes
- list all applications for a student
- fetch a single application by id

Keeps DB logic out of agent nodes.
"""

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from app.database.connection import get_db


VALID_STATUSES = {
    "interested",
    "preparing",
    "applied",
    "interview",
    "accepted",
    "rejected",
    "withdrawn",
}


def _clean(doc: dict | None) -> dict | None:
    if not doc:
        return None
    out = {k: v for k, v in doc.items() if k != "_id"}
    if "_id" in doc:
        out["id"] = str(doc["_id"])
    return out


async def get_application_by_id(application_id: str) -> dict | None:
    db = get_db()
    if db is None:
        return None
    try:
        oid = ObjectId(application_id)
    except Exception:
        return None
    doc = await db.applications.find_one({"_id": oid})
    return _clean(doc)


async def find_application(
    student_email: str, scholarship_slug: str
) -> dict | None:
    db = get_db()
    if db is None:
        return None
    doc = await db.applications.find_one({
        "student_email": student_email,
        "scholarship_slug": scholarship_slug,
    })
    return _clean(doc)


async def create_application(
    student_email: str,
    scholarship: dict,
    eligibility: dict | None = None,
    application_plan: dict | None = None,
    status: str = "interested",
) -> dict:
    db = get_db()
    if db is None:
        raise RuntimeError("Database not connected")

    now = datetime.now(timezone.utc)

    doc = {
        "student_email": student_email,
        "scholarship_slug": scholarship.get("slug"),
        "scholarship_name": scholarship.get("name"),
        "status": status if status in VALID_STATUSES else "interested",
        "eligibility": eligibility,
        "application_plan": application_plan,
        "checklist": [],
        "notes": "",
        "created_at": now,
        "updated_at": now,
    }

    result = await db.applications.update_one(
        {
            "student_email": student_email,
            "scholarship_slug": scholarship.get("slug"),
        },
        {"$setOnInsert": doc, "$set": {"updated_at": now}},
        upsert=True,
    )

    existing = await db.applications.find_one({
        "student_email": student_email,
        "scholarship_slug": scholarship.get("slug"),
    })
    return _clean(existing)


async def update_application(
    application_id: str, updates: dict[str, Any]
) -> dict | None:
    db = get_db()
    if db is None:
        return None

    try:
        oid = ObjectId(application_id)
    except Exception:
        return None

    if "status" in updates and updates["status"] not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {updates['status']}")

    updates["updated_at"] = datetime.now(timezone.utc)
    await db.applications.update_one({"_id": oid}, {"$set": updates})
    return await get_application_by_id(application_id)


async def list_applications(student_email: str) -> list[dict]:
    db = get_db()
    if db is None:
        return []
    cursor = db.applications.find(
        {"student_email": student_email}
    ).sort("updated_at", -1)
    docs = await cursor.to_list(length=200)
    return [_clean(d) for d in docs]