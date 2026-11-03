"""
ScholarMate AI — FastAPI entrypoint.

- Lifespan connects to MongoDB Atlas (Motor) on startup.
- /agent              : runs the LangGraph agent
- /students           : create a student profile
- /applications       : list / get / patch a tracked application
- /applications/{id}/checklist/{item_id} : toggle a checklist item
- /applications/{id}/essay               : fetch essay draft + review
"""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr

from app.agents.graph import graph
from app.database.connection import (
    connect_to_mongo,
    close_mongo_connection,
    get_db,
)
from app.services.applications import (
    list_applications,
    update_application,
    get_application_by_id,
    set_checklist,
    toggle_checklist_item,
    VALID_STATUSES,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield
    await close_mongo_connection()


app = FastAPI(title="ScholarMate AI", lifespan=lifespan)


# ---------------------------------------------------------------- #
# Schemas
# ---------------------------------------------------------------- #
class AgentRequest(BaseModel):
    user_goal: str
    student_email: str | None = None
    scholarship_name: str | None = None
    draft: str | None = None


class StudentCreate(BaseModel):
    full_name: str
    email: EmailStr
    country: str | None = None
    field_of_study: str | None = None
    degree_level: str | None = None


class ApplicationUpdate(BaseModel):
    status: str | None = None
    notes: str | None = None


class ChecklistToggle(BaseModel):
    done: bool


# ---------------------------------------------------------------- #
# Health
# ---------------------------------------------------------------- #
@app.get("/")
def root():
    return {"message": "ScholarMate AI Backend Running"}


# ---------------------------------------------------------------- #
# Context loading
# ---------------------------------------------------------------- #
async def _load_context(req: AgentRequest) -> dict:
    db = get_db()
    profile = None
    scholarship = None

    if req.student_email and db is not None:
        profile = await db.students.find_one(
            {"email": req.student_email}, {"_id": 0}
        )

    if req.scholarship_name and db is not None:
        scholarship = await db.scholarships.find_one(
            {"name": {"$regex": req.scholarship_name, "$options": "i"}},
            {"_id": 0},
        )

    return {
        "student_profile": profile,
        "target_scholarship": scholarship,
    }


# ---------------------------------------------------------------- #
# Agent endpoints
# ---------------------------------------------------------------- #
@app.get("/agent")
async def run_agent_default():
    ctx = await _load_context(
        AgentRequest(user_goal="Find AI scholarships in Germany")
    )
    return await graph.ainvoke({
        "user_goal": "Find AI scholarships in Germany",
        "intent": "",
        "needs_tool": False,
        "tool_name": None,
        "tool_result": None,
        "student_profile": ctx["student_profile"],
        "target_scholarship": ctx["target_scholarship"],
        "eligibility": None,
        "application_plan": None,
        "application_id": None,
        "application_record": None,
        "checklist": None,
        "essay_draft": None,
        "essay_review": None,
        "recommendations": None,
        "match_score": None,
        "plan": "",
        "final_response": "",
    })


@app.post("/agent")
async def run_agent(req: AgentRequest):
    ctx = await _load_context(req)
    return await graph.ainvoke({
        "user_goal": req.user_goal,
        "intent": "",
        "needs_tool": False,
        "tool_name": None,
        "tool_result": None,
        "student_profile": ctx["student_profile"],
        "target_scholarship": ctx["target_scholarship"],
        "eligibility": None,
        "application_plan": None,
        "application_id": None,
        "application_record": None,
        "checklist": None,
        "essay_draft": req.draft,
        "essay_review": None,
        "recommendations": None,
        "match_score": None,
        "plan": "",
        "final_response": "",
    })


# ---------------------------------------------------------------- #
# Students
# ---------------------------------------------------------------- #
@app.post("/students", status_code=201)
async def create_student(payload: StudentCreate):
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready")

    existing = await db.students.find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    doc = payload.model_dump()
    doc["created_at"] = datetime.utcnow()
    result = await db.students.insert_one(doc)

    return {"id": str(result.inserted_id), **doc}


# ---------------------------------------------------------------- #
# Applications
# ---------------------------------------------------------------- #
@app.get("/applications")
async def get_applications(student_email: str):
    if not student_email:
        raise HTTPException(status_code=400, detail="student_email required")
    return await list_applications(student_email)


@app.get("/applications/{application_id}")
async def get_application(application_id: str):
    record = await get_application_by_id(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Application not found")
    return record


@app.patch("/applications/{application_id}")
async def patch_application(application_id: str, payload: ApplicationUpdate):
    if payload.status and payload.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Allowed: {sorted(VALID_STATUSES)}",
        )

    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    record = await update_application(application_id, updates)
    if not record:
        raise HTTPException(status_code=404, detail="Application not found")
    return record


@app.patch("/applications/{application_id}/checklist/{item_id}")
async def patch_checklist_item(
    application_id: str, item_id: str, payload: ChecklistToggle
):
    record = await toggle_checklist_item(application_id, item_id, payload.done)
    if not record:
        raise HTTPException(
            status_code=404,
            detail="Application or checklist item not found",
        )
    return record


@app.get("/applications/{application_id}/essay")
async def get_essay(application_id: str):
    record = await get_application_by_id(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Application not found")
    return {
        "essay_draft": record.get("essay_draft"),
        "essay_review": record.get("essay_review"),
    }