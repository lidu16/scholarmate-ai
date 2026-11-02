"""
ScholarMate AI — FastAPI entrypoint.

- Lifespan connects to MongoDB Atlas (Motor) on startup.
- /agent    : runs the LangGraph agent (Gemini + tool + async DB search)
- /students : creates a student profile in MongoDB
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


# ---------------------------------------------------------------- #
# Lifespan — connect once at startup, close at shutdown
# ---------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield
    await close_mongo_connection()


app = FastAPI(title="ScholarMate AI", lifespan=lifespan)


# ---------------------------------------------------------------- #
# Request/Response schemas
# ---------------------------------------------------------------- #
class AgentRequest(BaseModel):
    user_goal: str


class StudentCreate(BaseModel):
    full_name: str
    email: EmailStr
    country: str | None = None
    field_of_study: str | None = None
    degree_level: str | None = None


# ---------------------------------------------------------------- #
# Health
# ---------------------------------------------------------------- #
@app.get("/")
def root():
    return {"message": "ScholarMate AI Backend Running"}


# ---------------------------------------------------------------- #
# Agent endpoints
# ---------------------------------------------------------------- #
@app.get("/agent")
async def run_agent_default():
    """Quick GET test with a default goal."""
    return await graph.ainvoke({
        "user_goal": "Find AI scholarships in Germany",
        "intent": "",
        "needs_tool": False,
        "tool_name": None,
        "tool_result": None,
        "plan": "",
        "final_response": "",
    })


@app.post("/agent")
async def run_agent(req: AgentRequest):
    """Main agent endpoint — accepts any user goal."""
    return await graph.ainvoke({
        "user_goal": req.user_goal,
        "intent": "",
        "needs_tool": False,
        "tool_name": None,
        "tool_result": None,
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

    return {
        "id": str(result.inserted_id),
        **doc,
    }