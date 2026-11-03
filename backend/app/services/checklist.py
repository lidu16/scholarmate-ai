"""
Document checklist service.

Gemini generates a scholarship-specific document checklist. Each item has:
  id, name, description, required (bool), category, done (bool), due_by (optional)
"""

import json
import uuid
from datetime import datetime, timezone

from app.config.llm import get_llm

llm = get_llm(temperature=0.2)


CHECKLIST_PROMPT = """You are ScholarMate AI's document checklist generator.

Given a STUDENT PROFILE, a SCHOLARSHIP, and an ELIGIBILITY RESULT, produce a
complete, scholarship-specific list of documents the student must prepare.

Return ONLY valid JSON with this exact shape (no markdown, no code fences):
{{
  "documents": [
    {{
      "name": "<short document name>",
      "description": "<one sentence explaining what to prepare>",
      "required": <true|false>,
      "category": "<one of: identity, academic, essays, recommendations, tests, financial, other>",
      "due_by": "<YYYY-MM-DD or null>"
    }},
    ...
  ]
}}

Rules:
- Include every document the student must submit for this scholarship.
- Cover standard items: transcripts, degree certificate, CV, statement of purpose, recommendation letters, language test (IELTS/TOEFL), passport, financial proof — but ONLY include ones the scholarship actually requires or reasonably expects.
- Address the eligibility "gaps" by adding documents that close them.
- When possible, set "due_by" to a realistic internal deadline before the scholarship's final deadline.
- Categories must be exactly one of: identity, academic, essays, recommendations, tests, financial, other.
- Keep descriptions short and actionable.
- Do NOT invent requirements not implied by the scholarship or its description.

TODAY: {today}
SCHOLARSHIP:
{scholarship}

STUDENT PROFILE:
{profile}

ELIGIBILITY RESULT:
{eligibility}
"""


def _safe_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass
    return {"documents": []}


async def generate_checklist(
    profile: dict,
    scholarship: dict,
    eligibility: dict,
) -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = CHECKLIST_PROMPT.format(
        today=today,
        profile=json.dumps(profile, default=str, indent=2),
        scholarship=json.dumps(scholarship, default=str, indent=2),
        eligibility=json.dumps(eligibility, default=str, indent=2),
    )

    try:
        response = llm.invoke(prompt)
        raw = _safe_json(response.content)
    except Exception:
        raw = {"documents": []}

    docs = raw.get("documents", [])

    checklist = []
    for doc in docs:
        checklist.append({
            "id": str(uuid.uuid4()),
            "name": doc.get("name", "Untitled document"),
            "description": doc.get("description", ""),
            "required": bool(doc.get("required", True)),
            "category": doc.get("category", "other"),
            "due_by": doc.get("due_by") or None,
            "done": False,
            "completed_at": None,
        })

    return checklist