"""
Scholarship matching service.

Two responsibilities:
1. recommend_scholarships — rank all scholarships by match, excluding ones
   the student already applied to.
2. score_match            — score a single (profile, scholarship) pair.
"""

import json
from app.config.llm import get_llm
from app.database.connection import get_db

llm = get_llm(temperature=0.15)


RECOMMEND_PROMPT = """You are ScholarMate AI's scholarship matcher.

You receive a STUDENT PROFILE and a list of candidate SCHOLARSHIPS the student
has NOT yet applied to. Rank the top 5 best matches.

Return ONLY valid JSON with this exact shape (no markdown, no code fences):
{{
  "recommendations": [
    {{
      "slug": "<scholarship slug>",
      "name": "<scholarship name>",
      "country": "<country>",
      "deadline": "<YYYY-MM-DD>",
      "amount": "<amount>",
      "match_score": <integer 0-100>,
      "reason": "<1 short sentence on why it fits>"
    }},
    ...
  ]
}}

Rules:
- Rank by fit: field alignment, level, country preference, deadline feasibility.
- Only use scholarships from the provided candidate list. Do NOT invent.
- match_score must reflect the student's actual fit.
- Sort recommendations by match_score descending.
- Return AT MOST 5 items.
- If nothing is a reasonable match (< 40), return an empty list.

STUDENT PROFILE:
{profile}

CANDIDATE SCHOLARSHIPS:
{scholarships}
"""


MATCH_PROMPT = """You are ScholarMate AI's scholarship matcher.

Evaluate how well this STUDENT fits this specific SCHOLARSHIP.

Return ONLY valid JSON with this exact shape (no markdown, no code fences):
{{
  "match_score": <integer 0-100>,
  "verdict": "strong" | "moderate" | "weak",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements": ["<improvement 1>", "<improvement 2>", "<improvement 3>"],
  "fit_summary": "<one sentence on overall fit>"
}}

Score guide:
- 80-100 -> "strong"
- 50-79  -> "moderate"
- 0-49   -> "weak"

STUDENT PROFILE:
{profile}

SCHOLARSHIP:
{scholarship}
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
    return {}


async def recommend_scholarships(
    profile: dict, limit: int = 5
) -> list[dict]:
    """Return top-N matched scholarships, excluding ones already applied to."""
    db = get_db()
    if db is None:
        return []

    # Scholarships already tracked by the student
    applied_cursor = db.applications.find(
        {"student_email": profile.get("email")}, {"scholarship_slug": 1}
    )
    applied = await applied_cursor.to_list(length=500)
    applied_slugs = {a.get("scholarship_slug") for a in applied if a.get("scholarship_slug")}

    # Fetch all scholarships and exclude applied ones
    cursor = db.scholarships.find({}, {"_id": 0})
    all_scholarships = await cursor.to_list(length=500)

    candidates = [
        {
            "slug": s.get("slug"),
            "name": s.get("name"),
            "country": s.get("country"),
            "level": s.get("level"),
            "field": s.get("field"),
            "deadline": s.get("deadline"),
            "amount": s.get("amount"),
            "description": s.get("description"),
        }
        for s in all_scholarships
        if s.get("slug") not in applied_slugs
    ]

    if not candidates:
        return []

    prompt = RECOMMEND_PROMPT.format(
        profile=json.dumps(profile, default=str, indent=2),
        scholarships=json.dumps(candidates, default=str, indent=2),
    )

    try:
        response = llm.invoke(prompt)
        raw = _safe_json(response.content)
    except Exception:
        raw = {}

    recs = raw.get("recommendations", [])[:limit]

    # Clamp scores
    for r in recs:
        try:
            r["match_score"] = max(0, min(100, int(r.get("match_score", 0))))
        except Exception:
            r["match_score"] = 0

    return recs


async def score_match(profile: dict, scholarship: dict) -> dict:
    """Score a single scholarship against the student profile."""
    prompt = MATCH_PROMPT.format(
        profile=json.dumps(profile, default=str, indent=2),
        scholarship=json.dumps(scholarship, default=str, indent=2),
    )

    try:
        response = llm.invoke(prompt)
        result = _safe_json(response.content)
    except Exception as e:
        return {
            "match_score": 0,
            "verdict": "weak",
            "strengths": [],
            "improvements": [f"Matching failed: {e}"],
            "fit_summary": "Could not compute match.",
        }

    try:
        result["match_score"] = max(0, min(100, int(result.get("match_score", 0))))
    except Exception:
        result["match_score"] = 0

    # Normalize verdict
    if result.get("verdict") not in {"strong", "moderate", "weak"}:
        score = result["match_score"]
        result["verdict"] = (
            "strong" if score >= 80 else "moderate" if score >= 50 else "weak"
        )

    result.setdefault("strengths", [])
    result.setdefault("improvements", [])
    result.setdefault("fit_summary", "")

    return result