"""
Eligibility evaluation service.

Given a student profile dict + a scholarship dict, ask Gemini to return
a structured eligibility verdict as JSON.
"""

import json
from app.config.llm import get_llm

llm = get_llm(temperature=0.1)


ELIGIBILITY_PROMPT = """You are ScholarMate AI's eligibility evaluator.

You receive a STUDENT PROFILE and a SCHOLARSHIP. Evaluate how well the student
matches the scholarship's stated requirements.

Return ONLY valid JSON with this exact shape (no markdown, no code fences):
{{
  "score": <integer 0-100>,
  "verdict": "eligible" | "partially_eligible" | "not_eligible" | "insufficient_info",
  "reasons": [<short string>, ...],
  "gaps": [<short string>, ...],
  "next_steps": [<short string>, ...]
}}

Rules:
- score 85-100 -> "eligible"
- score 50-84  -> "partially_eligible"
- score 0-49   -> "not_eligible"
- If the scholarship description or the profile is too vague to judge -> "insufficient_info"
- "reasons" = why the student matches (max 4 items)
- "gaps"    = what the student is missing or unclear (max 4 items)
- "next_steps" = concrete actions the student should take (max 4 items)
- Do NOT invent scholarship requirements not present in the data.
- Base everything on the description, country, level, field and deadline given.

STUDENT PROFILE:
{profile}

SCHOLARSHIP:
{scholarship}
"""


def _safe_json(raw: str) -> dict:
    """Extract the first JSON object from Gemini's response."""
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
    return {
        "score": 0,
        "verdict": "insufficient_info",
        "reasons": [],
        "gaps": ["Could not parse evaluation response."],
        "next_steps": ["Try rephrasing your question."],
    }


async def evaluate_eligibility(profile: dict, scholarship: dict) -> dict:
    """Return a structured eligibility verdict."""
    prompt = ELIGIBILITY_PROMPT.format(
        profile=json.dumps(profile, default=str, indent=2),
        scholarship=json.dumps(scholarship, default=str, indent=2),
    )

    try:
        response = llm.invoke(prompt)
        result = _safe_json(response.content)
    except Exception as e:
        result = {
            "score": 0,
            "verdict": "insufficient_info",
            "reasons": [],
            "gaps": [f"LLM error: {e}"],
            "next_steps": ["Retry in a moment."],
        }

    try:
        result["score"] = max(0, min(100, int(result.get("score", 0))))
    except Exception:
        result["score"] = 0

    return result