"""
Essay drafting and review service.

Two responsibilities:
1. generate_essay_draft — Gemini writes an SOP draft from profile + scholarship.
2. review_essay         — Gemini scores and critiques a provided draft.
"""

import json
from app.config.llm import get_llm

llm_draft = get_llm(temperature=0.7)   # higher temp for creative writing
llm_review = get_llm(temperature=0.2)  # low temp for consistent scoring


DRAFT_PROMPT = """You are ScholarMate AI, an experienced scholarship essay coach.

Write a compelling first-draft Statement of Purpose (550-650 words) for the
student below, targeting the specified scholarship.

Requirements:
- Open with a specific, personal hook (not "I am writing to apply...")
- Connect the student's background to the scholarship's field and values
- Include one concrete achievement or project (invent details ONLY if the
  profile is sparse, and keep them plausible given the field)
- Explain why THIS scholarship and why now
- Close with a forward-looking statement about impact
- Do NOT use clichés like "ever since I was a child" or "I have always been passionate"
- Write in first person, natural tone, no headings

Return ONLY the essay text. No title, no markdown fences, no commentary.

STUDENT PROFILE:
{profile}

SCHOLARSHIP:
{scholarship}

ELIGIBILITY CONTEXT:
{eligibility}
"""


REVIEW_PROMPT = """You are ScholarMate AI, a scholarship essay reviewer.

Review the STUDENT'S ESSAY DRAFT against the SCHOLARSHIP requirements.

Return ONLY valid JSON with this exact shape (no markdown, no code fences):
{{
  "score": <integer 0-100>,
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>", "<weakness 3>"],
  "suggestions": ["<actionable suggestion 1>", "<actionable suggestion 2>", "<actionable suggestion 3>"],
  "alignment": "<one of: strong, moderate, weak>"
}}

Scoring guide:
- 85-100: compelling, specific, well-aligned
- 60-84:  solid but generic or missing specifics
- 0-59:   off-topic, too vague, or misaligned with the scholarship

Be honest, specific, and actionable. Do NOT invent facts not in the draft.

SCHOLARSHIP:
{scholarship}

STUDENT PROFILE:
{profile}

ESSAY DRAFT:
{draft}
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
    return {
        "score": 0,
        "strengths": [],
        "weaknesses": ["Could not parse review response."],
        "suggestions": ["Try again."],
        "alignment": "weak",
    }


async def generate_essay_draft(
    profile: dict, scholarship: dict, eligibility: dict | None
) -> str:
    prompt = DRAFT_PROMPT.format(
        profile=json.dumps(profile, default=str, indent=2),
        scholarship=json.dumps(scholarship, default=str, indent=2),
        eligibility=json.dumps(eligibility or {}, default=str, indent=2),
    )
    try:
        response = llm_draft.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"[Draft generation failed: {e}]"


async def review_essay(
    draft: str, scholarship: dict, profile: dict | None
) -> dict:
    prompt = REVIEW_PROMPT.format(
        scholarship=json.dumps(scholarship, default=str, indent=2),
        profile=json.dumps(profile or {}, default=str, indent=2),
        draft=draft,
    )
    try:
        response = llm_review.invoke(prompt)
        result = _safe_json(response.content)
    except Exception as e:
        return {
            "score": 0,
            "strengths": [],
            "weaknesses": [f"Review failed: {e}"],
            "suggestions": ["Retry in a moment."],
            "alignment": "weak",
        }

    try:
        result["score"] = max(0, min(100, int(result.get("score", 0))))
    except Exception:
        result["score"] = 0
    return result