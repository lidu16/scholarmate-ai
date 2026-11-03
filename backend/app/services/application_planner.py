"""
Application Planner service.

Given a student profile, target scholarship, and eligibility result,
ask Gemini to return a week-by-week application plan as structured JSON.
"""

import json
from datetime import datetime, timezone
from app.config.llm import get_llm

llm = get_llm(temperature=0.3)


PLANNER_PROMPT = """You are ScholarMate AI's application planner.

Given a STUDENT PROFILE, a SCHOLARSHIP, an ELIGIBILITY RESULT, and the current date,
produce a realistic week-by-week application plan that gets the student to
submission before the scholarship deadline.

Return ONLY valid JSON with this exact shape (no markdown, no code fences):
{{
  "weeks_to_deadline": <integer>,
  "milestones": [
    {{
      "week": <integer starting at 1>,
      "title": "<short title>",
      "tasks": ["<task 1>", "<task 2>", ...],
      "focus": "<one of: documents, essays, recommendations, tests, submission, review>"
    }},
    ...
  ],
  "key_deadlines": [
    {{ "date": "YYYY-MM-DD", "label": "<what is due>" }},
    ...
  ],
  "risks": ["<risk 1>", "<risk 2>", ...]
}}

Rules:
- Total milestones should cover the whole period from today to the deadline.
- Each milestone covers exactly 1 week.
- Number weeks starting at 1 for the closest upcoming week.
- Address the eligibility "gaps" first (e.g. missing language test, missing documents).
- Do NOT invent requirements not present in the scholarship or eligibility.
- If the deadline has already passed, return "weeks_to_deadline": 0 and one milestone
  with a single task: "Deadline has passed — pick the next intake."
- "key_deadlines" must always include the final submission date.
- Keep task strings short, action-oriented, and specific.

TODAY: {today}
STUDENT PROFILE:
{profile}

SCHOLARSHIP:
{scholarship}

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
    return {
        "weeks_to_deadline": 0,
        "milestones": [],
        "key_deadlines": [],
        "risks": ["Could not parse planner response."],
    }


def _weeks_between(today: datetime, deadline_str: str) -> int:
    try:
        deadline = datetime.strptime(deadline_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    except Exception:
        return 0
    delta = (deadline - today).days
    return max(0, delta // 7)


async def build_application_plan(
    profile: dict,
    scholarship: dict,
    eligibility: dict,
) -> dict:
    today = datetime.now(timezone.utc)
    today_str = today.strftime("%Y-%m-%d")

    prompt = PLANNER_PROMPT.format(
        today=today_str,
        profile=json.dumps(profile, default=str, indent=2),
        scholarship=json.dumps(scholarship, default=str, indent=2),
        eligibility=json.dumps(eligibility, default=str, indent=2),
    )

    try:
        response = llm.invoke(prompt)
        result = _safe_json(response.content)
    except Exception as e:
        return {
            "weeks_to_deadline": 0,
            "milestones": [],
            "key_deadlines": [],
            "risks": [f"LLM error: {e}"],
        }

    # Sanity checks / defaults
    if "weeks_to_deadline" not in result or not isinstance(result["weeks_to_deadline"], int):
        result["weeks_to_deadline"] = _weeks_between(today, scholarship.get("deadline", ""))

    result.setdefault("milestones", [])
    result.setdefault("key_deadlines", [])
    result.setdefault("risks", [])

    # Always include the final submission date
    deadline = scholarship.get("deadline")
    if deadline and not any(d.get("date") == deadline for d in result["key_deadlines"]):
        result["key_deadlines"].append({
            "date": deadline,
            "label": f"Submit application — {scholarship.get('name', 'scholarship')}"
        })

    return result