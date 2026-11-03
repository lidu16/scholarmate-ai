"""
LangGraph nodes for ScholarMate AI.

Flow:
    classifier (Gemini)
        ├── scholarship_search  -> tool             -> planner
        ├── eligibility_check   -> eligibility      -> planner
        ├── application_plan    -> application_plan -> planner
        └── planning/general    ->                    planner
    planner -> response -> END

LLM failures fall back to safe defaults so a single Gemini hiccup
doesn't crash the whole request.
"""

from .state import AgentState
from ..tools.scholarship_search import search_scholarships
from ..services.eligibility import evaluate_eligibility
from ..services.application_planner import build_application_plan
from ..database.connection import get_db
from ..config.llm import get_llm

llm = get_llm(temperature=0.2)


# ---------------------------------------------------------------- #
# Node 1 — Intent classifier (Gemini)
# ---------------------------------------------------------------- #
def classifier_node(state: AgentState):
    goal = state["user_goal"]

    prompt = f"""You are the intent classifier for ScholarMate AI.

Classify the user's goal into EXACTLY ONE of these intents:
- "scholarship_search" : user wants to find, search, or discover scholarships
- "eligibility_check"  : user wants to know if they qualify / are eligible
- "application_plan"   : user wants a plan, timeline, roadmap, or preparation steps
                         to apply for a specific scholarship
- "planning"           : user wants general application advice that is NOT tied to
                         a specific scholarship
- "general"            : anything else

Return ONLY the intent word. No explanation. No punctuation. No quotes.

User goal: {goal}
Intent:"""

    try:
        response = llm.invoke(prompt)
        intent = response.content.strip().lower()
    except Exception:
        intent = "general"

    valid = {
        "scholarship_search",
        "eligibility_check",
        "application_plan",
        "planning",
        "general",
    }
    if intent not in valid:
        intent = "general"

    needs_tool = intent in {
        "scholarship_search",
        "eligibility_check",
        "application_plan",
    }
    tool_name = intent if needs_tool else None

    return {
        "intent": intent,
        "needs_tool": needs_tool,
        "tool_name": tool_name,
    }


# ---------------------------------------------------------------- #
# Node 2a — Scholarship search (async, MongoDB-backed)
# ---------------------------------------------------------------- #
async def tool_node(state: AgentState):
    if state.get("tool_name") == "scholarship_search":
        try:
            result = await search_scholarships(state["user_goal"])
        except Exception as e:
            result = f"Scholarship search failed: {e}"
    else:
        result = "No tool executed."
    return {"tool_result": result}


# ---------------------------------------------------------------- #
# Node 2b — Eligibility evaluation (async, Gemini + MongoDB)
# ---------------------------------------------------------------- #
async def eligibility_node(state: AgentState):
    db = get_db()
    if db is None:
        return {
            "eligibility": {
                "score": 0,
                "verdict": "insufficient_info",
                "reasons": [],
                "gaps": ["Database not connected."],
                "next_steps": [],
            }
        }

    profile = state.get("student_profile")
    if not profile:
        return {
            "eligibility": {
                "score": 0,
                "verdict": "insufficient_info",
                "reasons": [],
                "gaps": [
                    "No student profile provided. "
                    "Please create one first via POST /students."
                ],
                "next_steps": ["Register your profile with POST /students."],
            }
        }

    scholarship = state.get("target_scholarship")
    if not scholarship:
        goal = state["user_goal"]
        try:
            words = [w for w in goal.split() if len(w) > 3]
            if words:
                regex = "|".join(words)
                scholarship = await db.scholarships.find_one(
                    {"name": {"$regex": regex, "$options": "i"}},
                    {"_id": 0},
                )
        except Exception:
            scholarship = None

    if not scholarship:
        return {
            "eligibility": {
                "score": 0,
                "verdict": "insufficient_info",
                "reasons": [],
                "gaps": ["Could not find the scholarship you mentioned."],
                "next_steps": [
                    "Try searching first, then ask about a specific one."
                ],
            }
        }

    profile_clean = {k: v for k, v in profile.items() if k != "_id"}

    result = await evaluate_eligibility(profile_clean, scholarship)
    return {
        "student_profile": profile_clean,
        "target_scholarship": scholarship,
        "eligibility": result,
    }


# ---------------------------------------------------------------- #
# Node 2c — Application planner (async, Gemini + MongoDB)
# ---------------------------------------------------------------- #
async def application_plan_node(state: AgentState):
    db = get_db()
    if db is None:
        return {
            "application_plan": {
                "weeks_to_deadline": 0,
                "milestones": [],
                "key_deadlines": [],
                "risks": ["Database not connected."],
            }
        }

    profile = state.get("student_profile")
    if not profile:
        return {
            "application_plan": {
                "weeks_to_deadline": 0,
                "milestones": [],
                "key_deadlines": [],
                "risks": [
                    "No student profile provided. "
                    "Register via POST /students first."
                ],
            }
        }

    scholarship = state.get("target_scholarship")
    if not scholarship:
        goal = state["user_goal"]
        try:
            words = [w for w in goal.split() if len(w) > 3]
            if words:
                regex = "|".join(words)
                scholarship = await db.scholarships.find_one(
                    {"name": {"$regex": regex, "$options": "i"}},
                    {"_id": 0},
                )
        except Exception:
            scholarship = None

    if not scholarship:
        return {
            "application_plan": {
                "weeks_to_deadline": 0,
                "milestones": [],
                "key_deadlines": [],
                "risks": ["Could not identify the scholarship you want a plan for."],
            }
        }

    eligibility = state.get("eligibility")
    if not eligibility:
        eligibility = await evaluate_eligibility(
            {k: v for k, v in profile.items() if k != "_id"},
            scholarship,
        )

    plan = await build_application_plan(
        profile={k: v for k, v in profile.items() if k != "_id"},
        scholarship=scholarship,
        eligibility=eligibility,
    )

    return {
        "student_profile": {k: v for k, v in profile.items() if k != "_id"},
        "target_scholarship": scholarship,
        "eligibility": eligibility,
        "application_plan": plan,
    }


# ---------------------------------------------------------------- #
# Node 3 — Planner / response generator (Gemini)
# ---------------------------------------------------------------- #
def planner_node(state: AgentState):
    goal = state["user_goal"]
    tool_result = state.get("tool_result")
    eligibility = state.get("eligibility")
    application_plan = state.get("application_plan")
    scholarship = state.get("target_scholarship") or {}

    if application_plan and application_plan.get("milestones"):
        milestones_text = "\n".join(
            f"Week {m.get('week')}: {m.get('title')} ({m.get('focus', 'misc')})\n"
            + "\n".join(f"  - {t}" for t in m.get("tasks", []))
            for m in application_plan["milestones"]
        )
        key_dates_text = "\n".join(
            f"- {d.get('date')}: {d.get('label')}"
            for d in application_plan.get("key_deadlines", [])
        )
        risks_text = "\n".join(
            f"- {r}" for r in application_plan.get("risks", [])
        )

        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

Scholarship: {scholarship.get('name', 'Unknown')} (deadline {scholarship.get('deadline', 'unknown')})
Weeks to deadline: {application_plan.get('weeks_to_deadline')}

Your generated week-by-week plan:
{milestones_text}

Key dates:
{key_dates_text}

Risks:
{risks_text}

Write a warm, encouraging 5-7 sentence response that:
1. Opens with the number of weeks remaining and the final deadline
2. Summarizes the first 2 milestones as the immediate focus
3. Warns about the top 1-2 risks
4. Ends with one clear "start today" action

Do NOT reprint the whole table - keep it conversational."""
    elif eligibility:
        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

Eligibility: {eligibility.get('score')}/100 ({eligibility.get('verdict')}).
Reasons: {'; '.join(eligibility.get('reasons', []))}
Gaps: {'; '.join(eligibility.get('gaps', []))}

Write a friendly 4-6 sentence response with the verdict, top reasons, top gaps, and 1-2 next steps."""
    elif tool_result:
        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

Scholarships found:
{tool_result}

Write a short friendly response summarizing them with deadlines and offering eligibility checks or a plan."""
    else:
        prompt = f"""You are ScholarMate AI.

The student asked: "{goal}"

Write a helpful 3-5 sentence response with 2-3 concrete next steps.
Mention scholarship search, eligibility check, and application plan."""

    try:
        response = llm.invoke(prompt)
        plan_text = response.content.strip()
    except Exception:
        plan_text = "I'm having trouble responding right now. Please try again."

    return {"plan": plan_text}


# ---------------------------------------------------------------- #
# Node 4 — Final response assembly
# ---------------------------------------------------------------- #
def response_node(state: AgentState):
    return {"final_response": state.get("plan", "No response generated.")}


# ---------------------------------------------------------------- #
# Conditional router
# ---------------------------------------------------------------- #
def route_after_classify(state: AgentState) -> str:
    tool = state.get("tool_name")
    if tool == "scholarship_search":
        return "tool"
    if tool == "eligibility_check":
        return "eligibility"
    if tool == "application_plan":
        return "application_plan"
    return "planner"