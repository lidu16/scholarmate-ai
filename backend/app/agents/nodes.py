"""
LangGraph nodes for ScholarMate AI.

Flow:
    classifier (Gemini)
        ├── scholarship_search      -> tool             -> planner
        ├── recommend_scholarships  -> recommend        -> planner
        ├── match_score             -> match_score      -> planner
        ├── eligibility_check       -> eligibility      -> planner
        ├── application_plan        -> application_plan -> planner
        ├── track_application       -> tracker          -> planner
        ├── prepare_documents       -> checklist        -> planner
        ├── write_essay             -> essay_draft      -> planner
        ├── review_essay            -> essay_review     -> planner
        └── planning/general        ->                    planner
    planner -> response -> END
"""

from .state import AgentState
from ..tools.scholarship_search import search_scholarships
from ..services.eligibility import evaluate_eligibility
from ..services.application_planner import build_application_plan
from ..services.checklist import generate_checklist
from ..services.essay import generate_essay_draft, review_essay
from ..services.matching import recommend_scholarships, score_match
from ..services.applications import (
    create_application,
    set_checklist,
    save_essay_draft,
    save_essay_review,
    get_application_by_id,
)
from ..database.connection import get_db
from ..config.llm import get_llm

llm = get_llm(temperature=0.2)


# ---------------------------------------------------------------- #
# Node 1 — Intent classifier
# ---------------------------------------------------------------- #
def classifier_node(state: AgentState):
    goal = state["user_goal"]

    prompt = f"""You are the intent classifier for ScholarMate AI.

Classify the user's goal into EXACTLY ONE of these intents:
- "scholarship_search"      : user wants to find scholarships by keyword
- "recommend_scholarships"  : user wants personalized recommendations / "what else should I apply for"
- "match_score"             : user wants to know how well they fit a SPECIFIC scholarship
- "eligibility_check"       : user wants to know if they qualify / are eligible
- "application_plan"        : user wants a plan, timeline, or roadmap
- "track_application"       : user wants to save, track, or follow an application
- "prepare_documents"       : user wants a document checklist
- "write_essay"             : user wants to draft an SOP / essay
- "review_essay"            : user wants feedback on an existing essay
- "planning"                : general advice not tied to a scholarship
- "general"                 : anything else

Return ONLY the intent word. No punctuation. No quotes.

User goal: {goal}
Intent:"""

    try:
        response = llm.invoke(prompt)
        intent = response.content.strip().lower()
    except Exception:
        intent = "general"

    valid = {
        "scholarship_search",
        "recommend_scholarships",
        "match_score",
        "eligibility_check",
        "application_plan",
        "track_application",
        "prepare_documents",
        "write_essay",
        "review_essay",
        "planning",
        "general",
    }
    if intent not in valid:
        intent = "general"

    needs_tool = intent in {
        "scholarship_search",
        "recommend_scholarships",
        "match_score",
        "eligibility_check",
        "application_plan",
        "track_application",
        "prepare_documents",
        "write_essay",
        "review_essay",
    }
    tool_name = intent if needs_tool else None

    return {"intent": intent, "needs_tool": needs_tool, "tool_name": tool_name}


# ---------------------------------------------------------------- #
# Node 2a — Scholarship search
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
# Node 2b — Eligibility
# ---------------------------------------------------------------- #
async def eligibility_node(state: AgentState):
    db = get_db()
    if db is None:
        return {"eligibility": {"score": 0, "verdict": "insufficient_info",
                                "reasons": [], "gaps": ["Database not connected."],
                                "next_steps": []}}

    profile = state.get("student_profile")
    if not profile:
        return {"eligibility": {
            "score": 0, "verdict": "insufficient_info", "reasons": [],
            "gaps": ["No student profile provided. Please create one first via POST /students."],
            "next_steps": ["Register your profile with POST /students."]}}

    scholarship = state.get("target_scholarship")
    if not scholarship:
        goal = state["user_goal"]
        try:
            words = [w for w in goal.split() if len(w) > 3]
            if words:
                regex = "|".join(words)
                scholarship = await db.scholarships.find_one(
                    {"name": {"$regex": regex, "$options": "i"}}, {"_id": 0})
        except Exception:
            scholarship = None

    if not scholarship:
        return {"eligibility": {
            "score": 0, "verdict": "insufficient_info", "reasons": [],
            "gaps": ["Could not find the scholarship you mentioned."],
            "next_steps": ["Try searching first, then ask about a specific one."]}}

    profile_clean = {k: v for k, v in profile.items() if k != "_id"}
    result = await evaluate_eligibility(profile_clean, scholarship)
    return {"student_profile": profile_clean,
            "target_scholarship": scholarship,
            "eligibility": result}


# ---------------------------------------------------------------- #
# Node 2c — Application plan
# ---------------------------------------------------------------- #
async def application_plan_node(state: AgentState):
    db = get_db()
    if db is None:
        return {"application_plan": {"weeks_to_deadline": 0, "milestones": [],
                                     "key_deadlines": [], "risks": ["Database not connected."]}}

    profile = state.get("student_profile")
    if not profile:
        return {"application_plan": {"weeks_to_deadline": 0, "milestones": [],
                                     "key_deadlines": [],
                                     "risks": ["No student profile provided. Register via POST /students first."]}}

    scholarship = state.get("target_scholarship")
    if not scholarship:
        goal = state["user_goal"]
        try:
            words = [w for w in goal.split() if len(w) > 3]
            if words:
                regex = "|".join(words)
                scholarship = await db.scholarships.find_one(
                    {"name": {"$regex": regex, "$options": "i"}}, {"_id": 0})
        except Exception:
            scholarship = None

    if not scholarship:
        return {"application_plan": {"weeks_to_deadline": 0, "milestones": [],
                                     "key_deadlines": [],
                                     "risks": ["Could not identify the scholarship you want a plan for."]}}

    eligibility = state.get("eligibility")
    if not eligibility:
        eligibility = await evaluate_eligibility(
            {k: v for k, v in profile.items() if k != "_id"}, scholarship)

    plan = await build_application_plan(
        profile={k: v for k, v in profile.items() if k != "_id"},
        scholarship=scholarship, eligibility=eligibility)

    return {"student_profile": {k: v for k, v in profile.items() if k != "_id"},
            "target_scholarship": scholarship,
            "eligibility": eligibility,
            "application_plan": plan}


# ---------------------------------------------------------------- #
# Node 2d — Tracker
# ---------------------------------------------------------------- #
async def tracker_node(state: AgentState):
    db = get_db()
    if db is None:
        return {"application_record": {"error": "Database not connected."}}

    profile = state.get("student_profile")
    if not profile:
        return {"application_record": {"error": "No student profile — register first via POST /students."}}

    scholarship = state.get("target_scholarship")
    if not scholarship:
        goal = state["user_goal"]
        try:
            words = [w for w in goal.split() if len(w) > 3]
            if words:
                regex = "|".join(words)
                scholarship = await db.scholarships.find_one(
                    {"name": {"$regex": regex, "$options": "i"}}, {"_id": 0})
        except Exception:
            scholarship = None

    if not scholarship:
        return {"application_record": {"error": "Could not identify a scholarship to track."}}

    profile_clean = {k: v for k, v in profile.items() if k != "_id"}
    eligibility = state.get("eligibility")
    application_plan = state.get("application_plan")
    if not eligibility:
        eligibility = await evaluate_eligibility(profile_clean, scholarship)
    if not application_plan:
        application_plan = await build_application_plan(
            profile=profile_clean, scholarship=scholarship, eligibility=eligibility)

    record = await create_application(
        student_email=profile_clean["email"],
        scholarship=scholarship,
        eligibility=eligibility,
        application_plan=application_plan,
        status="interested")

    return {"student_profile": profile_clean,
            "target_scholarship": scholarship,
            "eligibility": eligibility,
            "application_plan": application_plan,
            "application_id": record.get("id"),
            "application_record": record}


# ---------------------------------------------------------------- #
# Node 2e — Checklist
# ---------------------------------------------------------------- #
async def checklist_node(state: AgentState):
    db = get_db()
    if db is None:
        return {"checklist": [], "application_record": {"error": "Database not connected."}}

    profile = state.get("student_profile")
    if not profile:
        return {"checklist": [], "application_record": {
            "error": "No student profile — register first via POST /students."}}

    scholarship = state.get("target_scholarship")
    if not scholarship:
        goal = state["user_goal"]
        try:
            words = [w for w in goal.split() if len(w) > 3]
            if words:
                regex = "|".join(words)
                scholarship = await db.scholarships.find_one(
                    {"name": {"$regex": regex, "$options": "i"}}, {"_id": 0})
        except Exception:
            scholarship = None

    if not scholarship:
        return {"checklist": [], "application_record": {
            "error": "Could not identify a scholarship to build a checklist for."}}

    profile_clean = {k: v for k, v in profile.items() if k != "_id"}
    eligibility = state.get("eligibility")
    if not eligibility:
        eligibility = await evaluate_eligibility(profile_clean, scholarship)

    checklist = await generate_checklist(profile_clean, scholarship, eligibility)

    application_record = state.get("application_record")
    application_id = state.get("application_id")
    if not application_record or application_record.get("error"):
        application_record = await create_application(
            student_email=profile_clean["email"],
            scholarship=scholarship,
            eligibility=eligibility,
            application_plan=state.get("application_plan"),
            status="preparing")
        application_id = application_record.get("id")

    if application_id:
        updated = await set_checklist(application_id, checklist)
        if updated:
            application_record = updated

    return {"student_profile": profile_clean,
            "target_scholarship": scholarship,
            "eligibility": eligibility,
            "application_id": application_id,
            "application_record": application_record,
            "checklist": checklist}


# ---------------------------------------------------------------- #
# Node 2f — Essay draft
# ---------------------------------------------------------------- #
async def essay_draft_node(state: AgentState):
    db = get_db()
    if db is None:
        return {"essay_draft": None, "application_record": {"error": "Database not connected."}}

    profile = state.get("student_profile")
    if not profile:
        return {"essay_draft": None, "application_record": {
            "error": "No student profile — register first via POST /students."}}

    scholarship = state.get("target_scholarship")
    if not scholarship:
        goal = state["user_goal"]
        try:
            words = [w for w in goal.split() if len(w) > 3]
            if words:
                regex = "|".join(words)
                scholarship = await db.scholarships.find_one(
                    {"name": {"$regex": regex, "$options": "i"}}, {"_id": 0})
        except Exception:
            scholarship = None

    if not scholarship:
        return {"essay_draft": None, "application_record": {"error": "Could not identify the scholarship."}}

    profile_clean = {k: v for k, v in profile.items() if k != "_id"}
    eligibility = state.get("eligibility")
    if not eligibility:
        eligibility = await evaluate_eligibility(profile_clean, scholarship)

    draft = await generate_essay_draft(profile_clean, scholarship, eligibility)

    application_record = state.get("application_record")
    application_id = state.get("application_id")
    if not application_record or application_record.get("error"):
        application_record = await create_application(
            student_email=profile_clean["email"],
            scholarship=scholarship,
            eligibility=eligibility,
            application_plan=state.get("application_plan"),
            status="preparing")
        application_id = application_record.get("id")

    if application_id:
        updated = await save_essay_draft(application_id, draft)
        if updated:
            application_record = updated

    return {"student_profile": profile_clean,
            "target_scholarship": scholarship,
            "eligibility": eligibility,
            "application_id": application_id,
            "application_record": application_record,
            "essay_draft": draft}


# ---------------------------------------------------------------- #
# Node 2g — Essay review
# ---------------------------------------------------------------- #
async def essay_review_node(state: AgentState):
    db = get_db()
    if db is None:
        return {"essay_review": None, "application_record": {"error": "Database not connected."}}

    profile = state.get("student_profile")
    scholarship = state.get("target_scholarship")
    draft = state.get("essay_draft")

    application_record = state.get("application_record")
    application_id = state.get("application_id")
    if not draft and application_id:
        record = await get_application_by_id(application_id)
        if record:
            draft = record.get("essay_draft")
            application_record = record

    if not draft:
        return {"essay_review": None, "application_record": {
            "error": "No essay draft found — generate one first."}}

    if not scholarship:
        return {"essay_review": None, "application_record": {
            "error": "Could not identify the scholarship."}}

    review = await review_essay(
        draft=draft,
        scholarship=scholarship,
        profile={k: v for k, v in (profile or {}).items() if k != "_id"})

    if application_id:
        updated = await save_essay_review(application_id, review)
        if updated:
            application_record = updated

    return {"application_id": application_id,
            "application_record": application_record,
            "essay_draft": draft,
            "essay_review": review}


# ---------------------------------------------------------------- #
# Node 2h — Recommendations
# ---------------------------------------------------------------- #
async def recommend_node(state: AgentState):
    db = get_db()
    if db is None:
        return {"recommendations": [], "application_record": {"error": "Database not connected."}}

    profile = state.get("student_profile")
    if not profile:
        return {"recommendations": [], "application_record": {
            "error": "No student profile — register first via POST /students."}}

    profile_clean = {k: v for k, v in profile.items() if k != "_id"}
    recs = await recommend_scholarships(profile_clean, limit=5)
    return {"student_profile": profile_clean, "recommendations": recs}


# ---------------------------------------------------------------- #
# Node 2i — Match score
# ---------------------------------------------------------------- #
async def match_score_node(state: AgentState):
    db = get_db()
    if db is None:
        return {"match_score": None, "application_record": {"error": "Database not connected."}}

    profile = state.get("student_profile")
    if not profile:
        return {"match_score": None, "application_record": {
            "error": "No student profile — register first via POST /students."}}

    scholarship = state.get("target_scholarship")
    if not scholarship:
        goal = state["user_goal"]
        try:
            words = [w for w in goal.split() if len(w) > 3]
            if words:
                regex = "|".join(words)
                scholarship = await db.scholarships.find_one(
                    {"name": {"$regex": regex, "$options": "i"}}, {"_id": 0})
        except Exception:
            scholarship = None

    if not scholarship:
        return {"match_score": None, "application_record": {
            "error": "Could not identify the scholarship."}}

    profile_clean = {k: v for k, v in profile.items() if k != "_id"}
    result = await score_match(profile_clean, scholarship)

    return {"student_profile": profile_clean,
            "target_scholarship": scholarship,
            "match_score": result}


# ---------------------------------------------------------------- #
# Node 3 — Planner
# ---------------------------------------------------------------- #
def planner_node(state: AgentState):
    goal = state["user_goal"]
    tool_result = state.get("tool_result")
    eligibility = state.get("eligibility")
    application_plan = state.get("application_plan")
    application_record = state.get("application_record")
    checklist = state.get("checklist")
    essay_draft = state.get("essay_draft")
    essay_review = state.get("essay_review")
    recommendations = state.get("recommendations")
    match_score = state.get("match_score")
    scholarship = state.get("target_scholarship") or {}

    if recommendations is not None:
        if not recommendations:
            prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

No strong new recommendations were found (all good matches may already be tracked,
or none scored above 40).

Write a warm 3-4 sentence response acknowledging this and suggesting they:
1) refine their profile (country, field, degree level), or
2) check their tracked applications for next steps."""
        else:
            rec_text = "\n".join(
                f"- {r['name']} ({r.get('country', '?')}) — {r.get('match_score')}/100 "
                f"— deadline {r.get('deadline', '?')} — {r.get('reason', '')}"
                for r in recommendations
            )
            prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

We found {len(recommendations)} strong personalized recommendations:

{rec_text}

Write a warm 5-7 sentence response that:
1. Opens with how many scholarships match their profile
2. Highlights the top match with its score and why it fits
3. Briefly names the other 1-2 strong options
4. Offers to run a deeper match check, check eligibility, or build a plan for any of them

Do NOT reprint the entire list - summarize with warmth."""
    elif match_score:
        strengths = "\n".join(f"- {s}" for s in match_score.get("strengths", []))
        improvements = "\n".join(f"- {s}" for s in match_score.get("improvements", []))
        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

Match score for {scholarship.get('name', 'the scholarship')}:
Score: {match_score.get('match_score')}/100 ({match_score.get('verdict')})
Fit summary: {match_score.get('fit_summary')}

Strengths:
{strengths}

Improvements:
{improvements}

Write a warm 4-6 sentence response that:
1. Opens with the score and verdict
2. Highlights the top 2 strengths
3. Names the top 1-2 improvements as concrete actions
4. Offers next steps (eligibility check, plan, or essay draft)"""
    elif essay_review:
        strengths = "\n".join(f"- {s}" for s in essay_review.get("strengths", []))
        weaknesses = "\n".join(f"- {w}" for w in essay_review.get("weaknesses", []))
        suggestions = "\n".join(f"- {s}" for s in essay_review.get("suggestions", []))
        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

We reviewed their essay. Score: {essay_review.get('score')}/100
Alignment: {essay_review.get('alignment')}

Strengths:
{strengths}

Weaknesses:
{weaknesses}

Suggestions:
{suggestions}

Write a warm 4-6 sentence response: open with the score and verdict,
highlight the strongest point, name the top weakness, give the 2 most
impactful suggestions as next actions. Do not rewrite the essay."""
    elif essay_draft:
        excerpt = essay_draft[:400] + ("..." if len(essay_draft) > 400 else "")
        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

We generated a first draft of their SOP for
{scholarship.get('name', 'the scholarship')}. Opening excerpt:

"{excerpt}"

Write a warm 4-5 sentence response: confirm a draft is ready and saved,
mention approximate length ({len(essay_draft.split())} words), suggest
personalizing the details, and offer to run a review."""
    elif checklist:
        required = [c for c in checklist if c.get("required")]
        optional = [c for c in checklist if not c.get("required")]
        req_text = "\n".join(
            f"- [ ] {c['name']} ({c['category']}) — {c.get('description', '')}"
            for c in required)
        opt_text = "\n".join(
            f"- [ ] {c['name']} ({c['category']})" for c in optional)
        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

We generated a document checklist for:
{scholarship.get('name', 'this scholarship')} (deadline {scholarship.get('deadline', 'unknown')}).

Required:
{req_text}

Optional:
{opt_text if opt_text else "(none)"}

Write a warm 4-6 sentence response opening with the number of required docs,
suggesting 3 to start with, reminding them they can tick items in the
dashboard, and ending with one clear action."""
    elif application_record and not application_record.get("error"):
        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

We just saved their application. Stored:
Scholarship: {application_record.get('scholarship_name')}
Status: {application_record.get('status')}
ID: {application_record.get('id')}

Write a friendly 3-5 sentence confirmation: it's tracked, current status,
next actions (checklist, essay), and the deadline {scholarship.get('deadline', 'for this scholarship')}."""
    elif application_record and application_record.get("error"):
        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

We could not save their application: {application_record.get('error')}

Write a short 2-3 sentence response explaining and asking them to register
via POST /students and name the scholarship."""
    elif application_plan and application_plan.get("milestones"):
        milestones_text = "\n".join(
            f"Week {m.get('week')}: {m.get('title')} ({m.get('focus', 'misc')})\n"
            + "\n".join(f"  - {t}" for t in m.get("tasks", []))
            for m in application_plan["milestones"])
        key_dates_text = "\n".join(
            f"- {d.get('date')}: {d.get('label')}"
            for d in application_plan.get("key_deadlines", []))
        risks_text = "\n".join(f"- {r}" for r in application_plan.get("risks", []))
        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

Scholarship: {scholarship.get('name', 'Unknown')} (deadline {scholarship.get('deadline', 'unknown')})
Weeks to deadline: {application_plan.get('weeks_to_deadline')}

Plan:
{milestones_text}

Key dates:
{key_dates_text}

Risks:
{risks_text}

Write a warm 5-7 sentence response opening with weeks remaining, summarizing
the first 2 milestones, warning about 1-2 risks, ending with one start-today action."""
    elif eligibility:
        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

Eligibility: {eligibility.get('score')}/100 ({eligibility.get('verdict')}).
Reasons: {'; '.join(eligibility.get('reasons', []))}
Gaps: {'; '.join(eligibility.get('gaps', []))}

Write a friendly 4-6 sentence response with verdict, top reasons, top gaps, and 1-2 next steps."""
    elif tool_result:
        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

Scholarships found:
{tool_result}

Write a short friendly response summarizing them with deadlines and offering
recommendations, eligibility checks, a plan, or a draft essay."""
    else:
        prompt = f"""You are ScholarMate AI.

The student asked: "{goal}"

Write a helpful 3-5 sentence response with 2-3 concrete next steps. Mention
search, personalized recommendations, match scoring, eligibility, planning,
documents, essay drafting, and tracking."""

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
    if tool == "recommend_scholarships":
        return "recommend"
    if tool == "match_score":
        return "match_score"
    if tool == "eligibility_check":
        return "eligibility"
    if tool == "application_plan":
        return "application_plan"
    if tool == "track_application":
        return "tracker"
    if tool == "prepare_documents":
        return "checklist"
    if tool == "write_essay":
        return "essay_draft"
    if tool == "review_essay":
        return "essay_review"
    return "planner"