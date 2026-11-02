from .state import AgentState
from ..tools.scholarship_search import search_scholarships
from ..config.llm import get_llm

# Single shared LLM instance
llm = get_llm(temperature=0.2)


# ---------- Node 1: Gemini intent classifier ----------
def classifier_node(state: AgentState):
    goal = state["user_goal"]

    prompt = f"""You are the intent classifier for ScholarMate AI, a scholarship assistant.

Classify the user's goal into EXACTLY ONE of these intents:
- "scholarship_search" : user wants to find, search, or discover scholarships
- "planning"           : user wants a plan, roadmap, timeline, or preparation steps
- "general"            : anything else (greeting, unclear, or unrelated)

Return ONLY the intent word. No explanation. No punctuation. No quotes.

User goal: {goal}
Intent:"""

    response = llm.invoke(prompt)
    intent = response.content.strip().lower()

    # Sanitize — if Gemini returns something unexpected, fall back to general
    if intent not in {"scholarship_search", "planning", "general"}:
        intent = "general"

    needs_tool = intent == "scholarship_search"
    tool_name = "scholarship_search" if needs_tool else None

    return {
        "intent": intent,
        "needs_tool": needs_tool,
        "tool_name": tool_name,
    }


# ---------- Node 2: Run tool ----------
def tool_node(state: AgentState):
    if state["tool_name"] == "scholarship_search":
        result = search_scholarships(state["user_goal"])
    else:
        result = "No tool executed."
    return {"tool_result": result}


# ---------- Node 3: Gemini planner ----------
def planner_node(state: AgentState):
    goal = state["user_goal"]
    intent = state.get("intent", "general")
    tool_result = state.get("tool_result")

    if tool_result:
        prompt = f"""You are ScholarMate AI. The student asked: "{goal}"

Here are the scholarships found:
{tool_result}

Write a short, friendly response (3-5 sentences) that:
1. Summarizes the scholarships found (name + deadline)
2. Suggests the student pick one to focus on
3. Offers to help with the application plan next

Do NOT invent scholarships beyond the list above."""
    else:
        prompt = f"""You are ScholarMate AI, a scholarship application assistant.

The student asked: "{goal}"

Write a helpful, friendly response (3-5 sentences) that:
1. Acknowledges their goal
2. Gives 2-3 concrete next steps they can take
3. Mentions you can help with scholarship search if they want

Keep it short and encouraging. Do not invent specific scholarship names or deadlines."""

    response = llm.invoke(prompt)
    return {"plan": response.content.strip()}


# ---------- Node 4: Final response assembly ----------
def response_node(state: AgentState):
    return {"final_response": state.get("plan", "No response generated.")}


# ---------- Router (unchanged — still decides tool vs planner) ----------
def route_after_classify(state: AgentState):
    if state.get("needs_tool"):
        return "tool"
    return "planner"