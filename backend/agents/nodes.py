from .state import AgentState
from ..tools.scholarship_search import search_scholarships


# ---------- Node 1: Classify intent ----------
def classifier_node(state: AgentState):
    goal = state["user_goal"].lower()

    if any(kw in goal for kw in ["scholarship", "find", "search", "funding"]):
        intent = "scholarship_search"
        needs_tool = True
        tool_name = "scholarship_search"
    elif any(kw in goal for kw in ["plan", "roadmap", "prepare"]):
        intent = "planning"
        needs_tool = False
        tool_name = None
    else:
        intent = "general"
        needs_tool = False
        tool_name = None

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


# ---------- Node 3: Planner ----------
def planner_node(state: AgentState):
    goal = state["user_goal"]
    intent = state.get("intent", "general")
    return {
        "plan": f"[{intent}] Build a plan for: {goal}"
    }


# ---------- Node 4: Response ----------
def response_node(state: AgentState):
    parts = [f"Goal: {state['user_goal']}"]

    if state.get("tool_result"):
        parts.append(f"Tool result:\n{state['tool_result']}")

    if state.get("plan"):
        parts.append(f"Plan:\n{state['plan']}")

    return {"final_response": "\n\n".join(parts)}


# ---------- Router (decides next node) ----------
def route_after_classify(state: AgentState):
    if state.get("needs_tool"):
        return "tool"
    return "planner"