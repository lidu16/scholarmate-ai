from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import (
    classifier_node,
    tool_node,
    eligibility_node,
    application_plan_node,
    tracker_node,
    checklist_node,
    essay_draft_node,
    essay_review_node,
    recommend_node,
    match_score_node,
    planner_node,
    response_node,
    route_after_classify,
)

builder = StateGraph(AgentState)

builder.add_node("classifier", classifier_node)
builder.add_node("tool", tool_node)
builder.add_node("eligibility", eligibility_node)
builder.add_node("application_plan", application_plan_node)
builder.add_node("tracker", tracker_node)
builder.add_node("checklist", checklist_node)
builder.add_node("essay_draft", essay_draft_node)
builder.add_node("essay_review", essay_review_node)
builder.add_node("recommend", recommend_node)
builder.add_node("match_score", match_score_node)
builder.add_node("planner", planner_node)
builder.add_node("response", response_node)

builder.add_edge(START, "classifier")

builder.add_conditional_edges(
    "classifier",
    route_after_classify,
    {
        "tool": "tool",
        "recommend": "recommend",
        "match_score": "match_score",
        "eligibility": "eligibility",
        "application_plan": "application_plan",
        "tracker": "tracker",
        "checklist": "checklist",
        "essay_draft": "essay_draft",
        "essay_review": "essay_review",
        "planner": "planner",
    },
)

for node in [
    "tool", "recommend", "match_score", "eligibility",
    "application_plan", "tracker", "checklist",
    "essay_draft", "essay_review",
]:
    builder.add_edge(node, "planner")

builder.add_edge("planner", "response")
builder.add_edge("response", END)

graph = builder.compile()