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
builder.add_node("planner", planner_node)
builder.add_node("response", response_node)

builder.add_edge(START, "classifier")

builder.add_conditional_edges(
    "classifier",
    route_after_classify,
    {
        "tool": "tool",
        "eligibility": "eligibility",
        "application_plan": "application_plan",
        "tracker": "tracker",
        "checklist": "checklist",
        "essay_draft": "essay_draft",
        "essay_review": "essay_review",
        "planner": "planner",
    },
)

builder.add_edge("tool", "planner")
builder.add_edge("eligibility", "planner")
builder.add_edge("application_plan", "planner")
builder.add_edge("tracker", "planner")
builder.add_edge("checklist", "planner")
builder.add_edge("essay_draft", "planner")
builder.add_edge("essay_review", "planner")
builder.add_edge("planner", "response")
builder.add_edge("response", END)

graph = builder.compile()