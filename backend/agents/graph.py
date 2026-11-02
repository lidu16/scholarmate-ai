from langgraph.graph import StateGraph
from langgraph.graph import START, END

from .state import AgentState
from .nodes import planner_node
from .nodes import response_node

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("response", response_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "response")
builder.add_edge("response", END)

graph = builder.compile()