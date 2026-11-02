from typing import TypedDict, Optional, List, Literal


class AgentState(TypedDict):
    user_goal: str                      # what the student wants
    intent: str                         # classified intent
    needs_tool: bool                    # whether a tool must run
    tool_name: Optional[str]            # which tool to call
    tool_result: Optional[str]          # what the tool returned
    plan: str                           # generated plan
    final_response: str                 # final answer to user