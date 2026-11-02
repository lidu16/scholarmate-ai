from typing import TypedDict


class AgentState(TypedDict):
    user_goal: str
    plan: str
    final_response: str