from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    user_goal: str
    intent: str
    needs_tool: bool
    tool_name: Optional[str]
    tool_result: Optional[str]
    student_profile: Optional[dict[str, Any]]
    target_scholarship: Optional[dict[str, Any]]
    eligibility: Optional[dict[str, Any]]
    application_plan: Optional[dict[str, Any]]
    application_id: Optional[str]
    application_record: Optional[dict[str, Any]]
    plan: str
    final_response: str