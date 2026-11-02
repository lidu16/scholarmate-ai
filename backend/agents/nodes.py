from .state import AgentState


def planner_node(state: AgentState):
    goal = state["user_goal"]

    return {
        "plan": f"Create a plan for: {goal}"
    }


def response_node(state: AgentState):
    plan = state["plan"]

    return {
        "final_response": f"Plan generated successfully. {plan}"
    }