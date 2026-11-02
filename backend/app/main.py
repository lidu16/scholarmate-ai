from fastapi import FastAPI
from pydantic import BaseModel

from app.agents.graph import graph

app = FastAPI(title="ScholarMate AI")


class AgentRequest(BaseModel):
    user_goal: str


@app.get("/")
def root():
    return {"message": "ScholarMate AI Backend Running"}


@app.get("/agent")
def run_agent_default():
    return graph.invoke({
        "user_goal": "Find AI scholarships in Germany",
        "intent": "",
        "needs_tool": False,
        "tool_name": None,
        "tool_result": None,
        "plan": "",
        "final_response": "",
    })


@app.post("/agent")
def run_agent(req: AgentRequest):
    return graph.invoke({
        "user_goal": req.user_goal,
        "intent": "",
        "needs_tool": False,
        "tool_name": None,
        "tool_result": None,
        "plan": "",
        "final_response": "",
    })