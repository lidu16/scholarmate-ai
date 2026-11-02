from fastapi import FastAPI

from app.agents.graph import graph

app = FastAPI(
    title="ScholarMate AI"
)


@app.get("/")
def root():
    return {
        "message": "ScholarMate AI Backend Running"
    }


@app.get("/agent")
def run_agent():

    result = graph.invoke(
        {
            "user_goal": "Find AI scholarships",
            "plan": "",
            "final_response": ""
        }
    )

    return result