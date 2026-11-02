from fastapi import FastAPI

app = FastAPI(
    title="ScholarMate AI"
)


@app.get("/")
def root():
    return {
        "message": "ScholarMate AI Backend Running"
    }