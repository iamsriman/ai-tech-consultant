from fastapi import FastAPI
from task2_inbox_agent.api.routes import router

app = FastAPI(title="AI Tech Consultant - Task 2 Inbox Agent", version="1.0.0")
app.include_router(router)

@app.get("/")
def root(): return {"service":"task2_inbox_agent", "docs":"/docs", "health":"/api/v1/health"}
