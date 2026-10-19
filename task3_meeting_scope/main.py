from fastapi import FastAPI
from task3_meeting_scope.api.routes import router
app=FastAPI(title="AI Tech Consultant - Task 3",version="1.0.0")
app.include_router(router)
@app.get("/")
def root(): return {"service":"task3_meeting_scope","docs":"/docs","health":"/api/v1/health"}
