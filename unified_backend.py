"""Single FastAPI entry point for the three AI Tech Consultant tasks.

The task packages remain separate.  This module only composes their existing
routers and starts the Task 1 scheduler once, which makes it suitable for a
single Render web service.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

# Task 2 and Task 3 deliberately call Task 1 over HTTP.  In a unified process
# that HTTP endpoint is this same service.  Use UNIFIED_TASK1_API_BASE only if
# a proxy or custom internal hostname is required; a stale TASK1_API_BASE from
# the old three-process setup must not silently send requests to port 8000.
_port = os.environ.get("PORT", "8000")
os.environ["TASK1_API_BASE"] = os.environ.get(
    "UNIFIED_TASK1_API_BASE", f"http://127.0.0.1:{_port}"
).rstrip("/")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from task1_knowledge_brain.api import routes as task1_routes
from task1_knowledge_brain.api.routes import router as task1_router
from task1_knowledge_brain.syncer.scheduler import DailySyncScheduler
from task2_inbox_agent.api.routes import router as task2_router, health as task2_health
from task3_meeting_scope.api.routes import router as task3_router, health as task3_health

logger = logging.getLogger("ai_tech_consultant.unified")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run one daily sync scheduler for the composed application."""
    scheduler = DailySyncScheduler(syncer=task1_routes.daily_syncer)
    app.state.daily_sync_scheduler = scheduler
    scheduler.start()
    logger.info("Unified backend started; Task 1 daily scheduler is running")
    try:
        yield
    finally:
        scheduler.stop()
        logger.info("Unified backend stopped")


app = FastAPI(
    title="AI Tech Consultant API",
    description="Unified API for Knowledge Brain, Inbox Agent, and Meeting to Proposal.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Task 1 is included first so its existing endpoint contracts remain the
# canonical ones.  The task routers have distinct paths apart from /health.
app.include_router(task1_router)
app.include_router(task2_router)
app.include_router(task3_router)


@app.get("/api/v1/unified-health", tags=["Unified"])
def unified_health() -> dict[str, Any]:
    """Return one status document for deployment probes and dashboards."""
    try:
        task1 = task1_routes.health_check().model_dump()
    except Exception as exc:
        logger.exception("Task 1 health check failed")
        task1 = {"status": "degraded", "error": type(exc).__name__}
    try:
        task2 = task2_health()
    except Exception as exc:
        logger.exception("Task 2 health check failed")
        task2 = {"status": "degraded", "error": type(exc).__name__}
    try:
        task3 = task3_health()
    except Exception as exc:
        logger.exception("Task 3 health check failed")
        task3 = {"status": "degraded", "error": type(exc).__name__}
    healthy = all(item.get("status") == "ok" for item in (task1, task2, task3))
    return {"status": "ok" if healthy else "degraded", "task1": task1, "task2": task2, "task3": task3}


@app.get("/", tags=["Unified"])
def root() -> dict[str, Any]:
    return {
        "service": "AI Tech Consultant unified backend",
        "docs": "/docs",
        "health": "/api/v1/unified-health",
        "tasks": {"knowledge_brain": "/api/v1/query", "inbox_agent": "/api/v1/emails", "meeting_to_proposal": "/api/v1/projects/{project_id}"},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("unified_backend:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
