"""
FastAPI Application Entrypoint for Task 1: The Knowledge Brain.
Offers a clean REST API for Task 2 (Inbox Agent), Task 3 (Scoping Agent), and evaluation tests.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from task1_knowledge_brain.api import routes as api_routes
from task1_knowledge_brain.api.routes import router as api_router
from task1_knowledge_brain.syncer.scheduler import DailySyncScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("knowledge_brain")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Knowledge Brain API Service...")
    scheduler = DailySyncScheduler(syncer=api_routes.daily_syncer)
    app.state.daily_sync_scheduler = scheduler
    scheduler.start()
    yield
    scheduler.stop()
    logger.info("Shutting down Knowledge Brain API Service...")


app = FastAPI(
    title="AI Tech Consultant - Knowledge Brain API",
    description=(
        "Public API for Task 1 (The Knowledge Brain). "
        "Provides grounded RAG queries, source citations, screenshot references, "
        "human-in-the-loop expert ticketing, and daily sitemap sync."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local dashboards or frontend agents
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_router)


@app.get("/", summary="Root Health Status")
def root():
    return {
        "service": "AI Tech Consultant - Task 1: Knowledge Brain",
        "docs": "/docs",
        "endpoints": {
            "query": "POST /api/v1/query",
            "settings_check": "POST /api/v1/settings/check",
            "tickets": "GET /api/v1/tickets",
            "resolve_ticket": "POST /api/v1/tickets/{id}/resolve",
            "sync": "POST /api/v1/sync",
            "health": "GET /api/v1/health",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("task1_knowledge_brain.main:app", host="0.0.0.0", port=8000, reload=True)
