import os
from pathlib import Path

from fastapi.testclient import TestClient
from types import SimpleNamespace

from unified_backend import app
from task2_inbox_agent.agent.knowledge_client import KnowledgeClient as InboxKnowledgeClient
from task3_meeting_scope.knowledge_client import KnowledgeClient as ScopeKnowledgeClient
from frontend.api_client import ApiClient, clients
from frontend import app as frontend_app
import yaml


def test_unified_app_exposes_all_task_routes():
    paths = {route.path for route in app.routes}
    assert "/api/v1/query" in paths
    assert "/api/v1/emails" in paths
    assert "/api/v1/projects/{project_id}" in paths
    assert "/api/v1/unified-health" in paths


def test_task_clients_use_the_unified_port():
    expected = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/api/v1/query"
    assert InboxKnowledgeClient().endpoint == expected
    assert ScopeKnowledgeClient().endpoint == expected


def test_unified_health_reports_task_statuses():
    with TestClient(app) as client:
        response = client.get("/api/v1/unified-health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["task1"]["chroma_db_healthy"] is True
    assert body["task2"]["status"] == "ok"
    assert body["task3"]["status"] == "ok"


def test_existing_task_two_and_task_three_routes_share_the_app():
    with TestClient(app) as client:
        emails = client.get("/api/v1/emails")
        project = client.get("/api/v1/projects/integration-project")
    assert emails.status_code == 200
    assert project.status_code == 200


def test_frontend_defaults_to_one_unified_backend(monkeypatch):
    for name in ("BACKEND_API_URL", "TASK1_API_URL", "TASK2_API_URL", "TASK3_API_URL"):
        monkeypatch.delenv(name, raising=False)
    result = clients()
    assert {client.base_url for client in result.values()} == {"http://127.0.0.1:8000"}


def test_frontend_backend_url_overrides_old_per_task_urls(monkeypatch):
    monkeypatch.setenv("BACKEND_API_URL", "https://consultant.example")
    monkeypatch.setenv("TASK1_API_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("TASK2_API_URL", "http://127.0.0.1:8001")
    monkeypatch.setenv("TASK3_API_URL", "http://127.0.0.1:8002")
    assert {client.base_url for client in clients().values()} == {"https://consultant.example"}


def test_frontend_ignores_stale_per_task_urls_in_unified_mode(monkeypatch):
    monkeypatch.delenv("BACKEND_API_URL", raising=False)
    monkeypatch.setenv("BACKEND_API_MODE", "unified")
    monkeypatch.setenv("TASK1_API_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("TASK2_API_URL", "http://127.0.0.1:8001")
    monkeypatch.setenv("TASK3_API_URL", "http://127.0.0.1:8002")
    assert {client.base_url for client in clients().values()} == {"http://127.0.0.1:8000"}


def test_frontend_health_reads_each_task_from_aggregate(monkeypatch):
    payload = {
        "status": "ok",
        "task1": {"status": "ok", "indexed_chunks_count": 1812},
        "task2": {"status": "ok", "provider": "groq"},
        "task3": {"status": "ok", "projects": 4},
    }
    monkeypatch.setattr(
        "frontend.api_client.httpx.get",
        lambda url, timeout: SimpleNamespace(status_code=200, json=lambda: payload),
    )
    assert ApiClient("http://127.0.0.1:8999").health("task2") == payload["task2"]


def test_frontend_fetches_one_aggregate_for_one_backend(monkeypatch):
    aggregate = {
        "status": "ok",
        "task1": {"status": "ok", "indexed_chunks_count": 1812},
        "task2": {"status": "ok", "provider": "groq"},
        "task3": {"status": "ok", "projects": 4},
    }
    calls = []

    class FakeClient:
        def __init__(self, url):
            self.url = url

        def health(self):
            calls.append(self.url)
            return aggregate

    monkeypatch.setattr(frontend_app, "ApiClient", FakeClient)
    frontend_app.fetch_health.clear()
    result = frontend_app.fetch_health(tuple((task, "http://one-backend") for task in ("task1", "task2", "task3")))
    assert calls == ["http://one-backend"]
    assert all(result[task]["ok"] for task in ("task1", "task2", "task3"))
    assert result["task2"]["data"]["provider"] == "groq"


def test_render_deploys_one_api_and_points_streamlit_to_it():
    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((root / "render.yaml").read_text(encoding="utf-8"))
    services = {service["name"]: service for service in manifest["services"]}

    backend = services["ai-tech-consultant-api"]
    frontend = services["ai-tech-consultant-ui"]
    assert backend["startCommand"] == "uvicorn unified_backend:app --host 0.0.0.0 --port $PORT"
    assert backend["healthCheckPath"] == "/api/v1/unified-health"
    assert frontend["startCommand"].endswith("--server.port $PORT --server.headless true")
    assert "frontend/requirements.txt" in frontend["buildCommand"]
    backend_url = next(item for item in frontend["envVars"] if item["key"] == "BACKEND_API_URL")
    assert backend_url["fromService"] == {
        "type": "web",
        "name": "ai-tech-consultant-api",
        "envVarKey": "RENDER_EXTERNAL_URL",
    }
