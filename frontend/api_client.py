"""Small HTTP client for the unified backend API.

The Streamlit app deliberately contains no retrieval, drafting, or proposal
logic. It only validates user input, calls the public APIs, and presents their
responses. In unified mode, every task shares one base URL and port.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx


class BackendError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 35.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = httpx.request(
                method,
                self.base_url + path,
                timeout=self.timeout,
                **kwargs,
            )
        except httpx.TimeoutException:
            raise BackendError(f"Request timed out ({self.base_url})")
        except httpx.RequestError as exc:
            raise BackendError(f"Service unavailable ({self.base_url}): {exc.__class__.__name__}")
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text[:500]}
        if response.status_code >= 400:
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
            raise BackendError(f"API error {response.status_code}: {detail}", response.status_code)
        return payload

    def health(self, task: Optional[str] = None) -> Any:
        # Health indicators should fail quickly so one offline service does not
        # make the whole dashboard appear frozen. Workflow calls retain the
        # longer client timeout above.
        try:
            response = httpx.get(
                self.base_url + "/api/v1/unified-health",
                timeout=min(self.timeout, 3.0),
            )
        except httpx.TimeoutException:
            raise BackendError(f"Health check timed out ({self.base_url})")
        except httpx.RequestError as exc:
            raise BackendError(f"Service unavailable ({self.base_url}): {exc.__class__.__name__}")

        # Keep compatibility with deployments that still run one task per
        # process. A 404 means this server predates the aggregate health route.
        if response.status_code == 404:
            try:
                response = httpx.get(
                    self.base_url + "/api/v1/health",
                    timeout=min(self.timeout, 3.0),
                )
            except httpx.TimeoutException:
                raise BackendError(f"Health check timed out ({self.base_url})")
            except httpx.RequestError as exc:
                raise BackendError(f"Service unavailable ({self.base_url}): {exc.__class__.__name__}")
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text[:500]}
        if response.status_code >= 400:
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
            raise BackendError(f"Health check error {response.status_code}: {detail}", response.status_code)
        if task and isinstance(payload, dict) and isinstance(payload.get(task), dict):
            return payload[task]
        return payload


def clients() -> dict[str, ApiClient]:
    unified = os.getenv("BACKEND_API_URL", "").strip()
    if unified:
        return {task: ApiClient(unified) for task in ("task1", "task2", "task3")}
    # Unified mode is the default. Old per-task variables are ignored unless
    # separate services are explicitly requested, preventing stale 8001/8002
    # settings from making a unified deployment look offline.
    if os.getenv("BACKEND_API_MODE", "unified").strip().lower() == "separate":
        return {
            "task1": ApiClient(os.getenv("TASK1_API_URL", "http://127.0.0.1:8000")),
            "task2": ApiClient(os.getenv("TASK2_API_URL", "http://127.0.0.1:8001")),
            "task3": ApiClient(os.getenv("TASK3_API_URL", "http://127.0.0.1:8002")),
        }
    local_backend = os.getenv("LOCAL_BACKEND_API_URL", "http://127.0.0.1:8000").strip()
    return {task: ApiClient(local_backend) for task in ("task1", "task2", "task3")}
