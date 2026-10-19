"""
Task 2 configuration.
Loads environment variables and provides paths used throughout the inbox-agent.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TASK2_ROOT = Path(__file__).resolve().parent
DATA_DIR = _TASK2_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_DIR = DATA_DIR / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

STREAK_FILE = DATA_DIR / "streak_state.json"
OUTBOX_DIR = DATA_DIR / "outbox"
OUTBOX_DIR.mkdir(parents=True, exist_ok=True)


def _load_env() -> None:
    """Load the repository-root .env without overriding existing env vars."""
    env_path = _TASK2_ROOT.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        name, value = entry.split("=", 1)
        name = name.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value


_load_env()

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "").strip()
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "").strip()
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip()
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "mixtral-8x7b-32768").strip()

# ---------------------------------------------------------------------------
# Task 1 Knowledge Brain API
# ---------------------------------------------------------------------------

TASK1_API_BASE: str = os.environ.get("TASK1_API_BASE", "http://127.0.0.1:8000").strip()
TASK1_QUERY_ENDPOINT: str = f"{TASK1_API_BASE}/api/v1/query"

# ---------------------------------------------------------------------------
# Autonomy thresholds
# ---------------------------------------------------------------------------

AUTO_SEND_STREAK_THRESHOLD: int = int(os.environ.get("AUTO_SEND_STREAK_THRESHOLD", "20"))
SENSITIVE_CATEGORIES: set = {"refund_request", "legal", "complaint", "escalation"}
