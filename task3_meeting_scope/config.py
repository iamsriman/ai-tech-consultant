import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORE_FILE = DATA_DIR / "projects.json"
TASK1_API_BASE = os.getenv("TASK1_API_BASE", "http://127.0.0.1:8000").strip()
LLM_PROVIDER = os.getenv("TASK3_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "claude")).strip().lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768").strip()

def load_env():
    path = ROOT.parent / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1); key = key.strip(); value = value.strip().strip('"\'')
                if key and key not in os.environ: os.environ[key] = value
load_env()
