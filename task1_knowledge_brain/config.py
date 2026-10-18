"""Task 1 configuration helpers. Never log values loaded from the environment."""
import os
from pathlib import Path


def load_project_environment() -> None:
    """Load the repository .env without overriding explicitly configured env vars."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    # Parse the small project configuration directly so an explicitly empty
    # environment value remains an intentional opt-out (python-dotenv treats
    # an existing empty value as absent on some versions).
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


def anthropic_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip()
