"""A local, idempotent sender used until a real mail provider is authorized."""
import json
from pathlib import Path
from task2_inbox_agent.config import OUTBOX_DIR

class SimulatedSender:
    def __init__(self, outbox_dir: Path = OUTBOX_DIR):
        self.outbox_dir = Path(outbox_dir); self.outbox_dir.mkdir(parents=True, exist_ok=True)
    def send(self, email_id: str, recipient: str, subject: str, body: str) -> dict:
        path = self.outbox_dir / f"{email_id}.json"
        if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
        record = {"email_id": email_id, "to": recipient, "subject": subject, "body": body, "simulated": True}
        tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(record, indent=2), encoding="utf-8"); tmp.replace(path)
        return record
