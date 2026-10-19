"""Durable per-category unchanged-send streaks with quality pause controls."""
import json
from pathlib import Path
from typing import Dict
from task2_inbox_agent.config import STREAK_FILE, AUTO_SEND_STREAK_THRESHOLD

class StreakTracker:
    def __init__(self, path: Path = STREAK_FILE, threshold: int = AUTO_SEND_STREAK_THRESHOLD):
        self.path, self.threshold = Path(path), threshold
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()
    def _load(self):
        if not self.path.exists(): return {"categories": {}, "disabled": {}}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            value.setdefault("categories", {}); value.setdefault("disabled", {})
            return value
        except (OSError, json.JSONDecodeError): return {"categories": {}, "disabled": {}}
    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.path)
    def record(self, category: str, unchanged: bool, success: bool = True):
        if not success or not unchanged: self.state["categories"][category] = 0
        else: self.state["categories"][category] = int(self.state["categories"].get(category, 0)) + 1
        self._save(); return self.count(category)
    def count(self, category): return int(self.state["categories"].get(category, 0))
    def eligible(self, category): return self.count(category) >= self.threshold and not self.state["disabled"].get(category, False)
    def set_disabled(self, category, disabled=True): self.state["disabled"][category] = disabled; self._save()
    def snapshot(self): return self.state.copy()
