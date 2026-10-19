"""
Feedback memory store.
Persists human corrections (the diff between the AI draft and the final approved
reply) so the drafting agent can learn from them.

Each record includes:
  - category, original_draft, final_reply, diff_summary
  - timestamp, email_id

The store supports retrieval by category so the agent can include relevant past
corrections in its prompt (few-shot learning from human feedback).
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from task2_inbox_agent.config import FEEDBACK_DIR

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    feedback_id: str
    email_id: str
    category: str
    original_draft: str
    final_reply: str
    diff_summary: str
    human_notes: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


class MemoryStore:
    """
    JSON-file-per-category feedback memory.
    Thread-safety is not required in this prototype (single uvicorn worker).
    """

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self.storage_dir = storage_dir or FEEDBACK_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _category_file(self, category: str) -> Path:
        safe = category.replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe}.json"

    def _load_category(self, category: str) -> List[dict]:
        p = self._category_file(category)
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt feedback file %s — starting fresh", p)
            return []

    def _save_category(self, category: str, records: List[dict]) -> None:
        p = self._category_file(category)
        p.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_feedback(self, record: FeedbackRecord) -> None:
        records = self._load_category(record.category)
        records.append(record.to_dict())
        self._save_category(record.category, records)
        logger.info("Saved feedback %s for category %s", record.feedback_id, record.category)

    def get_feedback_for_category(self, category: str, limit: int = 5) -> List[FeedbackRecord]:
        """Return the most recent *limit* feedback records for the category."""
        records = self._load_category(category)
        results: List[FeedbackRecord] = []
        for raw in records[-limit:]:
            try:
                results.append(FeedbackRecord(**raw))
            except TypeError:
                continue
        return results

    def get_all_feedback(self) -> List[FeedbackRecord]:
        """Return every stored feedback record across all categories."""
        all_records: List[FeedbackRecord] = []
        for fp in sorted(self.storage_dir.glob("*.json")):
            try:
                raw_list = json.loads(fp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for raw in raw_list:
                try:
                    all_records.append(FeedbackRecord(**raw))
                except TypeError:
                    continue
        return all_records

    def delete_feedback(self, feedback_id: str) -> bool:
        """Remove a single feedback record. Returns True if found and deleted."""
        for fp in self.storage_dir.glob("*.json"):
            try:
                raw_list = json.loads(fp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            before = len(raw_list)
            raw_list = [r for r in raw_list if r.get("feedback_id") != feedback_id]
            if len(raw_list) < before:
                fp.write_text(json.dumps(raw_list, indent=2, ensure_ascii=False), encoding="utf-8")
                return True
        return False

    def clear_category(self, category: str) -> int:
        """Delete all feedback for a category. Returns count removed."""
        records = self._load_category(category)
        count = len(records)
        if count:
            self._save_category(category, [])
        return count
