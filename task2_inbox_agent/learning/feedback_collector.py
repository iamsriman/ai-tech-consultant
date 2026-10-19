"""
Feedback collector.
Computes the diff between the AI draft and the human-edited final reply,
builds a ``FeedbackRecord``, and stores it via ``MemoryStore``.
"""

import difflib
import logging
import uuid
from typing import Optional

from task2_inbox_agent.learning.memory_store import FeedbackRecord, MemoryStore

logger = logging.getLogger(__name__)


def _compute_diff(original: str, final: str) -> str:
    """
    Return a human-readable unified diff summarising what the reviewer changed.
    If there is no difference, returns an empty string.
    """
    orig_lines = original.splitlines(keepends=True)
    final_lines = final.splitlines(keepends=True)
    diff = difflib.unified_diff(orig_lines, final_lines, fromfile="ai_draft", tofile="human_final", lineterm="")
    return "".join(diff)


class FeedbackCollector:
    """
    Receives the original AI draft and the final (possibly edited) reply,
    computes a diff, stores the feedback record, and returns the record id.
    """

    def __init__(self, memory: Optional[MemoryStore] = None) -> None:
        self.memory = memory or MemoryStore()

    def collect(
        self,
        email_id: str,
        category: str,
        original_draft: str,
        final_reply: str,
        human_notes: str = "",
    ) -> FeedbackRecord:
        diff = _compute_diff(original_draft, final_reply)
        record = FeedbackRecord(
            feedback_id=f"fb_{uuid.uuid4().hex[:12]}",
            email_id=email_id,
            category=category,
            original_draft=original_draft,
            final_reply=final_reply,
            diff_summary=diff,
            human_notes=human_notes,
        )
        self.memory.save_feedback(record)
        return record

    def was_draft_changed(self, original_draft: str, final_reply: str) -> bool:
        """Return True if the human edited the draft (ignoring whitespace)."""
        return original_draft.strip() != final_reply.strip()
