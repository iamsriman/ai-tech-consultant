"""
Human-in-the-loop fallback subpackage for Task 1: Knowledge Brain.
Contains TicketManager and FeedbackIngester.
"""

from .ticket_manager import TicketManager, ExpertTicket
from .feedback_ingest import FeedbackIngester

__all__ = [
    "TicketManager",
    "ExpertTicket",
    "FeedbackIngester",
]
