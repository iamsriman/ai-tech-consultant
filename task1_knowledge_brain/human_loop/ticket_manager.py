"""
Ticket Manager for Human-in-the-Loop Fallback.
Logs unanswered user questions when the Knowledge Brain has insufficient evidence.
Allows human experts to review, answer, and track resolutions.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

DEFAULT_TICKETS_FILE = Path("task1_knowledge_brain/data/tickets.json")


@dataclass
class ExpertTicket:
    ticket_id: str
    query: str
    reason: str
    status: str  # "pending", "resolved", "rejected"
    created_at: str
    context_attempted: List[str]
    resolved_at: Optional[str] = None
    expert_answer: Optional[str] = None
    expert_name: Optional[str] = None
    category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TicketManager:
    """
    Manages human expert tickets for unanswered queries.
    Persists tickets to disk in JSON format.
    """

    def __init__(self, filepath: Path = DEFAULT_TICKETS_FILE):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.tickets: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Loads tickets from disk."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.tickets = json.load(f)
            except Exception as e:
                logger.error(f"Error loading tickets from {self.filepath}: {e}")
                self.tickets = {}

    def save(self) -> None:
        """Atomically saves tickets to disk."""
        temp_path = self.filepath.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.tickets, f, indent=2, ensure_ascii=False)
            temp_path.replace(self.filepath)
        except Exception as e:
            logger.error(f"Error saving tickets to {self.filepath}: {e}")

    def create_ticket(
        self,
        query: str,
        reason: str = "insufficient_evidence",
        context_attempted: Optional[List[str]] = None,
        category: Optional[str] = None,
    ) -> ExpertTicket:
        """Creates and logs a new pending ticket for a human expert."""
        ticket_id = f"tkt_{uuid.uuid4().hex[:8]}"
        ticket = ExpertTicket(
            ticket_id=ticket_id,
            query=query,
            reason=reason,
            status="pending",
            created_at=datetime.utcnow().isoformat(),
            context_attempted=context_attempted or [],
            category=category,
        )
        self.tickets[ticket_id] = ticket.to_dict()
        self.save()
        logger.info(f"Created fallback ticket {ticket_id} for query: '{query}'")
        return ticket

    def get_ticket(self, ticket_id: str) -> Optional[ExpertTicket]:
        """Fetches a specific ticket by ID."""
        data = self.tickets.get(ticket_id)
        if not data:
            return None
        return ExpertTicket(**data)

    def list_tickets(self, status: Optional[str] = None) -> List[ExpertTicket]:
        """Lists all tickets, optionally filtered by status ('pending', 'resolved')."""
        results: List[ExpertTicket] = []
        for data in self.tickets.values():
            if status is None or data.get("status") == status:
                results.append(ExpertTicket(**data))
        return sorted(results, key=lambda t: t.created_at, reverse=True)

    def resolve_ticket(
        self,
        ticket_id: str,
        expert_answer: str,
        expert_name: str = "Expert Consultant",
    ) -> Optional[ExpertTicket]:
        """Marks a ticket as resolved and records the expert's answer."""
        if ticket_id not in self.tickets:
            logger.warning(f"Ticket {ticket_id} not found.")
            return None

        t = self.tickets[ticket_id]
        t["status"] = "resolved"
        t["resolved_at"] = datetime.utcnow().isoformat()
        t["expert_answer"] = expert_answer.strip()
        t["expert_name"] = expert_name.strip()
        self.save()
        logger.info(f"Ticket {ticket_id} resolved by {expert_name}.")
        return ExpertTicket(**t)

    def count(self, status: Optional[str] = None) -> int:
        """Returns the count of tickets, optionally filtered by status."""
        if status is None:
            return len(self.tickets)
        return sum(1 for t in self.tickets.values() if t.get("status") == status)
