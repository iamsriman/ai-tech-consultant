from task2_inbox_agent.config import SENSITIVE_CATEGORIES

class PolicyEngine:
    def __init__(self, streak_tracker): self.streak_tracker = streak_tracker
    def can_auto_send(self, category, has_answer, supported=True):
        return bool(has_answer and supported and category not in SENSITIVE_CATEGORIES and self.streak_tracker.eligible(category))
