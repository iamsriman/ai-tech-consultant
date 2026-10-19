import re
from task3_meeting_scope.models import TranscriptAnalysis, Requirement

class ConsultantAgent:
    def analyze(self, transcript: str) -> TranscriptAnalysis:
        text=transcript.strip(); lower=text.lower(); req=[]
        patterns=[("membership", "membership functionality"),("email", "email account functionality"),("search", "member search functionality"),("payment", "payment functionality"),("dashboard", "admin dashboard"),("migration", "data migration")]
        for i,(word,label) in enumerate(patterns,1):
            if word in lower: req.append(Requirement(id=f"REQ-{i}",statement=label))
        goals=[s.strip() for s in re.split(r"[.!?]", text) if any(w in s.lower() for w in ["want","need","goal","build"])][:5]
        constraints=[]; questions=[]; deadlines=[]; budget=None
        if "budget" in lower or "$" in text: budget=next((s.strip() for s in re.split(r"[.!?]",text) if "budget" in s.lower() or "$" in s), None)
        if "deadline" in lower or "by " in lower: deadlines=[s.strip() for s in re.split(r"[.!?]",text) if "deadline" in s.lower() or "by " in s.lower()]
        if not req: questions.append("Which product capabilities and user workflows are required?")
        if "admin" not in lower: questions.append("Which administrator roles and permissions are needed?")
        return TranscriptAnalysis(goals=goals or ["Clarify the client's desired outcome"], requested_features=req, problems=[s.strip() for s in re.split(r"[.!?]",text) if "problem" in s.lower() or "issue" in s.lower()][:5], constraints=constraints, deadlines=deadlines, budget=budget, assumptions=[], dependencies=[], unanswered_questions=questions)
