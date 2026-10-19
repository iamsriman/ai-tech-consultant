import json
from pathlib import Path
from fastapi.testclient import TestClient

from task2_inbox_agent.main import app
from task2_inbox_agent.autonomy.streak_tracker import StreakTracker
from task2_inbox_agent.agent.knowledge_client import KnowledgeClient
from task2_inbox_agent.agent.drafting_agent import DraftingAgent
from task2_inbox_agent.learning.memory_store import MemoryStore
from task2_inbox_agent.agent.llm_provider import get_llm_provider
from task2_inbox_agent.sender import SimulatedSender

def test_streak_requires_20_and_resets(tmp_path):
    tracker = StreakTracker(tmp_path / "streak.json", threshold=20)
    for _ in range(19): assert tracker.record("product_support", True) < 20
    assert not tracker.eligible("product_support")
    assert tracker.record("product_support", True) == 20
    assert tracker.eligible("product_support")
    tracker.record("product_support", False)
    assert tracker.count("product_support") == 0

def test_api_validation_and_idempotent_process(monkeypatch, tmp_path):
    import task2_inbox_agent.api.routes as routes
    monkeypatch.setattr(routes, "EMAIL_FILE", tmp_path / "emails.json")
    class FakeKnowledge:
        has_answer=True; answer="Use the documented email settings."; citations=[]; screenshots=[]
        confidence=0.9; ticket_id=None; model_used="test"; fallback_used=True; generation_error=None
        def __init__(self): self.__dict__.update(self.__class__.__dict__)
    class FakeAgent:
        def draft(self, *args):
            from task2_inbox_agent.agent.knowledge_client import KnowledgeResult
            return {"category":"product_support","draft":"Documented reply","knowledge":KnowledgeResult("q","Use the documented email settings.",True,.9),"provider":"test","model_used":"test","provider_error":None,"requires_human":False}
    monkeypatch.setattr(routes, "_agent", lambda: FakeAgent())
    client=TestClient(app)
    response=client.post("/api/v1/emails/process", json={"email_id":"e1","sender":"a@example.test","subject":"Help","body":"How?"})
    assert response.status_code == 200 and response.json()["status"] == "draft"
    assert client.post("/api/v1/emails/process", json={"email_id":"e1","sender":"a@example.test","subject":"Help","body":"How?"}).json()["email_id"] == "e1"
    assert client.post("/api/v1/emails/e1/send").status_code == 409
    assert client.post("/api/v1/emails/e1/review", json={"final_reply":"Documented reply"}).status_code == 200
    assert client.post("/api/v1/emails/e1/send").status_code == 200
    assert client.post("/api/v1/emails/e1/send").json()["status"] == "sent"
    feedback = client.post("/api/v1/emails/e1/feedback", json={"notes":"approved"})
    assert feedback.status_code == 200 and feedback.json()["category"] == "product_support"

def test_sample_and_eval_counts():
    samples=json.loads(Path("task2_inbox_agent/data/customer_emails.json").read_text())
    evals=json.loads(Path("task2_inbox_agent/evals/eval_cases.json").read_text())
    assert len(samples) >= 50 and len(evals) >= 20

def test_task1_failure_is_safe(monkeypatch):
    class Broken:
        def post(self, *a, **k): raise RuntimeError("offline")
    monkeypatch.setattr("httpx.Client", lambda **kw: Broken())
    result=KnowledgeClient("http://invalid.test").query("unknown")
    assert result.has_answer is False and "unreachable" in result.generation_error

def test_provider_selection(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert get_llm_provider("claude").provider_label == "claude"
    assert get_llm_provider("groq").provider_label == "groq"

def test_feedback_is_used_in_next_prompt(tmp_path):
    memory=MemoryStore(tmp_path)
    from task2_inbox_agent.learning.feedback_collector import FeedbackCollector
    FeedbackCollector(memory).collect("e1","product_support","old","preferred", "use concise wording")
    class Knowledge:
        def query(self,*a,**k):
            from task2_inbox_agent.agent.knowledge_client import KnowledgeResult
            return KnowledgeResult("q","verified answer",True,.9)
    class FakeLLM:
        provider_label="test"; model_name="test"
        def __init__(self): self.prompt=""
        def call(self,prompt,system_prompt=None): self.prompt=prompt; return {"text":"draft","input_tokens":1,"output_tokens":1}
    llm=FakeLLM(); DraftingAgent(Knowledge(),llm,memory).draft("e2","a@b.test","How do I configure this?","How do I configure this?")
    assert "preferred" in llm.prompt

def test_feedback_changes_later_generated_draft(tmp_path):
    memory=MemoryStore(tmp_path)
    from task2_inbox_agent.learning.feedback_collector import FeedbackCollector
    FeedbackCollector(memory).collect("e1", "product_support", "Draft with too much detail", "Use the short verified answer.")
    class Knowledge:
        def query(self,*args,**kwargs):
            from task2_inbox_agent.agent.knowledge_client import KnowledgeResult
            return KnowledgeResult("q", "Verified answer", True, .9)
    class LearningLLM:
        provider_label="test"; model_name="test"
        def call(self, prompt, system_prompt=None):
            text = "Use the short verified answer." if "Use the short verified answer." in prompt else "Generic draft"
            return {"text": text, "input_tokens": 2, "output_tokens": 3}
    result=DraftingAgent(Knowledge(), LearningLLM(), memory).draft(
        "e2", "customer@example.test", "How do I configure this?", "How do I configure this?"
    )
    assert result["draft"] == "Use the short verified answer."

def test_streaks_are_independent_and_edit_resets_category(tmp_path):
    tracker=StreakTracker(tmp_path / "streak.json", threshold=20)
    categories=["product_support", "new_project_request", "refund_request", "complaint", "quote_request", "follow_up", "legal", "general_inquiry"]
    for category in categories:
        for _ in range(20): tracker.record(category, True)
    assert tracker.eligible("product_support")
    assert all(tracker.eligible(category) for category in categories)
    tracker.record("product_support", False)
    assert tracker.count("product_support") == 0
    assert tracker.count("quote_request") == 20
    tracker2=StreakTracker(tmp_path / "streak.json", threshold=20)
    assert tracker2.count("product_support") == 0 and tracker2.count("quote_request") == 20

def test_simulated_send_is_duplicate_safe_after_restart(tmp_path):
    first=SimulatedSender(tmp_path)
    sent=first.send("email-1", "customer@example.test", "Subject", "Original body")
    second=SimulatedSender(tmp_path)
    duplicate=second.send("email-1", "customer@example.test", "Subject", "Different retry body")
    assert duplicate == sent
    assert len(list(tmp_path.glob("email-1.json"))) == 1
