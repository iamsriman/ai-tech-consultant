import json, time, uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException
from task2_inbox_agent.api.schemas import EmailIn, ReviewIn, FeedbackIn, AutoPolicyIn
from task2_inbox_agent.agent.drafting_agent import DraftingAgent
from task2_inbox_agent.agent.knowledge_client import KnowledgeClient
from task2_inbox_agent.agent.llm_provider import get_llm_provider
from task2_inbox_agent.autonomy.streak_tracker import StreakTracker
from task2_inbox_agent.autonomy.policy_engine import PolicyEngine
from task2_inbox_agent.learning.feedback_collector import FeedbackCollector
from task2_inbox_agent.config import DATA_DIR, LLM_PROVIDER
from task2_inbox_agent.sender import SimulatedSender

router = APIRouter(prefix="/api/v1")
EMAIL_FILE = DATA_DIR / "emails.json"

def _load():
    if not EMAIL_FILE.exists(): return {}
    try: return json.loads(EMAIL_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): raise HTTPException(500, "email store is corrupt")
def _save(data):
    tmp = EMAIL_FILE.with_suffix(".tmp"); tmp.write_text(json.dumps(data, indent=2), encoding="utf-8"); tmp.replace(EMAIL_FILE)
def _agent(provider_name=None, model_name=None, knowledge_provider=None, knowledge_model=None):
    selected = provider_name or LLM_PROVIDER
    return DraftingAgent(knowledge=KnowledgeClient(provider=knowledge_provider, model=knowledge_model), llm=get_llm_provider(selected, model_name))

@router.get("/health")
def health(): return {"status":"ok", "provider":LLM_PROVIDER}

@router.post("/emails/process")
def process(email: EmailIn):
    data = _load(); eid = email.email_id or f"email_{uuid.uuid4().hex[:12]}"
    if eid in data: return data[eid]
    agent = _agent(email.provider, email.model, email.knowledge_provider, email.knowledge_model) if (email.provider or email.model or email.knowledge_provider or email.knowledge_model) else _agent()
    result = agent.draft(eid, email.sender, email.subject, email.body)
    row = {"email_id":eid, "sender":email.sender, "subject":email.subject, "body":email.body,
           "category":result["category"], "original_draft":result["draft"], "final_reply":None,
           "status":"draft", "created_at":time.time(), "provider":result["provider"],
           "model_used":result["model_used"], "provider_error":result["provider_error"],
           "requires_human":result["requires_human"], "knowledge":result["knowledge"].__dict__}
    data[eid]=row; _save(data); return row

@router.get("/emails")
def list_emails(): return list(_load().values())

@router.get("/emails/{email_id}")
def get_email(email_id: str):
    row=_load().get(email_id)
    if not row: raise HTTPException(404,"email not found")
    return row

@router.post("/emails/{email_id}/review")
def review(email_id: str, review: ReviewIn):
    data=_load(); row=data.get(email_id)
    if not row: raise HTTPException(404,"email not found")
    if row["status"] == "sent": return row
    row["final_reply"] = review.final_reply; row["human_notes"] = review.human_notes
    row["status"] = "approved" if review.approve else "rejected"
    if not review.approve: row["status"]="needs_review"
    _save(data); return row

@router.post("/emails/{email_id}/send")
def send(email_id: str):
    data=_load(); row=data.get(email_id)
    if not row: raise HTTPException(404,"email not found")
    if row["status"] == "sent": return row
    if row["status"] != "approved": raise HTTPException(409,"email must be approved before sending")
    try:
        SimulatedSender().send(email_id, row["sender"], row["subject"], row["final_reply"] or "")
    except Exception as exc:
        raise HTTPException(502, f"sender failed: {type(exc).__name__}")
    row["status"]="sent"; row["sent_at"]=time.time(); _save(data)
    unchanged = row["original_draft"].strip() == (row["final_reply"] or "").strip()
    tracker=StreakTracker(); row["streak"] = tracker.record(row["category"], unchanged)
    _save(data); return row

@router.post("/emails/{email_id}/auto-send")
def auto_send(email_id: str):
    data=_load(); row=data.get(email_id)
    if not row: raise HTTPException(404,"email not found")
    tracker=StreakTracker()
    if not PolicyEngine(tracker).can_auto_send(row["category"], bool(row["knowledge"].get("has_answer")), not row.get("provider_error")):
        raise HTTPException(403,"category is not eligible for automatic sending")
    if row["status"] == "sent": return row
    row["final_reply"] = row["original_draft"]; row["status"] = "approved"; _save(data)
    return send(email_id)

@router.post("/emails/{email_id}/feedback")
def feedback(email_id: str, item: FeedbackIn):
    row=_load().get(email_id)
    if not row: raise HTTPException(404,"email not found")
    if row["status"] != "sent": raise HTTPException(409,"feedback requires a sent email")
    record=FeedbackCollector().collect(email_id,row["category"],row["original_draft"],row["final_reply"] or "",item.notes)
    return record.to_dict()

@router.get("/autonomy")
def autonomy(): return StreakTracker().snapshot()

@router.post("/autonomy/disable")
def disable(policy: AutoPolicyIn):
    StreakTracker().set_disabled(policy.category, policy.disabled); return StreakTracker().snapshot()
