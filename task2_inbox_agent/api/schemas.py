from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class EmailIn(BaseModel):
    email_id: Optional[str] = None
    sender: str = Field(min_length=3, max_length=320)
    subject: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=20000)
    provider: Optional[str] = Field(default=None, description="Optional provider override: claude or groq")
    model: Optional[str] = Field(default=None, description="Optional model override for the selected provider")
    knowledge_provider: Optional[str] = Field(default=None, description="Provider for the Task 1 HTTP knowledge call")
    knowledge_model: Optional[str] = Field(default=None, description="Model for the Task 1 HTTP knowledge call")

class ProcessResponse(BaseModel):
    email_id: str; category: str; draft: str; status: str
    requires_human: bool; provider: str; model_used: str
    provider_error: Optional[str] = None; knowledge: Dict[str, Any] = Field(default_factory=dict)

class ReviewIn(BaseModel):
    final_reply: str = Field(min_length=1, max_length=20000)
    approve: bool = True
    human_notes: str = Field(default="", max_length=5000)

class FeedbackIn(BaseModel):
    notes: str = Field(default="", max_length=5000)

class AutoPolicyIn(BaseModel):
    category: str; disabled: bool = True
