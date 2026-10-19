from typing import Optional
from pydantic import BaseModel, Field
class MeetingRequest(BaseModel): transcript: str = Field(min_length=20, max_length=200000); previous_version: Optional[int] = Field(default=None, ge=1); provider: Optional[str] = Field(default=None, description="Optional provider override: claude or groq"); model: Optional[str] = Field(default=None, description="Optional model override"); knowledge_provider: Optional[str] = Field(default=None, description="Provider for Task 1 HTTP feature checks"); knowledge_model: Optional[str] = Field(default=None, description="Model for Task 1 HTTP feature checks")
class DecisionRequest(BaseModel): approved: bool
