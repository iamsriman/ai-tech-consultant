from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class MeetingInput(BaseModel):
    transcript: str = Field(min_length=20, max_length=200000)
    previous_version: Optional[int] = Field(default=None, ge=1)

class Requirement(BaseModel):
    id: str; statement: str; status: str = "confirmed"; source: str = "transcript"
    clarification: Optional[str] = None

class TranscriptAnalysis(BaseModel):
    goals: List[str] = []; requested_features: List[Requirement] = []
    problems: List[str] = []; constraints: List[str] = []; deadlines: List[str] = []
    budget: Optional[str] = None; assumptions: List[str] = []; dependencies: List[str] = []
    unanswered_questions: List[str] = []

class FeatureCheck(BaseModel):
    requirement_id: str; supported: Optional[bool] = None; answer: str = ""
    citations: List[Dict[str, Any]] = []; status: str = "not_verified"

class DeveloperTask(BaseModel):
    task_id: str; description: str; expected_result: str; estimated_hours: float
    dependencies: List[str] = []; assumptions: List[str] = []

class Scope(BaseModel):
    version: int; objectives: List[str]; included_features: List[Requirement]
    deliverables: List[str]; assumptions: List[str]; dependencies: List[str]
    exclusions: List[str]; risks: List[str]; open_questions: List[str]
    feature_checks: List[FeatureCheck] = []; tasks: List[DeveloperTask] = []

class Proposal(BaseModel):
    version: int; status: str = "pending_human_approval"; overview: str
    deliverables: List[str]; estimated_hours: float; pricing: str = "Pending pricing information"
    assumptions: List[str]; exclusions: List[str]; risks: List[str]; open_questions: List[str]
    provider: str = "deterministic"; model_used: str = "deterministic"
    provider_error: Optional[str] = None; input_tokens: Optional[int] = None; output_tokens: Optional[int] = None

class ProjectVersion(BaseModel):
    version: int; transcript: str; analysis: TranscriptAnalysis; scope: Scope
    proposal: Proposal; change_summary: Optional[Dict[str, Any]] = None

class Project(BaseModel):
    project_id: str; versions: List[ProjectVersion] = []
