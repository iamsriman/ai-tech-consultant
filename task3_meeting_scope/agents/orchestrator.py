import json, uuid
from pathlib import Path
from task3_meeting_scope.models import Project, ProjectVersion, Scope
from task3_meeting_scope.config import STORE_FILE
from task3_meeting_scope.agents.consultant_agent import ConsultantAgent
from task3_meeting_scope.agents.settings_checker_agent import SettingsCheckerAgent
from task3_meeting_scope.agents.estimator_agent import EstimatorAgent
from task3_meeting_scope.agents.proposal_agent import ProposalAgent
from task3_meeting_scope.diff_engine.scope_diff import compare
from task3_meeting_scope.providers import get_provider

class Orchestrator:
    def __init__(self,path=STORE_FILE, provider=None): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.projects=self._load(); self.provider=provider if provider is not None else get_provider()
    def _load(self):
        if not self.path.exists(): return {}
        try: return {k:Project.model_validate(v) for k,v in json.loads(self.path.read_text()).items()}
        except Exception: return {}
    def _save(self): self.path.write_text(json.dumps({k:v.model_dump() for k,v in self.projects.items()},indent=2),encoding="utf-8")
    def process(self,pid,transcript,previous=None,provider_name=None,model_name=None,knowledge_provider=None,knowledge_model=None):
        project=self.projects.setdefault(pid,Project(project_id=pid)); analysis=ConsultantAgent().analyze(transcript); checks=SettingsCheckerAgent(provider_name=knowledge_provider, model_name=knowledge_model).check(analysis.requested_features); tasks=EstimatorAgent().estimate(analysis.requested_features,checks); version=len(project.versions)+1
        scope=Scope(version=version,objectives=analysis.goals,included_features=analysis.requested_features,deliverables=["Approved scope of work","Developer task list","Client proposal"],assumptions=analysis.assumptions,dependencies=analysis.dependencies,exclusions=["Unconfirmed requirements and pricing"],risks=["Estimates may change after unanswered questions are resolved"],open_questions=analysis.unanswered_questions,feature_checks=checks,tasks=tasks)
        proposal_provider = get_provider(provider_name, model_name) if provider_name else self.provider
        proposal=ProposalAgent(proposal_provider).create(scope); item=ProjectVersion(version=version,transcript=transcript,analysis=analysis,scope=scope,proposal=proposal)
        if previous: item.change_summary=compare(project.versions[previous-1],item)
        project.versions.append(item); self._save(); return item
    def get(self,pid): return self.projects.get(pid)
    def approve(self,pid,version,approved):
        project=self.projects.get(pid)
        if not project or version<1 or version>len(project.versions): raise KeyError("project/version not found")
        project.versions[version-1].proposal.status="approved" if approved else "rejected"; self._save(); return project.versions[version-1]
