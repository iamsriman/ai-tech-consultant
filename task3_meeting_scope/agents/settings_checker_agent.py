from task3_meeting_scope.knowledge_client import KnowledgeClient
from task3_meeting_scope.models import FeatureCheck, Requirement

class SettingsCheckerAgent:
    def __init__(self, client=None, provider_name=None, model_name=None): self.client=client or KnowledgeClient(provider=provider_name, model=model_name)
    def check(self, requirements):
        checks=[]
        for req in requirements:
            data=self.client.query(req.statement); has=bool(data.get("has_answer"));
            checks.append(FeatureCheck(requirement_id=req.id,supported=has,answer=data.get("answer", ""),citations=data.get("citations",[]),status="supported" if has else "not_verified"))
        return checks
