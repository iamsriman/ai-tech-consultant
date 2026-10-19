from task3_meeting_scope.models import Proposal, Scope
class ProposalAgent:
    def __init__(self, provider=None): self.provider=provider
    def create(self, scope: Scope) -> Proposal:
        overview="Proposed implementation based on the analyzed meeting transcript."; error=None; in_tokens=out_tokens=None
        provider_name=getattr(self.provider,"label","deterministic"); model=getattr(self.provider,"model_name","deterministic")
        if self.provider:
            try:
                result=self.provider.call("Prepare a concise proposal overview from this validated scope: " + str(scope.model_dump()))
                overview=str(result.get("text","")).strip() or overview; in_tokens=result.get("input_tokens"); out_tokens=result.get("output_tokens")
            except Exception as exc:
                error=f"{type(exc).__name__}: {exc}"
                provider_name = "fallback"
                model = "deterministic_fallback"
        return Proposal(version=scope.version,overview=overview,deliverables=scope.deliverables,estimated_hours=sum(t.estimated_hours for t in scope.tasks),assumptions=scope.assumptions,exclusions=scope.exclusions,risks=scope.risks,open_questions=scope.open_questions,provider=provider_name,model_used=model,provider_error=error,input_tokens=in_tokens,output_tokens=out_tokens)
