import json
from pathlib import Path
from task3_meeting_scope.agents.orchestrator import Orchestrator

def main():
    cases=json.loads(Path(__file__).with_name("eval_transcripts.json").read_text(encoding="utf-8")); service=Orchestrator(Path("task3_meeting_scope/data/eval_projects.json")); passed=failed=0
    for c in cases:
        if c.get("previous"):
            service.process(c["id"], c["previous"])
            item=service.process(c["id"], c["text"], previous=1)
        else:
            item=service.process(c["id"],c["text"])
        outcomes=[]
        if item.scope.included_features or item.analysis.goals: outcomes.append("scope")
        if item.analysis.unanswered_questions: outcomes.append("clarification")
        if item.proposal.status=="pending_human_approval": outcomes.append("pending")
        if item.change_summary: outcomes.append("change")
        outcomes.append("proposal")
        actual="+".join(outcomes) or "none"; ok=c["expected"] in outcomes
        passed+=ok; failed+=not ok; print(f"{c['id']}: expected={c['expected']} actual={actual} {'PASS' if ok else 'FAIL'}")
    print(f"SUMMARY: total={len(cases)} passed={passed} failed={failed} pass_rate={passed/len(cases)*100:.1f}%")
if __name__=="__main__": main()
