"""Run the Task 2 evaluation set through the real drafting workflow.

The evaluator deliberately does not copy Task 1 retrieval logic. It calls the
same DraftingAgent used by the API, which in turn calls Task 1 over HTTP.
"""
import json
from pathlib import Path
from task2_inbox_agent.agent.drafting_agent import DraftingAgent

def _actual_label(result):
    """Map the workflow result to the labels used by eval_cases.json."""
    category = result["category"]
    if result["requires_human"] and category in {
        "refund_request", "complaint", "legal", "escalation",
        "new_project_request", "quote_request", "follow_up"}:
        return "human_review"
    if not result["knowledge"].has_answer:
        return "unanswerable"
    if result["requires_human"]:
        return "human_review"
    return "answerable"

def _reason(result, actual):
    evidence = result["knowledge"]
    if actual == "answerable":
        provider_note = "; provider fallback used" if result.get("provider_error") else ""
        return f"Task 1 returned grounded evidence ({len(evidence.citations)} citation(s)){provider_note}"
    if evidence.generation_error:
        if "unreachable" in evidence.generation_error:
            return "Task 1 was unreachable; case skipped"
        if not evidence.has_answer:
            return f"Task 1 returned no answer ({evidence.generation_error}); human review is required"
        return evidence.generation_error
    if actual == "human_review":
        return f"category={result['category']} requires human approval"
    return "Task 1 reported no sufficient answer; human review is required"

def main():
    cases=json.loads(Path(__file__).with_name("eval_cases.json").read_text(encoding="utf-8"))
    agent=DraftingAgent()
    passed = failed = skipped = 0
    for case in cases:
        result=agent.draft(case["id"], "eval@example.test", case["query"], case["query"])
        actual = _actual_label(result)
        is_unreachable = bool(result["knowledge"].generation_error and "unreachable" in result["knowledge"].generation_error)
        status = "SKIP" if is_unreachable else ("PASS" if actual == case["expected"] else "FAIL")
        if status == "PASS": passed += 1
        elif status == "FAIL": failed += 1
        else: skipped += 1
        print(f"{case['id']}: expected={case['expected']} actual={actual} {status} - {_reason(result, actual)}")
    total = passed + failed + skipped
    rate = (passed / (total - skipped) * 100) if total != skipped else 0.0
    print(f"SUMMARY: total={total} passed={passed} failed={failed} skipped={skipped} pass_rate={rate:.1f}%")
    return 1 if failed else 0

if __name__ == "__main__": raise SystemExit(main())
