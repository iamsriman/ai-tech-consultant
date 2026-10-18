"""Run the local Task 1 answerability and retrieval evaluation set."""
import json
import argparse
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from task1_knowledge_brain.human_loop.ticket_manager import TicketManager
from task1_knowledge_brain.knowledge_base.query_engine import BDQueryEngine


def run(no_model: bool = False):
    cases_path = Path(__file__).with_name("test_cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if no_model:
        # Explicitly avoid outbound provider calls for offline corpus evaluation.
        os.environ["ANTHROPIC_API_KEY"] = ""
    # Keep evaluation runs from adding test tickets to the production ticket file.
    rows = []
    with TemporaryDirectory(prefix="knowledge-brain-eval-") as temp_dir:
        ticket_manager = TicketManager(Path(temp_dir) / "tickets.json")
        engine = BDQueryEngine(ticket_manager=ticket_manager)
        for case in cases:
            started = time.perf_counter()
            result = engine.query(case["query"])
            elapsed_ms = (time.perf_counter() - started) * 1000
            source_ok = (not case["expected_source_contains"] or any(
                case["expected_source_contains"].lower() in citation["title"].lower()
                for citation in result.citations
            ))
            rows.append({
            "id": case["id"],
            "expected_answer": case["expect_answer"],
            "actual_answer": result.has_answer,
            "answerability_correct": result.has_answer == case["expect_answer"],
            "source_correct": bool(source_ok),
            "latency_ms": round(elapsed_ms, 1),
            "model_used": result.model_used,
            "fallback_used": result.fallback_used,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            })
    total = len(rows)
    report = {
        "cases": total,
        "generation_mode": "fallback_only" if no_model else "configured_provider",
        "answerability_accuracy": sum(r["answerability_correct"] for r in rows) / total if total else 0,
        "source_accuracy": sum(r["source_correct"] for r in rows) / total if total else 0,
        "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / total, 1) if total else 0,
        "p95_latency_ms": sorted(r["latency_ms"] for r in rows)[max(0, int(total * .95) - 1)] if total else 0,
        "input_tokens_observed": sum(r["input_tokens"] or 0 for r in rows),
        "output_tokens_observed": sum(r["output_tokens"] or 0 for r in rows),
        "results": rows,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-model", action="store_true", help="Run retrieval/answerability evaluation without provider calls")
    run(no_model=parser.parse_args().no_model)
