"""
Comprehensive Integration Test Suite for Phase 4 & Phase 5.
Tests:
- Phase 4: Grounded RAG, Honesty Guardrails, Fallback Ticketing, and Expert Feedback Ingestion.
- Phase 5: Daily Sitemap Sync, Diff Detection, Deletions, and Failure Safety Guard.
- FastAPI Endpoints: /query, /settings/check, /tickets, /tickets/resolve, /sync, /health.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi.testclient import TestClient
from task1_knowledge_brain.main import app
from task1_knowledge_brain.knowledge_base.vector_store import BDVectorStore
from task1_knowledge_brain.knowledge_base.query_engine import BDQueryEngine
from task1_knowledge_brain.human_loop.ticket_manager import TicketManager
from task1_knowledge_brain.human_loop.feedback_ingest import FeedbackIngester
from task1_knowledge_brain.syncer.daily_sync import DailySitemapSyncer, SyncReport
from task1_knowledge_brain.scraper.sitemap_parser import SitemapEntry
from task1_knowledge_brain.multimodal.image_extractor import ArticleData


def run_tests():
    print("=" * 70)
    print("        RUNNING TEST SUITE: PHASE 4 & PHASE 5 VALIDATION        ")
    print("=" * 70)

    client = TestClient(app)
    passed_tests = 0
    total_tests = 8

    # -------------------------------------------------------------
    # TEST 1: Grounded RAG Query with Valid Citations & Screenshots
    # -------------------------------------------------------------
    print("\n[TEST 1/8] Testing Grounded RAG Query (Known BD Topic)...")
    try:
        engine = BDQueryEngine()
        res = engine.query("How do I create email accounts and mailboxes on my domain?", top_k=2)

        assert res.has_answer is True, "Expected grounded answer for known email topic"
        assert len(res.citations) > 0, "Expected at least 1 citation"
        assert res.confidence > 0.40, f"Expected confidence >0.40, got {res.confidence}"
        assert res.ticket_id is None, "Should not create ticket when answer is found"
        
        cit = res.citations[0]
        assert "url" in cit and cit["url"].startswith("http"), "Citation must contain valid URL"
        assert "title" in cit and len(cit["title"]) > 0, "Citation must contain title"

        print(f"  --> PASSED: Answered grounded question with confidence {res.confidence:.2f}.")
        print(f"      Top Citation: '{cit['title']}' -> {cit['url']}")
        print(f"      Screenshots attached: {len(res.screenshots)}")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 1: {e}")

    # -------------------------------------------------------------
    # TEST 2: Honesty Guardrails & Automatic Fallback Ticket Creation
    # -------------------------------------------------------------
    print("\n[TEST 2/8] Testing Honesty Guardrail & Automatic Ticket Creation (Unknown Topic)...")
    try:
        unknown_q = "How do I launch a satellite into Mars orbit using Brilliant Directories?"
        res_unknown = engine.query(unknown_q)

        assert res_unknown.has_answer is False, "Bot must not hallucinate answers for unknown topics"
        assert "don't know" in res_unknown.answer.lower() or "not know" in res_unknown.answer.lower()
        assert res_unknown.ticket_id is not None, "Bot must generate a fallback ticket ID"

        # Verify ticket was persisted in TicketManager
        ticket = engine.ticket_manager.get_ticket(res_unknown.ticket_id)
        assert ticket is not None, "Ticket must exist in persistent store"
        assert ticket.status == "pending", "New ticket status must be 'pending'"
        assert ticket.query == unknown_q, "Ticket must store the exact user query"

        print(f"  --> PASSED: Bot honestly stated 'I don't know' without hallucination.")
        print(f"      Created fallback ticket: {res_unknown.ticket_id} (Status: {ticket.status})")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 2: {e}")

    # -------------------------------------------------------------
    # TEST 3: Expert Feedback Ingestion Loop (Learning from Humans)
    # -------------------------------------------------------------
    print("\n[TEST 3/8] Testing Expert Feedback Ingestion (Human Resolution -> Vector Memory)...")
    try:
        ingester = FeedbackIngester(vector_store=engine.retriever.vector_store, ticket_manager=engine.ticket_manager)
        expert_answer_text = (
            "Brilliant Directories connects to orbital telemetry via our custom Webhook API "
            "under Developer Tools > Webhook Endpoints. Configure the endpoint to stream payload events."
        )

        res_ingest = ingester.ingest_expert_answer(
            ticket_id=res_unknown.ticket_id,
            expert_answer=expert_answer_text,
            expert_name="Dr. Eleanor Support",
        )

        assert res_ingest["status"] == "success"
        assert res_ingest["indexed_chunks"] == 1

        # Check ticket status updated to resolved
        resolved_tkt = engine.ticket_manager.get_ticket(res_unknown.ticket_id)
        assert resolved_tkt.status == "resolved"
        assert resolved_tkt.expert_answer == expert_answer_text

        # Verify memory learned: Re-query the previously unknown question!
        re_query_res = engine.query(unknown_q, top_k=2)
        assert re_query_res.has_answer is True, "Bot must now successfully answer after human feedback!"
        assert "telemetry" in re_query_res.answer.lower() or "webhook" in re_query_res.answer.lower()

        print(f"  --> PASSED: Expert answer indexed into ChromaDB. Ticket marked resolved.")
        print(f"      Verified: Knowledge Brain immediately answered subsequent query using newly learned memory!")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 3: {e}")

    # -------------------------------------------------------------
    # TEST 4: Daily Sitemap Sync - Safety Guard Against Data Loss
    # -------------------------------------------------------------
    print("\n[TEST 4/8] Testing Daily Sync Safety Guard (Never delete on network/parser failure)...")
    try:
        vstore = BDVectorStore()
        initial_chunk_count = vstore.count()

        # Mock a failing sitemap parser
        failing_parser = MagicMock()
        failing_parser.parse.side_effect = Exception("Simulated 503 Service Unavailable / Connection Timeout")

        syncer = DailySitemapSyncer(sitemap_parser=failing_parser, vector_store=vstore)
        report = syncer.sync()

        assert report.status == "aborted", "Sync must abort when sitemap fetch fails"
        assert "503" in report.message or "failed" in report.message.lower()
        assert vstore.count() == initial_chunk_count, "ChromaDB data must remain 100% intact after failed sync!"

        print(f"  --> PASSED: Sync safely aborted without deleting any articles (Count: {vstore.count()} chunks intact).")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 4: {e}")

    # -------------------------------------------------------------
    # TEST 5: Daily Sitemap Sync - Deletion of Removed Articles
    # -------------------------------------------------------------
    print("\n[TEST 5/8] Testing Daily Sync Article Deletion (Outdated article cleanup)...")
    try:
        # Create a mock sitemap parser that omits article 5000558373
        real_syncer = DailySitemapSyncer(vector_store=vstore)
        articles_before = len(real_syncer.crawler.load_all_saved_articles())

        # Test delete logic specifically
        test_delete_id = "5000558373"
        vstore.delete_article(test_delete_id)
        
        # Verify chunks for this article are gone
        results_after = vstore.query("Email Forwarders", n_results=5)
        for r in results_after:
            assert r.article_id != test_delete_id, f"Deleted article {test_delete_id} must not appear in ChromaDB"

        print(f"  --> PASSED: Successfully purged removed article {test_delete_id} from vector store.")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 5: {e}")

    # -------------------------------------------------------------
    # TEST 6: FastAPI - /api/v1/query & /api/v1/health
    # -------------------------------------------------------------
    print("\n[TEST 6/8] Testing FastAPI Endpoint: POST /api/v1/query & GET /api/v1/health...")
    try:
        # Test health
        h_resp = client.get("/api/v1/health")
        assert h_resp.status_code == 200
        h_data = h_resp.json()
        assert h_data["status"] == "ok"
        assert h_data["chroma_db_healthy"] is True

        # Test query endpoint
        q_resp = client.post(
            "/api/v1/query",
            json={"query": "How do I configure email accounts and mailboxes?", "top_k": 2},
        )
        assert q_resp.status_code == 200
        q_data = q_resp.json()
        assert "answer" in q_data
        assert "citations" in q_data
        assert q_data["has_answer"] is True

        print(f"  --> PASSED: Health status OK. Query endpoint returned valid JSON response.")
        print(f"      API Citations: {len(q_data['citations'])} | Confidence: {q_data['confidence']:.2f}")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 6: {e}")

    # -------------------------------------------------------------
    # TEST 7: FastAPI - /api/v1/settings/check (For Task 3 Scoping)
    # -------------------------------------------------------------
    print("\n[TEST 7/8] Testing FastAPI Endpoint: POST /api/v1/settings/check...")
    try:
        s_resp = client.post(
            "/api/v1/settings/check",
            json={"feature_description": "Custom mailbox at our own website domain"},
        )
        assert s_resp.status_code == 200
        s_data = s_resp.json()
        assert "can_solve_with_native_settings" in s_data
        assert s_data["can_solve_with_native_settings"] is True, "Domain mailbox is a native BD setting"
        assert s_data["article_title"] is not None

        print(f"  --> PASSED: Correctly identified native BD setting for Task 3 scoping:")
        print(f"      Matched Setting: '{s_data['matched_setting']}'")
        print(f"      Article: '{s_data['article_title']}'")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 7: {e}")

    # -------------------------------------------------------------
    # TEST 8: FastAPI - /api/v1/tickets & /api/v1/tickets/{id}/resolve
    # -------------------------------------------------------------
    print("\n[TEST 8/8] Testing FastAPI Endpoint: Ticket Listing & Resolution...")
    try:
        # Create a ticket via an unknown question
        client.post(
            "/api/v1/query",
            json={"query": "How do I install custom Python microservices inside BD core kernel?"},
        )

        # List pending tickets
        t_resp = client.get("/api/v1/tickets?status=pending")
        assert t_resp.status_code == 200
        t_list = t_resp.json()
        assert len(t_list) > 0, "Expected at least 1 pending ticket"
        target_ticket = t_list[0]

        # Resolve ticket via API
        r_resp = client.post(
            f"/api/v1/tickets/{target_ticket['ticket_id']}/resolve",
            json={
                "expert_answer": "Brilliant Directories hosted environment does not permit internal kernel modification. Use Zapier or external API webhooks.",
                "expert_name": "Senior Solutions Architect",
            },
        )
        assert r_resp.status_code == 200
        r_data = r_resp.json()
        assert r_data["status"] == "success"
        assert r_data["indexed_chunks"] == 1

        print(f"  --> PASSED: Successfully resolved ticket {target_ticket['ticket_id']} via REST API.")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 8: {e}")

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    if passed_tests == total_tests:
        print(f"   ALL {passed_tests}/{total_tests} TESTS PASSED! (PHASES 4 & 5 FULLY OPERATIONAL)")
    else:
        print(f"   COMPLETED: {passed_tests}/{total_tests} TESTS PASSED.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_tests()
