"""
Integration Test for Phase 3:
- HierarchyAwareChunker (600 token size, context header injection, screenshots)
- BDVectorStore (ChromaDB persistence, batch upsert, similarity search, delete)
- KnowledgeRetriever (score filtering, prompt formatting, citation tracking)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from task1_knowledge_brain.scraper import PoliteCrawler
from task1_knowledge_brain.knowledge_base import (
    HierarchyAwareChunker,
    BDVectorStore,
    KnowledgeRetriever,
)


def run_phase_3_tests():
    print("=" * 65)
    print("        RUNNING TEST SUITE: PHASE 3 (RAG & VECTOR DB)          ")
    print("=" * 65)

    passed_tests = 0
    total_tests = 5

    # 1. Test Chunker with 600 token target
    print("\n[TEST 1/5] Testing HierarchyAwareChunker (600 token target)...")
    try:
        crawler = PoliteCrawler()
        articles = crawler.load_all_saved_articles()
        assert len(articles) > 0, "Need at least 1 saved article to test chunker"

        chunker = HierarchyAwareChunker(target_chunk_size=600, overlap=60)
        all_chunks = []
        for art in articles:
            chunks = chunker.chunk_article(art)
            all_chunks.extend(chunks)

        assert len(all_chunks) >= len(articles), "Each article must produce at least 1 chunk"
        sample = all_chunks[0]

        # Verify context injection
        assert sample.text.startswith("[Article:"), "Chunk must begin with hierarchy context header"
        assert sample.category in sample.text, "Chunk must include category in header"
        assert sample.url in sample.text, "Chunk must include source URL"

        print(f"  --> PASSED: Generated {len(all_chunks)} chunks from {len(articles)} articles.")
        print(f"      Chunk ID: {sample.chunk_id}")
        print(f"      Header snippet: {sample.text.splitlines()[0]}")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 1: {e}")

    # 2. Test ChromaDB Persistence & Batch Upsert
    print("\n[TEST 2/5] Testing ChromaDB Vector Store indexing...")
    try:
        vstore = BDVectorStore()
        vstore.reset()
        added = vstore.add_chunks(all_chunks)

        assert added == len(all_chunks), f"Expected {len(all_chunks)} chunks added, got {added}"
        assert vstore.count() == len(all_chunks), "Vector store count must match indexed chunks"

        print(f"  --> PASSED: Upserted {added} chunks into persistent ChromaDB.")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 2: {e}")

    # 3. Test Vector Similarity Retrieval & Metadata Filtering
    print("\n[TEST 3/5] Testing KnowledgeRetriever similarity search...")
    try:
        retriever = KnowledgeRetriever(vstore)
        query = "How do I configure domain email accounts?"
        results = retriever.retrieve(query, top_k=3, min_score=0.2)

        assert len(results) > 0, "Query should return matching documents"
        top = results[0]
        assert top.score > 0.0, "Score must be positive"
        assert top.title != "", "Title must not be empty"
        assert top.url.startswith("http"), "URL must be valid"

        print(f"  --> PASSED: Retrieved top match for '{query}':")
        print(f"      Title: '{top.title}'")
        print(f"      Category: '{top.category}' | Folder: '{top.folder}'")
        print(f"      Similarity Score: {top.score:.3f}")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 3: {e}")

    # 4. Test Prompt Context & Citation Extraction
    print("\n[TEST 4/5] Testing Citations & LLM Context Formatting...")
    try:
        citations = retriever.get_unique_citations(results)
        assert len(citations) > 0, "Citations list should not be empty"
        assert "url" in citations[0] and "title" in citations[0]

        context_prompt = retriever.format_context_for_prompt(results)
        assert "--- SOURCE [1] ---" in context_prompt, "Context block must format numbered sources"
        assert "Title:" in context_prompt
        assert "URL:" in context_prompt

        print(f"  --> PASSED: Generated clean citations ({len(citations)} source articles cited).")
        print(f"      Sample citation: {citations[0]['title']} -> {citations[0]['url']}")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 4: {e}")

    # 5. Test Article Deletion (Daily Sync Requirement)
    print("\n[TEST 5/5] Testing Article Deletion (Outdated/Removed article handling)...")
    try:
        del_target = articles[0].article_id
        initial_count = vstore.count()
        vstore.delete_article(del_target)

        after_count = vstore.count()
        assert after_count < initial_count, "Chunk count must decrease after deletion"

        # Ensure deleted article never appears in query results
        post_del_results = retriever.retrieve(query, top_k=10, min_score=0.1)
        for r in post_del_results:
            assert r.article_id != del_target, f"Deleted article {del_target} must not appear in search!"

        print(f"  --> PASSED: Deleted article ID {del_target} (count went from {initial_count} to {after_count}).")
        print(f"      Verified: Deleted article was completely excluded from search results.")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 5: {e}")

    # SUMMARY
    print("\n" + "=" * 65)
    if passed_tests == total_tests:
        print(f"   ALL {passed_tests}/{total_tests} TESTS PASSED! (PHASE 3 FULLY OPERATIONAL)")
    else:
        print(f"   COMPLETED: {passed_tests}/{total_tests} TESTS PASSED.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_phase_3_tests()
