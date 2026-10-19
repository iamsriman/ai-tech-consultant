# AI Log

These are short excerpts from the prompts I used with Codex while building the project. I kept the requests focused on the code I already had and asked for tests instead of a rebuild.

1. “Complete Task 1 — Knowledge Brain: Find and Fix All Bugs.” I wanted the existing crawler, ChromaDB retrieval, citations, human tickets, image handling, and sync to work together without losing the saved article data.

2. “Now I want to complete Task 2 of my AI Tech Consultant project.” I asked for an email workflow that uses Task 1 through its API, keeps sending simulated, stores human edits, and tests the 20 unchanged-send rule.

3. “Now I want to start Task 3 of my AI Tech Consultant project.” I wanted existing code to turn transcripts into requirements, product checks, task estimates, proposals, and versioned follow-up comparisons, with a person approving each proposal.

4. “I want to clarify my main requirement: I need all three FastAPI backends to run on one port instead of three separate ports.” I wanted one FastAPI application for deployment, while Tasks 2 and 3 still called Task 1 through HTTP.

5. “I want to redesign the entire frontend of my AI Tech Consultant application.” I asked for a Streamlit interface that uses the real APIs, explains model failures honestly, and keeps the existing review and approval steps.

One correction I made was to insist on one actual backend listener. A launcher that starts three separate FastAPI servers would still use three ports, even if one command started them. I clarified that Task 1, Task 2, and Task 3 must be registered in the same FastAPI app so Render can host one backend service and the UI can use one backend URL.
