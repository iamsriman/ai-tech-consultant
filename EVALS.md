# Evaluations

The assignment requires 20 or more cases per task, model comparisons, and quality, cost, and speed measurements. The project has 22 Task 1 cases and 20 each for Tasks 2 and 3. The results below separate measured checks from missing measurements.

| Task | Cases and observed result | Speed, tokens, and cost |
|---|---|---|
| Knowledge Brain | 22 cases; 21/22 answerability matches (95.5%). Expected-title check: 21/22. | Mean 493.8 ms; p95 510.8 ms. No provider tokens. No model API call. |
| Inbox Agent | 20 cases; an earlier run reported 20/20 passing. | The runner did not save latency, token, or cost measurements. |
| Meeting to Proposal | 20 cases are present; no trustworthy pass summary is saved. | No reliable latency, token, or cost report. |

## Task 1: Knowledge Brain

I ran `python -m task1_knowledge_brain.evals.run_evals --no-model` against the current local index. It used the deterministic grounded fallback and a temporary ticket file. It did not call Claude or Groq. Fifteen cases expected an answer and seven expected a rejection. The system got 21 answerability decisions right out of 22. The failed case was `remote_mysql`: the case expected an answer, but the current index did not provide sufficient evidence. The satellite, Mars orbit, microservices, quantum, Bootstrap, medical, and weather cases were all rejected as expected.

The expected-title check was 21/22. This is not a human review of citations; cases without an expected source title count as a match by default. The first request took 1,425.7 ms, so startup affected the average. No model tokens or provider cost were used; local compute cost was not measured. This tests retrieval and answerability, not Claude or Groq writing quality.

The Task 1 regression test file also passed 23 tests in this documentation pass. The full Task 1 suite did not finish: it stalled while `test_vector_store_counts` built a temporary test index from the saved article corpus, so I stopped that run.

## Task 2: Inbox Agent

The evaluation file contains 20 cases: 9 expected to be answerable, 6 expected to require human review, and 5 expected to be unanswerable. Earlier in our project work, the evaluation runner was reported as 20/20 passing. That runner checks the workflow label against the expected label. It does not grade the factual quality or tone of each generated email. No machine-readable report from that run was saved.

The repository has 50 synthetic emails. The combined Task 2 and Task 3 test run passed 11 tests, including 9 for Task 2. They cover feedback and send rules, not model quality. The eval output does not preserve model per case, latency, tokens, or cost, so the 20/20 result cannot prove Groq or Claude quality.

The implementation can report provider failure and use a labelled grounded fallback for supported product questions. Unanswerable emails go for human review. It does not silently claim that a failed provider wrote the draft. The sender used in tests is simulated.

## Task 3: Meeting to Proposal

There are 20 evaluation transcripts. The runner checks whether basic output sections such as a scope, clarification questions, a pending proposal, or a change summary are present. It does not check if the extracted requirements, estimates, or proposal are correct. I did not run it against the saved `eval_projects.json` during this documentation pass because the runner writes into that file.

The saved evaluation-project data has 20 records. Each latest proposal identifies Claude Haiku but also has a provider error and remains pending. These are failed attempts, not successful Claude usage or a 20-case pass. The 2 Task 3 unit tests passed in the combined run. No model latency, token usage, or cost was established.

The unified backend integration tests passed 10/10 in an earlier run. They check that the three routers share one app and that Task 2 and Task 3 point to Task 1 on the same backend port. These are integration checks, not model evaluations.

## Model comparisons still needed

I have not run a controlled Claude-versus-Groq comparison on the same cases for any task. In earlier testing, Claude had credit or request errors, and Groq also returned HTTP 400 errors. The current saved UI settings select Groq for all three tasks, but the previous Task 2 evaluation did not save enough information to prove which model handled every case. The Task 1 results above use no model. The Task 3 saved records show failures. Therefore there is no measured model quality comparison, no measured cost per task, and no fair model speed comparison yet. I need successful provider calls, saved model and usage fields, and reviewed outputs before I can report those numbers.
