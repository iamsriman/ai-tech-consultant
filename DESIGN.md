# Design

## Why I built it this way

The slow part of a small software project is often the work around the code. A consultant has to search help pages, answer repeated customer questions, and turn a loose meeting into tasks and an estimate. I wanted one place to support those steps. I chose three connected tools because they share product knowledge but need different review rules.

I used the Brilliant Directories Solution Center as the first knowledge source because it is the help center named in the assignment. I kept each task in its own Python package so I could change one workflow without putting all its rules into one large module. Task 1 owns retrieval. Tasks 2 and 3 ask it for product evidence through HTTP, as required by the assignment. The brief names TypeScript for Task 2 but allows Python when a student does not use TypeScript. I kept all three tasks in Python and did not mix languages.

## How the parts connect

The three FastAPI routers are registered in `unified_backend.py`. They run in one backend process. Task 2 and Task 3 still make HTTP requests to Task 1, but they point to the same listener and port. The Streamlit UI is a separate process and calls the public APIs. Render gives the UI its own web service because Streamlit runs its own HTTP server; the three task APIs remain on one backend service.

```text
Streamlit UI
    | HTTP
    v
One FastAPI app and backend port
    |-- /api/v1/query ----------------> Task 1 --> ChromaDB
    |-- /api/v1/emails --> Task 2 --HTTP /api/v1/query--+
    |-- /api/v1/projects -> Task 3 --HTTP /api/v1/query-+--> Task 1
```

The unified health endpoint checks Task 1, Task 2, and Task 3. On Render, Uvicorn binds to the port supplied in `$PORT`. Task 1's daily sync scheduler starts with the FastAPI lifespan and uses the configured time and timezone.

## What each task does

Task 1 reads the sitemap, downloads and processes articles, and stores article chunks in ChromaDB. Its crawler checkpoint records completed work so a crawl can resume. The local article folder contains 1,540 article JSON files, and the current health response reports 1,812 indexed chunks. Images stay associated with their articles. The image pipeline has caching and tests with mocked vision calls; I have not verified a live visual-model request. The query endpoint retrieves evidence, returns article citations and screenshots when relevant, and uses a grounded fallback when generation fails. Unsupported questions can create tickets in the Task 1 ticket file. An approved expert answer is saved and indexed.

Task 2 classifies an email, asks Task 1 for product evidence, and uses that evidence to prepare a reply. It does not copy the RAG code. The sample email file contains 50 synthetic emails. The simulated sender records a successful send without contacting a customer. Human approval is required for sensitive categories and unsupported questions. A send only counts toward the automatic-send rule after a successful simulated send with the original draft unchanged. Each category has a separate count, and the policy can be disabled.

Human edits are saved as feedback records by category. The next draft can include recent examples from that category in its prompt. This is example-based prompt feedback. It does not train or fine-tune a model. The stored feedback can be removed, so the behavior can be reversed.

Task 3 separates transcript analysis, product checks, estimates, proposal creation, and change comparison. Pydantic models give these steps structured inputs and outputs. Each project version keeps its transcript, requirements, scope, tasks, and proposal. Proposals begin as pending and need an explicit human decision. The proposal is not sent to a client.

## Choices, things I left out, and limits

I did not connect Gmail or Outlook. The assignment workflow can be tested with sample emails and a simulated sender without asking for mailbox permissions. I also did not build a client proposal sender or an MCP server; the tasks connect through their HTTP APIs. These decisions keep external actions under human control and keep the current integration simple.

The meeting workflow is still a prototype. `ConsultantAgent` currently uses keyword rules to extract a small set of feature types. Although the request schema accepts transcripts up to 200,000 characters, this rule-based extraction can miss meaning in long or casually worded meetings. The feature checker treats a Task 1 answer as proof that the product supports a requirement; a relevant article does not always prove a built-in setting solves it. The Task 3 provider is used for the proposal overview, not for transcript analysis or hour estimates. `EstimatorAgent` assigns 4 hours when a feature check says supported and 12 hours otherwise. These are starter rules, not estimates reviewed by a developer. The change comparison uses requirement IDs and treats added items as outside the original scope. It needs stronger handling of renamed or reworded requirements.

Claude remains selectable because the assignment says it must be central. Render defaults to Claude Haiku (`claude-haiku-4-5-20251001`). I chose Groq for development after I could not use Claude reliably because of API credit and request errors. The saved local UI settings currently select Groq for all three tasks: `qwen/qwen3.8-27b` for Task 1, `openai/gpt-oss-20b` for Task 2, and `openai/gpt-oss-120b` for Task 3. The UI has these per-task choices, but I have not completed a controlled comparison or verified successful live use of either provider. Task 3 currently calls its selected model only for the proposal overview; its transcript extraction and estimates use rules. A provider label in settings is not proof that a request succeeded. This leaves the assignment's model requirement and evaluation requirement unfinished.

Most records are JSON files, and the article index is local ChromaDB. The Render configuration does not attach a persistent disk or external database. Before relying on Render for durable data, I need to set up persistent storage and test redeploy recovery. The app also has no user login or multi-tenant separation. The current help-center crawler is specific to Brilliant Directories, rather than a ready-made connector for any company's help center.

My next steps are to make transcript analysis more reliable, review estimates with a developer, run clean model comparisons, add durable deployment storage, and obtain the real recorded client meetings required by the assignment. The current evaluation transcripts and email samples are synthetic. They are useful for development, but they are not recordings of meetings with a friend.
