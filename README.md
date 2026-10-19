# AI Tech Consultant

I built this project to reduce repeated consultant work: searching product help, answering customer email, and writing project scopes from meetings. It uses the Brilliant Directories Solution Center.

The Knowledge Brain crawls and searches help articles, returns citations, and creates expert tickets for unanswered questions. The Inbox Agent uses its API to draft replies for human review. Meeting to Proposal checks product features, creates task estimates and a pending proposal, and compares follow-up versions. Email sending is simulated.

The backend uses Python, FastAPI, and ChromaDB. The UI uses Streamlit. All three task APIs share one backend app and port. JSON files hold tickets, feedback, email history, and project versions. Claude and Groq are selectable, but successful live use and model comparisons are not verified yet.

## Setup and run

In PowerShell at the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r frontend\requirements.txt
```

Put `ANTHROPIC_API_KEY` and/or `GROQ_API_KEY` in the root `.env` file. Do not commit it. Then run this single PowerShell line to start the backend, wait for health, and launch the UI:

```powershell
$env:PORT='8000'; $env:BACKEND_API_URL='http://127.0.0.1:8000'; `
Start-Process python -ArgumentList 'run.py' -WorkingDirectory $PWD; `
$ready=$false; for($i=0;$i -lt 90;$i++){try{if((Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/v1/unified-health' -TimeoutSec 2).status -eq 'ok'){$ready=$true;break}}catch{}; Start-Sleep -Seconds 1}; `
if(-not $ready){throw 'Backend did not become healthy; check the backend window'}; `
python -m streamlit run frontend/app.py --server.port 8501
```

Open `http://localhost:8501`. All task APIs use port 8000; the UI uses 8501. Render runs one backend service and a separate Streamlit service, connected by `BACKEND_API_URL`.
