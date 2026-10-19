"""Light, beginner-friendly Streamlit frontend for the AI Tech Consultant APIs.

The app contains no retrieval, drafting or proposal logic. It validates input,
calls the three backend services and presents their responses in plain language.
"""
from __future__ import annotations

import html
import inspect
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import streamlit as st

try:  # Support both `streamlit run frontend/app.py` and package-based tests.
    from .api_client import ApiClient, BackendError, clients
    from .provider_settings import load_settings, model_for, save_settings
except ImportError:  # Streamlit executes this file as a script.
    from api_client import ApiClient, BackendError, clients
    from provider_settings import load_settings, model_for, save_settings

st.set_page_config(
    page_title="AI Tech Consultant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# `use_container_width` was replaced by `width="stretch"` in newer Streamlit.
STRETCH: dict[str, Any] = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.button).parameters
    else {"use_container_width": True}
)

NOT_FOUND = object()  # returned by api_call(..., quiet_404=True) on HTTP 404

PAGES = ["Home", "Knowledge Brain", "Inbox Agent", "Meeting to Proposal", "Settings"]
PAGE_ICONS = {
    "Home": "🏠",
    "Knowledge Brain": "🔎",
    "Inbox Agent": "✉️",
    "Meeting to Proposal": "📝",
    "Settings": "⚙️",
}
SERVICES = {
    "task1": {"name": "Knowledge Brain", "icon": "🔎"},
    "task2": {"name": "Inbox Agent", "icon": "✉️"},
    "task3": {"name": "Meeting to Proposal", "icon": "📝"},
}
PROVIDER_LABELS = {"claude": "Claude", "groq": "Groq"}
EMAIL_STATUS = {
    "draft": ("Draft ready for review", "amber"),
    "needs_review": ("Needs your attention", "amber"),
    "approved": ("Approved, ready to send", "green"),
    "sent": ("Sent (simulated)", "green"),
}
PROPOSAL_STATUS = {
    "pending_human_approval": ("Waiting for your approval", "amber"),
    "approved": ("Approved", "green"),
    "rejected": ("Rejected", "red"),
}
EXAMPLE_QUESTIONS = [
    "How do I create and manage email accounts?",
    "How do I add a new membership plan?",
    "How can I customize my site's menu?",
]


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def css() -> None:
    st.markdown(
        """<style>
        :root{--ink:#172033;--muted:#5b6578;--blue:#1d4ed8;--line:#e2e8f0}
        html,body,.stApp{color-scheme:light}
        .stApp{background:#f8fafc;color:var(--ink)}
        .stApp h1,.stApp h2,.stApp h3,.stApp h4,
        .stApp [data-testid="stMarkdownContainer"] p,.stApp label p{color:var(--ink)}
        .block-container{padding-top:2.2rem;padding-bottom:4rem;max-width:1080px}
        footer{visibility:hidden}
        [data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
        [data-testid="stSidebar"] [role="radiogroup"] label{padding:.4rem .55rem;border-radius:8px}
        .brand{font-weight:800;font-size:1.25rem;letter-spacing:-.03em;color:var(--ink)}
        .brand-sub,.muted{color:var(--muted);font-size:.82rem}
        .eyebrow{color:var(--blue);font-size:.72rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase}
        .page-title{font-size:clamp(1.55rem,3vw,2.2rem);font-weight:800;letter-spacing:-.04em;margin:.2rem 0 .4rem;color:var(--ink)}
        .page-subtitle{color:var(--muted);line-height:1.55;max-width:760px;margin-bottom:1.2rem}
        .status-card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:.85rem 1rem;min-height:96px}
        .status-name{font-size:.84rem;color:var(--muted)}
        .status-value{font-size:1.08rem;font-weight:750;margin-top:.25rem;color:var(--ink)}
        .status-detail{color:var(--muted);font-size:.78rem;overflow-wrap:anywhere}
        .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#16a34a;margin-right:6px}
        .dot.off{background:#dc2626}
        .source{border-left:3px solid #93c5fd;background:#f8fbff;border-radius:0 9px 9px 0;padding:.65rem .85rem;margin:.5rem 0}
        .source a{color:var(--blue);font-weight:700;text-decoration:none}
        .source a:hover{text-decoration:underline}
        .source-url{color:var(--muted);font-size:.76rem;overflow-wrap:anywhere}
        .badge{display:inline-block;border-radius:999px;padding:.22rem .7rem;font-size:.78rem;font-weight:700;background:#f1f5f9;color:#334155}
        .badge.amber{background:#fffbeb;color:#92400e}
        .badge.green{background:#ecfdf3;color:#166534}
        .badge.red{background:#fef3f2;color:#b42318}
        .next-step{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:.7rem .9rem;color:#1e3a8a;margin:.6rem 0 1rem}
        .stButton button,.stFormSubmitButton button{border-radius:10px;font-weight:600}
        @media(max-width:700px){.block-container{padding-left:1rem;padding-right:1rem}.stButton button{width:100%}}
        </style>""",
        unsafe_allow_html=True,
    )


def header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{html.escape(kicker)}</div>'
        f'<div class="page-title">{html.escape(title)}</div>'
        f'<div class="page-subtitle">{html.escape(description)}</div>',
        unsafe_allow_html=True,
    )
    show_flash()


def badge(label: str, tone: str = "") -> str:
    return f'<span class="badge {tone}">{html.escape(label)}</span>'


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #
def settings_data() -> dict:
    if "provider_settings" not in st.session_state:
        st.session_state.provider_settings = load_settings()
    return st.session_state.provider_settings


def task_provider(task: str) -> str:
    return settings_data()[task]["provider"]


def task_model(task: str) -> str:
    return settings_data()[task]["model"]


def pname(value: Any) -> str:
    if not value or value == "not_used":
        return "Not used"
    if value == "fallback":
        return "Built-in fallback"
    return PROVIDER_LABELS.get(value, str(value).title())


def num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def flash(kind: str, message: str) -> None:
    """Queue a message that survives st.rerun() and shows on the next page draw."""
    st.session_state.setdefault("_flash", []).append((kind, message))


def show_flash() -> None:
    for kind, message in st.session_state.pop("_flash", []):
        {"success": st.success, "warning": st.warning, "error": st.error}.get(kind, st.info)(message)


def go(page: str) -> None:
    st.session_state.page = page


# --------------------------------------------------------------------------- #
# Backend calls
# --------------------------------------------------------------------------- #
def api_call(client: ApiClient, method: str, path: str, *, service: str = "The service",
             quiet_404: bool = False, **kwargs: Any) -> Any:
    """Call a backend and turn failures into plain-language messages."""
    try:
        return client.request(method, path, **kwargs)
    except BackendError as exc:
        if quiet_404 and exc.status_code == 404:
            return NOT_FOUND
        text = str(exc).lower()
        if exc.status_code == 422:
            message = "Some information is missing or not valid. Please check what you entered."
        elif exc.status_code == 404:
            message = "That item could not be found. It may have been removed."
        elif "timed out" in text:
            message = f"{service} took too long to answer. Please try again in a moment."
        elif exc.status_code is None:
            message = f"{service} can't be reached right now. Make sure it is running, then try again."
        else:
            message = "Something went wrong while handling your request. Please try again."
        st.error(message)
        with st.expander("Technical details (for developers)"):
            st.code(re.sub(r"(?i)(api[-_ ]?key|authorization)\s*[:=]\s*[^,; ]+", r"\1: [hidden]", str(exc)))
        return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health(targets: tuple) -> dict:
    """Use one aggregate check per unified backend URL.

    Legacy deployments may still provide a distinct URL per task. In that case
    each URL is checked once and its ordinary `/health` response is assigned
    to the task(s) configured for it.
    """
    grouped: dict[str, list[str]] = {}
    for task, url in targets:
        grouped.setdefault(url, []).append(task)

    def check(item: tuple) -> tuple:
        url, tasks = item
        try:
            data = ApiClient(url).health()
            data = data if isinstance(data, dict) else {}
            if any(task in data for task in ("task1", "task2", "task3")):
                result = {}
                for task in tasks:
                    task_data = data.get(task)
                    result[task] = {
                        "ok": isinstance(task_data, dict) and task_data.get("status") == "ok",
                        "data": task_data if isinstance(task_data, dict) else {},
                        "url": url,
                        **({} if isinstance(task_data, dict) else {"error": "Task status missing from unified health response"}),
                    }
                return result
            return {
                task: {"ok": data.get("status") == "ok", "data": data, "url": url}
                for task in tasks
            }
        except BackendError as exc:
            return {
                task: {"ok": False, "data": {}, "url": url, "error": str(exc)}
                for task in tasks
            }

    with ThreadPoolExecutor(max_workers=max(1, len(grouped))) as pool:
        checked = pool.map(check, grouped.items())
        return {task: info for group in checked for task, info in group.items()}


def status_cards(health: dict) -> None:
    for col, (task, svc) in zip(st.columns(3), SERVICES.items()):
        info = health[task]
        ok, data = info["ok"], info["data"]
        detail = "Not reachable. Start this service and refresh."
        if ok and task == "task1":
            detail = f"{int(num(data.get('indexed_chunks_count'))):,} help-article sections ready"
        elif ok and task == "task2":
            detail = "Ready to draft replies"
        elif ok and task == "task3":
            detail = f"{int(num(data.get('projects')))} saved project(s)"
        with col:
            st.markdown(
                f'<div class="status-card"><div class="status-name">'
                f'<span class="dot{"" if ok else " off"}"></span>{svc["icon"]} {html.escape(svc["name"])}</div>'
                f'<div class="status-value">{"Online" if ok else "Offline"}</div>'
                f'<div class="status-detail">{html.escape(detail)}</div></div>',
                unsafe_allow_html=True,
            )


def refresh_button(key: str) -> None:
    if st.button("↻ Refresh status", key=key):
        fetch_health.clear()
        st.rerun()


# --------------------------------------------------------------------------- #
# Shared display pieces
# --------------------------------------------------------------------------- #
def model_line(task: str) -> None:
    st.caption(f"Written by **{pname(task_provider(task))}** · you can change this in Settings")


def safe_url(value: Any) -> str:
    url = str(value or "").strip()
    return url if url.lower().startswith(("http://", "https://")) else ""


def sources(items: Any) -> None:
    items = [x for x in (items or []) if isinstance(x, dict)]
    if not items:
        st.caption("No help-center articles were linked to this answer.")
        return
    for x in items:
        title = html.escape(str(x.get("title") or "Help-center article"))
        url = safe_url(x.get("url"))
        link = (f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
                if url else f"<b>{title}</b>")
        meta = [html.escape(str(x["category"]))] if x.get("category") else []
        if isinstance(x.get("relevance_score"), (int, float)):
            meta.append(f"relevance {float(x['relevance_score']):.2f}")
        st.markdown(
            f'<div class="source">{link}'
            f'{"<br><span class=muted>" + " · ".join(meta) + "</span>" if meta else ""}'
            f'{"<br><span class=source-url>" + html.escape(url) + "</span>" if url else ""}</div>',
            unsafe_allow_html=True,
        )


def images(items: Any) -> None:
    for image in [x for x in (items or []) if safe_url(x)]:
        st.image(image, caption="Screenshot from the help center", width=520)


def bullets(title: str, values: Any) -> None:
    st.markdown(f"**{title}**")
    if not values:
        st.caption("Nothing recorded.")
        return
    for x in values:
        st.markdown(f"- {x if isinstance(x, str) else (x.get('statement') or x)}")


# --------------------------------------------------------------------------- #
# Home
# --------------------------------------------------------------------------- #
def home(api: dict, health: dict) -> None:
    header("AI Tech Consultant", "Make product work easier",
           "Three assistants in one place: find verified answers, prepare customer replies, "
           "and turn meeting notes into proposals. A person always reviews the final result.")

    st.subheader("Choose what you want to do")
    cards = [
        ("Knowledge Brain", "🔎", "Ask a product question",
         "Get an answer backed by help-center articles, with the sources shown."),
        ("Inbox Agent", "✉️", "Reply to a customer email",
         "The assistant drafts a reply. You edit it, approve it, then send."),
        ("Meeting to Proposal", "📝", "Plan a client project",
         "Paste meeting notes and get a scope, task estimates and a proposal to approve."),
    ]
    for col, (page, icon, title, desc) in zip(st.columns(3), cards):
        with col, st.container(border=True):
            st.markdown(f"#### {icon} {title}")
            st.write(desc)
            st.button(f"Open {page}", key=f"home_{page}", on_click=go, args=(page,), **STRETCH)

    st.subheader("Needs your attention")
    if not health["task2"]["ok"]:
        st.info("The Inbox Agent is offline, so email counts can't be shown right now.")
    else:
        emails = api_call(api["task2"], "GET", "/api/v1/emails", service="The Inbox Agent")
        if emails is not None:
            count = lambda *s: sum(isinstance(x, dict) and x.get("status") in s for x in emails)
            a, b, c = st.columns(3)
            a.metric("Waiting for review", count("draft", "needs_review"))
            b.metric("Approved, ready to send", count("approved"))
            c.metric("Sent (simulated)", count("sent"))
            if count("draft", "needs_review", "approved"):
                st.button("Go to the Inbox Agent", key="home_inbox", on_click=go, args=("Inbox Agent",))

    st.subheader("Service status")
    status_cards(health)
    refresh_button("refresh_home")


# --------------------------------------------------------------------------- #
# Knowledge Brain
# --------------------------------------------------------------------------- #
def use_example(text: str) -> None:
    st.session_state.kb_question = text


def knowledge(api: dict, health: dict) -> None:
    header("Knowledge Brain", "Ask a product question",
           "Type a question in plain English. You will get an answer plus the help-center "
           "articles it is based on, so you can check it before you act.")
    model_line("task1")
    if not health["task1"]["ok"]:
        st.warning("The Knowledge Brain service looks offline. You can still try, but it may fail.")

    st.caption("Not sure what to ask? Try an example:")
    for col, example in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
        col.button(example, key=f"ex_{example}", on_click=use_example, args=(example,), **STRETCH)

    with st.form("ask"):
        question = st.text_area("Your question", key="kb_question", height=100,
                                placeholder="Example: How do I create and manage email accounts?")
        with st.expander("Advanced options"):
            top = st.slider("How many help articles to check", 1, 10, 4,
                            help="More articles can give a fuller answer but may be slower.")
        ask = st.form_submit_button("Get answer", type="primary")

    if ask:
        if len(question.strip()) < 3:
            st.warning("Please type a question with at least three characters.")
        else:
            with st.spinner("Searching the documentation and preparing your answer…"):
                result = api_call(api["task1"], "POST", "/api/v1/query", service="The Knowledge Brain",
                                  json={"query": question.strip(), "top_k": top,
                                        "provider": task_provider("task1"), "model": task_model("task1")})
            if isinstance(result, dict):
                st.session_state.knowledge_result = {**result, "_question": question.strip()}

    result = st.session_state.get("knowledge_result")
    if not result:
        st.info("Your answer and its sources will appear here after you ask a question.")
        return

    st.divider()
    st.markdown(f"**Question:** {result.get('_question', '')}")
    if not result.get("has_answer"):
        st.warning("There isn't enough verified information to answer this confidently. "
                   "Try rephrasing with specific feature names.")
    elif result.get("fallback_used"):
        st.warning("Relevant articles were found, but the AI model couldn't write the answer, "
                   "so the text below comes directly from the articles. Please read it carefully.")
    else:
        st.success("Answer ready. It is based on the articles listed below.")

    with st.container(border=True):
        st.markdown(result.get("answer") or "No answer was returned.")

    st.markdown("#### Sources")
    sources(result.get("citations", []))
    images(result.get("screenshots", []))
    if result.get("ticket_id"):
        st.info(f"An expert ticket was created for follow-up: `{result['ticket_id']}`")

    with st.expander("Technical details (for developers)"):
        writer = "Built-in fallback" if result.get("fallback_used") else pname(task_provider("task1"))
        st.write(f"Written by: {writer} · model `{result.get('model_used', 'not reported')}`")
        if result.get("generation_error"):
            st.code(str(result["generation_error"]))


# --------------------------------------------------------------------------- #
# Inbox Agent
# --------------------------------------------------------------------------- #
def email_status(status: Any, error: bool = False) -> tuple[str, str]:
    if error:
        return "Reply could not be generated", "red"
    return EMAIL_STATUS.get(status, (str(status or "unknown").replace("_", " ").capitalize(), ""))


NEXT_STEP = {
    "draft": "Read the customer's message, edit the draft reply, tick “Approve”, then save.",
    "needs_review": "Read the customer's message, edit the draft reply, tick “Approve”, then save.",
    "approved": "This reply is approved. Click “Send reply” when you are ready (sending is simulated).",
}


def inbox(api: dict, health: dict) -> None:
    header("Inbox Agent", "Reply to customer emails",
           "The assistant drafts a reply, you review and approve it, then send. "
           "Sending is simulated, so no real customer is ever contacted.")
    model_line("task2")

    emails = api_call(api["task2"], "GET", "/api/v1/emails", service="The Inbox Agent")
    if emails is None:
        refresh_button("refresh_inbox")
        return
    emails = [x for x in emails if isinstance(x, dict) and x.get("email_id")]

    # Kept above the selector so a new email can be selected in the same run.
    with st.expander("➕ Add a customer email to answer", expanded=not emails):
        with st.form("process_email", clear_on_submit=True):
            c1, c2 = st.columns(2)
            sender = c1.text_input("Customer email address", "customer@example.test")
            subject = c2.text_input("Subject", "Question about my website")
            body = st.text_area("Message from the customer", height=130)
            submit = st.form_submit_button("Draft a reply", type="primary")
        if submit:
            if not all((sender.strip(), subject.strip(), body.strip())):
                st.warning("Please fill in the email address, subject and message.")
            else:
                with st.spinner("Drafting a reply…"):
                    result = api_call(api["task2"], "POST", "/api/v1/emails/process", service="The Inbox Agent",
                                      json={"email_id": f"ui_{uuid.uuid4().hex[:10]}", "sender": sender.strip(),
                                            "subject": subject.strip(), "body": body.strip(),
                                            "provider": task_provider("task2"), "model": task_model("task2"),
                                            "knowledge_provider": task_provider("task1"),
                                            "knowledge_model": task_model("task1")})
                if isinstance(result, dict) and result.get("email_id"):
                    st.session_state.email_id = result["email_id"]
                    flash("success", "A draft reply is ready below. Please review it.")
                    st.rerun()

    if not emails:
        st.info("No emails yet. Add one above to see the assistant draft a reply.")
        return

    ids = [x["email_id"] for x in emails]
    labels = {x["email_id"]: f"{x.get('subject') or '(no subject)'}  ·  {email_status(x.get('status'))[0]}"
              for x in emails}
    if st.session_state.get("email_id") not in ids:
        st.session_state.email_id = ids[0]
    waiting = sum(x.get("status") in ("draft", "needs_review") for x in emails)
    st.selectbox("Choose an email", ids, key="email_id", format_func=labels.get,
                 help="Pick an email to read it and review the drafted reply.")
    st.caption(f"{len(emails)} email(s) · {waiting} waiting for review")

    eid = st.session_state.email_id
    item = api_call(api["task2"], "GET", f"/api/v1/emails/{eid}", service="The Inbox Agent")
    if not item:
        return

    label, tone = email_status(item.get("status"), bool(item.get("provider_error")))
    st.divider()
    st.subheader(item.get("subject") or "Customer email")
    st.markdown(f"From **{html.escape(str(item.get('sender', '')))}** · Topic: {item.get('category') or 'unknown'}"
                f" &nbsp; {badge(label, tone)}", unsafe_allow_html=True)
    if item.get("status") in NEXT_STEP:
        st.markdown(f'<div class="next-step"><b>Next step:</b> {NEXT_STEP[item["status"]]}</div>',
                    unsafe_allow_html=True)

    if item.get("provider_error"):
        st.warning("The AI model couldn't write a reply this time and nothing was sent. "
                   "You can write the reply yourself below.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Customer's message")
        with st.container(border=True):
            st.write(item.get("body", ""))
        if (item.get("knowledge") or {}).get("citations"):
            with st.expander("Help articles used for the draft"):
                sources(item["knowledge"]["citations"])

    with right:
        if item.get("status") == "sent":
            st.markdown("#### Reply that was sent")
            with st.container(border=True):
                st.write(item.get("final_reply") or "")
            st.success("This reply was accepted by the simulated sender.")
            with st.form(f"feedback_{eid}"):
                notes = st.text_area("Feedback on this reply (optional)", height=80,
                                     help="Tell the team what worked or what to improve.")
                if st.form_submit_button("Save feedback"):
                    if api_call(api["task2"], "POST", f"/api/v1/emails/{eid}/feedback",
                                service="The Inbox Agent", json={"notes": notes}) is not None:
                        st.success("Feedback saved. Thank you!")
        else:
            st.markdown("#### Your reply")
            with st.form(f"review_{eid}"):
                reply = st.text_area("Reply (edit as needed)", height=210,
                                     value=item.get("final_reply") or item.get("original_draft") or "")
                notes = st.text_area("Notes for your team (optional)", value=item.get("human_notes") or "", height=70)
                approve = st.checkbox("I approve this reply for sending", value=item.get("status") == "approved")
                save = st.form_submit_button("Save reply", type="primary")
            if save:
                saved = api_call(api["task2"], "POST", f"/api/v1/emails/{eid}/review", service="The Inbox Agent",
                                 json={"final_reply": reply, "approve": approve, "human_notes": notes})
                if saved is not None:
                    flash("success", "Reply saved and approved. You can send it now." if approve
                          else "Reply saved. Tick “approve” when it is ready to send.")
                    st.rerun()
            if item.get("status") == "approved":
                if st.button("Send reply (simulated)", type="primary", **STRETCH):
                    if api_call(api["task2"], "POST", f"/api/v1/emails/{eid}/send",
                                service="The Inbox Agent") is not None:
                        flash("success", "The simulated sender accepted the email.")
                        st.rerun()

    with st.expander("Technical details (for developers)"):
        st.write(f"Draft written by: {pname(item.get('provider', 'not_used'))} · "
                 f"model `{item.get('model_used', 'not_used')}`")
        if item.get("provider_error"):
            st.code(str(item["provider_error"]))


# --------------------------------------------------------------------------- #
# Meeting to Proposal
# --------------------------------------------------------------------------- #
def proposal_status(status: Any) -> tuple[str, str]:
    return PROPOSAL_STATUS.get(status, (str(status or "pending").replace("_", " ").capitalize(), ""))


def load_project(api: dict, name: str) -> Any:
    return api_call(api["task3"], "GET", f"/api/v1/projects/{name}", service="The Meeting service", quiet_404=True)


def meeting(api: dict, health: dict) -> None:
    header("Meeting to Proposal", "Turn meeting notes into a project plan",
           "Paste what was discussed. The assistant lists the goals, checks what the product "
           "already supports, estimates the work and prepares a proposal for you to approve.")
    model_line("task3")
    if not health["task3"]["ok"]:
        st.warning("The Meeting to Proposal service looks offline. You can still try, but it may fail.")

    st.markdown("#### 1 · Name the project")
    st.session_state.setdefault("project_id", "client-project")
    project = st.text_input("Project name", key="project_id",
                            help="Use the same name for follow-up meetings so changes can be compared.").strip()
    if st.button("Load saved versions of this project") and project:
        saved = load_project(api, project)
        if saved is NOT_FOUND:
            st.info("No saved project with this name yet. It will be created with your first meeting.")
            st.session_state.pop("meeting_project", None)
        elif saved:
            st.session_state.meeting_project, st.session_state.meeting_project_name = saved, project

    saved = st.session_state.get("meeting_project")
    if saved and st.session_state.get("meeting_project_name") != project:
        saved = None
    versions = (saved or {}).get("versions") or []
    previous = None
    if versions:
        st.caption(f"This project has {len(versions)} saved version(s).")
        previous = st.selectbox("Compare with an earlier version (optional)",
                                [None] + [v.get("version") for v in versions],
                                format_func=lambda x: "Don't compare" if x is None else f"Version {x}")

    st.markdown("#### 2 · Add the meeting notes")
    transcript = st.text_area("Type or paste the notes", height=220,
                              placeholder="Paste at least 20 characters of notes or a transcript…")
    upload = st.file_uploader("…or upload a text file (.txt)", type=["txt"])
    if upload:
        transcript = upload.getvalue().decode("utf-8", errors="replace")
        st.caption(f"Using the uploaded file ({len(transcript):,} characters). Remove it to use the typed notes instead.")

    st.markdown("#### 3 · Create the plan")
    if st.button("Create scope and proposal", type="primary"):
        if not project:
            st.warning("Please enter a project name.")
        elif len(transcript.strip()) < 20:
            st.warning("Please provide at least 20 characters of meeting notes.")
        else:
            payload = {"transcript": transcript.strip(), "provider": task_provider("task3"),
                       "model": task_model("task3"), "knowledge_provider": task_provider("task1"),
                       "knowledge_model": task_model("task1")}
            if previous is not None:
                payload["previous_version"] = previous
            with st.spinner("Reading the notes and checking product capabilities. This can take a minute…"):
                result = api_call(api["task3"], "POST", f"/api/v1/projects/{project}/meetings",
                                  service="The Meeting service", json=payload)
            if isinstance(result, dict):
                st.session_state.meeting_result = result
                st.session_state.meeting_result_project = project
                refreshed = load_project(api, project)
                if isinstance(refreshed, dict):
                    st.session_state.meeting_project, st.session_state.meeting_project_name = refreshed, project
                st.success("Done. A new version of the project was saved. See the results below.")

    result = st.session_state.get("meeting_result")
    if result:
        show_meeting_result(api, result, st.session_state.get("meeting_result_project") or project)


def show_meeting_result(api: dict, result: dict, project: str) -> None:
    analysis, scope = result.get("analysis") or {}, result.get("scope") or {}
    proposal = result.get("proposal") or {}
    version = result.get("version")
    label, tone = proposal_status(proposal.get("status", "pending_human_approval"))

    st.divider()
    st.subheader(f"Results · {project} · Version {version}")
    m1, m2 = st.columns(2)
    m1.metric("Estimated effort", f"{num(proposal.get('estimated_hours')):g} hours")
    m2.markdown(f"**Proposal status**<br>{badge(label, tone)}", unsafe_allow_html=True)

    names = ["Overview", "Scope & tasks", "Proposal"]
    if result.get("change_summary"):
        names.append("What changed")
    project_data = st.session_state.get("meeting_project") or {}
    if project_data.get("versions"):
        names.append("Version history")
    tabs = dict(zip(names, st.tabs(names)))

    with tabs["Overview"]:
        a, b = st.columns(2, gap="large")
        with a:
            bullets("Client goals", analysis.get("goals"))
            bullets("Features the client asked for", analysis.get("requested_features"))
            bullets("Questions to clarify with the client", analysis.get("unanswered_questions"))
        with b:
            st.markdown("**Does the product already support this?**")
            feature_names = {x.get("id"): x.get("statement") for x in analysis.get("requested_features") or []
                             if isinstance(x, dict)}
            checks = scope.get("feature_checks") or []
            if not checks:
                st.caption("No feature checks were returned.")
            for check in checks:
                supported = check.get("supported")
                text = ("✅ Supported by existing settings" if supported is True
                        else "🛠️ May need custom development" if supported is False
                        else "❓ More information needed")
                feature = feature_names.get(check.get("requirement_id")) or "Requested feature"
                with st.expander(f"{feature} · {text}"):
                    st.write(check.get("answer") or "No verified product answer was returned.")
                    sources(check.get("citations"))

    with tabs["Scope & tasks"]:
        a, b = st.columns(2, gap="large")
        with a:
            bullets("Deliverables", scope.get("deliverables"))
            bullets("Assumptions", scope.get("assumptions"))
        with b:
            bullets("Not included", scope.get("exclusions"))
            bullets("Open questions", scope.get("open_questions"))
        st.markdown("**Developer tasks and effort**")
        rows = [{"Task": x.get("task_id"), "Description": x.get("description"),
                 "Estimated hours": x.get("estimated_hours"), "Depends on": ", ".join(x.get("dependencies") or [])}
                for x in scope.get("tasks") or [] if isinstance(x, dict)]
        if rows:
            st.dataframe(rows, hide_index=True, **STRETCH)
        else:
            st.caption("No tasks were returned.")

    with tabs["Proposal"]:
        generator = "Built-in fallback" if proposal.get("provider_error") else pname(proposal.get("provider"))
        st.caption(f"{num(proposal.get('estimated_hours')):g} estimated hours · written by {generator}")
        with st.container(border=True):
            st.write(proposal.get("overview") or "No overview was returned.")
        bullets("What the proposal delivers", proposal.get("deliverables"))
        if proposal.get("provider_error"):
            st.warning("The AI model couldn't write the overview, so a simpler built-in version is shown. "
                       "Please review it carefully before approving.")
            with st.expander("Technical details (for developers)"):
                st.code(str(proposal["provider_error"]))
        if proposal.get("status", "pending_human_approval") == "pending_human_approval":
            st.markdown("**Your decision**")
            c1, c2 = st.columns(2)
            for col, approved, text, kind in ((c1, True, "Approve proposal", "primary"),
                                              (c2, False, "Reject proposal", "secondary")):
                if col.button(text, type=kind, key=f"decide_{approved}", **STRETCH):
                    done = api_call(api["task3"], "POST", f"/api/v1/projects/{project}/versions/{version}/approval",
                                    service="The Meeting service", json={"approved": approved})
                    if done is not None:
                        result.setdefault("proposal", {})["status"] = "approved" if approved else "rejected"
                        refreshed = load_project(api, project)
                        if isinstance(refreshed, dict):
                            st.session_state.meeting_project = refreshed
                            st.session_state.meeting_project_name = project
                        flash("success", "Proposal approved." if approved else "Proposal rejected.")
                        st.rerun()

    if "What changed" in tabs:
        diff = result["change_summary"]
        with tabs["What changed"]:
            a, b = st.columns(2, gap="large")
            with a:
                bullets("Stayed the same", diff.get("unchanged"))
                bullets("Added", diff.get("added"))
            with b:
                bullets("Removed", diff.get("removed"))
                bullets("Outside the original scope", diff.get("outside_original_scope"))
            st.info(f"Estimated effort changed from {num(diff.get('old_hours')):g} to "
                    f"{num(diff.get('new_hours')):g} hours ({num(diff.get('hours_delta')):+.1f}).")

    if "Version history" in tabs:
        with tabs["Version history"]:
            for v in project_data["versions"]:
                v_label, v_tone = proposal_status((v.get("proposal") or {}).get("status", "pending_human_approval"))
                hours = num((v.get("proposal") or {}).get("estimated_hours"))
                st.markdown(f"**Version {v.get('version')}** · {hours:g} hours · {badge(v_label, v_tone)}",
                            unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def on_provider_change(task: str) -> None:
    choice = st.session_state[f"setting_{task}"]
    data = settings_data()
    data[task] = {"provider": choice, "model": model_for(task, choice)}
    name = SERVICES[task]["name"]
    try:
        save_settings(data)
        flash("success", f"{name} will now use {pname(choice)} for new requests.")
    except OSError:
        flash("warning", f"{name} switched to {pname(choice)}, but the choice couldn't be saved to disk.")


def settings(api: dict, health: dict) -> None:
    header("Settings", "Choose the AI model for each assistant",
           "Each assistant can use a different AI provider. Changes apply to new requests only. "
           "API keys are never typed here; they stay in the server's environment variables.")
    data = settings_data()
    blurbs = {"task1": "Answers product questions.", "task2": "Drafts customer replies.",
              "task3": "Creates scopes and proposals."}
    for task, svc in SERVICES.items():
        with st.container(border=True):
            st.markdown(f"#### {svc['icon']} {svc['name']}")
            st.caption(blurbs[task])
            st.session_state.setdefault(f"setting_{task}", data[task]["provider"])
            st.radio("AI provider", ["claude", "groq"], key=f"setting_{task}", horizontal=True,
                     on_change=on_provider_change, args=(task,),
                     format_func=lambda x: "Claude (recommended)" if x == "claude" else "Groq (for testing)")
            st.caption(f"Model for new requests: `{data[task]['model']}`")

    st.subheader("Service status")
    status_cards(health)
    refresh_button("refresh_settings")
    with st.expander("Service addresses (for developers)"):
        for task, svc in SERVICES.items():
            st.write(f"{svc['name']}: `{health[task]['url']}`")


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
PAGE_FUNCS = {"Home": home, "Knowledge Brain": knowledge, "Inbox Agent": inbox,
              "Meeting to Proposal": meeting, "Settings": settings}


def sidebar(health: dict) -> None:
    with st.sidebar:
        st.markdown('<div class="brand">✦ AI Tech Consultant</div>'
                    '<div class="brand-sub">Clear tools for client work</div>', unsafe_allow_html=True)
        st.write("")
        st.radio("Go to", PAGES, key="page", label_visibility="collapsed",
                 format_func=lambda p: f"{PAGE_ICONS[p]}  {p}")
        st.divider()
        online = sum(info["ok"] for info in health.values())
        if online == len(health):
            st.markdown('<span class="dot"></span> All services online', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="dot off"></span> {len(health) - online} service(s) offline',
                        unsafe_allow_html=True)
        st.write("")
        st.caption("AI model in use")
        for task, svc in SERVICES.items():
            st.caption(f"{svc['name']}: {pname(task_provider(task))}")


def main() -> None:
    css()
    settings_data()
    api = clients()
    st.session_state.setdefault("page", "Home")
    health = fetch_health(tuple((task, client.base_url) for task, client in api.items()))
    sidebar(health)
    PAGE_FUNCS.get(st.session_state.page, home)(api, health)


if __name__ == "__main__":
    main()
