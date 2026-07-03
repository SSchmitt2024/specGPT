"""
Phase 2 - Step 2.5: Web Application (FastAPI Backend)

Exposes the full retrieval + generation pipeline as a web service, gated
by a shared-password login (see src/pipeline/auth.py for the threat model).

Endpoints (auth-gated unless marked public):
  GET  /healthz       - public liveness check (Railway/k8s healthcheck)
  GET  /login         - public; renders the password form
  POST /login         - public; validates password, sets session cookie
  POST /logout        - public; clears session cookie
  GET  /              - gated; serves the web UI (or redirects to /login)
  POST /api/query     - gated; runs the pipeline
  GET  /api/config    - gated; returns default PipelineConfig

Required env vars:
  APP_PASSWORD     - plaintext shared password (hashed at startup, wiped from memory)
  SESSION_SECRET   - ≥16-byte string used as HMAC key for session cookies
  SUPABASE_URL / SUPABASE_KEY / VOYAGE_API_KEY / ANTHROPIC_API_KEY - pipeline backends

Optional env vars (model backends):
  GEMINI_API_KEY   - required when using any gemini-* model

Optional env vars:
  DEBUG_PIPELINE   - "1" to include full trace in responses (default: off)
  PORT             - server port (default: 8000)
  HOST             - server host (default: 127.0.0.1)
  COOKIE_SECURE    - "0" to allow non-HTTPS cookies for local dev (default: on)
  LOG_LEVEL        - Python logging level (default: INFO)

Run:
  python -m src.pipeline.app
  Then visit http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path


def _load_dotenv(path: str = ".env") -> int:
    """Populate `os.environ` from a KEY=value file. Production env vars win.

    Only meaningful for local dev - Railway/Cloudflare/etc. inject vars
    directly and there is no .env file to find. Without this, the Anthropic
    SDK (which reads `ANTHROPIC_API_KEY` from `os.environ`) and any other
    consumer of plain `os.environ` can't see `.env` settings, even though
    src/pipeline/search.py's `_load_env_var` happily reads them via its own
    helper. Loading once at import unifies the two.
    """
    env_path = Path(path)
    if not env_path.exists():
        return 0
    loaded = 0
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:  # explicit env wins
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
        loaded += 1
    return loaded


# Must run before anything imports `Anthropic()` or reads ANTHROPIC_API_KEY.
_load_dotenv()


from fastapi import Cookie, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline.auth import (
    SESSION_COOKIE,
    SESSION_LIFETIME_SECONDS,
    LoginThrottle,
    create_session_token,
    hash_password,
    verify_password,
    verify_session_token,
)
from src.pipeline.generator import DeepThoughtUnreachableError
from src.pipeline.orchestrator import (
    ALL_SPECS,
    CONCRETE_SPEC_IDS,
    GenerationError,
    orchestrate,
    PipelineConfig,
    PRESETS,
    DEFAULT_PRESET,
)
from src.pipeline.retriever import load_field_index, load_tables_by_figure


def _generation_error_detail(e: GenerationError, request_id: str, *, include_trace: bool) -> dict:
    """Build the JSON `detail` for a 502 generation failure.

    When the cause is a known network-config error (DeepThought off-VPN),
    include a `message` field with a human-readable hint so the UI can show
    something more useful than "Bad Gateway".
    """
    detail: dict = {
        "error": "generation_failed",
        "request_id": request_id,
        "cause_type": type(e.cause).__name__,
    }
    if isinstance(e.cause, DeepThoughtUnreachableError):
        detail["message"] = str(e.cause)
    if include_trace:
        detail["pipeline_trace"] = e.trace
    return detail


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# ============================================================================
# Auth: read + hash + wipe at module load (fail loud if missing)
# ============================================================================

def _bootstrap_auth() -> tuple[str, bytes, bool]:
    """
    Read auth env vars, hash the password, return ``(hash, secret_bytes, cookie_secure)``.

    Raises RuntimeError at import time if either secret is missing so the
    server never starts in a half-authenticated state. The plaintext
    password is dropped from the local namespace once hashed.
    """
    plain = os.getenv("APP_PASSWORD")
    secret = os.getenv("SESSION_SECRET")
    if not plain:
        raise RuntimeError(
            "APP_PASSWORD is not set. Pick a password and set it in the environment "
            "(or .env). The plaintext is hashed at startup and never written to disk."
        )
    if not secret or len(secret.encode("utf-8")) < 16:
        raise RuntimeError(
            "SESSION_SECRET must be set to a string of at least 16 bytes. Generate one with:\n"
            "  python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    hashed = hash_password(plain)
    # Be defensive about leaving the plaintext in module-globals or env.
    # (os.environ.pop is best-effort: any subprocess we spawned before now
    # would already have inherited it. For a single-process app it suffices.)
    os.environ.pop("APP_PASSWORD", None)
    return hashed, secret.encode("utf-8"), os.getenv("COOKIE_SECURE", "1").lower() in ("1", "true", "yes")


_PASSWORD_HASH, _SESSION_SECRET_BYTES, _COOKIE_SECURE = _bootstrap_auth()
_throttle = LoginThrottle()


# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request body for /api/query endpoint."""
    query: str
    config: dict | None = None
    debug: bool = True
    # When true, after the normal pipeline finishes the orchestrator runs a
    # gap-analysis LLM call, dispatches follow-up retrievals for any
    # under-covered aspects, merges + re-reranks the expanded chunk pool,
    # and regenerates with the agentic model. Adds ~30-60s + Opus cost.
    agentic: bool = False


class QueryResponse(BaseModel):
    """Response from /api/query endpoint."""
    query: str
    answer: str
    citations: list[dict]
    config: dict
    pipeline_trace: list[dict] | None = None
    latency_ms: float
    tokens_used: dict | None = None
    agentic: bool = False
    # Present when agentic mode is off and `auto_gap_check` ran. The UI
    # uses this to surface a one-click "run agentic refinement" prompt
    # when the model would have requested more context.
    gap_hint: dict | None = None
    # Opaque handle the UI passes to /api/refine to resume from the
    # already-computed first-pass state (no Stages 1-4 redo).
    request_id: str | None = None
    # Figures present in the retrieved context, so the UI can turn inline
    # "Figure N" mentions in the answer into clickable links that open the
    # spec PDF at that figure. Each: {figure_number, spec, pdf_pages, caption,
    # section_id}. Only figures we have a page for are included.
    figures: list[dict] = []


class FlagAnswerRequest(BaseModel):
    """Request body for /api/flag-answer.

    A snapshot of the last QueryResponse the user was viewing plus an optional
    free-text reason. Nothing is re-run server-side; the UI just POSTs what it
    already holds so a reviewer can reproduce and triage the flagged answer.
    """
    query: str
    answer: str
    config: dict
    pipeline_trace: list[dict] | None = None
    citations: list[dict] | None = None
    latency_ms: float | None = None
    tokens_used: dict | None = None
    agentic: bool = False
    reason: str | None = None          # optional user explanation


class DevNoteRequest(BaseModel):
    """Request body for POST /api/dev-notes (free-form dev scratchpad)."""
    body: str


class RefineRequest(BaseModel):
    """Request body for /api/refine - resumes the prior /api/query call by
    request_id, runs only Stage 5 (gap analysis + targeted fetch + follow-up
    retrieval + re-rerank + Opus regen) against the cached first-pass state.
    """
    request_id: str
    config: dict | None = None
    debug: bool = True


# ============================================================================
# Refine cache - in-process LRU mapping request_id → first-pass state.
#
# Cleared on process restart; bounded so a noisy session can't OOM the box.
# Single-worker uvicorn is the supported deploy, so the in-process cache is
# fine; if we ever scale to multiple workers we'll need an external store.
# ============================================================================
_REFINE_CACHE: OrderedDict[str, dict] = OrderedDict()
_REFINE_CACHE_LOCK = threading.Lock()
_REFINE_CACHE_MAX = 64


def _refine_cache_set(request_id: str, state: dict) -> None:
    with _REFINE_CACHE_LOCK:
        _REFINE_CACHE[request_id] = state
        _REFINE_CACHE.move_to_end(request_id)
        while len(_REFINE_CACHE) > _REFINE_CACHE_MAX:
            _REFINE_CACHE.popitem(last=False)


def _refine_cache_get(request_id: str) -> dict | None:
    with _REFINE_CACHE_LOCK:
        state = _REFINE_CACHE.get(request_id)
        if state is not None:
            _REFINE_CACHE.move_to_end(request_id)
        return state


# ============================================================================
# FastAPI App Setup
# ============================================================================

# Trace is only returned to authenticated callers (the API endpoints that
# include it all require a valid session). Default OFF to keep production
# responses small; flip on to populate the in-UI pipeline flow chart.
DEBUG_PIPELINE = os.getenv("DEBUG_PIPELINE", "0").lower() in ("1", "true", "yes")

# Separate switch from DEBUG_PIPELINE: /docs, /redoc, /openapi.json are auto-
# added by FastAPI with NO auth. They don't let anyone call gated endpoints,
# but they enumerate the API surface to unauthenticated visitors - different
# blast radius than the trace, so it gets its own variable. Off by default.
_EXPOSE_API_DOCS = os.getenv("EXPOSE_API_DOCS", "0").lower() in ("1", "true", "yes")

# Specifications the UI can search. The `id` is the value stored on every
# spec_chunks / lookup row (see scripts/load_lookup_data.py + indexer.py) and
# the value the retrievers filter on. `url` is the official nvmexpress.org PDF
# the frontend deep-links to (with a #page=N fragment) when a citation/figure is
# clicked - we reference the spec in place rather than re-hosting it. Add a row
# here when a new transport corpus (e.g. RDMA/TCP) is ingested.
AVAILABLE_SPECS = [
    {"id": "base", "label": "Base Specification", "version": "2.3", "url": "https://nvmexpress.org/wp-content/uploads/NVM-Express-Base-Specification-Revision-2.3-2025.08.01-Ratified.pdf"},
    {"id": "pcie", "label": "PCIe Transport", "version": "1.3", "url": "https://nvmexpress.org/wp-content/uploads/NVM-Express-NVMe-over-PCIe-Transport-Specification-Revision-1.3-2025.08.01-Ratified.pdf"},
    {"id": "command", "label": "NVM Command Set", "version": "1.2", "url": "https://nvmexpress.org/wp-content/uploads/NVM-Express-NVM-Command-Set-Specification-Revision-1.2-2025.08.01-Ratified.pdf"},
    # Sentinel: search every corpus at once (orchestrator.ALL_SPECS). There is
    # no single PDF, so url is None; citation deep-links use each chunk's own
    # spec provenance instead. Keep this entry LAST so the frontend's
    # _specData[0] fallback for spec-less citations stays the base spec.
    {"id": "all", "label": "All Specifications", "version": "", "url": None},
]
_VALID_SPEC_IDS = {s["id"] for s in AVAILABLE_SPECS}
# The all-specs merge in the orchestrator expands ALL_SPECS to
# CONCRETE_SPEC_IDS; fail loudly at import time if this list drifts from it.
assert _VALID_SPEC_IDS == set(CONCRETE_SPEC_IDS) | {ALL_SPECS}, (
    f"AVAILABLE_SPECS ids {_VALID_SPEC_IDS} out of sync with "
    f"orchestrator.CONCRETE_SPEC_IDS {CONCRETE_SPEC_IDS}"
)


app = FastAPI(
    title="specGPT Pipeline",
    description="NVMe Specification Q&A with Full Pipeline Visibility",
    version="2.0",
    docs_url="/docs" if _EXPOSE_API_DOCS else None,
    redoc_url="/redoc" if _EXPOSE_API_DOCS else None,
    openapi_url="/openapi.json" if _EXPOSE_API_DOCS else None,
)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ============================================================================
# Auth helpers + endpoints
# ============================================================================

def _client_ip(request: Request) -> str:
    """Best-effort client IP. Trusts X-Forwarded-For only if explicitly opted in
    via TRUST_PROXY_HEADERS=1 - otherwise the throttle can be bypassed by
    spoofing the header."""
    if os.getenv("TRUST_PROXY_HEADERS", "0").lower() in ("1", "true", "yes"):
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_LIFETIME_SECONDS,
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def require_auth(
    request: Request,
    specgpt_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> bool:
    """FastAPI dependency: 401 (JSON) if the session cookie is missing or invalid.

    Use on API routes. For HTML routes prefer the explicit cookie check +
    303 redirect to /login (see `frontend`).
    """
    del request  # only here to satisfy callers that want the request context
    if verify_session_token(specgpt_session, _SESSION_SECRET_BYTES):
        return True
    raise HTTPException(status_code=401, detail={"error": "auth_required"})


def _login_html(error: str | None = None, *, next_path: str = "/") -> str:
    """Standalone login page. Same look-and-feel as the main UI."""
    import html as _html
    error_html = (
        f'<div class="error">{_html.escape(error)}</div>' if error else ""
    )
    next_attr = _html.escape(next_path or "/", quote=True)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>specGPT - sign in</title>
  <link rel="icon" type="image/png" href="/static/favicon.png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #ffffff; --bg-soft: #fbfbfc; --bg-muted: #f1f1f4;
      --border: #e7e7eb; --border-strong: #dadadf;
      --text: #18181b; --text-muted: #52525b; --text-subtle: #71717a; --text-faint: #a1a1aa;
      --accent: #2d68e6; --danger: #dc2626;
      --font-sans: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      --font-mono: 'Geist Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #141417; --bg-soft: #0c0c0e; --bg-muted: #1d1d23;
        --border: #26262d; --border-strong: #33333c;
        --text: #f4f4f5; --text-muted: #a1a1aa; --text-subtle: #8a8a93; --text-faint: #5f5f68;
        --accent: #5b8cff; --danger: #f87171;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: var(--font-sans);
            font-feature-settings: "cv11", "ss01";
            background: var(--bg-soft); color: var(--text);
            font-size: 14px; line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            letter-spacing: -0.005em;
            margin: 0; min-height: 100vh;
            display: flex; align-items: center; justify-content: center; }}
    .card {{ background: var(--bg); padding: 28px 26px;
             border: 1px solid var(--border); border-radius: 8px;
             box-shadow: 0 1px 2px rgba(0,0,0,0.04);
             width: 340px; }}
    .brand {{ display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }}
    .brand-mark {{ width: 22px; height: 22px; border-radius: 5px;
                   background: var(--accent); color: white;
                   display: grid; place-items: center;
                   font-family: var(--font-mono);
                   font-size: 11px; font-weight: 700; letter-spacing: -0.02em; }}
    h1 {{ font-size: 14px; font-weight: 600; margin: 0;
          color: var(--text); letter-spacing: -0.01em; }}
    p.sub {{ font-size: 12.5px; color: var(--text-subtle);
             margin: 0 0 18px; line-height: 1.45; }}
    label {{ display: block; font-size: 10.5px; font-weight: 500;
             text-transform: uppercase; color: var(--text-subtle);
             margin-bottom: 5px; letter-spacing: 0.05em; }}
    input[type=password] {{ width: 100%; padding: 9px 11px;
                             border: 1px solid var(--border);
                             border-radius: 4px; font-size: 14px;
                             background: var(--bg); color: var(--text);
                             font-family: var(--font-mono);
                             letter-spacing: -0.005em;
                             transition: border-color 0.12s, box-shadow 0.12s; }}
    input[type=password]:focus {{ outline: none;
                                   border-color: var(--accent);
                                   box-shadow: 0 0 0 3px rgba(28,25,23,0.08); }}
    button {{ margin-top: 16px; width: 100%; padding: 10px 12px;
              background: var(--accent); color: white; border: 1px solid var(--accent);
              border-radius: 4px; font-size: 13px; font-weight: 500;
              cursor: pointer; font-family: inherit;
              letter-spacing: -0.005em;
              transition: background 0.12s, border-color 0.12s; }}
    button:hover {{ background: #000; border-color: #000; }}
    .error {{ background: #fef2f2; color: var(--danger);
              padding: 9px 12px; border-radius: 4px;
              border: 1px solid #fecaca;
              font-size: 12.5px; margin-bottom: 14px; }}
  </style>
</head>
<body>
  <form class="card" method="post" action="/login" autocomplete="off">
    <div class="brand">
      <div class="brand-mark">sG</div>
      <h1>specGPT</h1>
    </div>
    <p class="sub">Enter the access password to continue.</p>
    {error_html}
    <input type="hidden" name="next" value="{next_attr}">
    <label for="password">Password</label>
    <input id="password" type="password" name="password" autocomplete="current-password"
           required autofocus>
    <button type="submit">Sign in</button>
  </form>
</body>
</html>"""


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/healthz")
async def healthz() -> dict:
    """Liveness check - intentionally unauthenticated so external healthchecks work.

    Does NOT exercise Supabase/Voyage/Anthropic - those have cost. It just
    confirms the process is up and importing. Add a /readyz if you ever
    want a deep healthcheck.
    """
    return {"ok": True}


@app.get("/login", response_class=HTMLResponse)
async def login_page(next: str = "/") -> HTMLResponse:
    return HTMLResponse(_login_html(next_path=next))


@app.post("/login")
async def login(
    request: Request,
    password: str = Form(...),
    next: str = Form(default="/"),
) -> Response:
    ip = _client_ip(request)
    delay = _throttle.delay_for(ip)
    if delay > 0:
        await asyncio.sleep(delay)

    if not verify_password(password, _PASSWORD_HASH):
        _throttle.record_failure(ip)
        logger.info("Failed login from %s", ip)
        return HTMLResponse(
            _login_html("Incorrect password.", next_path=next),
            status_code=401,
        )

    _throttle.clear(ip)
    # Sanitise the next-URL so a phishing redirect can't pivot off the login.
    safe_next = next if isinstance(next, str) and next.startswith("/") and not next.startswith("//") else "/"
    token = create_session_token(_SESSION_SECRET_BYTES)
    resp = RedirectResponse(url=safe_next, status_code=303)
    _set_session_cookie(resp, token)
    return resp


@app.api_route("/logout", methods=["GET", "POST"])
async def logout() -> Response:
    resp = RedirectResponse(url="/login", status_code=303)
    _clear_session_cookie(resp)
    return resp


# A figure-reference span in answer prose: "Figure 328", "Figures 630 and 631",
# "Fig. 632", "Figures 630, 631, and 632". Captures the trailing number list so
# every figure in a list/pair is recognised, not just the one right after the
# "Figure(s)" keyword. Kept permissive on the connectors the LLM uses.
_ANSWER_FIG_REF_RE = re.compile(
    r"\bFig(?:ure)?s?\.?\s+(\d{1,4}(?:(?:\s*(?:,|and|&|/|or)\s*)+\d{1,4})*)",
    re.IGNORECASE,
)


def _referenced_figure_numbers(answer: str) -> set[str]:
    """Normalised figure numbers the answer text references, across singular,
    plural, abbreviated, and comma/and-separated list phrasings."""
    out: set[str] = set()
    for m in _ANSWER_FIG_REF_RE.finditer(answer or ""):
        for num in re.findall(r"\d{1,4}", m.group(1)):
            out.add(num.lstrip("0") or "0")
    return out


def _figures_from_sources(result: dict) -> list[dict]:
    """Slim, deduped list of figures the answer actually CITES, so the UI can
    link inline "Figure N" mentions to the PDF and list them in the sidebar.
    Only figures with a known page (so the link can jump there) AND that are
    referenced in the answer text are kept. Spec falls back to the query's spec
    for agentically-fetched figure chunks that don't carry it."""
    sources = result.get("sources") or []
    answer = result.get("answer") or ""
    default_spec = (result.get("config") or {}).get("spec")
    referenced = _referenced_figure_numbers(answer)
    seen: set[str] = set()
    figures: list[dict] = []
    for ch in sources:
        fn = ch.get("figure_number")
        if fn is None:
            continue
        fn = str(fn).strip()
        if not fn or fn in seen:
            continue
        pages = ch.get("pdf_pages") or []
        if not pages:
            continue
        # Only surface figures the answer references ("Figure 328", "Figures 630
        # and 631", "Fig. 632") - not every figure that happened to be retrieved
        # - so chips and the sidebar reflect what was actually cited.
        if (fn.lstrip("0") or "0") not in referenced:
            continue
        seen.add(fn)
        figures.append({
            "figure_number": fn,
            "spec": ch.get("spec") or default_spec,
            "pdf_pages": pages,
            "caption": ch.get("section_title") or "",
            "section_id": ch.get("section_id") or "",
        })
    return figures


# Strong refs to in-flight fire-and-forget logging tasks. asyncio only keeps a
# weak reference to a bare create_task() result, so without this the task could
# be garbage-collected before the DB write finishes.
_qa_log_tasks: set[asyncio.Task] = set()


def _qa_log_row(resp: QueryResponse, request_id: str) -> dict:
    """Map a QueryResponse to a qa_log row. Pure (no IO) so it's unit-testable;
    `spec`/`llm_model` are denormalized out of config for easy querying."""
    config = resp.config or {}
    return {
        "request_id": request_id,
        "query": resp.query,
        "answer": resp.answer,
        "config": config,
        "citations": resp.citations,
        "spec": config.get("spec"),
        "llm_model": config.get("llm_model"),
        "agentic": resp.agentic,
        "latency_ms": resp.latency_ms,
        "tokens_used": resp.tokens_used,
    }


async def _log_qa(resp: QueryResponse, request_id: str) -> None:
    """Persist one answered query to qa_log. Best-effort: any failure is logged
    and swallowed so a logging problem never affects the user's response. The
    Supabase insert is sync, so it runs in a worker thread off the event loop."""
    row = _qa_log_row(resp, request_id)
    try:
        from src.pipeline.search import supabase_client

        await asyncio.to_thread(
            lambda: supabase_client().table("qa_log").insert(row).execute()
        )
    except Exception:
        logger.exception("qa_log insert failed [%s]", request_id)


def _schedule_qa_log(resp: QueryResponse, request_id: str) -> None:
    """Fire-and-forget the qa_log write so every Q&A is recorded without adding
    latency to the request. Records *all* answers, unlike flagged_answers which
    is only the subset a user explicitly flags."""
    try:
        task = asyncio.create_task(_log_qa(resp, request_id))
    except RuntimeError:
        # No running loop (shouldn't happen inside a request handler).
        return
    _qa_log_tasks.add(task)
    task.add_done_callback(_qa_log_tasks.discard)


@app.post("/api/query")
async def query_endpoint(req: QueryRequest, _: bool = Depends(require_auth)) -> QueryResponse:
    """
    Run the full retrieval + generation pipeline.

    Returns answer with citations and optional pipeline trace for debugging.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    # Parse config from request, use defaults if not provided
    try:
        config_dict = req.config or {}
        config = PipelineConfig(**config_dict)
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"invalid config: {e}")

    if config.spec not in _VALID_SPEC_IDS:
        raise HTTPException(status_code=400, detail=f"unknown spec: {config.spec!r}")

    debug_trace = req.debug and DEBUG_PIPELINE
    request_id = uuid.uuid4().hex[:12]

    start = time.time()
    try:
        # orchestrate() is synchronous and CPU/IO-heavy (Voyage, Supabase,
        # cross-encoder, Anthropic). Run it in a worker thread so we don't
        # block the FastAPI event loop for the duration of the pipeline.
        result = await asyncio.to_thread(
            orchestrate, req.query, config=config, debug=debug_trace,
            agentic=req.agentic,
        )
    except GenerationError as e:
        # Retrieval worked; generation failed. 502 (bad upstream gateway)
        # is more accurate than 500 here, and the trace can still be returned
        # in debug mode so callers can see what *was* retrieved.
        logger.warning("Generation failure [%s]: %s", request_id, e.cause)
        detail = _generation_error_detail(e, request_id, include_trace=debug_trace)
        raise HTTPException(status_code=502, detail=detail)
    except Exception as e:
        logger.exception("Pipeline error [%s]: %s", request_id, e)
        # Don't leak internal error text to the client; surface only an id.
        raise HTTPException(
            status_code=500,
            detail={"error": "pipeline_error", "request_id": request_id},
        )
    latency_ms = (time.time() - start) * 1000

    # Cache first-pass state so /api/refine can resume without redoing
    # Stages 1-4. Only meaningful when the request landed in non-agentic
    # mode (agentic queries already ran the loop and have nothing to resume).
    if not req.agentic:
        _refine_cache_set(request_id, {
            "query": result["query"],
            "deduplicated": result.get("deduplicated") or [],
            "answer": result["answer"],
            "citations": result["citations"],
            "context_chunks": result.get("sources") or [],
            "tokens_used": result.get("tokens_used"),
        })

    resp = QueryResponse(
        query=result["query"],
        answer=result["answer"],
        citations=result["citations"],
        config=result["config"],
        pipeline_trace=result.get("pipeline_trace") if debug_trace else None,
        latency_ms=latency_ms,
        tokens_used=result.get("tokens_used"),
        agentic=bool(result.get("agentic")),
        gap_hint=result.get("gap_hint"),
        figures=_figures_from_sources(result),
        request_id=request_id if not req.agentic else None,
    )
    _schedule_qa_log(resp, request_id)
    return resp


def _dump_model(model: BaseModel) -> dict:
    """Serialize a pydantic model to a plain dict across pydantic v1/v2."""
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


@app.post("/api/query/stream")
async def query_stream_endpoint(req: QueryRequest, _: bool = Depends(require_auth)):
    """Streaming variant of /api/query.

    Emits newline-delimited JSON: a ``{"type":"progress","stage","took_ms"}``
    line as each pipeline stage completes, then a terminal
    ``{"type":"done","data": <QueryResponse>}`` (or ``{"type":"error",...}``).
    The final ``data`` payload is identical to what /api/query returns, so the
    client renders it the same way. Same auth + validation as /api/query.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    try:
        config = PipelineConfig(**(req.config or {}))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"invalid config: {e}")
    if config.spec not in _VALID_SPEC_IDS:
        raise HTTPException(status_code=400, detail=f"unknown spec: {config.spec!r}")

    debug_trace = req.debug and DEBUG_PIPELINE
    request_id = uuid.uuid4().hex[:12]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_progress(evt: dict) -> None:
        # Invoked from the orchestrate worker thread; hop back onto the loop.
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", **evt})

    async def _drive() -> None:
        start = time.time()
        try:
            result = await asyncio.to_thread(
                orchestrate, req.query, config=config, debug=debug_trace,
                agentic=req.agentic, on_progress=on_progress,
            )
            latency_ms = (time.time() - start) * 1000
            if not req.agentic:
                _refine_cache_set(request_id, {
                    "query": result["query"],
                    "deduplicated": result.get("deduplicated") or [],
                    "answer": result["answer"],
                    "citations": result["citations"],
                    "context_chunks": result.get("sources") or [],
                    "tokens_used": result.get("tokens_used"),
                })
            resp = QueryResponse(
                query=result["query"],
                answer=result["answer"],
                citations=result["citations"],
                config=result["config"],
                pipeline_trace=result.get("pipeline_trace") if debug_trace else None,
                latency_ms=latency_ms,
                tokens_used=result.get("tokens_used"),
                agentic=bool(result.get("agentic")),
                gap_hint=result.get("gap_hint"),
                figures=_figures_from_sources(result),
                request_id=request_id if not req.agentic else None,
            )
            _schedule_qa_log(resp, request_id)
            queue.put_nowait({"type": "done", "data": _dump_model(resp)})
        except GenerationError as e:
            logger.warning("Stream generation failure [%s]: %s", request_id, e.cause)
            detail = _generation_error_detail(e, request_id, include_trace=debug_trace)
            queue.put_nowait({"type": "error", "detail": detail})
        except Exception as e:
            logger.exception("Stream pipeline error [%s]: %s", request_id, e)
            queue.put_nowait({"type": "error",
                              "detail": {"error": "pipeline_error", "request_id": request_id}})
        finally:
            queue.put_nowait(None)  # sentinel: end of stream

    async def _gen():
        task = asyncio.create_task(_drive())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item) + "\n"
        finally:
            # Client disconnected or stream closed; stop driving. The orchestrate
            # worker thread can't be force-killed but its result is discarded.
            if not task.done():
                task.cancel()

    return StreamingResponse(
        _gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/refine", response_model=QueryResponse)
async def refine_endpoint(req: RefineRequest, _: bool = Depends(require_auth)) -> QueryResponse:
    """Resume a prior /api/query by request_id and run the agentic
    refinement against the cached first-pass state - no Stages 1-4 redo.
    Returns the same QueryResponse shape as /api/query with the refined
    Opus answer.
    """
    seed = _refine_cache_get(req.request_id)
    if seed is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "refine_state_missing",
                    "note": "request_id is unknown or has been evicted from the cache; resubmit the original query"},
        )

    try:
        config_dict = req.config or {}
        config = PipelineConfig(**config_dict)
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"invalid config: {e}")

    if config.spec not in _VALID_SPEC_IDS:
        raise HTTPException(status_code=400, detail=f"unknown spec: {config.spec!r}")

    debug_trace = req.debug and DEBUG_PIPELINE
    start = time.time()
    try:
        result = await asyncio.to_thread(
            orchestrate,
            seed["query"],
            config=config,
            debug=debug_trace,
            agentic=True,
            refine_seed=seed,
        )
    except GenerationError as e:
        logger.warning("Refine generation failure [%s]: %s", req.request_id, e.cause)
        detail = _generation_error_detail(e, req.request_id, include_trace=debug_trace)
        raise HTTPException(status_code=502, detail=detail)
    except Exception as e:
        logger.exception("Refine error [%s]: %s", req.request_id, e)
        raise HTTPException(
            status_code=500,
            detail={"error": "refine_error", "request_id": req.request_id},
        )
    latency_ms = (time.time() - start) * 1000

    # Refresh the cached state so a second click refines from the latest
    # Opus output, not the original Sonnet first-pass.
    _refine_cache_set(req.request_id, {
        "query": result["query"],
        "deduplicated": result.get("deduplicated") or [],
        "answer": result["answer"],
        "citations": result["citations"],
        "context_chunks": result.get("sources") or [],
        "tokens_used": result.get("tokens_used"),
    })

    resp = QueryResponse(
        query=result["query"],
        answer=result["answer"],
        citations=result["citations"],
        config=result["config"],
        pipeline_trace=result.get("pipeline_trace") if debug_trace else None,
        latency_ms=latency_ms,
        tokens_used=result.get("tokens_used"),
        agentic=True,
        gap_hint=None,
        figures=_figures_from_sources(result),
        request_id=req.request_id,
    )
    _schedule_qa_log(resp, req.request_id)
    return resp


@app.post("/api/refine/stream")
async def refine_stream_endpoint(req: RefineRequest, _: bool = Depends(require_auth)):
    """Streaming variant of /api/refine — same NDJSON protocol as
    /api/query/stream (progress lines, then a terminal done/error). Lets the
    UI show live agentic-loop progress during the Opus regen, which is the
    longest-running path (20–60s)."""
    seed = _refine_cache_get(req.request_id)
    if seed is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "refine_state_missing",
                    "note": "request_id is unknown or has been evicted from the cache; resubmit the original query"},
        )
    try:
        config = PipelineConfig(**(req.config or {}))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"invalid config: {e}")
    if config.spec not in _VALID_SPEC_IDS:
        raise HTTPException(status_code=400, detail=f"unknown spec: {config.spec!r}")

    debug_trace = req.debug and DEBUG_PIPELINE
    request_id = req.request_id
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_progress(evt: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", **evt})

    async def _drive() -> None:
        start = time.time()
        try:
            result = await asyncio.to_thread(
                orchestrate, seed["query"], config=config, debug=debug_trace,
                agentic=True, refine_seed=seed, on_progress=on_progress,
            )
            latency_ms = (time.time() - start) * 1000
            # Refresh cache so a second refine builds on the latest Opus output.
            _refine_cache_set(request_id, {
                "query": result["query"],
                "deduplicated": result.get("deduplicated") or [],
                "answer": result["answer"],
                "citations": result["citations"],
                "context_chunks": result.get("sources") or [],
                "tokens_used": result.get("tokens_used"),
            })
            resp = QueryResponse(
                query=result["query"],
                answer=result["answer"],
                citations=result["citations"],
                config=result["config"],
                pipeline_trace=result.get("pipeline_trace") if debug_trace else None,
                latency_ms=latency_ms,
                tokens_used=result.get("tokens_used"),
                agentic=True,
                gap_hint=None,
                figures=_figures_from_sources(result),
                request_id=request_id,
            )
            _schedule_qa_log(resp, request_id)
            queue.put_nowait({"type": "done", "data": _dump_model(resp)})
        except GenerationError as e:
            logger.warning("Stream refine generation failure [%s]: %s", request_id, e.cause)
            detail = _generation_error_detail(e, request_id, include_trace=debug_trace)
            queue.put_nowait({"type": "error", "detail": detail})
        except Exception as e:
            logger.exception("Stream refine error [%s]: %s", request_id, e)
            queue.put_nowait({"type": "error",
                              "detail": {"error": "refine_error", "request_id": request_id}})
        finally:
            queue.put_nowait(None)

    async def _gen():
        task = asyncio.create_task(_drive())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item) + "\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        _gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/config")
async def config_endpoint(_: bool = Depends(require_auth)) -> dict:
    """Return default PipelineConfig."""
    return PipelineConfig().to_dict()


@app.get("/api/presets")
async def presets_endpoint(_: bool = Depends(require_auth)) -> dict:
    """Named config presets for the UI dropdown, plus the default selection.

    Each preset bundles a subset of PipelineConfig overrides + the agentic
    flag; the frontend applies the chosen preset to its config inputs.
    """
    return {"presets": PRESETS, "default": DEFAULT_PRESET}


@app.get("/api/specs")
async def specs_endpoint(_: bool = Depends(require_auth)) -> dict:
    """Specs the UI can search, plus the default selection."""
    return {"specs": AVAILABLE_SPECS, "default": PipelineConfig().spec}


@app.get("/api/models")
async def models_endpoint(_: bool = Depends(require_auth)) -> dict:
    """Return model info and per-token pricing for all pipeline stages."""
    cfg = PipelineConfig()
    return {
        "embedding": {
            "model": "voyage-3-lite",
            "provider": "Voyage AI",
            "price_per_1m_input": 0.02,
            "price_per_1m_output": None,
            "note": "Query embedding only; doc embeddings pre-computed",
        },
        "reranker": {
            "model": cfg.cross_encoder_model,
            "provider": "Voyage AI",
            "price_per_1m_input": 0.05,
            "price_per_1m_output": None,
            "note": "rerank-2-lite; ~$0.05/1M tokens",
        },
        "llm": {
            "model": cfg.llm_model,
            "provider": "UNH DeepThought",
            "price_per_1m_input": 0.0,
            "price_per_1m_output": 0.0,
            "note": "Standard queries (UNH-hosted)",
        },
        "agentic_llm": {
            "model": cfg.agentic_model,
            "provider": "UNH DeepThought",
            "price_per_1m_input": 0.0,
            "price_per_1m_output": 0.0,
            "note": "Agentic mode only (UNH-hosted)",
        },
    }


@app.get("/api/figure/{spec}/{figure_number}")
async def render_figure_endpoint(spec: str, figure_number: str, _: bool = Depends(require_auth)) -> dict:
    """Parsed table JSON for one figure, for the in-app figure render popup."""
    for s in _define_specs(spec):
        try:
            table = load_tables_by_figure(s).get(figure_number)
        except Exception:  # noqa: BLE001 - a corpus without table data just contributes nothing
            continue
        if table:
            return table
    raise HTTPException(status_code=404, detail="Figure not found")


# ── Field-acronym definitions (answer popovers) ────────────────────────────
# Backs the clickable acronym chips in the rendered answer: the UI fetches the
# definable-term list once per spec, marks matching inline-code tokens, and
# resolves a clicked term against /api/define. Both reads come from the
# in-process field index cache (retriever.load_field_index), so after warmup
# there is no per-click DB round-trip.

# A field-index key that reads as a numeric/hex literal (bare number, 0x1F,
# 3FH) is never a definable acronym - drop it server-side so the UI can't
# mark hex values in answers as clickable terms.
_LITERAL_TERM_RE = re.compile(r"^(?:0X[0-9A-F]+|[0-9A-F]+H|[0-9]+)$")


def _define_specs(spec: str) -> tuple[str, ...]:
    if spec not in _VALID_SPEC_IDS:
        raise HTTPException(status_code=400, detail=f"unknown spec: {spec!r}")
    return CONCRETE_SPEC_IDS if spec == ALL_SPECS else (spec,)


@lru_cache(maxsize=8)
def _definable_terms(spec: str) -> tuple[str, ...]:
    terms: set[str] = set()
    for s in _define_specs(spec):
        try:
            index = load_field_index(s)
        except Exception:  # noqa: BLE001 - a corpus without lookup data just contributes nothing
            continue
        for name in index:
            name = str(name).strip().upper()
            if name and not _LITERAL_TERM_RE.match(name):
                terms.add(name)
    return tuple(sorted(terms))


def _truncate_definition(text: str | None, limit: int = 700) -> str | None:
    if not text:
        return None
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 0 else limit].rstrip() + "…"


@app.get("/api/define/terms")
async def define_terms_endpoint(spec: str = "base", _: bool = Depends(require_auth)) -> dict:
    """All field acronyms with a known definition, for client-side marking."""
    return {"spec": spec, "terms": list(_definable_terms(spec))}


@app.get("/api/define")
async def define_endpoint(term: str, spec: str = "base", _: bool = Depends(require_auth)) -> dict:
    """Resolve one acronym to its field definition(s) for the popover."""
    term_n = term.strip().upper()
    if not term_n or len(term_n) > 32:
        raise HTTPException(status_code=400, detail="bad term")
    matches: list[dict] = []
    for s in _define_specs(spec):
        try:
            records = load_field_index(s).get(term_n) or []
        except Exception:  # noqa: BLE001
            continue
        for rec in records:
            matches.append({
                "spec": s,
                "full_name": rec.get("full_name"),
                "description": _truncate_definition(rec.get("description")),
                "section_id": rec.get("section_id"),
                "figure_number": rec.get("parent_figure"),
                "parent_caption": rec.get("parent_caption"),
                "offset": rec.get("offset"),
                "offset_type": rec.get("offset_type"),
            })
            if len(matches) >= 4:
                return {"term": term_n, "matches": matches}
    return {"term": term_n, "matches": matches}


@app.post("/api/flag-answer")
async def flag_answer_endpoint(
    req: FlagAnswerRequest,
    _: bool = Depends(require_auth),
) -> dict:
    """Persist a user-reported answer-quality flag.

    Writes one row to flagged_answers snapshotting the response the user was
    viewing plus an optional reason. Fire-and-forget from the UI's view: a DB
    error returns 502 so the modal can surface it, but normal Q&A is unaffected.
    """
    from src.pipeline.search import supabase_client

    config = req.config or {}
    row = {
        "query": req.query,
        "answer": req.answer,
        "config": config,
        "pipeline_trace": req.pipeline_trace,
        "citations": req.citations,
        "spec": config.get("spec"),
        "llm_model": config.get("llm_model"),
        "agentic": req.agentic,
        "latency_ms": req.latency_ms,
        "tokens_used": req.tokens_used,
        "reason": (req.reason or "").strip() or None,
    }
    try:
        supabase_client().table("flagged_answers").insert(row).execute()
    except Exception as exc:  # don't 500 the UI over a logging-style write
        logger.exception("flag insert failed")
        raise HTTPException(status_code=502, detail={"error": "flag_failed"}) from exc
    return {"ok": True}


@app.get("/api/flags")
async def list_flags_endpoint(
    limit: int = 100,
    _: bool = Depends(require_auth),
) -> dict:
    """Return recent flagged answers for the in-app dev panel, newest first."""
    from src.pipeline.search import supabase_client

    limit = max(1, min(limit, 500))
    try:
        res = (
            supabase_client()
            .table("flagged_answers")
            .select(
                "id, created_at, query, answer, reason, status, spec, llm_model, "
                "agentic, latency_ms, tokens_used, citations, config, pipeline_trace"
            )
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        logger.exception("flags list failed")
        raise HTTPException(status_code=502, detail={"error": "flags_list_failed"}) from exc
    return {"flags": res.data or []}


@app.delete("/api/flags/{flag_id}")
async def delete_flag_endpoint(
    flag_id: int,
    _: bool = Depends(require_auth),
) -> dict:
    """Delete one flagged answer."""
    from src.pipeline.search import supabase_client

    try:
        supabase_client().table("flagged_answers").delete().eq("id", flag_id).execute()
    except Exception as exc:
        logger.exception("flag delete failed")
        raise HTTPException(status_code=502, detail={"error": "flag_delete_failed"}) from exc
    return {"ok": True}


@app.get("/api/dev-notes")
async def list_dev_notes_endpoint(_: bool = Depends(require_auth)) -> dict:
    """Return dev scratchpad notes, newest first."""
    from src.pipeline.search import supabase_client

    try:
        res = (
            supabase_client()
            .table("dev_notes")
            .select("id, created_at, body")
            .order("created_at", desc=True)
            .limit(200)
            .execute()
        )
    except Exception as exc:
        logger.exception("dev notes list failed")
        raise HTTPException(status_code=502, detail={"error": "notes_list_failed"}) from exc
    return {"notes": res.data or []}


@app.post("/api/dev-notes")
async def create_dev_note_endpoint(
    req: DevNoteRequest,
    _: bool = Depends(require_auth),
) -> dict:
    """Append one note to the dev scratchpad."""
    from src.pipeline.search import supabase_client

    body = (req.body or "").strip()
    if not body:
        raise HTTPException(status_code=400, detail={"error": "empty_note"})
    try:
        res = supabase_client().table("dev_notes").insert({"body": body}).execute()
    except Exception as exc:
        logger.exception("dev note insert failed")
        raise HTTPException(status_code=502, detail={"error": "note_failed"}) from exc
    note = (res.data or [None])[0]
    return {"ok": True, "note": note}


@app.delete("/api/dev-notes/{note_id}")
async def delete_dev_note_endpoint(
    note_id: int,
    _: bool = Depends(require_auth),
) -> dict:
    """Delete one note from the dev scratchpad."""
    from src.pipeline.search import supabase_client

    try:
        supabase_client().table("dev_notes").delete().eq("id", note_id).execute()
    except Exception as exc:
        logger.exception("dev note delete failed")
        raise HTTPException(status_code=502, detail={"error": "note_delete_failed"}) from exc
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def frontend(
    specgpt_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> Response:
    """Serve the web frontend, or redirect to /login if not signed in."""
    if not verify_session_token(specgpt_session, _SESSION_SECRET_BYTES):
        return RedirectResponse(url="/login", status_code=303)
    return HTMLResponse(FRONTEND_HTML)


# ============================================================================
# Frontend HTML/CSS/JS
# ============================================================================

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>specGPT - NVMe Spec Q&A</title>
    <link rel="icon" type="image/png" href="/static/favicon.png">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
:root {
  /* zinc neutrals */
  --canvas:   #fbfbfc;
  --surface:  #ffffff;
  --surface-2:#f6f6f8;
  --subtle:   #f1f1f4;
  --border:   #e7e7eb;
  --border-2: #dadadf;
  --ink:      #18181b;
  --t-strong: #27272a;
  --t-muted:  #52525b;
  --t-subtle: #71717a;
  --t-faint:  #a1a1aa;
  --accent:      #2d68e6;
  --accent-ink:  #1d4ed8;
  --accent-soft: #eef3fe;
  --accent-bd:   #cbdcfb;
  --ok:    #15803d;
  --ok-soft:#e8f6ee;
  --warn:  #b45309;
  --warn-soft:#fbf2e3;
  --danger:#dc2626;
  --danger-soft:#fdecec;
  /* aliases used by legacy engine CSS */
  --bg: var(--canvas); --bg-soft: var(--surface-2); --bg-muted: var(--subtle);
  --text: var(--ink); --text-muted: var(--t-muted); --text-subtle: var(--t-subtle); --text-faint: var(--t-faint);
  --border-strong: var(--border-2);
  --radius: 12px;
  --radius-sm: 8px;
  --radius-xs: 6px;
  --sans: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --mono: 'Geist Mono', ui-monospace, 'SF Mono', Menlo, monospace;
  --font-sans: var(--sans);
  --font-mono: var(--mono);
  --shadow-sm: 0 1px 2px rgba(24,24,27,.05);
  --shadow-md: 0 4px 16px -6px rgba(24,24,27,.12), 0 1px 3px rgba(24,24,27,.06);
  --shadow-pop: 0 12px 40px -8px rgba(24,24,27,.22), 0 2px 8px rgba(24,24,27,.10);
  --pad-shell: 32px;
  --gap-block: 22px;
  --composer-pad: 18px;
  --answer-fs: 15px;
}
.density-compact { --pad-shell: 22px; --gap-block: 14px; --composer-pad: 13px; --answer-fs: 14px; }
.density-comfy   { --pad-shell: 44px; --gap-block: 30px; --composer-pad: 24px; --answer-fs: 16px; }

[data-theme="dark"] {
  --canvas:   #0c0c0e;
  --surface:  #141417;
  --surface-2:#191920;
  --subtle:   #1d1d23;
  --border:   #26262d;
  --border-2: #33333c;
  --ink:      #f4f4f5;
  --t-strong: #e4e4e7;
  --t-muted:  #a1a1aa;
  --t-subtle: #8a8a93;
  --t-faint:  #5f5f68;
  --accent:      #5b8cff;
  --accent-ink:  #82a6ff;
  --accent-soft: #16213d;
  --accent-bd:   #284070;
  --ok:    #4ade80;
  --ok-soft:#13251a;
  --warn:  #d8a657;
  --warn-soft:#2a2113;
  --danger:#f87171;
  --danger-soft:#2c1717;
  --shadow-sm: 0 1px 2px rgba(0,0,0,.4);
  --shadow-md: 0 4px 18px -6px rgba(0,0,0,.6), 0 1px 3px rgba(0,0,0,.4);
  --shadow-pop: 0 16px 48px -8px rgba(0,0,0,.7), 0 2px 8px rgba(0,0,0,.5);
}

* { margin:0; padding:0; box-sizing:border-box; }
html { height:100%; background:var(--canvas); }
body { min-height:100%; }
body {
  font-family: var(--sans);
  background: var(--canvas);
  color: var(--ink);
  font-size: 14px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  letter-spacing: -0.006em;
  font-feature-settings: "ss01", "cv01";
}
::selection { background: var(--accent-bd); }
[data-theme="dark"] ::selection { background: rgba(91, 140, 255, 0.35); }
button { font-family: inherit; cursor: pointer; }
input, select, textarea { font-family: inherit; }
.mono { font-family: var(--mono); font-feature-settings: normal; letter-spacing: 0; }
a { color: var(--accent); text-decoration: none; }
.hidden { display: none !important; }

*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 8px; border: 2px solid var(--canvas); }
*::-webkit-scrollbar-thumb:hover { background: var(--t-faint); }

@keyframes spin { to { transform: rotate(360deg); } }
@keyframes fadeUp { from { opacity:0; transform: translateY(6px); } to { opacity:1; transform:none; } }
@keyframes pulse { 0%,100%{opacity:.45} 50%{opacity:1} }
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }

/* shell */
.app { min-height:100vh; display:flex; flex-direction:column; }
.wrap, .container { width:100%; max-width:1200px; margin:0 auto; padding:0 var(--pad-shell); }

/* topbar */
.topbar { position:sticky; top:0; z-index:30; background:color-mix(in srgb, var(--surface) 82%, transparent);
  backdrop-filter:saturate(1.4) blur(12px); border-bottom:1px solid var(--border); }
.topbar-inner { display:flex; align-items:center; justify-content:space-between; height:58px; gap:12px; }
.brand { display:flex; align-items:center; gap:11px; min-width:0; }
.brand-mark { width:30px; height:30px; border-radius:8px; background:var(--ink); color:var(--surface);
  display:grid; place-items:center; box-shadow:var(--shadow-sm); flex:none; }
.brand-mark svg { width:17px; height:17px; }
.brand-name { font-size:15px; font-weight:600; letter-spacing:-0.02em; white-space:nowrap; }
.brand-name b { color:var(--accent); font-weight:600; }
.brand-tag { font-size:11px; color:var(--t-faint); font-weight:500; margin-left:2px; white-space:nowrap;
  padding:2px 7px; border:1px solid var(--border); border-radius:99px; letter-spacing:.01em; }
.topbar-right { display:flex; align-items:center; gap:10px; flex:none; }
.topbar-form { margin:0; display:inline-flex; }

/* spec + model pickers */
.picker { display:flex; align-items:center; gap:8px; height:34px; padding:0 6px 0 11px;
  border:1px solid var(--border); border-radius:8px; background:var(--surface); transition:border-color .14s; }
.picker:hover { border-color:var(--border-2); }
.picker-lbl { font-size:10.5px; text-transform:uppercase; letter-spacing:.07em; color:var(--t-faint); font-weight:600; white-space:nowrap; }
.picker select { border:0; background:transparent; color:var(--ink); font-size:13px; font-weight:500;
  outline:none; cursor:pointer; max-width:180px; }
[data-theme="dark"] select option { background:var(--surface-2); color:var(--ink); }
.ghost-btn { height:34px; padding:0 13px; border:1px solid var(--border); border-radius:8px; background:var(--surface);
  color:var(--t-muted); font-size:13px; font-weight:500; transition:all .14s; white-space:nowrap; }
.ghost-btn:hover { background:var(--surface-2); border-color:var(--border-2); color:var(--ink); }
.icon-btn { width:34px; height:34px; display:grid; place-items:center; border:1px solid var(--border);
  border-radius:8px; background:var(--surface); color:var(--t-muted); transition:all .14s; flex:none; }
.icon-btn:hover { background:var(--surface-2); border-color:var(--border-2); color:var(--ink); }
.icon-btn svg { width:16px; height:16px; }
.icon-sun { display:none; }
[data-theme="dark"] .icon-sun { display:block; }
[data-theme="dark"] .icon-moon { display:none; }

/* main */
.main { flex:1; padding-top:var(--gap-block); padding-bottom:80px; }

/* composer */
.composer { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  box-shadow:var(--shadow-md); padding:var(--composer-pad); transition:border-color .15s, box-shadow .2s; }
.composer.focus { border-color:var(--accent-bd); box-shadow:var(--shadow-md), 0 0 0 4px var(--accent-soft); }
/* agentic on: highlight the composer with the accent + a slight glow */
.composer.agentic-active { border-color:var(--accent);
  background:radial-gradient(ellipse at center,
    color-mix(in srgb, var(--accent) 13%, transparent),
    color-mix(in srgb, var(--accent) 4%, transparent) 72%), var(--surface);
  box-shadow:var(--shadow-md), 0 0 0 1px var(--accent), 0 0 10px color-mix(in srgb, var(--accent) 25%, transparent); }
.composer-row { display:flex; align-items:flex-end; gap:10px; }
.composer-input-wrap { flex:1; display:flex; align-items:flex-end; gap:10px; padding-left:4px; min-width:0; }
.composer-input-wrap > svg { width:19px; height:19px; color:var(--t-faint); flex:none; margin-bottom:9px; }
.composer textarea { flex:1; border:0; background:transparent; color:var(--ink); font-size:17px;
  letter-spacing:-0.01em; outline:none; padding:8px 0; min-width:0; resize:none; overflow:hidden;
  line-height:1.5; min-height:36px; max-height:200px; }
.composer textarea::placeholder { color:var(--t-faint); }
.ask-btn { box-sizing:border-box; width:130px; height:56px; padding:0; border-radius:9px; border:1px solid var(--ink);
  background:var(--ink); color:var(--surface); font-size:14px; font-weight:600; letter-spacing:-0.01em;
  display:inline-flex; align-items:center; justify-content:center; gap:10px;
  transition:filter .14s, opacity .14s; flex:none; }
.ask-btn:hover { filter:brightness(1.18); }
.ask-btn:disabled { opacity:.4; cursor:not-allowed; filter:none; }
.ask-btn svg { width:15px; height:15px; flex:none; }
.ask-btn-inner { display:flex; flex-direction:column; align-items:center; gap:1px; width:46px; }
.ask-spec-label { font-size:10px; font-weight:400; opacity:0.6; letter-spacing:0.02em; }

/* control strip */
.controls { display:flex; align-items:center; gap:8px; margin-top:13px; padding-top:13px;
  border-top:1px dashed var(--border); flex-wrap:wrap; }
.controls-spacer { flex:1; }
.pill { display:inline-flex; align-items:center; gap:7px; height:32px; padding:0 12px; border-radius:99px;
  border:1px solid var(--border); background:var(--surface); color:var(--t-muted); font-size:12.5px;
  font-weight:500; transition:all .14s; position:relative; }
.pill:hover { border-color:var(--border-2); color:var(--ink); background:var(--surface-2); }
.pill svg { width:14px; height:14px; }
.pill.on { background:var(--accent-soft); border-color:var(--accent-bd); color:var(--accent-ink); }
.pill.on .pill-dot { background:var(--accent); }
.pill-dot { width:7px; height:7px; border-radius:50%; background:var(--t-faint); transition:background .14s; }
.cost-chip { display:inline-flex; align-items:center; gap:8px; height:32px; padding:0 12px 0 10px;
  border-radius:99px; border:1px solid var(--border); background:var(--surface); font-size:12px; color:var(--t-subtle); cursor:pointer; }
.cost-chip:hover { border-color:var(--border-2); }
.cost-chip .cost-total { color:var(--ink); font-weight:500; font-size:12.5px; font-family:var(--mono); }
.cost-chip .cost-total.cost-warn { color:var(--warn); }
.cost-chip .cost-total.cost-high { color:var(--danger); }
.cost-chip-lbl { font-size:10px; text-transform:uppercase; letter-spacing:.06em; color:var(--t-faint); font-weight:600; white-space:nowrap; }
.cost-chip .cost-context { display:none; }
.cost-chip .cost-toggle { color:var(--t-faint); font-size:9px; }

/* cost breakdown popover (under the cost chip) */
.cost-breakdown { display:none; }
.cost-estimator.open .cost-breakdown { display:block; position:absolute; right:0; margin-top:8px; z-index:40;
  width:380px; background:var(--surface); border:1px solid var(--border-2); border-radius:var(--radius);
  box-shadow:var(--shadow-pop); padding:14px; animation:fadeUp .14s ease; }
.cost-estimator { position:relative; }
.cost-row { display:flex; align-items:flex-start; justify-content:space-between; gap:10px; padding:6px 0; border-bottom:1px dashed var(--border); }
.cost-row:last-of-type { border-bottom:0; }
.cost-row-name { font-size:12.5px; color:var(--t-strong); display:flex; flex-direction:column; gap:2px; }
.cost-row-name small { color:var(--t-faint); font-size:11px; }
.cost-row-value { font-family:var(--mono); font-size:12px; color:var(--t-muted); white-space:nowrap; }
.cost-row-total { border-top:1px solid var(--border); border-bottom:0; margin-top:4px; padding-top:9px; }
.cost-row-total .cost-row-value { color:var(--ink); font-weight:600; }
.cost-disclaimer { margin-top:10px; font-size:11px; color:var(--t-faint); line-height:1.5; }

/* config popover */
.config-panel, .agentic-config { display:none; }
.config-panel.open { display:block; position:absolute; z-index:40; margin-top:8px; left:0; width:520px;
  background:var(--surface); border:1px solid var(--border-2); border-radius:var(--radius);
  box-shadow:var(--shadow-pop); padding:16px; animation:fadeUp .14s ease; }
.config-pop-wrap { position:relative; }
.config-panel > strong { display:block; font-size:11px; text-transform:uppercase; letter-spacing:.07em;
  color:var(--t-subtle); font-weight:600; margin-bottom:13px; }
.config-section-label { font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--t-faint);
  font-weight:600; margin:16px 0 9px; }
.config-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:11px 14px; }
.config-item.config-item-wide { grid-column:1 / -1; }
.config-item label { display:block; font-size:11px; color:var(--t-subtle); font-weight:500; margin-bottom:5px; }
.config-item input[type=number], .config-item select { width:100%; height:32px; padding:0 9px; border:1px solid var(--border);
  border-radius:var(--radius-xs); background:var(--surface); color:var(--ink); font-size:13px; outline:none; font-family:var(--mono); }
.config-item input:focus, .config-item select:focus { border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-soft); }
.config-item label > input[type=checkbox] { width:15px; height:15px; accent-color:var(--accent); vertical-align:-2px; margin-right:6px; }
.config-item label:has(input[type=checkbox]) { display:flex; align-items:center; font-size:13px; color:var(--ink); font-weight:500; margin-top:18px; }

/* agentic config is always visible inside the config popover and the
   refine-time overlay, regardless of the legacy .hidden toggle */
.config-panel .agentic-config,
.config-panel .agentic-config.hidden { display:block !important; }

/* agentic-row (hint text under composer when agentic on) */
.agentic-row { display:none; }
.agentic-row.active { display:flex; gap:10px; align-items:flex-start; margin-top:12px; padding:11px 13px;
  background:var(--accent-soft); border:1px solid var(--accent-bd); border-radius:var(--radius-sm); }
.agentic-row label { display:flex; align-items:center; gap:7px; font-size:13px; font-weight:600; color:var(--accent-ink); white-space:nowrap; }
.agentic-row label input { accent-color:var(--accent); width:15px; height:15px; }
.agentic-hint { font-size:12px; color:var(--t-muted); line-height:1.5; }

/* empty state */
.empty { text-align:center; padding:54px 20px 30px; }
.empty-mark { width:82px; height:82px; border-radius:22px; background:var(--surface); border:1px solid var(--border);
  display:grid; place-items:center; margin:0 auto 18px; box-shadow:var(--shadow-sm); color:var(--ink); }
.empty-mark svg { width:26px; height:26px; }
.empty-mark img { width:65px; height:65px; object-fit:contain; }
.empty h2 { font-size:22px; font-weight:600; letter-spacing:-0.025em; margin-bottom:8px; }
.empty p { font-size:14px; color:var(--t-subtle); max-width:460px; margin:0 auto 26px; line-height:1.6; }
.examples { display:flex; flex-wrap:wrap; gap:9px; justify-content:center; max-width:720px; margin:0 auto; }
.ex-chip { padding:9px 15px; border:1px solid var(--border); border-radius:99px; background:var(--surface);
  color:var(--t-muted); font-size:13px; transition:all .14s; box-shadow:var(--shadow-sm); }
.ex-chip:hover { border-color:var(--accent-bd); color:var(--accent-ink); background:var(--accent-soft); transform:translateY(-1px); }

/* results split */
#results { margin-top:var(--gap-block); }
.split { display:grid; grid-template-columns:minmax(0,1fr) 332px; gap:var(--gap-block); align-items:start; }
@media (max-width:880px){ .split { grid-template-columns:1fr; } }
.split-main { min-width:0; display:flex; flex-direction:column; gap:var(--gap-block); }

/* answer card */
.answer-box { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  box-shadow:var(--shadow-sm); overflow:hidden; }
#answer-section.answer-stale { opacity: 0.4; transition: opacity 0.2s; }
.answer-meta { display:flex; align-items:center; gap:9px; flex-wrap:wrap; padding:14px 22px; border-bottom:1px solid var(--border); background:var(--surface-2); }
.meta-q { font-size:13px; font-weight:600; color:var(--ink); margin-right:auto; letter-spacing:-0.01em;
  max-width:58%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.badge { display:inline-flex; align-items:center; gap:5px; height:22px; padding:0 8px; border-radius:99px;
  font-size:11px; font-weight:500; font-family:var(--mono); border:1px solid var(--border); color:var(--t-muted); background:var(--surface); }
.badge svg { width:11px; height:11px; }
.badge.ok { color:var(--ok); background:var(--ok-soft); border-color:transparent; }
.badge.warn { color:var(--warn); background:var(--warn-soft); border-color:transparent; }
.badge.accent { color:var(--accent-ink); background:var(--accent-soft); border-color:var(--accent-bd); }
.answer-box > h3 { display:none; }
.answer-text { padding:8px 22px 22px; }

/* markdown */
.answer-text { font-size:var(--answer-fs); line-height:1.72; color:var(--t-strong); }
.answer-text > *:first-child { margin-top:14px; }
.answer-text h1 { font-size:17px; font-weight:600; color:var(--ink); margin:22px 0 10px; }
.answer-text h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--t-subtle); font-weight:600; margin:24px 0 10px; }
.answer-text h3 { font-size:15px; font-weight:600; color:var(--ink); margin:20px 0 8px; }
.answer-text p { margin:0 0 13px; }
.answer-text strong { color:var(--ink); font-weight:600; }
.answer-text em { color:var(--t-muted); }
.answer-text ul, .answer-text ol { margin:0 0 14px; padding-left:22px; }
.answer-text li { margin:0 0 7px; }
.answer-text li::marker { color:var(--t-faint); }
.answer-text code { font-family:var(--mono); font-size:.88em; background:var(--subtle); border:1px solid var(--border);
  padding:1px 5px; border-radius:5px; color:var(--accent-ink); }
.answer-text pre { background:var(--subtle); border:1px solid var(--border); border-radius:var(--radius-xs); padding:12px 14px; overflow:auto; margin:0 0 14px; }
.answer-text pre code { background:transparent; border:0; padding:0; }
/* inline-code acronyms with a known field definition: click for a popover */
.answer-text code.def-term { cursor:pointer; border-bottom:1px dashed var(--accent-bd); }
.answer-text code.def-term:hover { background:var(--accent-soft); border-color:var(--accent-bd); }
.answer-text blockquote { margin:0 0 14px; padding:10px 16px; background:var(--warn-soft); border-left:3px solid var(--warn);
  border-radius:0 var(--radius-xs) var(--radius-xs) 0; color:var(--t-muted); font-size:.95em; }
.answer-text blockquote p { margin:0; }
.answer-text table { width:100%; border-collapse:collapse; margin:0 0 14px; font-size:.92em; }
.answer-text th, .answer-text td { border:1px solid var(--border); padding:7px 10px; text-align:left; }
.answer-text th { background:var(--surface-2); font-weight:600; }
.answer-text.streaming::after { content:"\\2589"; color:var(--accent); animation:pulse 1s infinite; margin-left:1px; }

/* citation chips inside answer */
.cite-chip { font-family:var(--mono); font-size:.82em; font-weight:500; color:var(--accent-ink);
  background:var(--accent-soft); border:1px solid var(--accent-bd); border-radius:5px; padding:0 5px 1px;
  cursor:pointer; white-space:nowrap; transition:background .12s, box-shadow .12s; }
.cite-chip::before { content:"\\00a7"; opacity:.6; margin-right:1px; }
.cite-chip:hover, .cite-chip.hot { background:var(--accent); color:#fff; border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-soft); }
/* Figure chips carry their own "Figure" label, so drop the leading section mark. */
.fig-chip::before { content:""; margin-right:0; }

/* "Source: [§…]" attribution line under a table or code block (rule 2b) —
   reads as a quiet caption tied to the block directly above it. */
.answer-text .block-attrib { font-size:.82em; color:var(--t-faint); margin:-8px 0 14px; }
.answer-text .block-attrib::before { content:"\\21B3"; margin-right:5px; opacity:.6; }
.answer-text table.has-attrib, .answer-text pre.has-attrib { margin-bottom:4px; }

/* latency tag in answer meta (legacy id #latency reused) */
#latency { display:none; }

/* sources sidebar */
.sources { position:sticky; top:74px; }
.sources-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:11px; padding:0 2px; }
.sources-head h3 { font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--t-subtle); font-weight:600; white-space:nowrap; }
.sources-count { font-size:11px; color:var(--t-faint); font-family:var(--mono); }
.src-list { display:flex; flex-direction:column; gap:8px; max-height:520px; overflow-y:auto; padding-right:4px; }
.src { display:block; text-align:left; width:100%; background:var(--surface); border:1px solid var(--border);
  border-radius:var(--radius-sm); padding:11px 12px; transition:all .14s; box-shadow:var(--shadow-sm); cursor:default; }
.src:hover, .src.hot { border-color:var(--accent-bd); box-shadow:var(--shadow-sm), 0 0 0 3px var(--accent-soft); transform:translateY(-1px); }
.src-top { display:flex; align-items:center; gap:8px; margin-bottom:5px; }
.src-sec { font-family:var(--mono); font-size:12.5px; font-weight:600; color:var(--accent-ink); }
.src-type { font-size:10px; text-transform:uppercase; letter-spacing:.05em; font-weight:600; padding:1px 6px;
  border-radius:4px; background:var(--subtle); color:var(--t-subtle); }
.src-dot { margin-left:auto; width:14px; height:14px; border-radius:50%; flex:none;
  display:grid; place-items:center; font-size:9px; font-weight:700; line-height:1; color:#fff; }
[data-theme="dark"] .src-dot { color:#0c0c0e; }
.src-dot.ok { background:var(--ok); }
.src-dot.ok::before { content:"\\2713"; }
.src-dot.warn { background:var(--warn); }
.src-dot.warn::before { content:"?"; }
.src-title { font-size:12.5px; color:var(--t-muted); line-height:1.45; }
.sources-foot { margin-top:12px; padding:10px 12px; border:1px dashed var(--border); border-radius:var(--radius-sm);
  font-size:11.5px; color:var(--t-faint); line-height:1.5; display:flex; gap:8px; align-items:flex-start; }
.sources-foot svg { width:13px; height:13px; flex:none; margin-top:2px; color:var(--t-faint); }

/* gap hint / agent strip (reused #agent-strip) */
.agent-strip { display:none; }
.agent-strip.has-content { display:block; }
.gap-card { display:flex; gap:13px; align-items:flex-start; padding:14px 16px; border-radius:var(--radius); }
.gap-card.warn { background:var(--warn-soft); border:1px solid color-mix(in srgb, var(--warn) 30%, transparent); }
.gap-card.ok   { background:var(--ok-soft);  border:1px solid color-mix(in srgb, var(--ok) 28%, transparent); }
.gap-card.accent { background:var(--accent-soft); border:1px solid var(--accent-bd); }
.gap-card.muted { background:var(--surface); border:1px solid var(--border); }
.gap-ico { width:30px; height:30px; border-radius:8px; display:grid; place-items:center; flex:none; }
.gap-card.warn .gap-ico { background:color-mix(in srgb, var(--warn) 16%, var(--surface)); color:var(--warn); }
.gap-card.ok .gap-ico   { background:color-mix(in srgb, var(--ok) 16%, var(--surface)); color:var(--ok); }
.gap-card.accent .gap-ico { background:color-mix(in srgb, var(--accent) 16%, var(--surface)); color:var(--accent); }
.gap-card.muted .gap-ico { background:var(--subtle); color:var(--t-subtle); }
.gap-ico svg { width:16px; height:16px; }
.gap-body { flex:1; min-width:0; }
.gap-title { font-size:13px; font-weight:600; color:var(--ink); margin-bottom:2px; display:flex; align-items:center; gap:8px; }
.gap-latency { font-family:var(--mono); font-size:11px; color:var(--t-faint); font-weight:400; }
.gap-note { font-size:12.5px; color:var(--t-muted); line-height:1.5; }
.gap-details { margin-top:9px; display:flex; flex-direction:column; gap:7px; }
.detail-group { display:flex; gap:8px; align-items:baseline; flex-wrap:wrap; }
.detail-label { font-size:10px; text-transform:uppercase; letter-spacing:.05em; color:var(--t-faint); font-weight:600; min-width:54px; }
.gap-chips { display:flex; gap:5px; flex-wrap:wrap; }
.gap-chip { font-family:var(--mono); font-size:11px; padding:1px 7px; border-radius:5px; background:var(--surface); border:1px solid var(--border); color:var(--t-muted); }
.gap-chip.chip-section { color:var(--accent-ink); border-color:var(--accent-bd); background:var(--accent-soft); }
.gap-act { height:32px; padding:0 14px; border-radius:8px; border:1px solid var(--warn); background:transparent;
  color:var(--warn); font-size:12.5px; font-weight:600; white-space:nowrap; transition:all .14s; flex:none; align-self:center; display:inline-flex; align-items:center; gap:6px; }
.gap-act:hover { background:var(--warn); color:#fff; }
.gap-act:disabled { opacity:.5; cursor:default; }
.gap-act svg { width:15px; height:15px; }

/* pipeline disclosure (wraps viz + trace + models) */
.pipe { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); box-shadow:var(--shadow-sm); overflow:hidden; }
.pipe-head { display:flex; align-items:center; gap:12px; padding:13px 18px; width:100%; background:transparent; border:0; text-align:left; transition:background .12s; }
.pipe-head:hover { background:var(--surface-2); }
.pipe-head > svg.lead { width:16px; height:16px; color:var(--t-subtle); flex:none; }
.pipe-title { font-size:13px; font-weight:600; color:var(--ink); white-space:nowrap; }
.pipe-summary { display:flex; gap:14px; font-size:12px; color:var(--t-subtle); font-family:var(--mono); white-space:nowrap; margin-left:6px; }
.pipe-summary b { color:var(--t-muted); font-weight:500; }
.pipe-chev { color:var(--t-faint); transition:transform .18s; margin-left:auto; display:grid; flex:none; }
.pipe-chev svg { width:16px; height:16px; }
.pipe.open .pipe-chev { transform:rotate(180deg); }
.pipe-body { display:none; padding:6px 18px 18px; border-top:1px solid var(--border); }
.pipe.open .pipe-body { display:block; animation:fadeUp .2s ease; }
/* When expanded, the trace breaks out of the answer column to (near) full
   screen width so the flow chart has room. The .split sidebar (332px) + gap
   shift the main column's center left of the viewport center, so we re-center
   by offsetting the transform by half that amount. */
.pipe.open { position:relative; z-index:20; width:min(96vw, 1500px);
  left:50%; transform:translateX(calc(-50% + (332px + var(--gap-block)) / 2)); }
@media (max-width:880px) { .pipe.open { width:auto; left:auto; transform:none; } }

/* pipeline viz */
.viz-section { margin-top:8px; }
.viz-header-row { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:10px; }
.viz-header-row h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--t-subtle); font-weight:600; }
.viz-sub { font-size:11.5px; color:var(--t-faint); line-height:1.5; max-width:560px; margin-top:4px; }
.viz-nav { display:flex; align-items:center; gap:8px; flex:none; }
.viz-nav-btn { width:28px; height:28px; border:1px solid var(--border); border-radius:7px; background:var(--surface); color:var(--t-muted); display:grid; place-items:center; }
.viz-nav-btn:hover { border-color:var(--border-2); color:var(--ink); }
.viz-nav-label { font-size:11.5px; color:var(--t-subtle); font-family:var(--mono); white-space:nowrap; }
.viz-container { position:relative; border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--surface-2); padding:14px; overflow:auto; }
.viz-container svg { max-width:100%; height:auto; }
.viz-empty { color:var(--t-faint); font-size:12.5px; padding:24px; text-align:center; }
/* node hover highlight */
#pipeline-viz g.node { transition:filter .12s ease; }
#pipeline-viz g.node:hover { filter:drop-shadow(0 3px 9px rgba(0,0,0,.24)); }
#pipeline-viz g.node:hover rect,
#pipeline-viz g.node:hover polygon,
#pipeline-viz g.node:hover circle,
#pipeline-viz g.node:hover path { stroke-width:2.5px !important; }
/* per-iteration pass navigation (top-right of the chart) */
.viz-pass-nav { position:absolute; right:10px; top:10px; z-index:5; display:flex; align-items:center; gap:5px; padding:4px 6px; background:var(--surface); border:1px solid var(--border); border-radius:9px; box-shadow:var(--shadow-sm); }
.viz-pass-nav button { width:26px; height:26px; border:1px solid var(--border); border-radius:6px; background:var(--surface); color:var(--t-muted); display:grid; place-items:center; font-size:14px; line-height:1; padding:0; cursor:pointer; transition:border-color .12s, color .12s; }
.viz-pass-nav button:hover:not(:disabled) { border-color:var(--border-2); color:var(--ink); }
.viz-pass-nav button:disabled { opacity:.35; cursor:default; }
.viz-pass-label { font-size:11.5px; color:var(--t-subtle); font-family:var(--mono); white-space:nowrap; padding:0 3px; }
.viz-legend { display:flex; flex-wrap:wrap; gap:10px 16px; margin-top:10px; }
.viz-legend-item { display:inline-flex; align-items:center; gap:6px; font-size:11px; color:var(--t-subtle); }
.viz-legend-swatch { width:11px; height:11px; border-radius:3px; border:1px solid; flex:none; }

/* trace details */
.trace-details { margin-top:14px; border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--surface); overflow:hidden; }
.trace-summary { display:flex; align-items:center; gap:9px; padding:11px 14px; cursor:pointer; font-size:12.5px; font-weight:600; color:var(--ink); list-style:none; }
.trace-summary::-webkit-details-marker { display:none; }
.trace-summary:hover { background:var(--surface-2); }
.trace-count { color:var(--t-faint); font-weight:400; font-family:var(--mono); font-size:11.5px; }
.trace-summary-chevron { margin-left:auto; color:var(--t-faint); font-size:10px; transition:transform .18s; }
.trace-details[open] .trace-summary-chevron { transform:rotate(180deg); }
#pipeline-stages { padding:6px 14px 14px; }
.pipeline-stage { border-bottom:1px dashed var(--border); }
.pipeline-stage:last-child { border-bottom:0; }
.pipeline-stage.stage-group-agentic { background:color-mix(in srgb, var(--accent) 4%, transparent); border-radius:var(--radius-xs); }
.stage-header { display:flex; align-items:center; gap:12px; padding:11px 8px; cursor:pointer; }
.stage-header:hover { background:var(--surface-2); }
.stage-title-block { flex:1; min-width:0; display:flex; flex-direction:column; gap:1px; }
.stage-name { font-size:13px; font-weight:500; color:var(--ink); }
.stage-index { color:var(--t-faint); font-family:var(--mono); margin-right:6px; font-size:11.5px; }
.stage-subtitle { font-size:11.5px; color:var(--t-faint); }
.stage-time { font-family:var(--mono); font-size:11.5px; color:var(--t-faint); flex:none; }
.stage-time-slow { color:var(--warn); }
.stage-toggle { color:var(--t-faint); font-size:10px; transition:transform .18s; flex:none; }
.stage-header.open .stage-toggle { transform:rotate(180deg); }
.stage-content { display:none; padding:4px 8px 14px; }
.stage-content.open { display:block; }
.stage-metrics { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
.stage-chip { font-family:var(--mono); font-size:11px; padding:2px 8px; border-radius:5px; background:var(--subtle); border:1px solid var(--border); color:var(--t-muted); }
.stage-chip-ok { color:var(--ok); background:var(--ok-soft); border-color:transparent; }
.stage-chip-warn { color:var(--warn); background:var(--warn-soft); border-color:transparent; }
.stage-chip-error { color:var(--danger); background:var(--danger-soft); border-color:transparent; }
.stage-chip-info { color:var(--accent-ink); background:var(--accent-soft); border-color:var(--accent-bd); }
.stage-chip-skipped { color:var(--t-faint); }
.stage-kv { display:flex; gap:12px; padding:6px 0; font-size:12.5px; align-items:flex-start; }
.stage-kv-label { min-width:120px; flex:none; color:var(--t-subtle); font-weight:500; }
.stage-kv-value { flex:1; min-width:0; color:var(--t-strong); }
.stage-kv-value code { font-family:var(--mono); font-size:.9em; background:var(--subtle); padding:1px 5px; border-radius:4px; border:1px solid var(--border); }
.stage-list { margin:0; padding-left:18px; }
.stage-list li { margin:2px 0; }
.stage-tag { display:inline-flex; align-items:center; gap:5px; font-size:11.5px; padding:2px 8px; border-radius:99px; background:var(--surface-2); border:1px solid var(--border); margin:2px 4px 2px 0; }
.stage-tag-kind { font-size:9.5px; text-transform:uppercase; letter-spacing:.04em; color:var(--t-faint); }
.stage-mono { font-family:var(--mono); }
.stage-meta { color:var(--t-faint); }
.stage-hits { width:100%; border-collapse:collapse; font-size:11.5px; margin-top:4px; }
.stage-hits th, .stage-hits td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--border); }
.stage-hits th { color:var(--t-faint); font-weight:600; text-transform:uppercase; font-size:10px; letter-spacing:.04em; }
.stage-raw { margin-top:10px; }
.stage-raw summary { font-size:11.5px; color:var(--t-subtle); cursor:pointer; }
.stage-json { font-family:var(--mono); font-size:11px; background:var(--subtle); border:1px solid var(--border); border-radius:var(--radius-xs); padding:10px; overflow:auto; margin-top:6px; max-height:340px; color:var(--t-muted); }

/* model picker locked while agentic is on (always uses the strongest model) */
select.locked-agentic { opacity:.55; cursor:not-allowed; }

/* model panel */
.model-panel { margin-top:14px; border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--surface); overflow:hidden; }
.model-panel-header { display:flex; align-items:center; gap:9px; padding:11px 14px; width:100%; background:transparent; border:0; text-align:left; font-size:12.5px; font-weight:600; color:var(--ink); }
.model-panel-header:hover { background:var(--surface-2); }
.model-panel-badge { font-family:var(--mono); font-size:11px; color:var(--ok); background:var(--ok-soft); padding:1px 8px; border-radius:99px; }
.model-panel-chevron { margin-left:auto; color:var(--t-faint); font-size:10px; }
.model-panel-body { display:none; padding:0 14px 14px; }
.model-panel-body.open { display:block; }
.model-table { width:100%; border-collapse:collapse; font-size:11.5px; margin-top:6px; }
.model-table th, .model-table td { text-align:left; padding:7px 9px; border-bottom:1px solid var(--border); }
.model-table th { color:var(--t-faint); font-weight:600; text-transform:uppercase; font-size:10px; letter-spacing:.04em; }
.model-table code { font-family:var(--mono); font-size:.92em; color:var(--accent-ink); }
.model-table .model-note { color:var(--t-faint); }
.model-row-active { background:var(--accent-soft); }
.model-cost-row { display:flex; gap:8px; align-items:baseline; margin-top:12px; padding-top:10px; border-top:1px dashed var(--border); font-size:12px; color:var(--t-muted); font-family:var(--mono); }
.model-cost-label { color:var(--t-subtle); font-family:var(--sans); font-weight:500; }
.model-cost-sep { color:var(--t-faint); }
.model-cost-total { color:var(--ink); font-weight:600; }

/* loading */
.loading { display:flex; align-items:center; gap:13px; background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); box-shadow:var(--shadow-sm); padding:18px 20px; }
.loading-spinner { width:18px; height:18px; border:2px solid var(--border-2); border-top-color:var(--accent); border-radius:50%; animation:spin .7s linear infinite; flex:none; }
.loading-body { flex:1; min-width:0; }
.loading-title { font-size:14px; font-weight:600; color:var(--ink); }
.loading-meta { font-size:12px; color:var(--t-subtle); font-family:var(--mono); }
.loading-cancel { height:30px; padding:0 13px; border:1px solid var(--border); border-radius:7px; background:var(--surface); color:var(--t-muted); font-size:12.5px; font-weight:500; }
.loading-cancel:hover:not(:disabled) { border-color:var(--danger); color:var(--danger); }
.loading-cancel:disabled { opacity:.5; cursor:default; }
/* live progress — a single "Thinking…" line with a cycling activity ticker */
.loading { align-items:flex-start; }
.loading-ticker { margin-top:3px; font-size:13px; color:var(--accent-ink); line-height:1.5;
  min-height:1.5em; will-change:opacity, transform; }
.loading-ticker.ticker-in { animation:tickerIn .42s cubic-bezier(.16,.84,.44,1); }
@keyframes tickerIn { from { opacity:0; transform:translateY(7px); } to { opacity:1; transform:none; } }

/* error */
.error { background:var(--danger-soft); color:var(--danger); border:1px solid color-mix(in srgb, var(--danger) 30%, transparent); border-radius:var(--radius-sm); padding:12px 15px; font-size:13px; }

/* stage popup (multiple open at once, draggable, FIFO-capped) */
.stage-popup { position:fixed; z-index:2600; width:min(560px,92vw); max-height:78vh;
  background:var(--surface); border:1px solid var(--border-2); border-radius:var(--radius); box-shadow:var(--shadow-pop); display:flex; flex-direction:column; }
.stage-popup-header { display:flex; align-items:center; justify-content:space-between; gap:10px; padding:12px 16px; border-bottom:1px solid var(--border); cursor:move; font-weight:600; font-size:13px; color:var(--ink); }
.stage-popup-title { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.stage-popup-actions { display:flex; align-items:center; gap:2px; flex:none; }
.stage-popup-close { width:26px; height:26px; border:0; background:transparent; color:var(--t-faint);
  border-radius:6px; font-size:14px; cursor:pointer; display:flex; align-items:center; justify-content:center; line-height:1; }
.stage-popup-close:hover { background:var(--surface-2); color:var(--ink); }
/* "close all" is a small TEXT button to the LEFT of the per-popup close, only
   shown when 2+ popups are open — text vs icon reads clearer than two X's. */
.stage-popup-close-all { display:none; align-items:center; height:24px; padding:0 9px; border:0; background:transparent;
  color:var(--t-faint); border-radius:6px; font-size:11px; font-weight:500; cursor:pointer; white-space:nowrap; line-height:1; }
.stage-popup-close-all:hover { background:var(--warn-soft); color:var(--warn); }
.stage-popup.multi .stage-popup-close-all { display:inline-flex; }
/* hairline divider so the text button and the close icon read as separate */
.stage-popup.multi .stage-popup-close { margin-left:4px; border-left:1px solid var(--border); border-radius:0 6px 6px 0; padding-left:2px; width:28px; }
.stage-popup-body { padding:14px 16px; overflow:auto; }
/* instant, brief hover hint for the close buttons (native title is too slow) */
.popup-tip { position:fixed; z-index:3000; pointer-events:none; background:var(--ink); color:var(--surface);
  font-size:11px; font-weight:500; padding:3px 7px; border-radius:5px; white-space:nowrap;
  opacity:0; transform:translateY(2px); transition:opacity .08s ease, transform .08s ease; box-shadow:var(--shadow-pop); }
.popup-tip.show { opacity:1; transform:translateY(0); }

/* citation preview popover (click a blue section chip in the answer) */
.cite-pop { position:fixed; z-index:3000; width:min(420px,90vw); background:var(--surface);
  border:1px solid var(--border); border-radius:10px; box-shadow:0 12px 32px rgba(0,0,0,.18);
  padding:12px 14px; }
.cite-pop-title { font-size:12.5px; font-weight:600; color:var(--ink); margin-bottom:6px; }
.cite-pop-title .mono { font-family:var(--mono); }
.cite-pop-body { font-size:12.5px; color:var(--t-muted); line-height:1.55; max-height:180px; overflow:auto; }
.cite-pop-foot { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-top:10px; }
.cite-pop-page { font-size:11.5px; color:var(--t-faint); }
.cite-pop-open { border:1px solid var(--border); background:var(--surface-2); color:var(--ink);
  font-size:12px; font-weight:600; padding:5px 10px; border-radius:7px; cursor:pointer; }
.cite-pop-open:hover { background:var(--accent); border-color:var(--accent); color:#fff; }

/* overlays (agentic confirm + config) */
.ag-confirm-overlay, .ag-config-overlay { position:fixed; inset:0; z-index:2100; background:color-mix(in srgb, var(--ink) 38%, transparent);
  display:flex; align-items:center; justify-content:center; padding:20px; backdrop-filter:blur(2px); animation:fadeUp .14s ease; }
.ag-confirm, .ag-config { width:min(560px,94vw); max-height:88vh; overflow:auto; background:var(--surface); border:1px solid var(--border-2);
  border-radius:var(--radius); box-shadow:var(--shadow-pop); padding:22px; }
.ag-confirm-title, .ag-config-title { font-size:16px; font-weight:600; letter-spacing:-0.02em; color:var(--ink); }
.ag-confirm-sub, .ag-config-sub { font-size:12.5px; color:var(--t-subtle); margin-top:4px; }
.ag-config-header { margin-bottom:16px; }
.ag-confirm-rows { display:grid; grid-template-columns:auto 1fr; gap:8px 16px; margin:16px 0; padding:14px 16px; background:var(--surface-2); border:1px solid var(--border); border-radius:var(--radius-sm); }
.ag-confirm-key { font-size:12px; color:var(--t-subtle); font-weight:500; }
.ag-confirm-val { font-size:12.5px; color:var(--ink); font-family:var(--mono); text-align:right; }
.ag-confirm-actions, .ag-config-footer { display:flex; gap:9px; justify-content:flex-end; margin-top:18px; }
.ag-confirm-btn { height:36px; padding:0 16px; border-radius:8px; font-size:13px; font-weight:600; border:1px solid var(--border); background:var(--surface); color:var(--t-muted); transition:all .14s; display:inline-flex; align-items:center; gap:6px; }
.ag-confirm-btn:hover { border-color:var(--border-2); color:var(--ink); background:var(--surface-2); }
.ag-confirm-btn-run { background:var(--accent); border-color:var(--accent); color:#fff; }
.ag-confirm-btn-run:hover { filter:brightness(1.1); background:var(--accent); color:#fff; }
.ag-confirm-btn-edit { border-color:var(--accent-bd); color:var(--accent-ink); background:var(--accent-soft); }
.ag-config-body { display:flex; flex-direction:column; gap:14px; }
.ag-config-body .agentic-config,
.ag-config-body .agentic-config.hidden { display:block !important; }
.ag-config-body .config-grid { grid-template-columns:1fr 1fr 1fr; }

/* ── Flag answer: sticky FAB + modal ─────────────────────────────────────── */
.flag-fab { position:fixed; right:20px; bottom:20px; z-index:1500; width:44px; height:44px;
  display:inline-flex; align-items:center; justify-content:center; border-radius:999px;
  background:var(--surface); color:var(--text); border:1px solid var(--border);
  box-shadow:0 2px 10px rgba(0,0,0,.18); cursor:pointer;
  transition:transform .12s ease, color .12s ease, border-color .12s ease; }
.flag-fab:hover { transform:translateY(-1px); color:var(--accent); border-color:var(--accent); }
.flag-fab.flagged { color:var(--danger); border-color:var(--danger); cursor:default; }
.flag-fab[hidden] { display:none; }
.flag-modal-overlay { position:fixed; inset:0; z-index:2200; background:color-mix(in srgb, var(--ink) 38%, transparent);
  display:flex; align-items:center; justify-content:center; padding:20px; backdrop-filter:blur(2px); animation:fadeUp .14s ease; }
.flag-modal-overlay[hidden] { display:none; }
.flag-modal { width:min(460px,94vw); background:var(--surface); border:1px solid var(--border-2);
  border-radius:var(--radius); box-shadow:var(--shadow-pop); padding:22px; }
.flag-modal-title { font-size:16px; font-weight:600; letter-spacing:-0.02em; color:var(--ink); }
.flag-modal-sub { font-size:12.5px; color:var(--t-subtle); margin-top:4px; }
.flag-modal textarea { width:100%; margin-top:14px; min-height:84px; resize:vertical; padding:10px 12px;
  font:inherit; font-size:13px; color:var(--ink); background:var(--surface-2);
  border:1px solid var(--border); border-radius:var(--radius-sm); box-sizing:border-box; }
.flag-modal textarea:focus { outline:none; border-color:var(--accent); }
.flag-modal-err { font-size:12px; color:var(--danger); margin-top:10px; min-height:14px; }
.flag-modal-actions { display:flex; gap:9px; justify-content:flex-end; margin-top:14px; }
.flag-toast { position:fixed; right:20px; bottom:74px; z-index:2300; padding:9px 14px; border-radius:8px;
  background:var(--surface); border:1px solid var(--border-2); box-shadow:var(--shadow-pop);
  font-size:12.5px; color:var(--ink); animation:fadeUp .14s ease; }
.flag-toast[hidden] { display:none; }

/* ── Dev panel: bottom-left FAB + flags/notes drawer ─────────────────────── */
.dev-fab { position:fixed; left:20px; bottom:20px; z-index:1500; width:40px; height:40px;
  display:inline-flex; align-items:center; justify-content:center; border-radius:10px;
  background:var(--surface); color:var(--t-subtle); border:1px solid var(--border);
  box-shadow:0 2px 10px rgba(0,0,0,.14); cursor:pointer; opacity:.7;
  transition:transform .12s ease, color .12s ease, border-color .12s ease, opacity .12s ease; }
.dev-fab:hover { transform:translateY(-1px); color:var(--accent); border-color:var(--accent); opacity:1; }
.dev-overlay { position:fixed; inset:0; z-index:2400; background:color-mix(in srgb, var(--ink) 38%, transparent);
  display:flex; align-items:center; justify-content:center; padding:20px; backdrop-filter:blur(2px); animation:fadeUp .14s ease; }
.dev-overlay[hidden] { display:none; }
.dev-panel { width:min(940px,96vw); max-height:90vh; display:flex; flex-direction:column;
  background:var(--surface); border:1px solid var(--border-2); border-radius:var(--radius); box-shadow:var(--shadow-pop); overflow:hidden; }
.dev-head { display:flex; align-items:center; gap:10px; padding:12px 16px; border-bottom:1px solid var(--border); }
.dev-tabs { display:flex; gap:4px; }
.dev-tab { height:30px; padding:0 13px; border:1px solid transparent; border-radius:7px; background:transparent;
  font-size:12.5px; font-weight:600; color:var(--t-subtle); cursor:pointer; }
.dev-tab:hover { color:var(--ink); background:var(--surface-2); }
.dev-tab.active { color:var(--accent-ink); background:var(--accent-soft); border-color:var(--accent-bd); }
.dev-head-title { font-size:13px; font-weight:600; color:var(--ink); }
.dev-close { margin-left:auto; width:28px; height:28px; border:0; background:transparent; color:var(--t-faint); border-radius:6px; font-size:15px; cursor:pointer; }
.dev-close:hover { background:var(--surface-2); color:var(--ink); }
.dev-body { padding:14px 16px; overflow:auto; }
.dev-toolbar { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
.dev-btn { height:30px; padding:0 12px; border:1px solid var(--border); border-radius:7px; background:var(--surface);
  color:var(--t-muted); font-size:12px; font-weight:600; cursor:pointer; }
.dev-btn:hover { border-color:var(--border-2); color:var(--ink); background:var(--surface-2); }
.dev-btn-primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.dev-btn-primary:hover { filter:brightness(1.08); background:var(--accent); color:#fff; }
.dev-btn-danger { color:var(--danger); border-color:color-mix(in srgb, var(--danger) 40%, var(--border)); }
.dev-btn-danger:hover { color:#fff; background:var(--danger); border-color:var(--danger); }
.dev-count { font-size:12px; color:var(--t-subtle); font-family:var(--mono); }
.dev-empty { font-size:12.5px; color:var(--t-subtle); padding:18px 4px; }
/* flag rows (master list — click a row to open its detail page) */
.dev-row { display:flex; align-items:center; gap:10px; width:100%; text-align:left;
  border:1px solid var(--border); border-radius:var(--radius-sm); margin-bottom:8px;
  background:var(--surface); padding:10px 12px; cursor:pointer;
  transition:border-color .12s ease, background .12s ease; }
.dev-row:hover { background:var(--surface-2); border-color:var(--border-2); }
.dev-row-main { flex:1; min-width:0; }
.dev-row-q { font-size:13px; font-weight:600; color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.dev-row-sub { font-size:11.5px; color:var(--t-subtle); margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.dev-row-meta { display:flex; flex-direction:column; align-items:flex-end; gap:4px; flex:none; }
.dev-chip { font-family:var(--mono); font-size:10px; padding:1px 7px; border-radius:99px; background:var(--surface-2); border:1px solid var(--border); color:var(--t-muted); }
.dev-chip.agentic { color:var(--accent-ink); }
.dev-row-date { font-size:10.5px; color:var(--t-faint); font-family:var(--mono); }
.dev-row-arrow { flex:none; color:var(--t-faint); display:flex; align-items:center; }
/* flag detail page (drills in over the list, back button returns) */
.dev-back { display:inline-flex; align-items:center; gap:6px; }
.dev-detail-q { font-size:14.5px; font-weight:600; color:var(--ink); line-height:1.45; word-break:break-word; margin:2px 0 8px; }
.dev-detail-meta { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:2px; }
.dev-field { margin-top:12px; }
.dev-field-label { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.04em; color:var(--t-faint); margin-bottom:5px; }
.dev-field-val { font-size:12.5px; color:var(--ink); white-space:pre-wrap; word-break:break-word; }
.dev-answer { font-size:12.5px; color:var(--text); background:var(--surface-2); border:1px solid var(--border); border-radius:var(--radius-xs); padding:10px 12px; max-height:280px; overflow:auto; line-height:1.5; }
.dev-kv { display:grid; grid-template-columns:auto 1fr; gap:4px 14px; font-size:12px; }
.dev-kv dt { color:var(--t-subtle); font-family:var(--mono); }
.dev-kv dd { margin:0; color:var(--ink); font-family:var(--mono); word-break:break-word; }
.dev-list { list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:4px; }
.dev-list li { font-size:11.5px; color:var(--t-muted); font-family:var(--mono); }
.dev-stage-num { color:var(--t-faint); margin-right:6px; }
/* notes */
.dev-note-input { width:100%; min-height:74px; resize:vertical; padding:10px 12px; font:inherit; font-size:13px;
  color:var(--ink); background:var(--surface-2); border:1px solid var(--border); border-radius:var(--radius-sm); box-sizing:border-box; }
.dev-note-input:focus { outline:none; border-color:var(--accent); }
.dev-note-bar { display:flex; justify-content:flex-end; gap:9px; margin:10px 0 16px; align-items:center; }
.dev-note-err { font-size:12px; color:var(--danger); margin-right:auto; }
.dev-note { border:1px solid var(--border); border-radius:var(--radius-sm); padding:10px 12px; margin-bottom:8px; background:var(--surface); }
.dev-note-head { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:5px; }
.dev-note-date { font-size:10.5px; color:var(--t-faint); font-family:var(--mono); }
.dev-note-del { width:22px; height:22px; flex:none; border:0; background:transparent; color:var(--t-faint); border-radius:6px; cursor:pointer; font-size:12px; line-height:1; }
.dev-note-del:hover { background:var(--surface-2); color:var(--danger); }
.dev-note-body { font-size:12.5px; color:var(--ink); white-space:pre-wrap; word-break:break-word; }
/* dev flow chart (reconstructed from stored pipeline_trace) */
.dev-flow-host { position:relative; margin-top:8px; border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--surface-2); padding:12px; overflow:auto; }
.dev-flow-host:empty { display:none; }
.dev-flow-host svg { max-width:100%; height:auto; }
    </style>
    <script>
        // Apply the saved (or system-preferred) theme before first paint to
        // avoid a light-mode flash on load.
        (function () {
            try {
                var saved = localStorage.getItem("specgpt-theme");
                var dark = saved ? saved === "dark"
                    : window.matchMedia("(prefers-color-scheme: dark)").matches;
                if (dark) document.documentElement.setAttribute("data-theme", "dark");
            } catch (e) { /* localStorage unavailable - default to light */ }
        })();
    </script>
</head>
<body>
<div id="root" class="app">
    <header class="topbar">
        <div class="wrap topbar-inner">
            <div class="brand">
                <div class="brand-name">spec<b>GPT</b></div>
                <span class="brand-tag">NVMe Spec Q&amp;A</span>
            </div>
            <div class="topbar-right">
                <button id="theme-toggle" class="icon-btn" type="button" role="switch"
                        title="Toggle dark mode" aria-label="Toggle dark mode">
                    <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
                    <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
                </button>
                <label class="picker" title="Which NVMe specification to search">
                    <span class="picker-lbl">Spec</span>
                    <select id="global-spec-select"></select>
                </label>
                <form class="topbar-form" method="post" action="/logout">
                    <button type="submit" class="ghost-btn">Sign out</button>
                </form>
            </div>
        </div>
    </header>

    <main class="main">
        <div class="wrap">
            <div id="composer" class="composer">
                <div class="composer-row">
                    <div class="composer-input-wrap">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.2-3.2"/></svg>
                        <textarea id="query-input" autocomplete="off" rows="1"
                               placeholder="Ask about the NVMe spec...  e.g. What does CIRN indicate?"></textarea>
                    </div>
                    <button id="search-btn" class="ask-btn">
                        <span class="ask-btn-inner">Ask<span id="ask-spec-label" class="ask-spec-label">Base</span></span>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
                    </button>
                </div>

                <div class="controls">
                    <div class="config-pop-wrap">
                        <button id="config-toggle" class="pill" type="button">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5"/><circle cx="16" cy="6" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="13" cy="18" r="2"/></svg>
                            Config
                        </button>
                        <div id="config-panel" class="config-panel">
                            <strong>Pipeline configuration</strong>
                            <div class="config-grid">
                                <div class="config-item config-item-wide">
                                    <label>Regular LLM</label>
                                    <select id="config-llm_model" data-model-select="llm"></select>
                                </div>
                                <div class="config-item">
                                    <label>Context Chunks</label>
                                    <input type="number" id="config-final_rerank_topk" value="10" min="1" max="20" title="How many top-ranked sections are sent to the model. More = better coverage, higher cost.">
                                </div>
                                <div class="config-item">
                                    <label>Sub-queries</label>
                                    <input type="number" id="config-max_subqueries" value="3" min="1" max="5" title="Max focused sub-questions a relational/procedural query is split into before retrieval. More = wider coverage, higher cost.">
                                </div>
                                <div class="config-item">
                                    <label title="After answering, a cheap model checks for gaps and offers agentic refinement when coverage looks thin"><input type="checkbox" id="config-auto_gap_check" checked> Auto Gap Check</label>
                                </div>
                            </div>

                            <!-- Advanced retrieval internals: still applied per request (presets
                                 drive them; JS reads them when building the config payload), just
                                 not user-editable. Hidden, not removed, so every
                                 getElementById("config-*") call site keeps working. -->
                            <div hidden aria-hidden="true">
                                <input type="number" id="config-vector_topk" value="5">
                                <input type="number" id="config-tsvector_topk" value="5">
                                <input type="number" id="config-bm25_topk" value="5">
                                <input type="number" id="config-rrf_k" value="60">
                                <input type="number" id="config-rrf_output_topk" value="20">
                            </div>

                            <div class="config-section-label">Agentic refinement</div>
                            <div id="agentic-config" class="agentic-config">
                                <div class="config-grid">
                                    <div class="config-item config-item-wide">
                                        <label>Agentic LLM</label>
                                        <select id="config-agentic_model" data-model-select="agentic"></select>
                                    </div>
                                    <div class="config-item">
                                        <label title="Keep refining until the gap analyser reports no missing context"><input type="checkbox" id="config-agentic_recursive" checked> Recursive</label>
                                    </div>
                                    <div class="config-item">
                                        <label>Max Iterations</label>
                                        <input type="number" id="config-agentic_max_iterations" value="4" min="1" max="10">
                                    </div>
                                </div>
                                <div hidden aria-hidden="true">
                                    <input type="number" id="config-agentic_max_followups" value="3">
                                    <input type="number" id="config-agentic_rerank_topk" value="14">
                                    <input type="number" id="config-agentic_max_context_tokens" value="16000">
                                    <input type="number" id="config-agentic_max_output_tokens" value="2048">
                                    <input type="checkbox" id="config-agentic_targeted_fetch" checked>
                                    <input type="number" id="config-figure_reserve_tokens" value="3000">
                                </div>
                            </div>
                        </div>
                    </div>

                    <label class="pill preset-pill" title="Speed vs. depth preset — applies a bundle of pipeline settings">
                        <span style="font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--t-faint);font-weight:600;">Preset</span>
                        <select id="preset-select" style="border:0;background:transparent;color:inherit;font:inherit;font-weight:500;outline:none;cursor:pointer;"></select>
                    </label>
                    <script>
                    (function () {
                        // Config presets (#3): a dropdown that applies a bundle of
                        // pipeline settings to the existing config inputs. The server
                        // (/api/presets, mirroring orchestrator.PRESETS) is the source
                        // of truth; this fallback keeps the selector working if that
                        // fetch fails. agentic_model is listed before llm_model on
                        // purpose so the global picker (synced on each change, last
                        // one wins) ends up displaying the regular model.
                        var FALLBACK = {
                            fast:     { label: "Fast",     agentic: false, config: { agentic_model: "deepthought-claude-sonnet-4-6", llm_model: "deepthought-claude-sonnet-4-6", vector_topk: 6, tsvector_topk: 6, bm25_topk: 6, rrf_k: 60, rrf_output_topk: 12, final_rerank_topk: 5, max_subqueries: 1, auto_gap_check: false, agentic_max_followups: 2, agentic_rerank_topk: 10, agentic_max_context_tokens: 8000, agentic_max_output_tokens: 1024, agentic_targeted_fetch: true, agentic_recursive: false, agentic_max_iterations: 1, figure_reserve_tokens: 1500 } },
                            balanced: { label: "Balanced", agentic: false, config: { agentic_model: "deepthought-claude-sonnet-4-6", llm_model: "deepthought-claude-sonnet-4-6", vector_topk: 5, tsvector_topk: 5, bm25_topk: 5, rrf_k: 60, rrf_output_topk: 20, final_rerank_topk: 10, max_subqueries: 3, auto_gap_check: true, agentic_max_followups: 3, agentic_rerank_topk: 14, agentic_max_context_tokens: 16000, agentic_max_output_tokens: 2048, agentic_targeted_fetch: true, agentic_recursive: true, agentic_max_iterations: 4, figure_reserve_tokens: 3000 } },
                            thorough: { label: "Thorough", agentic: true,  config: { agentic_model: "deepthought-claude-sonnet-4-6", llm_model: "deepthought-claude-sonnet-4-6", vector_topk: 8, tsvector_topk: 8, bm25_topk: 8, rrf_k: 60, rrf_output_topk: 20, final_rerank_topk: 10, max_subqueries: 3, auto_gap_check: true, agentic_max_followups: 3, agentic_rerank_topk: 14, agentic_max_context_tokens: 16000, agentic_max_output_tokens: 2048, agentic_targeted_fetch: true, agentic_recursive: true, agentic_max_iterations: 4, figure_reserve_tokens: 3000 } }
                        };
                        var PRESETS = FALLBACK, DEFAULT = "balanced";
                        // True while apply() is programmatically writing inputs, so the
                        // divergence listeners below don't mistake it for a hand-edit.
                        var applying = false;
                        function setInput(key, val) {
                            var el = document.getElementById("config-" + key);
                            if (!el) return;
                            if (el.type === "checkbox") el.checked = !!val; else el.value = val;
                            el.dispatchEvent(new Event("change", { bubbles: true }));
                        }
                        function apply(name) {
                            var p = PRESETS[name] || PRESETS[DEFAULT];
                            if (!p) return;
                            applying = true;
                            try {
                                var cfg = p.config || {};
                                for (var k in cfg) if (Object.prototype.hasOwnProperty.call(cfg, k)) setInput(k, cfg[k]);
                                var tog = document.getElementById("agentic-toggle");
                                if (tog && tog.checked !== !!p.agentic) {
                                    tog.checked = !!p.agentic;
                                    tog.dispatchEvent(new Event("change", { bubbles: true }));
                                }
                            } finally { applying = false; }
                            try { localStorage.setItem("specgpt_preset", name); } catch (e) {}
                        }
                        function render(sel, cur) {
                            sel.innerHTML = Object.keys(PRESETS).map(function (k) {
                                return '<option value="' + k + '">' + (PRESETS[k].label || k) + "</option>";
                            }).join("") + '<option value="custom" disabled hidden>Custom</option>';
                            sel.value = (cur === "custom" || PRESETS[cur]) ? cur : DEFAULT;
                        }
                        // Hand-editing any config knob diverges from the applied preset;
                        // show "Custom" instead of letting the selector lie. Re-picking a
                        // preset re-applies the bundle. Not persisted: the last real
                        // preset stays in localStorage.
                        function markCustom() {
                            if (applying) return;
                            var sel = document.getElementById("preset-select");
                            if (sel && sel.value !== "custom") sel.value = "custom";
                        }
                        document.addEventListener("DOMContentLoaded", function () {
                            var sel = document.getElementById("preset-select");
                            if (!sel) return;
                            var cur = "balanced";
                            try { cur = localStorage.getItem("specgpt_preset") || "balanced"; } catch (e) {}
                            render(sel, cur);
                            apply(sel.value);
                            sel.addEventListener("change", function () {
                                if (sel.value !== "custom") apply(sel.value);
                            });
                            // #agentic-config is watched separately because the refine
                            // overlay moves that node out of #config-panel.
                            ["config-panel", "agentic-config"].forEach(function (id) {
                                var host = document.getElementById(id);
                                if (!host) return;
                                host.addEventListener("change", markCustom);
                                host.addEventListener("input", markCustom);
                            });
                            var tog = document.getElementById("agentic-toggle");
                            if (tog) tog.addEventListener("change", markCustom);
                            fetch("/api/presets").then(function (r) { return r.ok ? r.json() : null; }).then(function (d) {
                                if (d && d.presets) { PRESETS = d.presets; DEFAULT = d.default || "balanced"; render(sel, sel.value); }
                            }).catch(function () {});
                        });
                    })();
                    </script>

                    <button id="agentic-pill" class="pill" type="button"
                            title="Toggle agentic refinement for the next query">
                        <span class="pill-dot"></span>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="3" x2="12" y2="6"/><circle cx="12" cy="2.6" r="1" fill="currentColor" stroke="none"/><rect x="4" y="7" width="16" height="12" rx="3"/><circle cx="9" cy="13" r="1.3" fill="currentColor" stroke="none"/><circle cx="15" cy="13" r="1.3" fill="currentColor" stroke="none"/><line x1="10" y1="16.5" x2="14" y2="16.5"/></svg>
                        <span id="agentic-pill-label">Agentic</span>
                    </button>

                    <div class="controls-spacer"></div>

                    <div id="cost-estimator" class="cost-estimator" role="button" aria-expanded="false" tabindex="0" onclick="toggleCostBreakdown(event)">
                        <div class="cost-chip cost-summary">
                            <span class="cost-chip-lbl">Est. cost</span>
                            <span class="cost-total" id="cost-total">&ndash;</span>
                            <span class="cost-context" id="cost-context"></span>
                            <span class="cost-toggle" id="cost-toggle">&#9662;</span>
                        </div>
                        <div class="cost-breakdown" id="cost-breakdown"></div>
                    </div>
                </div>

                <div id="agentic-row" class="agentic-row">
                    <label>
                        <input type="checkbox" id="agentic-toggle">
                        <span>Agentic mode</span>
                    </label>
                    <span class="agentic-hint">
                        Decomposes the answer, runs follow-up retrieval to fill gaps, then
                        regenerates with the agentic model and a larger context. Slower
                        (about 30 to 90 seconds) and many times the cost, so leave it off
                        for routine queries. The Est. cost chip shows the price for the
                        current settings.
                    </span>
                </div>
            </div>

            <div id="empty-state" class="empty">
                <div class="empty-mark">
                    <img src="/static/favicon.png" alt="">
                </div>
                <h2>Ask the NVMe specification anything</h2>
                <p>Type a question and get a grounded, citation-backed answer. Every source
                   section is listed alongside, and you can inspect exactly how the answer
                   was retrieved.</p>
                <div class="examples" id="examples"></div>
            </div>

            <div id="results" class="hidden">
                <div id="error" class="error hidden"></div>

                <div id="loading" class="loading hidden" role="status" aria-live="polite">
                    <div class="loading-spinner" aria-hidden="true"></div>
                    <div class="loading-body">
                        <div class="loading-title" id="loading-title">Thinking…</div>
                        <div class="loading-ticker" id="loading-ticker"></div>
                        <div class="loading-meta"><span id="loading-elapsed">0.0s elapsed</span></div>
                    </div>
                    <button type="button" class="loading-cancel" id="loading-cancel">Cancel</button>
                </div>

                <div id="answer-section" class="hidden">
                    <div class="split">
                        <div class="split-main">
                            <div class="answer-box">
                                <div class="answer-meta" id="answer-meta"></div>
                                <h3>Answer</h3>
                                <div id="latency"></div>
                                <div id="answer-text" class="answer-text"></div>
                            </div>

                            <div id="agent-strip" class="agent-strip" aria-label="Agent activity"></div>

                            <div id="pipeline-disclosure" class="pipe">
                                <button class="pipe-head" id="pipe-head" type="button">
                                    <svg class="lead" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="6" height="5" rx="1.2"/><rect x="15" y="3" width="6" height="5" rx="1.2"/><rect x="9" y="16" width="6" height="5" rx="1.2"/><path d="M6 8v3a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V8M12 13v3"/></svg>
                                    <span class="pipe-title">How this answer was found</span>
                                    <span class="pipe-summary" id="pipe-summary"></span>
                                    <span class="pipe-chev"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg></span>
                                </button>
                                <div class="pipe-body" id="pipe-body">
                                    <div class="viz-section">
                                        <div class="viz-header-row">
                                            <div>
                                                <h2>Pipeline flow</h2>
                                                <div class="viz-sub">
                                                    Each color is a stage family. Branches show per-sub-query
                                                    retrieval (semantic, keyword, BM25); all paths merge through
                                                    rank fusion, dedup, rerank, generation. Click any node to inspect it.
                                                </div>
                                            </div>
                                            <div id="viz-nav" class="viz-nav" style="display:none">
                                                <button id="viz-nav-prev" class="viz-nav-btn" onclick="vizPrev()" title="Previous pass">&#8592;</button>
                                                <span id="viz-nav-label" class="viz-nav-label">Pass 1 / 1</span>
                                                <button id="viz-nav-next" class="viz-nav-btn" onclick="vizNext()" title="Next pass">&#8594;</button>
                                            </div>
                                        </div>
                                        <div id="pipeline-viz" class="viz-container">
                                            <div class="viz-empty">Run a query to see the pipeline flow.</div>
                                        </div>
                                        <div id="pipeline-legend" class="viz-legend" style="display:none">
                                            <span class="viz-legend-item"><span class="viz-legend-swatch" style="background:#ede9fe;border-color:#a78bfa"></span>Understand</span>
                                            <span class="viz-legend-item"><span class="viz-legend-swatch" style="background:#d1fae5;border-color:#34d399"></span>Structured lookup</span>
                                            <span class="viz-legend-item"><span class="viz-legend-swatch" style="background:#dbeafe;border-color:#93c5fd"></span>Semantic / keyword / BM25</span>
                                            <span class="viz-legend-item"><span class="viz-legend-swatch" style="background:#fef3c7;border-color:#fbbf24"></span>Fuse &amp; dedup</span>
                                            <span class="viz-legend-item"><span class="viz-legend-swatch" style="background:#fecaca;border-color:#fca5a5"></span>Rerank</span>
                                            <span class="viz-legend-item"><span class="viz-legend-swatch" style="background:#1e40af;border-color:#1e3a8a"></span>Generate</span>
                                            <span class="viz-legend-item"><span class="viz-legend-swatch" style="background:#f3e8ff;border-color:#c084fc"></span>Agentic pass</span>
                                            <span class="viz-legend-item"><span class="viz-legend-swatch" style="background:#10b981;border-color:#047857"></span>Final answer</span>
                                        </div>
                                    </div>

                                    <details id="trace-details" class="trace-details">
                                        <summary class="trace-summary">
                                            Pipeline trace
                                            <span id="trace-stage-count" class="trace-count"></span>
                                            <span class="trace-summary-chevron">&#9662;</span>
                                        </summary>
                                        <div id="pipeline-stages"></div>
                                    </details>

                                    <div class="model-panel" id="model-panel">
                                        <button class="model-panel-header" onclick="toggleModelPanel()" type="button">
                                            Models &amp; cost
                                            <span class="model-panel-badge" id="model-cost-badge" style="display:none"></span>
                                            <span class="model-panel-chevron" id="model-panel-chevron">&#9662;</span>
                                        </button>
                                        <div class="model-panel-body" id="model-panel-body">
                                            <table class="model-table" id="model-table">
                                                <thead>
                                                    <tr>
                                                        <th>Stage</th><th>Model</th><th>Provider</th>
                                                        <th>$/1M in</th><th>$/1M out</th><th>Note</th>
                                                    </tr>
                                                </thead>
                                                <tbody id="model-table-body">
                                                    <tr><td colspan="6" class="model-note">Loading...</td></tr>
                                                </tbody>
                                            </table>
                                            <div class="model-cost-row" id="model-cost-row" style="display:none"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <aside class="sources">
                            <div id="citations-box" class="citations hidden">
                                <div class="sources-head">
                                    <h3>Sources cited</h3>
                                    <span class="sources-count" id="sources-count"></span>
                                </div>
                                <div id="citations-list" class="src-list"></div>
                                <div class="sources-foot">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 16v-4m0-4h.01"/></svg>
                                    <span>A green check marks a section verified in the retrieved context. An amber question mark means the model cited it without it being in the retrieved context.</span>
                                </div>
                            </div>
                        </aside>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Stage-detail popups are created dynamically (up to 4, draggable, FIFO);
         see showStagePopup() / the stage-popup manager in the script below. -->

    <div id="ag-confirm-overlay" class="ag-confirm-overlay hidden" role="dialog" aria-modal="true">
        <div class="ag-confirm">
            <div class="ag-confirm-title">Run agentic refinement?</div>
            <div class="ag-confirm-sub">It will run with these settings. Edit the config first if you need to.</div>
            <div class="ag-confirm-rows" id="ag-confirm-rows"></div>
            <div class="ag-confirm-actions">
                <button class="ag-confirm-btn ag-confirm-btn-cancel" id="ag-confirm-cancel">Cancel</button>
                <button class="ag-confirm-btn ag-confirm-btn-edit" id="ag-confirm-edit">Edit config</button>
                <button class="ag-confirm-btn ag-confirm-btn-run" id="ag-confirm-run">Confirm &#8594;</button>
            </div>
        </div>
    </div>

    <div id="ag-config-overlay" class="ag-config-overlay hidden" role="dialog" aria-modal="true">
        <div class="ag-config">
            <div class="ag-config-header">
                <div class="ag-config-title">Agentic refinement config</div>
                <div class="ag-config-sub">Review and tweak these settings, then run the refinement.</div>
            </div>
            <div class="ag-config-body" id="ag-config-body">
            </div>
            <div class="ag-config-footer">
                <button class="ag-confirm-btn ag-confirm-btn-cancel" id="ag-config-cancel">Cancel</button>
                <button class="ag-confirm-btn ag-confirm-btn-run" id="ag-config-run">Run &#8594;</button>
            </div>
        </div>
    </div>
</div>

    <!-- Flag answer: sticky FAB, hidden until the first answer renders. -->
    <button id="flag-fab" class="flag-fab" type="button"
            aria-label="Flag this answer" hidden>
      <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
        <path d="M5 3v18M5 4h11l-2 4 2 4H5" fill="none"
              stroke="currentColor" stroke-width="2"
              stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </button>

    <div id="flag-modal-overlay" class="flag-modal-overlay" role="dialog" aria-modal="true"
         aria-labelledby="flag-modal-title" hidden>
        <div class="flag-modal">
            <div class="flag-modal-title" id="flag-modal-title">Flag this answer</div>
            <div class="flag-modal-sub">What was wrong with this answer? (optional)</div>
            <textarea id="flag-reason" placeholder="Optional. Describe the problem so we can reproduce it."></textarea>
            <div class="flag-modal-err" id="flag-modal-err"></div>
            <div class="flag-modal-actions">
                <button class="ag-confirm-btn ag-confirm-btn-cancel" id="flag-cancel" type="button">Cancel</button>
                <button class="ag-confirm-btn ag-confirm-btn-run" id="flag-submit" type="button">Submit flag</button>
            </div>
        </div>
    </div>

    <div id="flag-toast" class="flag-toast" hidden>Thanks, flagged</div>

    <!-- Dev panel: bottom-left FAB opens a drawer over flagged_answers + notes. -->
    <button id="dev-fab" class="dev-fab" type="button" aria-label="Open dev panel">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="m8 9-3 3 3 3M16 9l3 3-3 3M13 6l-2 12"/>
      </svg>
    </button>

    <div id="dev-overlay" class="dev-overlay" role="dialog" aria-modal="true" aria-label="Dev panel" hidden>
        <div class="dev-panel">
            <div class="dev-head">
                <div class="dev-tabs">
                    <button class="dev-tab active" type="button" data-devtab="flags">Flagged answers</button>
                    <button class="dev-tab" type="button" data-devtab="notes">Notes</button>
                </div>
                <button class="dev-close" id="dev-close" type="button" aria-label="Close">&#x2715;</button>
            </div>
            <div class="dev-body">
                <div id="dev-pane-flags" class="dev-pane">
                    <div id="dev-flags-listview">
                        <div class="dev-toolbar">
                            <button class="dev-btn" id="dev-refresh" type="button">Refresh</button>
                            <span class="dev-count" id="dev-flags-count"></span>
                        </div>
                        <div id="dev-flags-list"></div>
                    </div>
                    <div id="dev-flags-detailview" hidden>
                        <div class="dev-toolbar">
                            <button class="dev-btn dev-back" id="dev-flag-back" type="button">
                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m15 18-6-6 6-6"/></svg>
                                Back to flags
                            </button>
                        </div>
                        <div id="dev-flag-detail"></div>
                    </div>
                </div>
                <div id="dev-pane-notes" class="dev-pane" hidden>
                    <textarea class="dev-note-input" id="dev-note-input" placeholder="Write a note..."></textarea>
                    <div class="dev-note-bar">
                        <span class="dev-note-err" id="dev-note-err"></span>
                        <button class="dev-btn dev-btn-primary" id="dev-note-add" type="button">Add note</button>
                    </div>
                    <div id="dev-notes-list"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Markdown rendering: marked (parser) + DOMPurify (XSS sanitiser).
         LLM output is partially user-influenced via prompt injection, so we
         MUST sanitise the marked-generated HTML before injecting it into the
         DOM. Pinned to specific versions so the URL is effectively immutable. -->
    <script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"
            integrity="sha384-/TQbtLCAerC3jgaim+N78RZSDYV7ryeoBCVqTuzRrFec2akfBkHS7ACQ3PQhvMVi"
            crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.5/dist/purify.min.js"
            integrity="sha384-nszIONF2FGC59kn+pPFaRa6WUNGwsZgXZiJxJwQbym+TzcH7smolUviLgpPbNx7V"
            crossorigin="anonymous"></script>

    <!-- Mermaid: renders the pipeline_trace as a downward-facing DAG.
         securityLevel:'strict' so any text we interpolate into node labels
         is encoded; click events disabled. Mermaid's own renderer never
         executes user-supplied HTML. -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.0/dist/mermaid.min.js"
            integrity="sha384-Wm9qzEgq4j1jEnuFK2FxKTlwuhbV2QqtGhcchvjDoKxeJ7WWAW7fysBq+1s6myfX"
            crossorigin="anonymous"></script>

    <script>
        // GitHub-flavored markdown (tables, fenced code, autolinks).
        marked.setOptions({ gfm: true, breaks: false });

        // Mermaid: strict mode so any interpolated label text is encoded by
        // Mermaid itself; we also defensively scrub our own input. The base
        // theme variables are swapped per light/dark so the flow chart tracks
        // the page theme (re-applied before every render + on theme toggle).
        function applyMermaidTheme() {
            if (typeof mermaid === "undefined") return;
            var dark = document.documentElement.getAttribute("data-theme") === "dark";
            mermaid.initialize({
                startOnLoad: false,
                securityLevel: "strict",
                theme: "base",
                themeVariables: dark ? {
                    fontFamily: "'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                    fontSize: "12.5px",
                    primaryColor: "#1b2740",
                    primaryTextColor: "#e7eaf0",
                    primaryBorderColor: "#33415e",
                    lineColor: "#64748b",
                    textColor: "#cbd5e1",
                    nodeBorder: "#33415e",
                    mainBkg: "#1b2740",
                    clusterBkg: "#141d33",
                    clusterBorder: "#2a3956",
                } : {
                    fontFamily: "'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                    fontSize: "12.5px",
                    primaryColor: "#ffffff",
                    primaryTextColor: "#1c1917",
                    primaryBorderColor: "#d6d3d1",
                    lineColor: "#a8a29e",
                    textColor: "#1c1917",
                    nodeBorder: "#d6d3d1",
                    mainBkg: "#ffffff",
                    clusterBkg: "#fafaf9",
                    clusterBorder: "#e7e5e4",
                },
                flowchart: {
                    curve: "basis",
                    htmlLabels: true,
                    useMaxWidth: true,
                    diagramPadding: 24,
                    nodeSpacing: 48,
                    rankSpacing: 56,
                    padding: 14,
                },
            });
        }
        applyMermaidTheme();

        // ─── Model panel ──────────────────────────────────────────────────
        const MODEL_STAGE_LABELS = {
            embedding: "Embedding",
            reranker: "Reranker",
            llm: "LLM (standard)",
            agentic_llm: "LLM (agentic)",
        };
        let _modelsData = null;
        let _lastIsAgentic = false;

        function toggleModelPanel() {
            const body = document.getElementById("model-panel-body");
            const chevron = document.getElementById("model-panel-chevron");
            const open = body.classList.toggle("open");
            chevron.textContent = open ? "▲" : "▼";
        }

        function _fmtPrice(v) {
            if (v === null || v === undefined) return "n/a";
            if (v === 0) return "free";
            return "$" + v;
        }

        function _fmtCost(dollars) {
            if (dollars === null || dollars === undefined) return "n/a";
            if (dollars < 0.0001) return "<$0.0001";
            return "$" + dollars.toFixed(4);
        }

        // Single source of truth for selectable models. Both dropdowns and the
        // top-right global model picker render the exact same option list,
        // grouped by provider. Tags ("cheapest"/"fastest") are computed
        // per-provider from the price/latency fields so changing this catalog
        // automatically refreshes the badges everywhere.
        //
        //   in/out      - per-1M-token prices (USD)
        //   label       - short display name
        //   defaultFor  - which select(s) default to this model
        //   tags        - additional tags surfaced in the label
        // DeepThought (UNH on-prem gateway) is the only generation backend.
        // All models are UNH-hosted, so in/out price is 0 (the cost chip reads
        // these). Ids match generator.DEEPTHOUGHT_MODELS; keep the two in sync.
        // speed is a rough relative rank (1=fastest) for the "fastest" tag only.
        const MODEL_CATALOG = [
            // Claude via DeepThought's AWS Bedrock route - best citation fidelity
            {id: "deepthought-claude-sonnet-4-6", label: "Claude Sonnet 4.6", provider: "DeepThought (Bedrock)", in: 0.0, out: 0.0, speed: 5, tags: ["most capable"], defaultFor: ["llm", "agentic"]},
            {id: "deepthought-claude-sonnet-4-5", label: "Claude Sonnet 4.5", provider: "DeepThought (Bedrock)", in: 0.0, out: 0.0, speed: 5, tags: []},
            {id: "deepthought-claude-sonnet-4",   label: "Claude Sonnet 4",   provider: "DeepThought (Bedrock)", in: 0.0, out: 0.0, speed: 5, tags: []},
            // Open-weight models on UNH's own GPUs (free local compute)
            {id: "deepthought-llama-3.3-70b",        label: "Llama 3.3 70B",        provider: "DeepThought (Local)", in: 0.0, out: 0.0, speed: 4, tags: []},
            {id: "deepthought-qwen3-30b",            label: "Qwen3 30B Instruct",   provider: "DeepThought (Local)", in: 0.0, out: 0.0, speed: 3, tags: []},
        ];

        // Tag the per-provider cheapest (by input price) and fastest (by speed
        // rank) so the dropdown labels surface the obvious picks at a glance.
        (function _annotateCatalog() {
            const byProvider = {};
            for (const m of MODEL_CATALOG) {
                (byProvider[m.provider] = byProvider[m.provider] || []).push(m);
            }
            for (const list of Object.values(byProvider)) {
                if (list.length < 2) continue;
                const cheapest = list.reduce((a, b) => (a.in <= b.in ? a : b));
                const fastest  = list.reduce((a, b) => (a.speed <= b.speed ? a : b));
                if (cheapest.in > 0) cheapest.tags = [...cheapest.tags, "cheapest"];
                if (fastest !== cheapest) fastest.tags = [...fastest.tags, "fastest"];
            }
        })();

        const MODEL_PRICING = Object.fromEntries(
            MODEL_CATALOG.map(m => [m.id, {in: m.in, out: m.out}])
        );

        // Build the same set of <optgroup>/<option> entries for any select.
        // Pulls default selection from `defaultFor` so both dropdowns can share
        // this catalog while keeping their own per-role defaults.
        function _modelOptionLabel(m) {
            return m.tags.length ? `${m.label} (${m.tags.join(", ")})` : m.label;
        }
        function populateModelSelect(selectEl, role) {
            if (!selectEl) return;
            const byProvider = {};
            for (const m of MODEL_CATALOG) {
                if (!m.provider || !m.provider.startsWith("DeepThought")) continue;
                (byProvider[m.provider] = byProvider[m.provider] || []).push(m);
            }
            // Stable provider order matching the catalog declaration (only included providers).
            const providerOrder = [];
            for (const m of MODEL_CATALOG) {
                if (byProvider[m.provider] && !providerOrder.includes(m.provider)) providerOrder.push(m.provider);
            }
            let html = "";
            let defaultId = null;
            for (const provider of providerOrder) {
                html += `<optgroup label="${provider}">`;
                for (const m of byProvider[provider]) {
                    const isDefault = role && m.defaultFor && m.defaultFor.includes(role);
                    if (isDefault) defaultId = m.id;
                    html += `<option value="${m.id}">${_modelOptionLabel(m)}</option>`;
                }
                html += `</optgroup>`;
            }
            selectEl.innerHTML = html;
            if (defaultId) selectEl.value = defaultId;
        }

        // Render both model selects from the same catalog.
        populateModelSelect(document.getElementById("config-llm_model"),     "llm");
        populateModelSelect(document.getElementById("config-agentic_model"), "agentic");

        // ── Spec picker (Base vs PCIe Transport) ───────────────────────────
        // Scopes every query to one specification. Persisted in localStorage so
        // the choice survives reloads; sent as config.spec on each request.
        // Attached to window so runQuery / refine can read it regardless of
        // script-scope nesting.
        window.getSelectedSpec = function () {
            const el = document.getElementById("global-spec-select");
            if (el && el.value) return el.value;
            return localStorage.getItem("specgpt_spec") || "base";
        };
        (function _wireSpecPicker() {
            const el = document.getElementById("global-spec-select");
            if (!el) return;
            const _specShortNames = { base: "Base", pcie: "PCIe", command: "Cmd-Set", all: "All" };
            function _updateAskSpecLabel(specId) {
                const lbl = document.getElementById("ask-spec-label");
                if (lbl) lbl.textContent = _specShortNames[specId] || specId;
            }
            fetch("/api/specs")
                .then((r) => (r.ok ? r.json() : null))
                .then((data) => {
                    if (!data || !Array.isArray(data.specs)) return;
                    window._specData = data.specs;
                    const saved = localStorage.getItem("specgpt_spec");
                    el.innerHTML = data.specs
                        .map((s) => `<option value="${s.id}">${s.label}${s.version ? " " + s.version : ""}</option>`)
                        .join("");
                    el.value = (saved && data.specs.some((s) => s.id === saved))
                        ? saved
                        : (data.default || "base");
                    _updateAskSpecLabel(el.value);
                })
                .catch(() => {});
            el.addEventListener("change", () => {
                localStorage.setItem("specgpt_spec", el.value);
                _updateAskSpecLabel(el.value);
            });
            _updateAskSpecLabel(localStorage.getItem("specgpt_spec") || "base");
        })();

        // Overlay the model selectors onto `_modelsData` so the model panel +
        // cost calc reflect whatever the user picked. No-op until both the
        // /api/models response and the selectors are in the DOM.
        function _modelProvider(id) {
            // DeepThought is the only generation backend; every model id is
            // UNH-hosted. (gemini/gpt branches kept harmless for legacy ids.)
            if (!id) return "UNH DeepThought";
            if (id.startsWith("deepthought")) return "UNH DeepThought";
            if (id.startsWith("gemini-")) return "Google";
            if (id.startsWith("gpt-") || id.startsWith("o1") || id.startsWith("o3") || id.startsWith("o4")) return "OpenAI";
            return "UNH DeepThought";
        }

        function _applySelectedModels() {
            if (!_modelsData) return;
            const llmEl = document.getElementById("config-llm_model");
            const agEl  = document.getElementById("config-agentic_model");
            if (llmEl && _modelsData.llm) {
                const id = llmEl.value;
                const p  = MODEL_PRICING[id];
                _modelsData.llm.model    = id;
                _modelsData.llm.provider = _modelProvider(id);
                if (p) {
                    _modelsData.llm.price_per_1m_input  = p.in;
                    _modelsData.llm.price_per_1m_output = p.out;
                }
            }
            if (agEl && _modelsData.agentic_llm) {
                const id = agEl.value;
                const p  = MODEL_PRICING[id];
                _modelsData.agentic_llm.model    = id;
                _modelsData.agentic_llm.provider = _modelProvider(id);
                if (p) {
                    _modelsData.agentic_llm.price_per_1m_input  = p.in;
                    _modelsData.agentic_llm.price_per_1m_output = p.out;
                }
            }
        }

        function renderModelTable(isAgentic) {
            if (!_modelsData) return;
            _lastIsAgentic = isAgentic;
            _applySelectedModels();
            const tbody = document.getElementById("model-table-body");
            tbody.innerHTML = Object.entries(_modelsData).map(([key, info]) => {
                const active = (key === "llm" && !isAgentic) || (key === "agentic_llm" && isAgentic);
                return `<tr class="${active ? "model-row-active" : ""}">
                    <td>${MODEL_STAGE_LABELS[key] || key}</td>
                    <td><code>${escapeHtml(info.model)}</code></td>
                    <td>${escapeHtml(info.provider)}</td>
                    <td>${_fmtPrice(info.price_per_1m_input)}</td>
                    <td>${_fmtPrice(info.price_per_1m_output)}</td>
                    <td class="model-note">${escapeHtml(info.note || "")}</td>
                </tr>`;
            }).join("");
        }

        // Cost = Σ over every LLM call in the response. Each call carries its
        // own model so query-processor / gap-analysis (cheap Gemini) and the
        // final generation (Claude/Opus/etc.) get priced separately. Falls
        // back to the legacy single-model calc when the server hasn't sent a
        // per-call breakdown.
        function _costForCall(call) {
            const price = MODEL_PRICING[call.model] || {in: 0, out: 0};
            const inCost  = ((call.prompt     || 0) / 1e6) * price.in;
            const outCost = ((call.completion || 0) / 1e6) * price.out;
            return {inCost, outCost, total: inCost + outCost};
        }
        function _stageLabel(stage) {
            const s = String(stage || "");
            if (s === "query_processor")                  return "Query processor";
            if (s === "generation")                        return "Generation";
            if (s === "gap_hint")                          return "Gap hint";
            if (s.startsWith("agentic.gap_analysis"))      return "Gap analysis";
            if (s.startsWith("agentic.regenerate"))         return "Regenerate (agentic)";
            if (s.startsWith("agentic.followup_decomp"))    return "Follow-up decompose";
            return s.replace(/\\.iter\\d+$/, "");
        }
        function renderModelCost(tokensUsed, isAgentic) {
            if (!tokensUsed || !_modelsData) return;
            const calls = Array.isArray(tokensUsed.calls) ? tokensUsed.calls : null;

            let totalIn = 0, totalOut = 0, totalCost = 0;
            const perStage = [];

            if (calls && calls.length) {
                for (const c of calls) {
                    const {total} = _costForCall(c);
                    totalIn   += (c.prompt     || 0);
                    totalOut  += (c.completion || 0);
                    totalCost += total;
                    perStage.push({
                        stage: _stageLabel(c.stage),
                        model: c.model || "(unknown)",
                        prompt: c.prompt || 0,
                        completion: c.completion || 0,
                        cost: total,
                    });
                }
            } else {
                // Legacy fallback: single-model calc using the active panel model.
                const llm = isAgentic ? _modelsData.agentic_llm : _modelsData.llm;
                if (!llm) return;
                totalIn   = tokensUsed.prompt     || 0;
                totalOut  = tokensUsed.completion || 0;
                totalCost = (totalIn / 1e6) * llm.price_per_1m_input
                          + (totalOut / 1e6) * llm.price_per_1m_output;
            }

            const badge = document.getElementById("model-cost-badge");
            badge.textContent = _fmtCost(totalCost) + " / query";
            badge.style.display = "";

            const row = document.getElementById("model-cost-row");
            row.style.display = "flex";
            row.style.flexWrap = "wrap";
            const stageHtml = perStage.length
                ? `<div style="flex-basis:100%; margin-top:6px; padding-top:6px; border-top:1px dashed var(--border); display:flex; flex-direction:column; gap:3px;">
                    ${perStage.map(s => `
                        <div style="display:flex; gap:8px; font-size:11.5px; color:var(--text-muted); align-items:baseline;">
                            <span style="min-width:160px;">${escapeHtml(s.stage)}</span>
                            <code style="font-size:11px;">${escapeHtml(s.model)}</code>
                            <span style="margin-left:auto;">${s.prompt.toLocaleString()} in · ${s.completion.toLocaleString()} out</span>
                            <span class="model-cost-total" style="min-width:72px; text-align:right;">${_fmtCost(s.cost)}</span>
                        </div>`).join("")}
                  </div>`
                : "";
            row.innerHTML = `
                <span class="model-cost-label">Last query:</span>
                <span>${totalIn.toLocaleString()} in</span>
                <span class="model-cost-sep">+</span>
                <span>${totalOut.toLocaleString()} out</span>
                <span class="model-cost-sep">=</span>
                <span class="model-cost-total">${_fmtCost(totalCost)}</span>
                ${stageHtml}
            `;
        }

        // Fetch model info once on page load
        fetch("/api/models")
            .then(r => r.ok ? r.json() : null)
            .then(data => {
                if (!data) return;
                _modelsData = data;
                renderModelTable(false);
            })
            .catch(() => {});

        // Re-render the model panel whenever the user picks a different model
        // so the table + cost-per-query badge reflect the live selection.
        ["config-llm_model", "config-agentic_model"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener("change", () => renderModelTable(_lastIsAgentic));
        });

        // ─── Cost estimator ───────────────────────────────────────────────
        // Typical-case estimate of the *next* query, calibrated against real
        // per-stage token counts in qa_log. Updates live on every config
        // change. A worst-case figure (all iterations, max output) is shown
        // alongside in the breakdown but the headline number is the typical
        // case, since most agentic queries converge with 0-2 refinement
        // passes and never come close to the configured maximums.
        //
        // Calibration notes (qa_log, n=48 queries):
        //   • Support calls (query processor, gap analysis, follow-up
        //     decomposition) always run on the cheap utility model
        //     (src/llm/client DEFAULT_MODEL), not the selected models.
        //   • Targeted fetch is a deterministic DB lookup, no LLM call.
        //   • Median agentic query does 0 regeneration passes (verdict says
        //     complete on the first try); mean is ~1.8. We bill 2 as typical.
        //   • Real completion lengths sit far below the configured maximums:
        //     ~400 tok for generation, ~330 tok per agentic regeneration.
        //   • Figure reserve (figure_reserve_tokens) is additive context that
        //     only fills when the answer references figures. We bill ~1.5
        //     trimmed figure tables as typical per generation pass; worst case
        //     is the full configured reserve.
        const COST_ASSUMPTIONS = {
            support_price: {in: 0.15, out: 0.60}, // gpt-4o-mini utility model
            qp_in: 550,     qp_out: 70,           // query processor
            decomp_in: 550, decomp_out: 70,       // per follow-up decomposition
            decomp_calls: 2,                      // avg follow-ups decomposed
            gap_in: 1000,   gap_out: 130,         // gap-analysis verdict
            base_prompt: 2200,                    // system prompt + query + scaffolding
            avg_chunk_tokens: 450,                // effective per-chunk prompt cost
            gen_out: 400,                         // typical generation output
            regen_out: 330,                       // typical agentic regen output
            typical_iters: 2,                     // refinement passes billed as typical
            fig_typical_count: 1.5,               // referenced figures typically pulled into the reserve
            embedding_price_per_1m: 0.02,
        };

        function _fmtCostShort(d) {
            if (d === null || d === undefined || isNaN(d)) return "n/a";
            if (d < 0.0001) return "<$0.0001";
            if (d < 0.01)   return "$" + d.toFixed(4);
            if (d < 1)      return "$" + d.toFixed(3);
            return "$" + d.toFixed(2);
        }

        function _modelPrice(id, fallback) {
            return MODEL_PRICING[id] || fallback;
        }

        function _llmCallCost(inTok, outTok, price) {
            return (inTok / 1e6) * price.in + (outTok / 1e6) * price.out;
        }

        function _readCostInputs() {
            const v = id => document.getElementById(id);
            const num = (id, def) => {
                const el = v(id);
                if (!el) return def;
                const n = parseInt(el.value, 10);
                return isNaN(n) ? def : n;
            };
            const chk = (id) => { const el = v(id); return el ? el.checked : false; };
            const str = (id, def) => { const el = v(id); return el ? el.value : def; };
            return {
                agentic: v("agentic-toggle") ? v("agentic-toggle").checked : false,
                llm_model:                  str("config-llm_model", "deepthought-claude-sonnet-4-6"),
                agentic_model:              str("config-agentic_model", "deepthought-claude-sonnet-4-6"),
                final_rerank_topk:          num("config-final_rerank_topk", 10),
                auto_gap_check:             chk("config-auto_gap_check"),
                agentic_rerank_topk:        num("config-agentic_rerank_topk", 14),
                agentic_max_context_tokens: num("config-agentic_max_context_tokens", 16000),
                agentic_max_output_tokens:  num("config-agentic_max_output_tokens", 2048),
                agentic_max_followups:      num("config-agentic_max_followups", 3),
                agentic_targeted_fetch:     chk("config-agentic_targeted_fetch"),
                agentic_recursive:          chk("config-agentic_recursive"),
                agentic_max_iterations:     num("config-agentic_max_iterations", 4),
                figure_reserve_tokens:      num("config-figure_reserve_tokens", 3000),
            };
        }

        function estimateCost(cfg) {
            const A = COST_ASSUMPTIONS;
            const regPrice = _modelPrice(cfg.llm_model,     {in: 3,  out: 15});
            const agPrice  = _modelPrice(cfg.agentic_model, {in: 15, out: 75});
            const supPrice = A.support_price;
            const rows = [];
            // Figure reserve: additive context that fills only when the answer
            // references figures. Bill ~1.5 trimmed figures as typical per gen
            // pass; worst case is the full configured reserve.
            const figReserve = Math.max(0, cfg.figure_reserve_tokens || 0);
            const figTypical = Math.min(A.fig_typical_count * A.avg_chunk_tokens, figReserve);
            const figWorstExtraTok = figReserve - figTypical;
            // First-pass generation always runs, so its figure worst-case is
            // always in play.
            let worstExtra = _llmCallCost(figWorstExtraTok, 0, regPrice);

            // Embedding the query (negligible, shown for completeness).
            const embCost = (A.qp_in / 1e6) * A.embedding_price_per_1m;
            rows.push({
                name: "Query embedding",
                sub: `Voyage · ~${A.qp_in} tok`,
                value: embCost,
            });

            // Query processor (classification + decomposition, utility model).
            const qpCost = _llmCallCost(A.qp_in, A.qp_out, supPrice);
            rows.push({
                name: "Query processor",
                sub: `utility model · ~${A.qp_in} in / ~${A.qp_out} out`,
                value: qpCost,
            });

            // First-pass generation (always runs).
            const normalIn  = A.base_prompt + cfg.final_rerank_topk * A.avg_chunk_tokens + figTypical;
            const normalCost = _llmCallCost(normalIn, A.gen_out, regPrice);
            rows.push({
                name: "Generate (regular)",
                sub: `${cfg.llm_model} · ${cfg.final_rerank_topk} chunks → ~${normalIn.toLocaleString()} in / ~${A.gen_out} out`,
                value: normalCost,
            });

            // Optional auto-gap-check (regular mode only - agentic has its own).
            if (cfg.auto_gap_check && !cfg.agentic) {
                const gapCost = _llmCallCost(A.gap_in, A.gap_out, supPrice);
                rows.push({
                    name: "Auto gap check",
                    sub: `utility model · ~${A.gap_in.toLocaleString()} in / ~${A.gap_out} out`,
                    value: gapCost,
                });
            }

            // Agentic loop. Bill the typical number of refinement passes,
            // not the configured maximum: most queries converge immediately.
            // Targeted fetch is a free DB lookup so it gets no row.
            if (cfg.agentic) {
                const maxIters = cfg.agentic_recursive ? Math.max(1, cfg.agentic_max_iterations) : 1;
                const iters = Math.min(A.typical_iters, maxIters);
                const agIn = A.base_prompt + Math.min(
                    cfg.agentic_rerank_topk * A.avg_chunk_tokens,
                    cfg.agentic_max_context_tokens
                ) + figTypical;

                const decompCalls = Math.min(A.decomp_calls, Math.max(1, cfg.agentic_max_followups));
                const decompCost = _llmCallCost(A.decomp_in, A.decomp_out, supPrice) * decompCalls;
                rows.push({
                    name: "Follow-up decomposition",
                    sub: `utility model · ~${decompCalls}× ~${A.decomp_in} in / ~${A.decomp_out} out`,
                    value: decompCost,
                });

                const gapCostOne   = _llmCallCost(A.gap_in, A.gap_out, supPrice);
                const regenCostOne = _llmCallCost(agIn, A.regen_out, agPrice);
                rows.push({
                    name: "Agentic gap analysis",
                    sub: `utility model · ~${iters}× ~${A.gap_in.toLocaleString()} in / ~${A.gap_out} out`,
                    value: gapCostOne * iters,
                });
                rows.push({
                    name: "Regenerate (agentic)",
                    sub: `${cfg.agentic_model} · ~${iters}× ${cfg.agentic_rerank_topk} chunks → ~${agIn.toLocaleString()} in / ~${A.regen_out} out`,
                    value: regenCostOne * iters,
                });

                // Worst case: every iteration runs and each regen emits the
                // full configured output budget.
                const regenWorstOne = _llmCallCost(agIn + figWorstExtraTok, cfg.agentic_max_output_tokens, agPrice);
                worstExtra += (gapCostOne + regenWorstOne) * maxIters
                            - (gapCostOne + regenCostOne) * iters;
            }

            const total = rows.reduce((s, r) => s + (typeof r.value === "number" ? r.value : 0), 0);
            return { rows, total, worst: total + Math.max(0, worstExtra), cfg };
        }

        function renderCostEstimate() {
            const cfg = _readCostInputs();
            const est = estimateCost(cfg);

            const totalEl = document.getElementById("cost-total");
            const ctxEl   = document.getElementById("cost-context");
            const breakEl = document.getElementById("cost-breakdown");
            if (!totalEl || !ctxEl || !breakEl) return;

            totalEl.classList.remove("cost-warn", "cost-high");
            if (!isFinite(est.total) || est.total <= 0) {
                totalEl.textContent = "\\u2013";
            } else {
                totalEl.textContent = "~" + _fmtCostShort(est.total);
                if (est.total >= 1.0)      totalEl.classList.add("cost-high");
                else if (est.total >= 0.10) totalEl.classList.add("cost-warn");
            }

            const ctxParts = [];
            ctxParts.push(cfg.agentic ? "Agentic mode" : "Regular mode");
            ctxParts.push(cfg.agentic ? cfg.agentic_model : cfg.llm_model);
            if (cfg.agentic && cfg.agentic_recursive) ctxParts.push(`up to ${cfg.agentic_max_iterations}× iter`);
            ctxEl.textContent = " · " + ctxParts.join(" · ");

            const worstRow = (est.worst - est.total) > 0.0005 ? `
                <div class="cost-row">
                    <div class="cost-row-name">Worst case<small>all ${est.cfg.agentic_max_iterations} iterations, max output</small></div>
                    <div class="cost-row-value">${_fmtCostShort(est.worst)}</div>
                </div>` : "";
            breakEl.innerHTML = est.rows.map(r => `
                <div class="cost-row">
                    <div class="cost-row-name">${escapeHtml(r.name)}<small>${escapeHtml(r.sub)}</small></div>
                    <div class="cost-row-value">${typeof r.value === "number" ? _fmtCostShort(r.value) : ""}</div>
                </div>
            `).join("") + `
                <div class="cost-row cost-row-total">
                    <div class="cost-row-name"><b>Typical total</b></div>
                    <div class="cost-row-value">${_fmtCostShort(est.total)}</div>
                </div>
                ${worstRow}
                <div class="cost-disclaimer">
                    Typical-case estimate calibrated from logged queries. Most agentic queries converge in 0-2 refinement passes, so the configured iteration and output maximums are rarely reached. Rerank and targeted-fetch lookups are free.
                </div>
            `;
        }

        function toggleCostBreakdown(ev) {
            // Don't toggle when the click bubbled from an interactive child.
            if (ev && ev.target && ev.target.closest("button, input, select, a")) return;
            const card = document.getElementById("cost-estimator");
            const open = card.classList.toggle("open");
            card.setAttribute("aria-expanded", String(open));
            const tog = document.getElementById("cost-toggle");
            if (tog) tog.textContent = open ? "▲" : "▼";
        }

        // Wire every config knob + the agentic toggle to refresh the estimate.
        // `input` covers number-input typing in real time; `change` catches
        // checkbox + select tweaks.
        const COST_INPUT_IDS = [
            "agentic-toggle",
            "config-llm_model", "config-agentic_model",
            "config-final_rerank_topk", "config-auto_gap_check",
            "config-agentic_rerank_topk", "config-agentic_max_context_tokens",
            "config-agentic_max_output_tokens", "config-agentic_max_followups",
            "config-agentic_targeted_fetch", "config-agentic_recursive",
            "config-agentic_max_iterations",
        ];
        COST_INPUT_IDS.forEach(id => {
            const el = document.getElementById(id);
            if (!el) return;
            el.addEventListener("change", renderCostEstimate);
            el.addEventListener("input",  renderCostEstimate);
        });
        // Initial paint as soon as the DOM is wired up.
        renderCostEstimate();

        // ─── Pipeline trace → Mermaid graph definition ────────────────────
        // Strips anything that would break Mermaid's label parser (newlines,
        // pipes, quotes, brackets) and caps length so a giant entity list
        // doesn't blow up the layout.
        function _vizText(s, max = 50) {
            if (s === null || s === undefined) return "";
            return String(s)
                .replace(/[\\n\\r]+/g, " ")
                .replace(/[|"<>\\[\\]{}`]/g, "")
                .trim()
                .slice(0, max);
        }
        function _ms(stage) {
            return (stage && typeof stage.took_ms === "number")
                ? `${stage.took_ms.toFixed(0)}ms` : "";
        }

        // ─── Node color palette (theme-aware) ─────────────────────────────
        // The classDef set is emitted into the Mermaid definition. Light mode
        // uses soft pastel fills with dark text; dark mode mirrors each hue as
        // a deep tint with light text + a mid-saturation stroke so the chart
        // reads cleanly against the dark surface instead of glowing pastel.
        function vizClassDefs() {
            var dark = document.documentElement.getAttribute("data-theme") === "dark";
            if (dark) {
                return [
                    "  classDef input    fill:#e7e5e4,color:#1c1917,stroke:#a8a29e,stroke-width:1.5px,rx:6,ry:6",
                    "  classDef output   fill:#0f2418,color:#86efac,stroke:#22c55e,stroke-width:1.5px,rx:8,ry:8",
                    "  classDef stage_qp     fill:#241830,color:#e9d5ff,stroke:#7e22ce,stroke-width:1px,rx:5,ry:5",
                    "  classDef stage_struct fill:#0f2418,color:#bbf7d0,stroke:#16a34a,stroke-width:1px,rx:5,ry:5",
                    "  classDef stage_skipped fill:#1c1c1a,color:#a8a29e,stroke:#44403c,stroke-width:1px,stroke-dasharray:3 3,rx:5,ry:5",
                    "  classDef stage_subq   fill:#0c1f2e,color:#bae6fd,stroke:#0284c7,stroke-width:1px,rx:5,ry:5",
                    "  classDef stage_vector fill:#10203f,color:#bfdbfe,stroke:#2563eb,stroke-width:1px,rx:5,ry:5",
                    "  classDef stage_tsv    fill:#0a2226,color:#a5f3fc,stroke:#0891b2,stroke-width:1px,rx:5,ry:5",
                    "  classDef stage_bm25   fill:#0a221f,color:#99f6e4,stroke:#0d9488,stroke-width:1px,rx:5,ry:5",
                    "  classDef stage_rrf    fill:#241f0a,color:#fde68a,stroke:#ca8a04,stroke-width:1px,rx:5,ry:5",
                    "  classDef stage_dedup  fill:#26160c,color:#fed7aa,stroke:#ea580c,stroke-width:1px,rx:5,ry:5",
                    "  classDef stage_rerank fill:#260f0f,color:#fecaca,stroke:#dc2626,stroke-width:1px,rx:5,ry:5",
                    "  classDef stage_gen    fill:#e7e5e4,color:#1c1917,stroke:#a8a29e,stroke-width:1.5px,rx:6,ry:6",
                    "  classDef stage_resume   fill:#1c1c1a,color:#a8a29e,stroke:#57534e,stroke-width:1px,stroke-dasharray:4 3,rx:5,ry:5",
                    "  classDef stage_gap      fill:#161a3a,color:#c7d2fe,stroke:#4f46e5,stroke-width:1px",
                    "  classDef stage_followup fill:#161a3a,color:#c7d2fe,stroke:#4f46e5,stroke-width:1px",
                    "  classDef stage_agen     fill:#c7d2fe,color:#1e1b4b,stroke:#818cf8,stroke-width:1.5px,rx:6,ry:6",
                    "  classDef stage_tfetch   fill:#0a221f,color:#99f6e4,stroke:#0d9488,stroke-width:1px,rx:5,ry:5",
                    "  classDef stage_stop      fill:#241f0a,color:#fde68a,stroke:#ca8a04,stroke-width:1px,stroke-dasharray:4 3,rx:5,ry:5",
                    "  classDef stage_converged fill:#0f2418,color:#86efac,stroke:#22c55e,stroke-width:1px,rx:5,ry:5",
                ];
            }
            return [
                "  classDef input    fill:#1c1917,color:#fff,stroke:#1c1917,stroke-width:1.5px,rx:6,ry:6",
                "  classDef output   fill:#ffffff,color:#15803d,stroke:#15803d,stroke-width:1.5px,rx:8,ry:8",
                "  classDef stage_qp     fill:#fdf4ff,color:#581c87,stroke:#e9d5ff,stroke-width:1px,rx:5,ry:5",
                "  classDef stage_struct fill:#f0fdf4,color:#166534,stroke:#bbf7d0,stroke-width:1px,rx:5,ry:5",
                "  classDef stage_skipped fill:#fafaf9,color:#a8a29e,stroke:#e7e5e4,stroke-width:1px,stroke-dasharray:3 3,rx:5,ry:5",
                "  classDef stage_subq   fill:#f0f9ff,color:#0c4a6e,stroke:#bae6fd,stroke-width:1px,rx:5,ry:5",
                "  classDef stage_vector fill:#eff6ff,color:#1e3a8a,stroke:#bfdbfe,stroke-width:1px,rx:5,ry:5",
                "  classDef stage_tsv    fill:#ecfeff,color:#155e75,stroke:#a5f3fc,stroke-width:1px,rx:5,ry:5",
                "  classDef stage_bm25   fill:#f0fdfa,color:#115e59,stroke:#99f6e4,stroke-width:1px,rx:5,ry:5",
                "  classDef stage_rrf    fill:#fefce8,color:#854d0e,stroke:#fde68a,stroke-width:1px,rx:5,ry:5",
                "  classDef stage_dedup  fill:#fff7ed,color:#9a3412,stroke:#fed7aa,stroke-width:1px,rx:5,ry:5",
                "  classDef stage_rerank fill:#fef2f2,color:#991b1b,stroke:#fecaca,stroke-width:1px,rx:5,ry:5",
                "  classDef stage_gen    fill:#1c1917,color:#fff,stroke:#1c1917,stroke-width:1.5px,rx:6,ry:6",
                "  classDef stage_resume   fill:#f5f5f4,color:#57534e,stroke:#a8a29e,stroke-width:1px,stroke-dasharray:4 3,rx:5,ry:5",
                "  classDef stage_gap      fill:#eef2ff,color:#3730a3,stroke:#c7d2fe,stroke-width:1px",
                "  classDef stage_followup fill:#eef2ff,color:#3730a3,stroke:#c7d2fe,stroke-width:1px",
                "  classDef stage_agen     fill:#312e81,color:#fff,stroke:#312e81,stroke-width:1.5px,rx:6,ry:6",
                "  classDef stage_tfetch   fill:#f0fdfa,color:#115e59,stroke:#99f6e4,stroke-width:1px,rx:5,ry:5",
                "  classDef stage_stop      fill:#fefce8,color:#854d0e,stroke:#ca8a04,stroke-width:1px,stroke-dasharray:4 3,rx:5,ry:5",
                "  classDef stage_converged fill:#f0fdf4,color:#166534,stroke:#22c55e,stroke-width:1px,rx:5,ry:5",
            ];
        }

        // Compose a node label with title (bold) + optional subtitle (italic)
        // + time. Keeps the visual rhythm consistent across all node types.
        function _label(title, subtitle, stage) {
            const parts = [`<b>${title}</b>`];
            if (subtitle) parts.push(`<i>${subtitle}</i>`);
            const t = _ms(stage);
            if (t) parts.push(t);
            return parts.join("<br/>");
        }

        // ─── Split a full trace into per-iteration sub-traces ────────────
        // When recursive agentic mode is on, stages get `.iter0`, `.iter1`, …
        // suffixes. This splits them so each iteration gets its own chart.
        // Returns null when there are no iteration suffixes (single chart).
        function splitTraceByIteration(trace) {
            // Iteration suffix appears either at the end of a stage name
            // (e.g. "agentic.gap_analysis.iter1") or in the middle of a
            // namespaced follow-up sub-stage (e.g.
            // "agentic.followup_q0.iter1.hybrid_search.vector_search_q0"),
            // so the regex must match `.iterN` followed by `.` OR end.
            const ITER_RE = /\\.iter(\\d+)(?:\\.|$)/;
            // Replacement strips the `.iterN` while preserving a trailing
            // `.` when it's mid-name. "$1" captures either "." or "".
            const stripIter = stage => stage.replace(/\\.iter(\\d+)(\\.|$)/, "$2");

            // Which page a stage belongs on. Usually its own iteration
            // number, but the convergence verdict is emitted with the
            // *skipped* iteration's suffix (the loop checked it at the top
            // of iterN and stopped before doing anything), so it belongs on
            // the previous page - the pass whose regenerate produced that
            // verdict. Without this it becomes the sole stage of a final
            // page that renders as an empty chart.
            const effIter = s => {
                const m = s.stage.match(ITER_RE);
                if (!m) return null;
                const n = parseInt(m[1], 10);
                return /^agentic\\.verdict_converged\\./.test(s.stage)
                    ? Math.max(0, n - 1) : n;
            };

            let maxIter = -1;
            for (const s of trace) {
                const n = effIter(s);
                if (n !== null) maxIter = Math.max(maxIter, n);
            }
            if (maxIter < 0) return null;

            const result = [];
            for (let i = 0; i <= maxIter; i++) {
                // Page 1 carries the full base pipeline + iter0 agentic
                // stages. Pages 2+ carry only that iteration's agentic
                // stages (the base pipeline didn't re-run). Suffix-less
                // stages are base-pipeline stages, except the terminal
                // cap-reached marker, which describes the END of the loop
                // and so belongs on the last page.
                result.push(trace.filter(s => {
                    const n = effIter(s);
                    if (n === null) {
                        return s.stage === "agentic.cap_reached"
                            ? i === maxIter : i === 0;
                    }
                    return n === i;
                }).map(s => ({...s, stage: stripIter(s.stage)})));
            }
            return result;
        }

        // ─── Build Mermaid flowchart definition from a pipeline trace ─────
        // Returns {def, nodeMap} where nodeMap maps each Mermaid node ID
        // (e.g. "RRF", "GEN") to the stage object that generated it so the
        // click-to-inspect popup can retrieve data for the clicked node.
        function buildMermaidFromTrace(trace, query) {
            const stages = {};
            for (const s of trace) stages[s.stage] = s;

            const L = ["flowchart TD"];
            const nodeMap = {};

            const refineSeed = stages["refine.seed"];
            const qp         = stages.query_processor;
            const gapEarly   = stages["agentic.gap_analysis"];

            // Agentic-only trace (recursive pass 2+): no base pipeline stages
            // ran. The "query" for this pass is the follow-up queries gap
            // analysis identified, so we use those as the top node label.
            const isAgenticOnly = !qp && !refineSeed && gapEarly;
            let _qLabel, _qSub;
            if (isAgenticOnly) {
                const fqs = (gapEarly.output && gapEarly.output.queries) || [];
                _qLabel = "Follow-up queries";
                _qSub   = fqs.length
                    ? _vizText(fqs[0], 56) + (fqs.length > 1 ? ` +${fqs.length - 1} more` : "")
                    : _vizText((gapEarly.output && gapEarly.output.reason) || "", 56);
            } else {
                _qLabel = "Query";
                _qSub   = _vizText(query, 60);
            }
            // Populate the Q-node popup with the actual query text. For
            // agentic-only passes also surface the follow-up queries and the
            // gap reason that triggered this iteration so clicking the node
            // explains what the recursive pass is searching for.
            const _qStage = isAgenticOnly
                ? {
                    stage: "query.followup",
                    input: {query: query, note: "Original user query"},
                    output: {
                        queries: (gapEarly.output && gapEarly.output.queries) || [],
                        reason:  (gapEarly.output && gapEarly.output.reason)  || "",
                        requested_resources: (gapEarly.output && gapEarly.output.requested_resources) || null,
                    },
                    took_ms: 0,
                  }
                : {stage: "query", input: {query}, output: {}, took_ms: 0};
            L.push(`  Q["${_label(_qLabel, _qSub, null)}"]:::input`);
            nodeMap["Q"] = _qStage;

            // Refine-mode trace: /api/refine reused a prior /api/query's
            // first-pass state, so Stages 1-4 didn't run. Emit a "Resume"
            // marker so the diagram still has a visible upstream node feeding
            // into the agentic branch below. Without this we bail with just
            // the Query node and the user sees an empty canvas.
            if (!qp && !refineSeed && !gapEarly) {
                vizClassDefs().forEach(function (c) { L.push(c); });
                return {def: L.join("\\n"), nodeMap};
            }
            if (refineSeed && !qp) {
                const ddCount = (refineSeed.output && refineSeed.output.deduplicated_count) || 0;
                const ctxCount = (refineSeed.output && refineSeed.output.context_chunk_count) || 0;
                L.push(`  RESUME[/"${_label("Resume from cache", ddCount + " pooled chunks · " + ctxCount + " in prior context", refineSeed)}"/]:::stage_resume`);
                L.push("  Q --> RESUME");
                nodeMap["RESUME"] = refineSeed;
            }
            if (qp) {
                const qpType = _vizText(qp.output.type, 20);
                const qpEnts = (qp.output.entities || []).length;
                const qpSubs = (qp.output.sub_queries || []).length;
                const qpSub = `${qpType ? "type: " + qpType : ""}${qpEnts ? " · " + qpEnts + " entit" + (qpEnts===1?"y":"ies") : ""}${qpSubs ? " · " + qpSubs + " sub-quer" + (qpSubs===1?"y":"ies") : ""}`.replace(/^ · /, "");
                L.push(`  QP["${_label("Understand query", qpSub, qp)}"]:::stage_qp`);
                L.push("  Q --> QP");
                nodeMap["QP"] = qp;
            }

            // Structured lookup - side branch that merges back into dedup
            const sl = stages.structured_lookup;
            let slActive = false;
            if (sl) {
                if (sl.output.skipped) {
                    L.push(`  SL["${_label("Structured lookup", "skipped: " + _vizText(sl.output.reason, 40), null)}"]:::stage_skipped`);
                } else {
                    slActive = true;
                    const conf = _vizText(sl.output.confidence, 12);
                    const flds = sl.output.field_count || 0;
                    const tbls = sl.output.table_count || 0;
                    const lookupSub = `${sl.output.found ? "found" : "not found"}${conf ? " · " + conf : ""} · ${flds} field${flds===1?"":"s"} · ${tbls} table${tbls===1?"":"s"}`;
                    L.push(`  SL["${_label("Structured lookup", lookupSub, sl)}"]:::stage_struct`);
                }
                L.push("  QP --> SL");
                nodeMap["SL"] = sl;
            }

            // Per-sub-query branches: vector + tsvector + BM25
            const subIds = new Set();
            for (const s of trace) {
                const m = s.stage.match(/^hybrid_search\\.\\w+_q(\\d+)$/);
                if (m) subIds.add(parseInt(m[1], 10));
            }
            const sortedSubs = [...subIds].sort((a, b) => a - b);

            const rrf = stages["hybrid_search.rrf_merge"];
            for (const i of sortedSubs) {
                const v = stages[`hybrid_search.vector_search_q${i}`];
                const t = stages[`hybrid_search.tsvector_search_q${i}`];
                const b = stages[`hybrid_search.bm25_search_q${i}`];
                const sqText = (v && v.input && v.input.query) || `q${i}`;
                const sqStage = {stage: `sub_query_${i}`, input: {query: sqText}, output: {}, took_ms: 0};
                L.push(`  SQ${i}{{"${_label("Sub-query " + (i+1), _vizText(sqText, 60), null)}"}}:::stage_subq`);
                L.push(`  QP --> SQ${i}`);
                nodeMap[`SQ${i}`] = sqStage;

                if (v) {
                    L.push(`  V${i}["${_label("Semantic search", (v.output.count || 0) + " hits · Voyage", v)}"]:::stage_vector`);
                    L.push(`  SQ${i} --> V${i}`);
                    if (rrf) L.push(`  V${i} --> RRF`);
                    nodeMap[`V${i}`] = v;
                }
                if (t) {
                    L.push(`  T${i}["${_label("Keyword search", (t.output.count || 0) + " hits · tsvector", t)}"]:::stage_tsv`);
                    L.push(`  SQ${i} --> T${i}`);
                    if (rrf) L.push(`  T${i} --> RRF`);
                    nodeMap[`T${i}`] = t;
                }
                if (b) {
                    L.push(`  B${i}["${_label("BM25 search", (b.output.count || 0) + " hits · Okapi", b)}"]:::stage_bm25`);
                    L.push(`  SQ${i} --> B${i}`);
                    if (rrf) L.push(`  B${i} --> RRF`);
                    nodeMap[`B${i}`] = b;
                }
            }

            if (rrf) {
                L.push(`  RRF["${_label("Fuse results", (rrf.output.count || 0) + " merged · RRF", rrf)}"]:::stage_rrf`);
                nodeMap["RRF"] = rrf;
            }

            const dd = stages.result_dedup;
            if (dd) {
                L.push(`  DEDUP["${_label("Deduplicate", (dd.output.deduped_count || 0) + " unique chunks", dd)}"]:::stage_dedup`);
                if (rrf) L.push("  RRF --> DEDUP");
                if (slActive) L.push("  SL --> DEDUP");
                nodeMap["DEDUP"] = dd;
            }

            const rr = stages.final_rerank;
            if (rr) {
                L.push(`  RR["${_label("Rerank", "top " + (rr.output.count || 0) + " · voyage", rr)}"]:::stage_rerank`);
                if (dd) L.push("  DEDUP --> RR");
                nodeMap["RR"] = rr;
            }

            const gen = stages.generation;
            if (gen) {
                const cits = (gen.output.citation_count !== undefined) ? gen.output.citation_count : 0;
                const ans = gen.output.answer_length || 0;
                L.push(`  GEN["${_label("Generate answer", ans.toLocaleString() + " chars · " + cits + " citation" + (cits===1?"":"s") + " · Claude", gen)}"]:::stage_gen`);
                if (rr) L.push("  RR --> GEN");
                nodeMap["GEN"] = gen;
            }

            // ─── Agentic refinement branch (only present when agentic=true) ───
            const gap = stages["agentic.gap_analysis"];
            const tfetch = stages["agentic.targeted_fetch"];
            const ag_rr = stages["agentic.rerank"];
            const ag_gen = stages["agentic.regenerate"];
            // In refine mode there's no GEN node, so the final-answer arrow
            // falls back to RESUME until the agentic regen succeeds and
            // promotes itself to GEN2. For agentic-only charts (recursive
            // pass 2+) start from Q - the only node guaranteed to exist -
            // and promote to GEN2 only once that node is actually declared.
            // (A stalled pass has no regenerate, so a bare "GEN2" here would
            // make Mermaid auto-create an unstyled box literally labelled
            // GEN2.)
            let agAnswerNode = gen ? "GEN" : (refineSeed ? "RESUME" : "Q");
            if (gap) {
                const needs = gap.output && gap.output.needs_followup;
                const reason = _vizText((gap.output && gap.output.reason) || "", 60);
                const gapSub = needs ? ("needs follow-up: " + reason) : "answer covers the question";
                L.push(`  GAP{"${_label("Gap analysis", gapSub, gap)}"}:::stage_gap`);
                // Normal: first-pass GEN → gap. Refine: RESUME → gap.
                // Agentic-only (pass 2+): Q (follow-up queries) → gap.
                if (gen) L.push("  GEN --> GAP");
                else if (refineSeed) L.push("  RESUME --> GAP");
                else if (isAgenticOnly) L.push("  Q --> GAP");
                nodeMap["GAP"] = gap;

                if (needs) {
                    // (a) Targeted resource fetch - direct table/field lookup
                    if (tfetch) {
                        const req = (tfetch.input && tfetch.input.requested) || {};
                        const figs = (req.figures || []).length;
                        const flds = (req.fields || []).length;
                        const secs = (req.sections || []).length;
                        const got = (tfetch.output && tfetch.output.fetched_count) || 0;
                        const reqSummary = [
                            figs ? `${figs} fig${figs===1?"":"s"}` : "",
                            flds ? `${flds} field${flds===1?"":"s"}` : "",
                            secs ? `${secs} section${secs===1?"":"s"}` : "",
                        ].filter(Boolean).join(" · ") || "(none)";
                        L.push(`  TFETCH[/"${_label("Targeted fetch", "asked: " + reqSummary + " · got: " + got, tfetch)}"/]:::stage_tfetch`);
                        L.push("  GAP --> TFETCH");
                        if (ag_rr) L.push("  TFETCH --> RR2");
                        nodeMap["TFETCH"] = tfetch;
                    }

                    // (b) Per-followup natural-language retrieval branches.
                    // Each follow-up gets its own subgraph showing the full
                    // decompose → vector/keyword/bm25 per sub-query → mini-RRF
                    // path. Sub-stages are namespaced as
                    // `agentic.followup_q{fi}.hybrid_search.*` server-side
                    // so they don't collide with the main query's stages.
                    const fqIds = new Set();
                    for (const s of trace) {
                        const m = s.stage.match(/^agentic\\.followup_search_q(\\d+)$/);
                        if (m) fqIds.add(parseInt(m[1], 10));
                    }
                    for (const i of [...fqIds].sort((a, b) => a - b)) {
                        const fq = stages[`agentic.followup_search_q${i}`];
                        const decomp = stages[`agentic.followup_decomp_q${i}`];
                        const chunks = (fq && fq.output && fq.output.chunk_count) || 0;

                        // Collect this follow-up's sub-query indices from the
                        // namespaced hybrid_search stages.
                        const nsPrefix = `agentic.followup_q${i}.`;
                        const fqSubIds = new Set();
                        for (const s of trace) {
                            if (!s.stage.startsWith(nsPrefix)) continue;
                            const m2 = s.stage.slice(nsPrefix.length).match(/^hybrid_search\\.\\w+_q(\\d+)$/);
                            if (m2) fqSubIds.add(parseInt(m2[1], 10));
                        }
                        const fqSorted = [...fqSubIds].sort((a, b) => a - b);
                        const fqRrf = stages[`${nsPrefix}hybrid_search.rrf_merge`];

                        // Open a subgraph for this follow-up. The title is kept
                        // short (number + chunk count) so it stays a single,
                        // centered line — the long follow-up question wraps into
                        // 3 lines and overlaps the nodes inside. The actual
                        // question text is shown on the Sub-query node within.
                        L.push(`  subgraph FQGRP${i}["Follow-up ${i+1} · ${chunks} chunks"]`);
                        // Follow-up header node (clickable target for the GAP edge).
                        const decompSub = decomp && decomp.output && decomp.output.sub_queries
                            ? `${decomp.output.sub_queries.length} sub-quer${decomp.output.sub_queries.length===1?"y":"ies"}`
                            : "verbatim";
                        L.push(`    FQ${i}{{"${_label("Decompose", decompSub, decomp)}"}}:::stage_followup`);
                        if (decomp) nodeMap[`FQ${i}`] = decomp;

                        for (const j of fqSorted) {
                            const v = stages[`${nsPrefix}hybrid_search.vector_search_q${j}`];
                            const t = stages[`${nsPrefix}hybrid_search.tsvector_search_q${j}`];
                            const b = stages[`${nsPrefix}hybrid_search.bm25_search_q${j}`];
                            const sqTxt = (v && v.input && v.input.query) || `q${j}`;
                            L.push(`    FQ${i}SQ${j}{{"${_label("Sub-query " + (j+1), _vizText(sqTxt, 48), null)}"}}:::stage_subq`);
                            L.push(`    FQ${i} --> FQ${i}SQ${j}`);
                            if (v) {
                                L.push(`    FQ${i}V${j}["${_label("Semantic", (v.output.count || 0) + " hits", v)}"]:::stage_vector`);
                                L.push(`    FQ${i}SQ${j} --> FQ${i}V${j}`);
                                if (fqRrf) L.push(`    FQ${i}V${j} --> FQ${i}RRF`);
                                nodeMap[`FQ${i}V${j}`] = v;
                            }
                            if (t) {
                                L.push(`    FQ${i}T${j}["${_label("Keyword", (t.output.count || 0) + " hits", t)}"]:::stage_tsv`);
                                L.push(`    FQ${i}SQ${j} --> FQ${i}T${j}`);
                                if (fqRrf) L.push(`    FQ${i}T${j} --> FQ${i}RRF`);
                                nodeMap[`FQ${i}T${j}`] = t;
                            }
                            if (b) {
                                L.push(`    FQ${i}B${j}["${_label("BM25", (b.output.count || 0) + " hits", b)}"]:::stage_bm25`);
                                L.push(`    FQ${i}SQ${j} --> FQ${i}B${j}`);
                                if (fqRrf) L.push(`    FQ${i}B${j} --> FQ${i}RRF`);
                                nodeMap[`FQ${i}B${j}`] = b;
                            }
                        }
                        if (fqRrf) {
                            L.push(`    FQ${i}RRF["${_label("Fuse", (fqRrf.output.count || 0) + " merged", fqRrf)}"]:::stage_rrf`);
                            nodeMap[`FQ${i}RRF`] = fqRrf;
                        }
                        L.push(`  end`);

                        L.push(`  GAP --> FQ${i}`);
                        // Wire the deepest output of this subgraph into RR2.
                        const exit = fqRrf ? `FQ${i}RRF` : `FQ${i}`;
                        if (ag_rr) L.push(`  ${exit} --> RR2`);
                    }
                    if (ag_rr) {
                        const newC = (ag_rr.input && ag_rr.input.added_by_followups) || 0;
                        const topC = (ag_rr.output && ag_rr.output.count) || 0;
                        L.push(`  RR2["${_label("Rerank expanded pool", "top " + topC + " · " + newC + " new chunk" + (newC===1?"":"s"), ag_rr)}"]:::stage_rerank`);
                        nodeMap["RR2"] = ag_rr;
                    }
                    if (ag_gen) {
                        const c2 = (ag_gen.output && ag_gen.output.citation_count !== undefined) ? ag_gen.output.citation_count : 0;
                        const a2 = (ag_gen.output && ag_gen.output.answer_length) || 0;
                        const errType = ag_gen.output && ag_gen.output.error_type;
                        const label = errType
                            ? _label("Regenerate", _vizText(errType, 30) + " (fell back)", ag_gen)
                            : _label("Regenerate", a2.toLocaleString() + " chars · " + c2 + " citation" + (c2===1?"":"s") + " · Opus", ag_gen);
                        L.push(`  GEN2["${label}"]:::stage_agen`);
                        if (ag_rr) L.push("  RR2 --> GEN2");
                        // On an errored regenerate the prior answer is kept,
                        // so the upstream node (GEN/RESUME) stays the answer
                        // source - unless this is an agentic-only page where
                        // GEN2 is the only sensible terminus.
                        if (!errType || (!gen && !refineSeed)) agAnswerNode = "GEN2";
                        nodeMap["GEN2"] = ag_gen;
                    }
                }
            }

            // Non-agentic advisory gap hint (auto_gap_check): rendered as a
            // side note off the generated answer - it never changes the
            // answer flow, it just reports whether a follow-up would help.
            const hint = stages.gap_hint;
            if (hint && gen) {
                const hNeeds = hint.output && hint.output.needs_followup;
                const hSub = hNeeds
                    ? "needs follow-up: " + _vizText((hint.output && hint.output.reason) || "", 60)
                    : "answer covers the question";
                L.push(`  HINT{"${_label("Gap hint", hSub, hint)}"}:::stage_gap`);
                L.push("  GEN --> HINT");
                nodeMap["HINT"] = hint;
            }

            // ─── Terminal markers from the recursive loop ─────────────────
            // Stall guard: the reranked context matched the previous pass
            // byte-for-byte, so regenerate was skipped and the loop stopped.
            const stall = stages["agentic.stalled"];
            if (stall) {
                L.push(`  STALL[["${_label("Stopped: stalled", "context unchanged · kept previous answer", null)}"]]:::stage_stop`);
                L.push(`  ${ag_rr ? "RR2" : agAnswerNode} --> STALL`);
                nodeMap["STALL"] = stall;
                agAnswerNode = "STALL";
            }
            // Convergence verdict: the generator's self-assessment marked
            // the answer complete, ending the loop.
            const vc = stages["agentic.verdict_converged"];
            if (vc) {
                L.push(`  VC[["${_label("Converged", "model judged the answer complete", null)}"]]:::stage_converged`);
                L.push(`  ${agAnswerNode} --> VC`);
                nodeMap["VC"] = vc;
                agAnswerNode = "VC";
            }
            // Iteration cap: the loop ran out of passes before converging.
            const cap = stages["agentic.cap_reached"];
            if (cap) {
                const capIters = (cap.output && cap.output.iterations_run) || 0;
                L.push(`  CAP[["${_label("Iteration cap reached", capIters + " pass" + (capIters === 1 ? "" : "es") + " · stopped before converging", null)}"]]:::stage_stop`);
                L.push(`  ${agAnswerNode} --> CAP`);
                nodeMap["CAP"] = cap;
                agAnswerNode = "CAP";
            }

            L.push('  ANS(["<b>Final answer</b>"]):::output');
            L.push(`  ${agAnswerNode} --> ANS`);

            // ─── Color palette ───────────────────────────────────────────────
            // Theme-aware classDefs (see vizClassDefs): pre-search "thinking"
            // stages run cool (blues/greens), retrieval sweeps through warm
            // hues, and the agentic loop uses purples to signal "second pass".
            // Light mode = soft pastels on white; dark mode = deep tints on the
            // dark surface, each hue mirrored so families stay distinguishable.
            vizClassDefs().forEach(function (c) { L.push(c); });

            return {def: L.join("\\n"), nodeMap};
        }

        // ─── Viz nav state ────────────────────────────────────────────────
        let _vizPages  = [];  // [{def, nodeMap}, …]
        let _vizPageIdx = 0;
        let _vizCounter = 0;
        let _vizQuery   = "";

        // Attach click listeners to SVG nodes so clicking opens the popup.
        function attachNodeClickListeners(nodeMap) {
            const svg = document.querySelector("#pipeline-viz svg");
            if (!svg) return;
            svg.querySelectorAll("g.node").forEach(g => {
                // Mermaid 11 sets data-id on each node <g>.
                let nodeId = g.dataset.id || "";
                // Fallback: parse from the element id like "flowchart-RRF-42"
                if (!nodeId && g.id) {
                    nodeId = g.id.replace(/^flowchart-/, "").replace(/-\\d+$/, "");
                }
                if (!nodeId || !nodeMap[nodeId]) return;
                const stage = nodeMap[nodeId];
                g.style.cursor = "pointer";
                g.addEventListener("click", e => {
                    e.stopPropagation();
                    showStagePopup(stage, e.clientX, e.clientY);
                });
            });
        }

        // Per-iteration pass navigation, top-right of the chart. Only shown
        // when a recursive run produced more than one iteration — each page is
        // one iteration's flow, navigated with the two arrows.
        function attachPassNav(host) {
            if (_vizPages.length <= 1) return;
            const nav = document.createElement("div");
            nav.className = "viz-pass-nav";
            nav.innerHTML =
                '<button type="button" data-p="prev" title="Previous iteration">\\u2190</button>' +
                `<span class="viz-pass-label">Pass ${_vizPageIdx + 1} / ${_vizPages.length}</span>` +
                '<button type="button" data-p="next" title="Next iteration">\\u2192</button>';
            const prevB = nav.querySelector('[data-p="prev"]');
            const nextB = nav.querySelector('[data-p="next"]');
            if (prevB) prevB.disabled = _vizPageIdx === 0;
            if (nextB) nextB.disabled = _vizPageIdx === _vizPages.length - 1;
            nav.addEventListener("click", e => {
                const b = e.target.closest("button"); if (!b) return;
                if (b.dataset.p === "prev") vizPrev(); else vizNext();
            });
            host.appendChild(nav);
        }

        // Render the page at _vizPageIdx and update nav UI.
        async function vizGoTo(idx) {
            if (!_vizPages.length) return;
            _vizPageIdx = Math.max(0, Math.min(idx, _vizPages.length - 1));
            const page = _vizPages[_vizPageIdx];

            const host   = document.getElementById("pipeline-viz");
            const legend = document.getElementById("pipeline-legend");
            const nav    = document.getElementById("viz-nav");
            const label  = document.getElementById("viz-nav-label");
            const prevBtn = document.getElementById("viz-nav-prev");
            const nextBtn = document.getElementById("viz-nav-next");

            if (legend) legend.style.display = "none";

            if (typeof mermaid === "undefined") {
                host.innerHTML = '<div class="viz-empty">Mermaid failed to load (CDN blocked?). The expanded trace below still has every stage.</div>';
                return;
            }

            const id = `viz-${++_vizCounter}`;
            try {
                applyMermaidTheme();
                const {svg} = await mermaid.render(id, page.def);
                host.innerHTML = svg;
                if (legend) legend.style.display = "";
                attachNodeClickListeners(page.nodeMap);
                attachPassNav(host);
            } catch (err) {
                console.error("mermaid render failed:", err, page.def);
                host.innerHTML = `<div class="viz-empty">Could not render flow: ${escapeHtml(err.message || String(err))}</div>`;
            }

            // Update navigation controls. The header-row nav is superseded by
            // the in-chart top-right pass nav (built in attachPassNav), so keep
            // it hidden to avoid a duplicate control.
            if (nav) nav.style.display = "none";
            if (label) label.textContent = `Pass ${_vizPageIdx + 1} / ${_vizPages.length}`;
            if (prevBtn) prevBtn.disabled = _vizPageIdx === 0;
            if (nextBtn) nextBtn.disabled = _vizPageIdx === _vizPages.length - 1;
        }

        function vizPrev() { vizGoTo(_vizPageIdx - 1); }
        function vizNext() { vizGoTo(_vizPageIdx + 1); }

        async function renderPipelineViz(trace, query) {
            _vizQuery = query || "";
            _vizPages = [];
            _vizPageIdx = 0;

            const host = document.getElementById("pipeline-viz");
            const legend = document.getElementById("pipeline-legend");
            const nav = document.getElementById("viz-nav");
            if (legend) legend.style.display = "none";
            if (nav) nav.style.display = "none";

            if (!trace || !trace.length) {
                host.innerHTML = '<div class="viz-empty">No pipeline trace returned. Set <code>DEBUG_PIPELINE=1</code> on the server to enable.</div>';
                return;
            }

            // Use the split pages even when there's only one: its stage
            // names have the `.iterN` suffixes stripped, which the builder
            // requires. Rendering the raw trace would silently drop every
            // agentic stage of a single-iteration recursive run.
            const iters = splitTraceByIteration(trace);
            if (iters) {
                for (const sub of iters) {
                    _vizPages.push(buildMermaidFromTrace(sub, _vizQuery));
                }
            } else {
                _vizPages.push(buildMermaidFromTrace(trace, _vizQuery));
            }

            await vizGoTo(0);
        }

        // ─── Draggable stage-detail popups (manager) ──────────────────────
        // Several popups can be open at once; each is independently draggable.
        // The set is FIFO-capped at STAGE_POPUP_CAP — opening one past the cap
        // evicts the oldest. They persist across viz re-renders and pass nav;
        // the only thing that clears them is running a new query (runQuery).
        const STAGE_POPUP_CAP = 4;
        let _stagePopups = [];   // [{el, key}] oldest-first
        let _popupZ = 2600;      // running z so the last interacted popup is on top
        let _popupTip = null;    // one shared hover-hint element

        function _ensurePopupTip() {
            if (!_popupTip) {
                _popupTip = document.createElement("div");
                _popupTip.className = "popup-tip";
                document.body.appendChild(_popupTip);
            }
            return _popupTip;
        }
        // Show the brief hint right away (no native-title delay), above the
        // button (or below if there's no room).
        function _showPopupTip(btn) {
            const tip = _ensurePopupTip();
            tip.textContent = btn.getAttribute("data-tip") || "";
            tip.classList.add("show");
            const r = btn.getBoundingClientRect();
            const tw = tip.offsetWidth, th = tip.offsetHeight;
            let left = r.left + r.width / 2 - tw / 2;
            left = Math.max(6, Math.min(left, window.innerWidth - tw - 6));
            let top = r.top - th - 6;
            if (top < 6) top = r.bottom + 6;
            tip.style.left = left + "px";
            tip.style.top  = top + "px";
        }
        function _hidePopupTip() { if (_popupTip) _popupTip.classList.remove("show"); }

        function _raisePopup(rec) { rec.el.style.zIndex = String(++_popupZ); }

        // The "close all" control only makes sense with 2+ popups open; toggle
        // the .multi class on every popup so each shows/hides its own button.
        function _updatePopupMultiState() {
            const multi = _stagePopups.length > 1;
            _stagePopups.forEach(p => p.el.classList.toggle("multi", multi));
        }

        function _makePopupDraggable(el, handle) {
            let ox = 0, oy = 0;
            function onMove(e) {
                const cx = e.touches ? e.touches[0].clientX : e.clientX;
                const cy = e.touches ? e.touches[0].clientY : e.clientY;
                el.style.left = Math.max(0, cx - ox) + "px";
                el.style.top  = Math.max(0, cy - oy) + "px";
            }
            function onUp() {
                document.removeEventListener("mousemove", onMove);
                document.removeEventListener("mouseup",   onUp);
                document.removeEventListener("touchmove", onMove);
                document.removeEventListener("touchend",  onUp);
            }
            handle.addEventListener("mousedown", e => {
                if (e.target.closest("button")) return;   // let the close buttons work
                const r = el.getBoundingClientRect();
                ox = e.clientX - r.left; oy = e.clientY - r.top;
                document.addEventListener("mousemove", onMove);
                document.addEventListener("mouseup",   onUp);
                e.preventDefault();
            });
            handle.addEventListener("touchstart", e => {
                if (e.target.closest("button")) return;
                const r = el.getBoundingClientRect();
                ox = e.touches[0].clientX - r.left; oy = e.touches[0].clientY - r.top;
                document.addEventListener("touchmove", onMove, {passive: false});
                document.addEventListener("touchend",  onUp);
                e.preventDefault();
            }, {passive: false});
        }

        // Identity for a popup so clicking the same node twice just raises the
        // existing one instead of stacking a duplicate.
        function _stageKey(stage) {
            const s = stage || {};
            return (s.stage || "") + "#" + (s.took_ms != null ? s.took_ms : "");
        }

        function showStagePopup(stage, clickX, clickY) {
            const key = _stageKey(stage);
            const existing = _stagePopups.find(p => p.key === key);
            if (existing) { _raisePopup(existing); return; }

            const display = formatStageDisplay(stage.stage);

            const el = document.createElement("div");
            el.className = "stage-popup";
            el.setAttribute("role", "dialog");

            const header = document.createElement("div");
            header.className = "stage-popup-header";

            const titleEl = document.createElement("span");
            titleEl.className = "stage-popup-title";
            titleEl.textContent = display.title;

            const actions = document.createElement("div");
            actions.className = "stage-popup-actions";

            const closeAllBtn = document.createElement("button");
            closeAllBtn.type = "button";
            closeAllBtn.className = "stage-popup-close-all";
            closeAllBtn.textContent = "Close all";   // labeled, so no hover hint needed

            const closeBtn = document.createElement("button");
            closeBtn.type = "button";
            closeBtn.className = "stage-popup-close";
            closeBtn.setAttribute("data-tip", "Close this");
            closeBtn.innerHTML = "&#x2715;";

            actions.appendChild(closeAllBtn);   // left of the per-popup close
            actions.appendChild(closeBtn);
            header.appendChild(titleEl);
            header.appendChild(actions);

            const body = document.createElement("div");
            body.className = "stage-popup-body";
            body.innerHTML = renderStageBody(stage);

            el.appendChild(header);
            el.appendChild(body);
            document.body.appendChild(el);

            // Position near the click, clamped to the viewport, cascaded a touch
            // so a stack opened near one spot doesn't perfectly overlap.
            const w = el.offsetWidth || 480;
            const h = el.offsetHeight || 300;
            const off = _stagePopups.length * 14;
            const baseX = (clickX == null ? window.innerWidth / 2 - w / 2 : clickX) + off;
            const baseY = (clickY == null ? 90 : clickY + 14) + off;
            el.style.left = Math.max(8, Math.min(baseX, window.innerWidth  - w - 8)) + "px";
            el.style.top  = Math.max(8, Math.min(baseY, window.innerHeight - h - 8)) + "px";

            const rec = {el, key};
            _stagePopups.push(rec);

            closeBtn.addEventListener("click", ev => { ev.stopPropagation(); closeStagePopup(rec); });
            closeAllBtn.addEventListener("click", ev => { ev.stopPropagation(); closeAllStagePopups(); });
            // Only the icon needs a hover hint; the "Close all" button is labeled.
            closeBtn.addEventListener("mouseenter", () => _showPopupTip(closeBtn));
            closeBtn.addEventListener("mouseleave", _hidePopupTip);
            closeBtn.addEventListener("focus", () => _showPopupTip(closeBtn));
            closeBtn.addEventListener("blur", _hidePopupTip);
            el.addEventListener("mousedown", () => _raisePopup(rec));
            _makePopupDraggable(el, header);
            _raisePopup(rec);

            // FIFO: opening past the cap drops the oldest popup.
            while (_stagePopups.length > STAGE_POPUP_CAP) {
                const oldest = _stagePopups.shift();
                if (oldest.el && oldest.el.parentNode) oldest.el.parentNode.removeChild(oldest.el);
            }
            _updatePopupMultiState();
        }

        function closeStagePopup(rec) {
            _hidePopupTip();
            if (!rec) rec = _stagePopups[_stagePopups.length - 1];   // legacy no-arg call
            if (!rec) return;
            const i = _stagePopups.indexOf(rec);
            if (i !== -1) _stagePopups.splice(i, 1);
            if (rec.el && rec.el.parentNode) rec.el.parentNode.removeChild(rec.el);
            _updatePopupMultiState();
        }

        function closeAllStagePopups() {
            _hidePopupTip();
            _stagePopups.forEach(p => { if (p.el && p.el.parentNode) p.el.parentNode.removeChild(p.el); });
            _stagePopups = [];
        }

        let _agConfirmCallback = null;

        function showAgenticConfirm(onRun) {
            _agConfirmCallback = onRun;
            const val = id => { const el = document.getElementById(id); return el ? el.value : ""; };
            const chk = id => { const el = document.getElementById(id); return el ? el.checked : false; };
            const modelEl = document.getElementById("config-agentic_model");
            const modelLabel = modelEl ? (modelEl.options[modelEl.selectedIndex]?.text || modelEl.value) : "-";
            const rows = [
                ["Model",        modelLabel],
                ["Max follow-ups", val("config-agentic_max_followups") || "3"],
                ["Rerank top-k", val("config-agentic_rerank_topk") || "14"],
                ["Max ctx tok",  val("config-agentic_max_context_tokens") || "16000"],
                ["Targeted fetch", chk("config-agentic_targeted_fetch") ? "on" : "off"],
                ["Recursive",    chk("config-agentic_recursive") ? "on (max " + (val("config-agentic_max_iterations") || "4") + " iter)" : "off"],
            ];
            document.getElementById("ag-confirm-rows").innerHTML = rows.map(([k, v]) =>
                `<span class="ag-confirm-key">${escapeHtml(k)}</span><span class="ag-confirm-val">${escapeHtml(String(v))}</span>`
            ).join("");
            document.getElementById("ag-confirm-overlay").classList.remove("hidden");
        }

        function closeAgenticConfirm() {
            document.getElementById("ag-confirm-overlay").classList.add("hidden");
            _agConfirmCallback = null;
        }

        // Remembers where agentic-config came from so we can put it back
        // after the config popup closes.
        let _agConfigOrigParent = null;
        let _agConfigOrigNext = null;
        let _agConfigWasHidden = false;
        // Snapshot of input values at popup-open, used to revert on Cancel.
        let _agConfigSnapshot = null;

        // IDs of every agentic-* input the popup edits. Stays in sync with
        // the cost-estimator's COST_INPUT_IDS subset for agentic settings.
        const _AG_CONFIG_INPUT_IDS = [
            "config-agentic_model",
            "config-agentic_max_followups",
            "config-agentic_rerank_topk",
            "config-agentic_max_context_tokens",
            "config-agentic_max_output_tokens",
            "config-agentic_targeted_fetch",
            "config-agentic_recursive",
            "config-agentic_max_iterations",
        ];

        function _snapshotAgenticConfig() {
            const snap = { agenticToggle: false, inputs: {} };
            const tog = document.getElementById("agentic-toggle");
            snap.agenticToggle = !!(tog && tog.checked);
            _AG_CONFIG_INPUT_IDS.forEach(id => {
                const el = document.getElementById(id);
                if (!el) return;
                snap.inputs[id] = el.type === "checkbox" ? el.checked : el.value;
            });
            return snap;
        }

        function _restoreAgenticConfig(snap) {
            if (!snap) return;
            _AG_CONFIG_INPUT_IDS.forEach(id => {
                const el = document.getElementById(id);
                if (!el || !(id in snap.inputs)) return;
                const v = snap.inputs[id];
                if (el.type === "checkbox") {
                    if (el.checked !== v) el.checked = v;
                } else {
                    if (el.value !== v) el.value = v;
                }
            });
            const tog = document.getElementById("agentic-toggle");
            if (tog && tog.checked !== snap.agenticToggle) {
                tog.checked = snap.agenticToggle;
                tog.dispatchEvent(new Event("change"));
            }
            // Recompute cost estimator after a bulk revert.
            if (typeof renderCostEstimate === "function") renderCostEstimate();
        }

        function openAgenticConfigPopup() {
            const agConfig = document.getElementById("agentic-config");
            const body = document.getElementById("ag-config-body");
            if (!agConfig || !body) return;
            // Snapshot first so any in-popup edits can be reverted on Cancel.
            _agConfigSnapshot = _snapshotAgenticConfig();
            // Stash the original DOM position + visibility so we can restore.
            _agConfigOrigParent = agConfig.parentNode;
            _agConfigOrigNext = agConfig.nextSibling;
            _agConfigWasHidden = agConfig.classList.contains("hidden");
            // Force visible while it lives inside the popup.
            agConfig.classList.remove("hidden");
            body.appendChild(agConfig);
            // Opening the config popup implies agentic refinement is wanted, so
            // turn agentic mode on automatically (no in-popup enable toggle).
            const tog = document.getElementById("agentic-toggle");
            if (tog && !tog.checked) {
                tog.checked = true;
                tog.dispatchEvent(new Event("change"));
            }
            // Primary button label: "Run →" when there's a refinement
            // callback waiting, "Done" otherwise.
            const runBtn = document.getElementById("ag-config-run");
            if (runBtn) runBtn.innerHTML = _agConfirmCallback ? "Run &#8594;" : "Done";
            document.getElementById("ag-config-overlay").classList.remove("hidden");
        }

        // Hide the popup and return agentic-config to its home in the page.
        function _detachAgenticConfigPopup() {
            const agConfig = document.getElementById("agentic-config");
            if (agConfig && _agConfigOrigParent) {
                _agConfigOrigParent.insertBefore(agConfig, _agConfigOrigNext);
                const agToggle = document.getElementById("agentic-toggle");
                const shouldHide = agToggle ? !agToggle.checked : _agConfigWasHidden;
                agConfig.classList.toggle("hidden", shouldHide);
            }
            _agConfigOrigParent = null;
            _agConfigOrigNext = null;
            document.getElementById("ag-config-overlay").classList.add("hidden");
        }

        function cancelAgenticConfigPopup() {
            // Revert any in-popup edits, then close.
            _restoreAgenticConfig(_agConfigSnapshot);
            _agConfigSnapshot = null;
            _detachAgenticConfigPopup();
        }

        function commitAgenticConfigPopup() {
            // Agentic mode is enabled when the popup opens, so there's nothing
            // to commit here beyond keeping the tweaked config settings.
            _agConfigSnapshot = null;
            _detachAgenticConfigPopup();
        }

        document.addEventListener("DOMContentLoaded", () => {
            // Dark-mode toggle (top-left). The initial theme is applied by the
            // boot script in <head>; here we just handle clicks + persistence.
            const themeToggle = document.getElementById("theme-toggle");
            if (themeToggle) {
                const syncChecked = () =>
                    themeToggle.setAttribute("aria-checked",
                        document.documentElement.getAttribute("data-theme") === "dark");
                syncChecked();
                themeToggle.addEventListener("click", () => {
                    const root = document.documentElement;
                    const dark = root.getAttribute("data-theme") === "dark";
                    if (dark) root.removeAttribute("data-theme");
                    else root.setAttribute("data-theme", "dark");
                    syncChecked();
                    try { localStorage.setItem("specgpt-theme", dark ? "light" : "dark"); }
                    catch (e) { /* localStorage unavailable - toggle still works for the session */ }
                    // Re-render the flow chart so its node colors track the new
                    // theme (Mermaid bakes colors into the SVG at render time).
                    if (window._pipeTrace && typeof renderPipelineViz === "function") {
                        try { renderPipelineViz(window._pipeTrace, window._pipeQuery); } catch (e) {}
                    }
                });
            }

            document.getElementById("ag-confirm-cancel").addEventListener("click", closeAgenticConfirm);
            document.getElementById("ag-confirm-run").addEventListener("click", () => {
                closeAgenticConfirm();
                if (_agConfirmCallback) _agConfirmCallback();
            });
            document.getElementById("ag-confirm-edit").addEventListener("click", () => {
                closeAgenticConfirm();
                openAgenticConfigPopup();
            });
            // Click outside to dismiss
            document.getElementById("ag-confirm-overlay").addEventListener("click", e => {
                if (e.target === e.currentTarget) closeAgenticConfirm();
            });

            // Config popup wiring - Cancel reverts in-popup edits; Done/Run
            // commits them. Edits are NOT applied until commit.
            document.getElementById("ag-config-cancel").addEventListener("click", () => {
                cancelAgenticConfigPopup();
                _agConfirmCallback = null;
            });
            document.getElementById("ag-config-run").addEventListener("click", () => {
                const cb = _agConfirmCallback;
                commitAgenticConfigPopup();
                _agConfirmCallback = null;
                if (cb) cb();
            });
            document.getElementById("ag-config-overlay").addEventListener("click", e => {
                if (e.target === e.currentTarget) {
                    cancelAgenticConfigPopup();
                    _agConfirmCallback = null;
                }
            });

        });

        function renderMarkdown(md) {
            if (typeof md !== "string") return "";
            const html = marked.parse(md);
            return DOMPurify.sanitize(html, {
                // Whitelist the tags + attributes Claude actually emits. The
                // default DOMPurify whitelist already strips <script>, event
                // handlers, javascript: URLs, etc. This narrows it further so
                // a malicious answer can't, say, drop a giant <iframe>.
                ALLOWED_TAGS: [
                    "p", "br", "hr", "strong", "em", "code", "pre",
                    "ul", "ol", "li", "blockquote",
                    "h1", "h2", "h3", "h4", "h5", "h6",
                    "table", "thead", "tbody", "tr", "th", "td",
                    "a", "del", "ins", "sup", "sub", "span"
                ],
                ALLOWED_ATTR: ["href", "title", "align", "colspan", "rowspan"],
                ALLOW_DATA_ATTR: false,
                FORBID_ATTR: ["style", "onerror", "onload", "onclick"],
            });
        }

        const queryInput = document.getElementById("query-input");
        const searchBtn = document.getElementById("search-btn");
        const configToggle = document.getElementById("config-toggle");
        const configPanel = document.getElementById("config-panel");
        const resultsDiv = document.getElementById("results");
        const loadingDiv = document.getElementById("loading");
        const errorDiv = document.getElementById("error");
        const answerSection = document.getElementById("answer-section");
        const agenticToggle = document.getElementById("agentic-toggle");
        const agenticConfig = document.getElementById("agentic-config");
        agenticToggle.addEventListener("change", () => {
            const composerEl = document.getElementById("composer");
            if (composerEl) composerEl.classList.toggle("agentic-active", agenticToggle.checked);
            agenticConfig.classList.toggle("hidden", !agenticToggle.checked);
        });

        // Config panel toggle
        configToggle.addEventListener("click", () => {
            configPanel.classList.toggle("open");
        });

        // Auto-resize textarea
        const TEXTAREA_MAX_H = 200;
        function autoResizeInput() {
            queryInput.style.height = "auto";
            queryInput.style.height = queryInput.scrollHeight + "px";
            queryInput.style.overflowY = queryInput.scrollHeight > TEXTAREA_MAX_H ? "auto" : "hidden";
        }
        queryInput.addEventListener("input", autoResizeInput);

        // Enter submits, Shift+Enter inserts newline
        queryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                runQuery();
            }
        });

        // Search button
        searchBtn.addEventListener("click", runQuery);

        // ─── In-flight request tracking ───────────────────────────────────
        // One AbortController per active fetch. Cancel button calls .abort();
        // the corresponding fetch rejects with an AbortError which we treat
        // as a user-initiated cancellation (no error banner).
        let _activeAbort = null;
        let _loadingTimer = null;
        let _loadingStart = 0;

        function _startLoading(title) {
            const titleEl   = document.getElementById("loading-title");
            const elapsedEl = document.getElementById("loading-elapsed");
            const cancelBtn = document.getElementById("loading-cancel");
            if (titleEl) titleEl.textContent = title;
            if (cancelBtn) {
                cancelBtn.disabled = false;
                cancelBtn.textContent = "Cancel";
            }
            _startThinking();
            loadingDiv.classList.remove("hidden");
            // Disable search to prevent concurrent submits while a query runs.
            if (searchBtn) searchBtn.disabled = true;
            _loadingStart = Date.now();
            if (elapsedEl) elapsedEl.textContent = "0.0s elapsed";
            if (_loadingTimer) clearInterval(_loadingTimer);
            _loadingTimer = setInterval(() => {
                if (!elapsedEl) return;
                const sec = (Date.now() - _loadingStart) / 1000;
                elapsedEl.textContent = sec.toFixed(1) + "s elapsed";
            }, 100);
        }

        function _stopLoading() {
            loadingDiv.classList.add("hidden");
            if (_loadingTimer) { clearInterval(_loadingTimer); _loadingTimer = null; }
            _stopThinking();
            if (searchBtn) searchBtn.disabled = false;
            _activeAbort = null;
            answerSection.classList.remove("answer-stale");
        }

        // ─── Favicon badge ────────────────────────────────────────────────
        var _faviconEl = null;
        var _origFaviconHref = "/static/favicon.png";
        function _getFaviconEl() {
            if (!_faviconEl) _faviconEl = document.querySelector("link[rel='icon']");
            return _faviconEl;
        }
        function _setBadgeFavicon() {
            var img = new Image();
            img.src = _origFaviconHref;
            img.onload = function () {
                var sz = img.width || 32;
                var canvas = document.createElement("canvas");
                canvas.width = sz; canvas.height = sz;
                var ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0, sz, sz);
                var r = Math.max(5, sz * 0.22);
                ctx.beginPath();
                ctx.arc(sz - r, r, r, 0, 2 * Math.PI);
                ctx.fillStyle = "#5b8cff";
                ctx.fill();
                var el = _getFaviconEl();
                if (el) el.href = canvas.toDataURL("image/png");
            };
        }
        function _restoreFavicon() {
            var el = _getFaviconEl();
            if (el) el.href = _origFaviconHref;
        }
        window.addEventListener("focus", _restoreFavicon);

        function _notifyDone(title, body) {
            _setBadgeFavicon();
            if (!("Notification" in window)) return;
            if (Notification.permission === "granted") {
                try { new Notification(title, { body, icon: _origFaviconHref }); } catch (e) {}
            }
        }

        function _requestNotifyPermission() {
            if (!("Notification" in window) || Notification.permission !== "default") return;
            Notification.requestPermission();
        }

        // ─── "Thinking about…" cycling ticker ─────────────────────────────
        // Instead of listing every pipeline stage, show one calm line whose
        // text cycles. Generic phrases rotate on a timer so it feels alive;
        // when a real stage arrives we surface its phrase immediately.
        var THINK_MAP = {
            "query_processor":               "breaking your question into parts",
            "structured_lookup":             "looking up named fields and tables",
            "hybrid_search.vector_search":   "searching the specification",
            "hybrid_search.tsvector_search": "scanning for key terms",
            "hybrid_search.bm25_search":     "ranking keyword matches",
            "hybrid_search.rrf_merge":       "fusing the search results",
            "hybrid_search.total":           "searching the specification",
            "result_dedup":                  "gathering the most relevant sections",
            "final_rerank":                  "ranking the best matches",
            "generation":                    "writing a grounded answer",
            "query.followup":                "planning follow-up questions",
            "sub_query":                     "working through a sub-question",
            "refine.seed":                   "resuming from earlier work",
            "agentic.gap_analysis":          "checking the answer for gaps",
            "agentic.targeted_fetch":        "fetching specific figures and fields",
            "agentic.followup_search":       "following up on a sub-question",
            "agentic.rerank":                "re-ranking the expanded context",
            "agentic.regenerate":            "refining the final answer",
            "agentic.cap_reached":           "wrapping up"
        };
        var THINK_DEFAULT = [
            "reading your question",
            "searching the specification",
            "gathering relevant sections",
            "reasoning over the spec"
        ];
        var _thinkPool = [], _thinkIdx = 0, _thinkTimer = null;

        function _showThink(phrase) {
            var t = document.getElementById("loading-ticker");
            if (!t || !phrase) return;
            t.textContent = phrase;
            t.classList.remove("ticker-in");
            void t.offsetWidth;            // reflow so the animation restarts
            t.classList.add("ticker-in");
        }

        function _thinkPhrase(stage) {
            var name = String(stage || "")
                .replace(/_q\\d+(?=(?:\\.iter\\d+)?$)/, "")
                .replace(/\\.iter\\d+$/, "");
            return THINK_MAP[name] || null;
        }

        function _startThinking() {
            _thinkPool = THINK_DEFAULT.slice();
            _thinkIdx = 0;
            _showThink(_thinkPool[0]);
            if (_thinkTimer) clearInterval(_thinkTimer);
            _thinkTimer = setInterval(function () {
                if (!_thinkPool.length) return;
                _thinkIdx = (_thinkIdx + 1) % _thinkPool.length;
                _showThink(_thinkPool[_thinkIdx]);
            }, 2400);
        }

        function _stopThinking() {
            if (_thinkTimer) { clearInterval(_thinkTimer); _thinkTimer = null; }
        }

        // A real stage completed: surface its phrase now and keep it in rotation.
        function _onThinkStage(stage) {
            var phrase = _thinkPhrase(stage);
            if (!phrase) return;
            var at = _thinkPool.indexOf(phrase);
            if (at === -1) { _thinkPool.push(phrase); at = _thinkPool.length - 1; }
            _thinkIdx = at;
            _showThink(phrase);
        }

        // POST to an NDJSON streaming endpoint. Calls onProgress(evt) for each
        // {"type":"progress"} line; resolves with the {"type":"done"} data or
        // rejects on {"type":"error"} / HTTP error. Falls back gracefully if
        // the body can't be streamed (parses whatever arrived at the end).
        async function _streamPipeline(url, body, signal, onProgress) {
            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
                signal: signal,
            });
            if (response.status === 401) {
                window.location.href = "/login?next=" + encodeURIComponent(window.location.pathname);
                return null;
            }
            if (!response.ok) {
                let detail;
                try { detail = (await response.json()).detail; } catch (e) { detail = null; }
                const message = typeof detail === "string" ? detail
                    : (detail && typeof detail.message === "string" ? detail.message : "Request failed");
                const httpErr = new Error(message);
                httpErr.status = response.status;  // let callers branch (e.g. 404 → re-run)
                throw httpErr;
            }

            const reader = response.body && response.body.getReader ? response.body.getReader() : null;
            const decoder = new TextDecoder();
            let buf = "", done = null, errored = null;

            function handleLine(line) {
                line = line.trim();
                if (!line) return;
                let evt;
                try { evt = JSON.parse(line); } catch (e) { return; }
                if (evt.type === "progress") { if (onProgress) onProgress(evt); }
                else if (evt.type === "done") { done = evt.data; }
                else if (evt.type === "error") { errored = evt.detail; }
            }

            if (reader) {
                for (;;) {
                    const { value, done: rdDone } = await reader.read();
                    if (rdDone) break;
                    buf += decoder.decode(value, { stream: true });
                    let nl;
                    while ((nl = buf.indexOf("\\n")) !== -1) {
                        handleLine(buf.slice(0, nl));
                        buf = buf.slice(nl + 1);
                    }
                }
            } else {
                // No streaming reader: take the whole body, parse line-by-line.
                buf = await response.text();
            }
            buf.split("\\n").forEach(handleLine);

            if (errored) {
                const msg = typeof errored === "string" ? errored
                    : (errored && errored.message) ? errored.message : "Pipeline error";
                throw new Error(msg);
            }
            return done;
        }

        // Cancel button: aborts the in-flight fetch. Hook it up once on load.
        (function () {
            const btn = document.getElementById("loading-cancel");
            if (!btn) return;
            btn.addEventListener("click", () => {
                if (_activeAbort) {
                    btn.disabled = true;
                    btn.textContent = "Cancelling…";
                    _activeAbort.abort();
                }
            });
        })();

        async function runQuery() {
            const query = queryInput.value.trim();
            if (!query) return;
            _requestNotifyPermission();
            _restoreFavicon();

            // A new prompt is the only thing that clears open stage popups.
            closeAllStagePopups();

            // Reset the flag FAB for a fresh answer (hide + clear confirmation).
            if (typeof resetFlagFab === "function") resetFlagFab();

            // Collect config
            const config = {
                spec: window.getSelectedSpec(),
                llm_model: document.getElementById("config-llm_model").value,
                agentic_model: document.getElementById("config-agentic_model").value,
                vector_topk: parseInt(document.getElementById("config-vector_topk").value),
                tsvector_topk: parseInt(document.getElementById("config-tsvector_topk").value),
                bm25_topk: parseInt(document.getElementById("config-bm25_topk").value),
                rrf_k: parseInt(document.getElementById("config-rrf_k").value),
                rrf_output_topk: parseInt(document.getElementById("config-rrf_output_topk").value),
                final_rerank_topk: parseInt(document.getElementById("config-final_rerank_topk").value),
                max_subqueries: parseInt(document.getElementById("config-max_subqueries").value),
                agentic_max_followups: parseInt(document.getElementById("config-agentic_max_followups").value),
                agentic_rerank_topk: parseInt(document.getElementById("config-agentic_rerank_topk").value),
                agentic_max_context_tokens: parseInt(document.getElementById("config-agentic_max_context_tokens").value),
                agentic_max_output_tokens: parseInt(document.getElementById("config-agentic_max_output_tokens").value),
                agentic_targeted_fetch: document.getElementById("config-agentic_targeted_fetch").checked,
                agentic_recursive: document.getElementById("config-agentic_recursive").checked,
                agentic_max_iterations: parseInt(document.getElementById("config-agentic_max_iterations").value),
                auto_gap_check: document.getElementById("config-auto_gap_check").checked,
                figure_reserve_tokens: (function () {
                    var el = document.getElementById("config-figure_reserve_tokens");
                    var n = el ? parseInt(el.value, 10) : NaN;
                    return isNaN(n) ? 3000 : n;
                })(),
            };

            // Show loading
            resultsDiv.classList.remove("hidden");
            errorDiv.classList.add("hidden");
            if (!answerSection.classList.contains("hidden")) {
                answerSection.classList.add("answer-stale");
            } else {
                answerSection.classList.add("hidden");
            }

            const agentic = agenticToggle.checked;
            const title = agentic ? "Thinking deeply…" : "Thinking…";
            _startLoading(title);

            _activeAbort = new AbortController();
            try {
                const data = await _streamPipeline(
                    "/api/query/stream",
                    { query, config, debug: true, agentic },
                    _activeAbort.signal,
                    (evt) => _onThinkStage(evt.stage)
                );
                if (data) displayResults(data);
            } catch (err) {
                if (err.name === "AbortError") {
                    // User cancelled: leave the previous answer (if any) visible
                    // and don't show an error banner.
                    return;
                }
                errorDiv.textContent = `Error: ${err.message}`;
                errorDiv.classList.remove("hidden");
            } finally {
                _stopLoading();
            }
        }

        function escapeHtml(value) {
            if (value === null || value === undefined) return "";
            return String(value)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#39;");
        }

        // Holds the cancel function for the in-flight typewriter roll-out
        // so a follow-up query / refine can abort the previous animation.
        let _cancelAnswerStream = null;

        // Progressively reveal `fullText` into `el`. Re-renders markdown each
        // tick so partial syntax doesn't show through. Fast by default - the
        // intent is "chatbot rollout", not a leisurely typewriter.
        function streamAnswerInto(el, fullText, opts = {}) {
            const charsPerTick = opts.charsPerTick ?? 6;
            const tickMs       = opts.tickMs       ?? 20;
            const onDone       = typeof opts.onDone === "function" ? opts.onDone : null;
            const useMd = (typeof marked !== "undefined" && typeof DOMPurify !== "undefined");
            let i = 0;
            let cancelled = false;
            let timer = null;
            el.classList.add("streaming");
            function step() {
                if (cancelled) return;
                i = Math.min(fullText.length, i + charsPerTick);
                const slice = fullText.slice(0, i);
                if (useMd) {
                    el.innerHTML = renderMarkdown(slice);
                } else {
                    el.textContent = slice;
                }
                if (i < fullText.length) {
                    timer = setTimeout(step, tickMs);
                } else {
                    el.classList.remove("streaming");
                    if (onDone) onDone();
                }
            }
            step();
            return () => {
                cancelled = true;
                if (timer) clearTimeout(timer);
                // Flush to the full text so the user keeps the complete answer.
                if (useMd) {
                    el.innerHTML = renderMarkdown(fullText);
                } else {
                    el.textContent = fullText;
                }
                el.classList.remove("streaming");
            };
        }

        // ─── Stage name → human-friendly display ──────────────────────────
        // Maps the internal stage identifier to {title, subtitle, group}. Sub-
        // query and iteration suffixes are stripped before lookup so e.g.
        // `hybrid_search.vector_search_q2.iter1` maps to the "Semantic search"
        // entry with " (sub-query 3)" and " · pass 2" annotations.
        const STAGE_INFO = {
            "query":                        {t: "Query",                         s: "The question the user asked",                              g: "normal"},
            "query.followup":               {t: "Follow-up queries",             s: "What gap analysis decided to search for in this pass",     g: "agentic"},
            "sub_query":                    {t: "Sub-query",                     s: "One slice of the decomposed query",                        g: "normal"},
            "refine.seed":                  {t: "Resume from cache",             s: "Reused a prior /api/query first-pass state",               g: "normal"},
            "query_processor":              {t: "Understand the question",       s: "Decompose into sub-queries and extract entities",        g: "normal"},
            "structured_lookup":            {t: "Structured lookup",             s: "Direct hit against named fields, tables, or figures",     g: "normal"},
            "hybrid_search.vector_search":  {t: "Semantic search",               s: "Embeddings via Voyage (vector similarity)",               g: "normal"},
            "hybrid_search.tsvector_search":{t: "Keyword search",                s: "Postgres full-text tsvector match",                       g: "normal"},
            "hybrid_search.bm25_search":    {t: "BM25 search",                   s: "Classic Okapi BM25 ranking over the corpus",              g: "normal"},
            "hybrid_search.rrf_merge":      {t: "Fuse search branches",          s: "Reciprocal Rank Fusion across semantic + keyword + BM25", g: "normal"},
            "hybrid_search.total":          {t: "Hybrid search (total)",         s: "End-to-end time across all retrieval branches",           g: "normal"},
            "result_dedup":                 {t: "Deduplicate combined pool",     s: "Merge structured + hybrid chunks, drop duplicates",        g: "normal"},
            "final_rerank":                 {t: "Rerank",                        s: "Score each chunk against the query (Voyage rerank-2-lite)", g: "normal"},
            "figure_ref_expansion":         {t: "Pull referenced figures",       s: "Fetch data-structure tables the context cites but doesn't contain", g: "normal"},
            "agentic.figure_ref_expansion": {t: "Pull referenced figures",       s: "Fetch data-structure tables the context cites but doesn't contain", g: "agentic"},
            "generation":                   {t: "Generate answer",               s: "Synthesize the answer with citations (Claude)",            g: "normal"},
            "agentic.gap_analysis":         {t: "Agentic gap analysis",          s: "Does the answer fully cover the question?",                g: "agentic"},
            "agentic.targeted_fetch":       {t: "Targeted fetch",                s: "Pull figures, fields, or sections the model named",        g: "agentic"},
            "agentic.followup_search":      {t: "Follow-up search",              s: "Retrieve more chunks for a remaining gap",                 g: "agentic"},
            "agentic.rerank":               {t: "Rerank expanded pool",          s: "Rescore everything collected so far",                      g: "agentic"},
            "agentic.regenerate":           {t: "Regenerate answer",             s: "Synthesize the final answer with a larger context",        g: "agentic"},
            "agentic.cap_reached":          {t: "Iteration cap reached",         s: "Agentic loop stopped at its max-iterations setting",       g: "agentic"},
            "agentic.verdict_converged":    {t: "Converged",                     s: "Generator self-assessment marked the answer complete",     g: "agentic"},
            "agentic.stalled":              {t: "Stalled",                       s: "Reranked context unchanged from the previous pass; stopped early", g: "agentic"},
        };

        function formatStageDisplay(name) {
            let title, subtitle, group;
            const subqMatch = name.match(/_q(\\d+)(?=(?:\\.iter\\d+)?$)/);
            const iterMatch = name.match(/\\.iter(\\d+)$/);
            const stripped = name.replace(/_q\\d+(?=(?:\\.iter\\d+)?$)/, "")
                                  .replace(/\\.iter\\d+$/, "");
            const info = STAGE_INFO[stripped];
            if (info) {
                title = info.t;
                subtitle = info.s;
                group = info.g;
            } else {
                title = name.replace(/[._]/g, " ").replace(/\\b\\w/g, c => c.toUpperCase());
                subtitle = "";
                group = "other";
            }
            const annotations = [];
            if (subqMatch) annotations.push(`sub-query ${parseInt(subqMatch[1], 10) + 1}`);
            if (iterMatch) annotations.push(`pass ${parseInt(iterMatch[1], 10) + 1}`);
            if (annotations.length) title += " · " + annotations.join(" · ");
            return {title, subtitle, group};
        }

        // ─── Stage card rendering ─────────────────────────────────────────
        function renderStageCard(stage, idx) {
            const display = formatStageDisplay(stage.stage);
            const ms = (typeof stage.took_ms === "number") ? stage.took_ms : 0;
            const slowClass = ms > 1500 ? " stage-time-slow" : "";
            const groupClass = display.group === "agentic" ? " stage-group-agentic"
                              : display.group === "normal" ? " stage-group-normal"
                              : "";
            return `
                <div class="pipeline-stage${groupClass}">
                    <div class="stage-header" onclick="toggleStage(this)">
                        <div class="stage-title-block">
                            <span class="stage-name"><span class="stage-index">${idx + 1}.</span>${escapeHtml(display.title)}</span>
                            ${display.subtitle ? `<span class="stage-subtitle">${escapeHtml(display.subtitle)}</span>` : ""}
                        </div>
                        <span class="stage-time${slowClass}">${ms.toFixed(0)}ms</span>
                        <span class="stage-toggle">▼</span>
                    </div>
                    <div class="stage-content">
                        ${renderStageBody(stage)}
                    </div>
                </div>
            `;
        }

        function _kv(label, value) {
            return `<div class="stage-kv">
                <div class="stage-kv-label">${escapeHtml(label)}</div>
                <div class="stage-kv-value">${value}</div>
            </div>`;
        }

        function _chip(text, kind) {
            const cls = kind ? ` stage-chip-${kind}` : "";
            return `<span class="stage-chip${cls}">${escapeHtml(text)}</span>`;
        }

        function _fmtNum(n) {
            return (typeof n === "number") ? n.toLocaleString() : escapeHtml(String(n));
        }

        function _scoreString(r) {
            const s = (r.rerank_score !== null && r.rerank_score !== undefined) ? r.rerank_score
                    : (r.rrf_score !== null && r.rrf_score !== undefined) ? r.rrf_score
                    : (r.score !== null && r.score !== undefined) ? r.score
                    : null;
            return (typeof s === "number") ? s.toFixed(3) : "";
        }

        function _renderHitsTable(rows) {
            const TOP = 8;
            const top = rows.slice(0, TOP);
            const more = rows.length - top.length;
            const body = top.map(r => {
                const sid = r.section_id || r.id || "";
                const title = r.section_title || r.figure_number || "";
                const method = r.method || r.content_type || "";
                const score = _scoreString(r);
                return `<tr>
                    <td class="stage-mono">${sid !== "" ? "§" + escapeHtml(String(sid)) : ""}</td>
                    <td>${escapeHtml(String(title))}</td>
                    <td class="stage-meta">${escapeHtml(String(method))}</td>
                    <td class="stage-mono stage-meta">${escapeHtml(score)}</td>
                </tr>`;
            }).join("");
            const moreRow = more > 0
                ? `<tr><td colspan="4" class="stage-meta">…and ${more} more</td></tr>`
                : "";
            return `<table class="stage-hits">
                <thead><tr><th>Section</th><th>Title</th><th>Method</th><th>Score</th></tr></thead>
                <tbody>${body}${moreRow}</tbody>
            </table>`;
        }

        function renderStageBody(stage) {
            const out = stage.output || {};
            const inp = stage.input || {};
            const sections = [];

            // ─── Headline chips: top-level metrics at a glance ────────────
            const chips = [];
            if (out.error_type)                    chips.push(_chip("error: " + out.error_type, "error"));
            if (out.skipped)                       chips.push(_chip("skipped", "skipped"));
            if (typeof out.found === "boolean")    chips.push(_chip(out.found ? "found" : "not found", out.found ? "ok" : null));
            if (typeof out.needs_followup === "boolean") chips.push(_chip(out.needs_followup ? "needs follow-up" : "no gaps", out.needs_followup ? "warn" : "ok"));
            if (out.type)                          chips.push(_chip("type: " + out.type, "info"));
            if (out.confidence)                    chips.push(_chip("confidence: " + out.confidence, "info"));
            if (typeof out.count === "number")             chips.push(_chip(out.count + " hits"));
            if (typeof out.chunk_count === "number")       chips.push(_chip(out.chunk_count + " chunks"));
            if (typeof out.deduped_count === "number")     chips.push(_chip(out.deduped_count + " unique chunks"));
            if (typeof out.fetched_count === "number")     chips.push(_chip(out.fetched_count + " fetched"));
            if (typeof out.field_count === "number")       chips.push(_chip(out.field_count + " field" + (out.field_count===1?"":"s")));
            if (typeof out.table_count === "number")       chips.push(_chip(out.table_count + " table" + (out.table_count===1?"":"s")));
            if (typeof out.citation_count === "number")    chips.push(_chip(out.citation_count + " citation" + (out.citation_count===1?"":"s")));
            if (typeof out.answer_length === "number")     chips.push(_chip(out.answer_length.toLocaleString() + " chars"));
            if (typeof out.iterations_run === "number")    chips.push(_chip(out.iterations_run + " iteration" + (out.iterations_run===1?"":"s"), "warn"));
            if (chips.length) sections.push(`<div class="stage-metrics">${chips.join("")}</div>`);

            // ─── Inputs / queries / reasoning ─────────────────────────────
            if (inp.query)             sections.push(_kv("Query",   `<code>${escapeHtml(inp.query)}</code>`));
            if (out.reason)            sections.push(_kv("Reason",  escapeHtml(out.reason)));
            if (out.last_gap_reason)   sections.push(_kv("Last gap reason", escapeHtml(out.last_gap_reason)));
            if (out.rationale)         sections.push(_kv("Rationale", escapeHtml(out.rationale)));
            if (out.notes)             sections.push(_kv("Notes",   escapeHtml(out.notes)));
            if (out.note)              sections.push(_kv("Note",    escapeHtml(out.note)));

            // ─── Sub-queries / follow-up queries ──────────────────────────
            const sqList = (Array.isArray(out.sub_queries) && out.sub_queries.length) ? out.sub_queries
                         : (Array.isArray(inp.sub_queries) && inp.sub_queries.length) ? inp.sub_queries
                         : null;
            if (sqList) {
                sections.push(_kv("Sub-queries",
                    `<ol class="stage-list">${sqList.map(q => `<li>${escapeHtml(String(q))}</li>`).join("")}</ol>`));
            }
            if (Array.isArray(out.queries) && out.queries.length) {
                sections.push(_kv("Follow-up queries",
                    `<ol class="stage-list">${out.queries.map(q => `<li>${escapeHtml(String(q))}</li>`).join("")}</ol>`));
            }

            // ─── Entities ──────────────────────────────────────────────────
            if (Array.isArray(out.entities) && out.entities.length) {
                const tags = out.entities.map(e => {
                    if (e && typeof e === "object" && e.text) {
                        const kind = e.kind ? `<span class="stage-tag-kind">${escapeHtml(String(e.kind))}</span>` : "";
                        return `<span class="stage-tag">${escapeHtml(String(e.text))}${kind}</span>`;
                    }
                    return `<span class="stage-tag">${escapeHtml(String(e))}</span>`;
                }).join("");
                sections.push(_kv("Entities", tags));
            }

            // ─── Requested resources (agentic targeted-fetch) ─────────────
            const reqRes = out.requested_resources || (inp.requested && typeof inp.requested === "object" ? inp.requested : null);
            if (reqRes && (reqRes.figures || reqRes.fields || reqRes.sections)) {
                const parts = [];
                if (Array.isArray(reqRes.figures)  && reqRes.figures.length)  parts.push(`<li><b>Figures:</b> ${reqRes.figures.map(f => escapeHtml(String(f))).join(", ")}</li>`);
                if (Array.isArray(reqRes.fields)   && reqRes.fields.length)   parts.push(`<li><b>Fields:</b> ${reqRes.fields.map(f => escapeHtml(String(f))).join(", ")}</li>`);
                if (Array.isArray(reqRes.sections) && reqRes.sections.length) parts.push(`<li><b>Sections:</b> ${reqRes.sections.map(f => escapeHtml(String(f))).join(", ")}</li>`);
                if (parts.length) sections.push(_kv("Requested", `<ul class="stage-list">${parts.join("")}</ul>`));
            }
            if (out.by_method && typeof out.by_method === "object") {
                const items = Object.entries(out.by_method)
                    .map(([k, v]) => `<li>${escapeHtml(k)}: ${escapeHtml(String(v))}</li>`).join("");
                if (items) sections.push(_kv("By method", `<ul class="stage-list">${items}</ul>`));
            }

            // ─── Top hits table ────────────────────────────────────────────
            const hits = (Array.isArray(out.results) && out.results.length) ? out.results
                       : (Array.isArray(out.sources) && out.sources.length) ? out.sources
                       : (Array.isArray(out.fetched) && out.fetched.length) ? out.fetched
                       : null;
            if (hits) sections.push(_kv("Top hits", _renderHitsTable(hits)));
            if (Array.isArray(out.context_used) && out.context_used.length) {
                sections.push(_kv("Context used", _renderHitsTable(out.context_used)));
            }

            // ─── Tokens (model cost) ───────────────────────────────────────
            if (out.tokens && typeof out.tokens === "object") {
                const t = out.tokens;
                const prompt = t.prompt || 0;
                const completion = t.completion || 0;
                const total = prompt + completion;
                sections.push(_kv("Tokens",
                    `<span class="stage-mono">${prompt.toLocaleString()} prompt + ${completion.toLocaleString()} completion = <b>${total.toLocaleString()}</b></span>`));
            }

            // ─── Model / config knobs worth showing ───────────────────────
            const cfgRows = [];
            if (inp.model)                                       cfgRows.push(`<code>${escapeHtml(inp.model)}</code>`);
            if (typeof inp.top_k === "number")                   cfgRows.push(`top-k: ${inp.top_k}`);
            if (typeof inp.max_followups === "number")           cfgRows.push(`max follow-ups: ${inp.max_followups}`);
            if (typeof inp.max_context_tokens === "number")      cfgRows.push(`max context: ${inp.max_context_tokens.toLocaleString()} tok`);
            if (typeof inp.iteration === "number")               cfgRows.push(`pass ${inp.iteration + 1}`);
            if (cfgRows.length) sections.push(_kv("Config", cfgRows.join(" · ")));

            // ─── Raw JSON fallback for power users ────────────────────────
            const json = JSON.stringify(stage, null, 2);
            sections.push(`<details class="stage-raw"><summary>Show raw JSON</summary><pre class="stage-json">${escapeHtml(json)}</pre></details>`);

            return sections.join("");
        }

        function toggleStage(header) {
            header.classList.toggle("open");
            header.nextElementSibling.classList.toggle("open");
        }

        // ─── Agent strip (in-banner): render gap-hint from last response ─
        async function runRefine(requestId) {
            if (!requestId) {
                errorDiv.textContent = "Error: no request_id available for refine";
                errorDiv.classList.remove("hidden");
                return;
            }
            // Reuse the same config payload the user has set, so agentic_*
            // tweaks (max_followups, rerank_topk, recursive, etc.) apply.
            const config = {
                spec: window.getSelectedSpec(),
                vector_topk: parseInt(document.getElementById("config-vector_topk").value),
                tsvector_topk: parseInt(document.getElementById("config-tsvector_topk").value),
                bm25_topk: parseInt(document.getElementById("config-bm25_topk").value),
                rrf_k: parseInt(document.getElementById("config-rrf_k").value),
                rrf_output_topk: parseInt(document.getElementById("config-rrf_output_topk").value),
                final_rerank_topk: parseInt(document.getElementById("config-final_rerank_topk").value),
                max_subqueries: parseInt(document.getElementById("config-max_subqueries").value),
                agentic_max_followups: parseInt(document.getElementById("config-agentic_max_followups").value),
                agentic_rerank_topk: parseInt(document.getElementById("config-agentic_rerank_topk").value),
                agentic_max_context_tokens: parseInt(document.getElementById("config-agentic_max_context_tokens").value),
                agentic_max_output_tokens: parseInt(document.getElementById("config-agentic_max_output_tokens").value),
                agentic_targeted_fetch: document.getElementById("config-agentic_targeted_fetch").checked,
                agentic_recursive: document.getElementById("config-agentic_recursive").checked,
                agentic_max_iterations: parseInt(document.getElementById("config-agentic_max_iterations").value),
                auto_gap_check: document.getElementById("config-auto_gap_check").checked,
                figure_reserve_tokens: (function () {
                    var el = document.getElementById("config-figure_reserve_tokens");
                    var n = el ? parseInt(el.value, 10) : NaN;
                    return isNaN(n) ? 3000 : n;
                })(),
            };

            errorDiv.classList.add("hidden");
            if (!answerSection.classList.contains("hidden")) {
                answerSection.classList.add("answer-stale");
            } else {
                answerSection.classList.add("hidden");
            }
            _startLoading("Refining the answer…");

            _activeAbort = new AbortController();
            try {
                const data = await _streamPipeline(
                    "/api/refine/stream",
                    { request_id: requestId, config, debug: true },
                    _activeAbort.signal,
                    (evt) => _onThinkStage(evt.stage)
                );
                if (data) displayResults(data);
            } catch (err) {
                if (err.name === "AbortError") return;
                if (err.status === 404) {
                    // Cache evicted (older session or restart). Fall back to a
                    // full re-run so the user always has a working path. Hand
                    // off to runQuery which manages its own loading lifecycle.
                    _stopLoading();
                    runQuery();
                    return;
                }
                errorDiv.textContent = `Error: ${err.message}`;
                errorDiv.classList.remove("hidden");
            } finally {
                _stopLoading();
            }
        }
    </script>
    <script>
        /* ───────────────────────────────────────────────────────────────────
           Redesign glue: new two-column layout, sources sidebar, citation
           chips, gap-hint card, pipeline disclosure, tweaks, and the
           edit-config-before-refine flow. Reuses the engine defined above.
           ─────────────────────────────────────────────────────────────────── */

        var I_check = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 5 5L20 7"/></svg>';
        var I_warn  = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4m0 4h.01M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>';
        var I_info  = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 16v-4m0-4h.01"/></svg>';
        var I_robot = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="3" x2="12" y2="6"/><circle cx="12" cy="2.6" r="1" fill="currentColor" stroke="none"/><rect x="4" y="7" width="16" height="12" rx="3"/><circle cx="9" cy="13" r="1.3" fill="currentColor" stroke="none"/><circle cx="15" cy="13" r="1.3" fill="currentColor" stroke="none"/><line x1="10" y1="16.5" x2="14" y2="16.5"/></svg>';
        var I_arrow = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';

        /* ── citation cross-highlighting ─────────────────────────────────── */
        function setActiveSec(sec) {
            document.querySelectorAll(".cite-chip").forEach(function (c) {
                c.classList.toggle("hot", !!sec && c.getAttribute("data-sec") === sec);
            });
            document.querySelectorAll(".src").forEach(function (s) {
                s.classList.toggle("hot", !!sec && s.getAttribute("data-sec") === sec);
            });
        }
        /* Figure analog of setActiveSec: hovering a fig-chip (or a figure card
           in the sidebar) highlights every chip/card for that figure, exactly
           like section chips do. Matches on data-fig instead of data-sec. */
        function setActiveFig(num) {
            document.querySelectorAll(".fig-chip").forEach(function (c) {
                c.classList.toggle("hot", !!num && c.getAttribute("data-fig") === num);
            });
            document.querySelectorAll(".src").forEach(function (s) {
                s.classList.toggle("hot", !!num && s.getAttribute("data-fig") === num);
            });
        }

        /* ── citation preview popover ─────────────────────────────────────
           Clicking a section chip opens a small anchored popover showing the
           cited section's title, a snippet of its text (shipped with the
           citations payload, so no extra fetch), the page, and an Open PDF
           button. Esc or clicking elsewhere dismisses it. */
        var _citePop = null;
        function closeCitePop() {
            if (_citePop) { _citePop.remove(); _citePop = null; }
        }
        function showCitePop(chip, c) {
            closeCitePop();
            if (!chip || !c) return;
            var pop = document.createElement("div");
            pop.className = "cite-pop";

            var title = document.createElement("div");
            title.className = "cite-pop-title";
            var sid = document.createElement("span");
            sid.className = "mono";
            sid.textContent = String.fromCharCode(167) + String(c.section_id || "");
            title.appendChild(sid);
            if (c.section_title && c.section_title !== c.section_id) {
                title.appendChild(document.createTextNode("  " + c.section_title));
            }
            pop.appendChild(title);

            if (c.snippet) {
                var body = document.createElement("div");
                body.className = "cite-pop-body";
                body.textContent = c.snippet;
                pop.appendChild(body);
            }

            var foot = document.createElement("div");
            foot.className = "cite-pop-foot";
            var page = document.createElement("span");
            page.className = "cite-pop-page";
            if (c.pdf_pages && c.pdf_pages.length > 0) {
                var p0 = parseInt(c.pdf_pages[0], 10);
                if (!isNaN(p0)) page.textContent = "p. " + (p0 + 1);
            }
            foot.appendChild(page);
            if (pdfPageJumpSupported()) {
                var btn = document.createElement("button");
                btn.type = "button";
                btn.className = "cite-pop-open";
                btn.textContent = "Open PDF";
                btn.addEventListener("click", function (e) {
                    e.stopPropagation();
                    openCitationPdf(c);
                });
                foot.appendChild(btn);
            }
            pop.appendChild(foot);

            document.body.appendChild(pop);
            _citePop = pop;
            // Anchor under the chip, clamped to the viewport.
            var r = chip.getBoundingClientRect();
            var left = Math.max(8, Math.min(r.left, window.innerWidth - pop.offsetWidth - 8));
            var top = r.bottom + 8;
            if (top + pop.offsetHeight > window.innerHeight - 8) {
                top = Math.max(8, r.top - pop.offsetHeight - 8);
            }
            pop.style.left = left + "px";
            pop.style.top = top + "px";
        }
        function showFigPop(chip, f) {
            closeCitePop();
            if (!chip || !f) return;
            var pop = document.createElement("div");
            pop.className = "cite-pop";

            var title = document.createElement("div");
            title.className = "cite-pop-title";
            var sid = document.createElement("span");
            sid.className = "mono";
            sid.textContent = "Figure " + String(f.figure_number || "");
            title.appendChild(sid);
            if (f.caption) {
                title.appendChild(document.createTextNode("  " + f.caption));
            }
            pop.appendChild(title);

            var foot = document.createElement("div");
            foot.className = "cite-pop-foot";
            var page = document.createElement("span");
            page.className = "cite-pop-page";
            if (f.pdf_pages && f.pdf_pages.length > 0) {
                var p0 = parseInt(f.pdf_pages[0], 10);
                if (!isNaN(p0)) page.textContent = "p. " + (p0 + 1);
            }
            foot.appendChild(page);

            var actions = document.createElement("div");
            actions.className = "cite-pop-actions";
            actions.style.display = "flex";
            actions.style.gap = "6px";

            var renderBtn = document.createElement("button");
            renderBtn.type = "button";
            renderBtn.className = "cite-pop-open";
            renderBtn.textContent = "Render";
            renderBtn.addEventListener("click", function (e) {
                e.stopPropagation();
                openFigureRenderPopup(f);
                closeCitePop();
            });
            actions.appendChild(renderBtn);

            if (pdfPageJumpSupported()) {
                var btn = document.createElement("button");
                btn.type = "button";
                btn.className = "cite-pop-open";
                btn.textContent = "Open PDF";
                btn.addEventListener("click", function (e) {
                    e.stopPropagation();
                    openFigurePdf(f);
                });
                actions.appendChild(btn);
            }
            foot.appendChild(actions);
            pop.appendChild(foot);

            document.body.appendChild(pop);
            _citePop = pop;
            // Anchor under the chip, clamped to the viewport.
            var r = chip.getBoundingClientRect();
            var left = Math.max(8, Math.min(r.left, window.innerWidth - pop.offsetWidth - 8));
            var top = r.bottom + 8;
            if (top + pop.offsetHeight > window.innerHeight - 8) {
                top = Math.max(8, r.top - pop.offsetHeight - 8);
            }
            pop.style.left = left + "px";
            pop.style.top = top + "px";
        }
        document.addEventListener("click", function (e) {
            if (!_citePop) return;
            if (_citePop.contains(e.target)) return;
            if (e.target.closest && e.target.closest(".cite-chip")) return;
            // def-term clicks open their own popover (handler below); letting
            // this dismiss run too would close it in the same click.
            if (e.target.closest && e.target.closest("code.def-term")) return;
            closeCitePop();
        });
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape") closeCitePop();
        });

        /* Replace the model's bracketed citation tags ([§5.2.1] or
           [§5.2.1, §5.3]) in the rendered answer with clean citation chips.
           Walks text nodes only (skips code/pre/anchors). A bracket is only
           treated as a citation when every token inside it is a section id
           that the backend actually returned, so stray "[...]" is left alone.
           Regex-free to avoid backslash/unicode escaping inside this template. */
        function linkifyCitations(root, citations, figures) {
            // Figure-only answers arrive with citations=[] but a populated
            // figures payload - still linkify those brackets, so don't bail
            // unless BOTH are empty.
            if (!root) return;
            if (!((citations && citations.length) || (figures && figures.length))) return;
            var citeMap = {};
            var aliasMap = {};  // text the model wrote -> resolved section id
            (citations || []).forEach(function (c) {
                var id = String(c.section_id || "");
                // Only resolved citations become clickable chips. Hallucinated
                // ones (referenced but not in retrieved context) stay as plain
                // text so they don't masquerade as a verified, linkable source.
                if (id && !c.hallucinated) {
                    citeMap[id] = c;
                    aliasMap[id] = id;
                    // The backend resolves near-miss cites ("5.3") to the
                    // in-context section ("5.3.2.1") and reports the original
                    // as cited_as - index it so the inline text the user
                    // actually sees gets chipped and tied to this source.
                    if (c.cited_as) aliasMap[String(c.cited_as)] = id;
                }
            });
            // Figures the model may cite inside a bracket ("[Figure 328]" or
            // "[§8.1.13, Figure 328]"). Resolved via the figures payload, which
            // carries the page. linkifyFigures handles the bare "Figure N" form.
            var figMap = {};
            (figures || []).forEach(function (f) {
                var k = String(f.figure_number || "").trim();
                if (k) figMap[k] = f;
            });
            // If a token is "Figure 328" / "Fig. 12a", return its number, else "".
            function figTokNum(t) {
                var m = /^fig(?:ure)?\\.?\\s*([0-9]+[a-z]?)$/i.exec(t);
                return m ? m[1] : "";
            }

            // Strip a leading § (U+00A7 = 167) and surrounding spaces.
            function cleanTok(t) {
                var a = 0, b = t.length;
                while (a < b && (t.charCodeAt(a) === 32 || t.charCodeAt(a) === 167)) a++;
                while (b > a && t.charCodeAt(b - 1) === 32) b--;
                return t.slice(a, b);
            }
            // Normalize a whole bracket body to compare against a title-style
            // citation id (mirrors the backend: drop the section mark, collapse
            // whitespace, trim trailing dots). Lets a §-title that contains a
            // comma resolve as one citation instead of being split on the comma.
            function normWhole(s) {
                return s.split(String.fromCharCode(167)).join(" ")
                        .replace(/\\s+/g, " ").trim().replace(/\\.+$/, "");
            }

            var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
            var nodes = [], n;
            while ((n = walker.nextNode())) {
                if (n.parentNode && n.parentNode.closest && n.parentNode.closest(".cite-chip, code, pre, a")) continue;
                nodes.push(n);
            }

            nodes.forEach(function (node) {
                var text = node.nodeValue;
                if (text.indexOf("[") === -1) return;
                var frag = document.createDocumentFragment();
                var pos = 0, i = 0, changed = false;
                while (i < text.length) {
                    if (text.charAt(i) !== "[") { i++; continue; }
                    var close = text.indexOf("]", i + 1);
                    if (close === -1) break;
                    var inner = text.slice(i + 1, close);
                    // Each citation is "§"-prefixed; split on § (not comma) so a
                    // section title containing a comma stays one token. Brackets
                    // with no § (e.g. "[Figure 328]") fall back to comma split.
                    var SEC = String.fromCharCode(167);
                    var rawToks = inner.indexOf(SEC) !== -1 ? inner.split(SEC) : inner.split(",");
                    var items = [], ok = true, any = false;
                    for (var p = 0; p < rawToks.length; p++) {
                        var tok = cleanTok(rawToks[p]).replace(/^[\\s,]+/, "").replace(/[\\s,]+$/, "");
                        if (!tok) continue;        // empty (e.g. stray separator)
                        any = true;
                        if (aliasMap[tok]) { items.push({ sec: aliasMap[tok], label: tok }); continue; }
                        var fn = figTokNum(tok);
                        if (fn && figMap[fn]) { items.push({ fig: fn }); continue; }
                        ok = false; break;
                    }
                    // Nothing resolved cleanly: the bracket may be a single
                    // section TITLE containing a comma in a §-less form. Try the
                    // whole body as one id.
                    if (!ok || !any) {
                        var whole = normWhole(inner);
                        if (whole && aliasMap[whole]) { items = [{ sec: aliasMap[whole], label: whole }]; ok = true; }
                        else { ok = false; }
                    }
                    if (ok && items.length) {
                        if (i > pos) frag.appendChild(document.createTextNode(text.slice(pos, i)));
                        items.forEach(function (it, k) {
                            if (k > 0) frag.appendChild(document.createTextNode(" "));
                            var span = document.createElement("span");
                            span.setAttribute("tabindex", "0");
                            if (it.fig) {
                                span.className = "cite-chip fig-chip mono";
                                span.setAttribute("data-fig", it.fig);
                                span.textContent = "Figure " + it.fig;
                            } else {
                                span.className = "cite-chip mono";
                                span.setAttribute("data-sec", it.sec);
                                // Show what the answer wrote (the alias) while
                                // linking/highlighting via the resolved id.
                                span.textContent = it.label || it.sec;
                            }
                            frag.appendChild(span);
                        });
                        pos = close + 1;
                        changed = true;
                    }
                    i = close + 1;
                }
                if (!changed) return;
                if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
                node.parentNode.replaceChild(frag, node);
            });

            // Second pass: bare section ids in prose. The model sometimes lists
            // related sections without brackets ("... supports 8.2.6 8.1.9").
            // Only dotted ids (≥1 dot) that are in the resolved citation set are
            // linkified, so byte ranges ("257:256"), bit ranges ("15:12") and
            // hex values ("10h") are never touched.
            var bareRe = /\\b[A-Z]?[0-9]+(?:\\.[0-9]+)+[a-z]?\\b/g;
            var bnodes = [], bn;
            var bwalker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
            while ((bn = bwalker.nextNode())) {
                if (bn.parentNode && bn.parentNode.closest &&
                    bn.parentNode.closest(".cite-chip, code, pre, a")) continue;
                bnodes.push(bn);
            }
            bnodes.forEach(function (node) {
                var text = node.nodeValue;
                bareRe.lastIndex = 0;
                var frag = document.createDocumentFragment();
                var pos = 0, m, changed = false;
                while ((m = bareRe.exec(text)) !== null) {
                    var tok = m[0];
                    var rid = aliasMap[tok];
                    if (!rid) continue;
                    if (m.index > pos) frag.appendChild(document.createTextNode(text.slice(pos, m.index)));
                    var span = document.createElement("span");
                    span.className = "cite-chip mono";
                    span.setAttribute("data-sec", rid);
                    span.setAttribute("tabindex", "0");
                    span.textContent = tok;
                    frag.appendChild(span);
                    pos = m.index + tok.length;
                    changed = true;
                }
                if (!changed) return;
                if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
                node.parentNode.replaceChild(frag, node);
            });

            root.querySelectorAll(".cite-chip").forEach(function (c) {
                // Figure chips (incl. ones we made for bracketed figures) are
                // wired by linkifyFigures, which runs next - skip them here so
                // their click handler isn't bound twice (which opens two tabs).
                if (c.classList.contains("fig-chip")) return;
                c.addEventListener("mouseenter", function () { setActiveSec(c.getAttribute("data-sec")); });
                c.addEventListener("mouseleave", function () { setActiveSec(null); });
                c.addEventListener("click", function () {
                    var sec = c.getAttribute("data-sec");
                    showCitePop(c, citeMap[sec]);
                });
            });
        }

        /* Style the "Source: [§…]" attribution line the model emits under a
           table or code block (see system-prompt rule 2b). Marks the paragraph
           so it reads as a subtle caption tied to the block above it, and
           strips the leading "Source:" label since the chip already says it. */
        function styleBlockAttributions(root) {
            if (!root) return;
            var paras = root.querySelectorAll("p");
            for (var i = 0; i < paras.length; i++) {
                var p = paras[i];
                if (!p.querySelector(".cite-chip")) continue;
                var first = p.firstChild;
                if (!first || first.nodeType !== 3) continue;       // must lead with text
                var lead = first.nodeValue.replace(/^\\s+/, "");
                if (lead.slice(0, 7).toLowerCase() !== "source:") continue;
                first.nodeValue = first.nodeValue.replace(/^\\s*[Ss]ource:\\s*/, "");
                if (!first.nodeValue) p.removeChild(first);
                p.classList.add("block-attrib");
                // Tie it visually to the block directly above, if any.
                var prev = p.previousElementSibling;
                if (prev && (prev.tagName === "TABLE" || prev.tagName === "PRE")) {
                    prev.classList.add("has-attrib");
                }
            }
        }

        /* ── field-acronym definition popovers ────────────────────────────
           The model already code-formats spec acronyms (AUS, PKAS, CDPALG).
           After an answer renders, inline-code tokens that match a known
           field acronym (/api/define/terms, fetched once per spec) get a
           dashed underline; clicking one shows the field's full name +
           description in a cite-pop style popover (/api/define, cached).
           Only acronym-shaped tokens are eligible: uppercase alphanumerics
           starting with a letter. Hex/bit literals ("0x1F", "3Fh", '1',
           "257:256") never match, so values stay plain code. */
        var _defTerms = {};   // spec → Set of acronyms, or an array of waiting callbacks while fetching
        var _defCache = {};   // spec + "|" + term → matches array
        function _ensureDefTerms(spec, cb) {
            var have = _defTerms[spec];
            if (have instanceof Set) { cb(have); return; }
            if (Array.isArray(have)) { have.push(cb); return; }  // fetch already in flight
            _defTerms[spec] = [cb];
            fetch("/api/define/terms?spec=" + encodeURIComponent(spec))
                .then(function (r) { return r.ok ? r.json() : null; })
                .then(function (data) {
                    // A non-ok response must not cache an empty set, or one
                    // transient failure disables popovers for the session.
                    if (!data) { delete _defTerms[spec]; return; }
                    var waiting = _defTerms[spec];
                    var set = new Set(data.terms || []);
                    _defTerms[spec] = set;
                    (Array.isArray(waiting) ? waiting : []).forEach(function (fn) { fn(set); });
                })
                .catch(function () { delete _defTerms[spec]; });
        }
        function markDefinableTerms(root, spec) {
            if (!root || !spec) return;
            _ensureDefTerms(spec, function (terms) {
                if (!terms.size) return;
                root.querySelectorAll("code").forEach(function (c) {
                    if (c.closest("pre") || c.classList.contains("def-term")) return;
                    var t = (c.textContent || "").trim();
                    if (!/^[A-Z][A-Z0-9]{1,11}$/.test(t)) return;
                    if (!terms.has(t)) return;
                    c.classList.add("def-term");
                    c.setAttribute("tabindex", "0");
                    c.setAttribute("data-term", t);
                    c.setAttribute("data-spec", spec);
                });
            });
        }
        function renderDefPop(el, term, matches) {
            closeCitePop();
            var m = (matches && matches.length) ? matches[0] : null;
            const key = "def-" + term;
            const existing = _stagePopups.find(function(p) { return p.key === key; });
            if (existing) { _raisePopup(existing); return; }

            const pop = document.createElement("div");
            pop.className = "stage-popup";
            pop.setAttribute("role", "dialog");
            pop.style.width = "420px";

            const header = document.createElement("div");
            header.className = "stage-popup-header";

            const titleEl = document.createElement("span");
            titleEl.className = "stage-popup-title";
            titleEl.innerHTML = "<span class='mono'>" + escapeHtml(term) + "</span>" + (m && m.full_name ? "  " + escapeHtml(m.full_name) : "");

            const actions = document.createElement("div");
            actions.className = "stage-popup-actions";

            const closeAllBtn = document.createElement("button");
            closeAllBtn.type = "button";
            closeAllBtn.className = "stage-popup-close-all";
            closeAllBtn.textContent = "Close all";

            const closeBtn = document.createElement("button");
            closeBtn.type = "button";
            closeBtn.className = "stage-popup-close";
            closeBtn.setAttribute("data-tip", "Close this");
            closeBtn.innerHTML = "&#x2715;";

            actions.appendChild(closeAllBtn);
            actions.appendChild(closeBtn);
            header.appendChild(titleEl);
            header.appendChild(actions);

            const body = document.createElement("div");
            body.className = "stage-popup-body";
            body.textContent = (m && m.description) || "No definition found in the field index.";

            if (m) {
                var foot = document.createElement("div");
                foot.className = "cite-pop-foot";
                foot.style.marginTop = "12px";
                foot.style.borderTop = "1px solid var(--border)";
                foot.style.paddingTop = "8px";
                var ctx = document.createElement("span");
                ctx.className = "cite-pop-page";
                var bits = [];
                if (m.spec) {
                    var sd = (window._specData || []).filter(function (s) { return s.id === m.spec; })[0];
                    bits.push((sd && sd.label) || m.spec);
                }
                if (m.figure_number) bits.push("Figure " + m.figure_number);
                if (m.offset) bits.push((m.offset_type === "bits" ? "bits " : "offset ") + m.offset);
                if (matches.length > 1) bits.push("+" + (matches.length - 1) + " more context" + (matches.length > 2 ? "s" : ""));
                ctx.textContent = bits.join(" \u00b7 ");
                foot.appendChild(ctx);
                body.appendChild(foot);
            }

            pop.appendChild(header);
            pop.appendChild(body);
            document.body.appendChild(pop);

            var r = el.getBoundingClientRect();
            var left = Math.max(8, Math.min(r.left, window.innerWidth - 420 - 8));
            var top = r.bottom + 8;
            if (top + pop.offsetHeight > window.innerHeight - 8) {
                top = Math.max(8, r.top - pop.offsetHeight - 8);
            }
            pop.style.left = left + "px";
            pop.style.top = top + "px";

            const rec = {el: pop, key};
            _stagePopups.push(rec);

            closeBtn.addEventListener("click", function(ev) { ev.stopPropagation(); closeStagePopup(rec); });
            closeAllBtn.addEventListener("click", function(ev) { ev.stopPropagation(); closeAllStagePopups(); });
            closeBtn.addEventListener("mouseenter", function() { _showPopupTip(closeBtn); });
            closeBtn.addEventListener("mouseleave", _hidePopupTip);
            closeBtn.addEventListener("focus", function() { _showPopupTip(closeBtn); });
            closeBtn.addEventListener("blur", _hidePopupTip);
            pop.addEventListener("mousedown", function() { _raisePopup(rec); });
            _makePopupDraggable(pop, header);
            _raisePopup(rec);

            while (_stagePopups.length > STAGE_POPUP_CAP) {
                const oldest = _stagePopups.shift();
                if (oldest.el && oldest.el.parentNode) oldest.el.parentNode.removeChild(oldest.el);
            }
            _updatePopupMultiState();
        }
        function showDefPop(el) {
            var term = el.getAttribute("data-term");
            var spec = el.getAttribute("data-spec") || "base";
            if (!term) return;
            var key = spec + "|" + term;
            if (_defCache[key]) { renderDefPop(el, term, _defCache[key]); return; }
            fetch("/api/define?term=" + encodeURIComponent(term) + "&spec=" + encodeURIComponent(spec))
                .then(function (r) { return r.ok ? r.json() : null; })
                .then(function (data) {
                    if (!data) return;
                    _defCache[key] = data.matches || [];
                    renderDefPop(el, term, _defCache[key]);
                })
                .catch(function () {});
        }
        document.addEventListener("click", function (e) {
            var el = e.target.closest && e.target.closest("code.def-term");
            if (!el) return;
            showDefPop(el);
        });
        document.addEventListener("keydown", function (e) {
            if (e.key !== "Enter") return;
            var el = e.target && e.target.closest && e.target.closest("code.def-term");
            if (el) showDefPop(el);
        });

        /* ── citations click handling ─────────────────────────────────────── */
        /* Resolve a spec id to its official nvmexpress.org PDF URL (delivered by
           /api/specs into window._specData). We deep-link to that URL rather
           than re-hosting the PDF: the user's browser fetches the file straight
           from NVM Express, so we only ever "reference/cite" the spec. Returns
           "" when the spec or its url is unknown. */
        function specPdfUrl(specId) {
            var specs = window._specData || [];
            for (var i = 0; i < specs.length; i++) {
                if (specs[i].id === specId && specs[i].url) return specs[i].url;
            }
            return "";
        }

        /* PDF "#page=N" deep links are honoured by Chrome/Firefox/Edge but
           silently ignored by Safari and every iOS browser (all WebKit), which
           would open the spec at page 1 - a confusing jump-to-nowhere. On those
           we make the citation/figure click a no-op instead of mis-opening. */
        function pdfPageJumpSupported() {
            var ua = navigator.userAgent || "";
            var iOS = /iPad|iPhone|iPod/.test(ua) ||
                (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
            var safari = /safari/i.test(ua) &&
                !/chrome|chromium|crios|fxios|edg|opr|android/i.test(ua);
            return !(iOS || safari);
        }

        /* Open a URL in a new tab via a synthesized anchor click. Chrome's PDF
           viewer honours the "#page=N" fragment far more reliably from a real
           link navigation than from window.open(...,"noopener"), whose two-step
           popup navigation routinely drops the fragment and lands on page 1. */
        function openInNewTab(url) {
            var a = document.createElement("a");
            a.href = url;
            a.target = "_blank";
            a.rel = "noopener noreferrer";
            document.body.appendChild(a);
            a.click();
            a.remove();
        }

        /* Open the official spec PDF at the cited page in a new tab. pdf_pages
           are stored 0-indexed, so +1 for the 1-indexed "#page=N" fragment that
           browser PDF viewers honour. No page → opens at the top of the doc.
           No-op on Safari/iOS (see pdfPageJumpSupported). */
        function openCitationPdf(c) {
            if (!c || c.hallucinated) return;
            if (!pdfPageJumpSupported()) return;
            var specId = c.spec
                || (window._specData && window._specData[0] && window._specData[0].id)
                || "base";
            var url = specPdfUrl(specId);
            if (!url) return;
            if (c.pdf_pages && c.pdf_pages.length > 0) {
                var p0 = parseInt(c.pdf_pages[0], 10);
                if (!isNaN(p0)) url += "#page=" + (p0 + 1);
            }
            openInNewTab(url);
        }

        async function openFigureRenderPopup(f) {
            if (!f) return;
            var specId = f.spec
                || (window._specData && window._specData[0] && window._specData[0].id)
                || "base";
            var num = encodeURIComponent(String(f.figure_number || "").trim());
            var s = encodeURIComponent(specId);
            
            try {
                var res = await fetch("/api/figure/" + s + "/" + num);
                if (!res.ok) {
                    var msg = "HTTP " + res.status;
                    try { msg = (await res.json()).detail || msg; } catch (e) {}
                    alert("Could not load figure: " + msg);
                    return;
                }
                var table = await res.json();
                
                const figNum = String(f.figure_number || "").trim();
                const key = "fig-" + specId + "-" + figNum;
                const existing = _stagePopups.find(function(p) { return p.key === key; });
                if (existing) { _raisePopup(existing); return; }

                const el = document.createElement("div");
                el.className = "stage-popup";
                el.setAttribute("role", "dialog");
                el.style.width = "auto";
                el.style.maxWidth = "90vw";
                el.style.maxHeight = "90vh";

                const header = document.createElement("div");
                header.className = "stage-popup-header";

                const titleEl = document.createElement("span");
                titleEl.className = "stage-popup-title";
                titleEl.textContent = "Figure " + figNum + (table.caption ? " — " + table.caption : "");

                const actions = document.createElement("div");
                actions.className = "stage-popup-actions";

                const closeAllBtn = document.createElement("button");
                closeAllBtn.type = "button";
                closeAllBtn.className = "stage-popup-close-all";
                closeAllBtn.textContent = "Close all";

                const closeBtn = document.createElement("button");
                closeBtn.type = "button";
                closeBtn.className = "stage-popup-close";
                closeBtn.setAttribute("data-tip", "Close this");
                closeBtn.innerHTML = "&#x2715;";

                actions.appendChild(closeAllBtn);
                actions.appendChild(closeBtn);
                header.appendChild(titleEl);
                header.appendChild(actions);

                const body = document.createElement("div");
                body.className = "stage-popup-body";
                body.style.padding = "0";
                
                let hhtml = "";
                if (table.headers && table.rows) {
                    let hasAcronyms = false;
                    const newRows = table.rows.map(function(r) {
                        let newR = r.slice();
                        let extracted = "";
                        for (let i = 0; i < newR.length; i++) {
                            let cell = String(newR[i]);
                            // Look for "Name (ACRONYM): Description" in any cell
                            let m = cell.match(/^([^\\n:(]{1,100}?)\\(([^)\\n]{1,30})\\)\\s*:(.*)$/s);
                            if (m) {
                                extracted = m[2].trim();
                                newR[i] = m[1].trim() + ":" + m[3];
                                hasAcronyms = true;
                                break;
                            }
                        }
                        return [extracted].concat(newR);
                    });
                    
                    if (hasAcronyms) {
                        table.headers = ["Symbol"].concat(table.headers);
                        table.rows = newRows;
                    }
                }

                if (table.headers && table.headers.length) {
                    hhtml = "<thead><tr>" + table.headers.map(function(h) { return "<th>" + escapeHtml(h) + "</th>"; }).join("") + "</tr></thead>";
                }
                let bhtml = "<tbody>";
                
                function renderCellContent(c) {
                    c = String(c);
                    if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
                        try { return DOMPurify.sanitize(marked.parse(c)); } catch (e) {}
                    }
                    return escapeHtml(c);
                }

                if (table.rows && table.rows.length) {
                    table.rows.forEach(function(r) {
                        bhtml += "<tr>" + r.map(function(c) { return "<td>" + renderCellContent(c) + "</td>"; }).join("") + "</tr>";
                    });
                }
                bhtml += "</tbody>";
                
                body.innerHTML = `
                <div style="padding: 16px;">
                    <style>
                        .fig-rendered-table { border-collapse: collapse; font-size: 13px; width: 100%; font-family: var(--sans); }
                        .fig-rendered-table th, .fig-rendered-table td { border: 1px solid var(--border); padding: 8px 10px; text-align: left; vertical-align: top; }
                        .fig-rendered-table th { background: var(--surface-2); font-weight: 600; color: var(--t-muted); }
                        .fig-rendered-table p { margin: 0 0 8px 0; }
                        .fig-rendered-table p:last-child { margin: 0; }
                        .fig-rendered-table table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 12px; }
                        .fig-rendered-table table th, .fig-rendered-table table td { border: 1px solid var(--border); padding: 4px 6px; }
                    </style>
                    <table class="fig-rendered-table">${hhtml}${bhtml}</table>
                </div>
                `;

                el.appendChild(header);
                el.appendChild(body);
                document.body.appendChild(el);

                const w = el.offsetWidth || 480;
                const h = el.offsetHeight || 300;
                const off = _stagePopups.length * 14;
                const baseX = window.innerWidth / 2 - w / 2 + off;
                const baseY = 90 + off;
                el.style.left = Math.max(8, Math.min(baseX, window.innerWidth  - w - 8)) + "px";
                el.style.top  = Math.max(8, Math.min(baseY, window.innerHeight - h - 8)) + "px";

                const rec = {el, key};
                _stagePopups.push(rec);

                closeBtn.addEventListener("click", function(ev) { ev.stopPropagation(); closeStagePopup(rec); });
                closeAllBtn.addEventListener("click", function(ev) { ev.stopPropagation(); closeAllStagePopups(); });
                closeBtn.addEventListener("mouseenter", function() { _showPopupTip(closeBtn); });
                closeBtn.addEventListener("mouseleave", _hidePopupTip);
                closeBtn.addEventListener("focus", function() { _showPopupTip(closeBtn); });
                closeBtn.addEventListener("blur", _hidePopupTip);
                el.addEventListener("mousedown", function() { _raisePopup(rec); });
                _makePopupDraggable(el, header);
                _raisePopup(rec);

                while (_stagePopups.length > STAGE_POPUP_CAP) {
                    const oldest = _stagePopups.shift();
                    if (oldest.el && oldest.el.parentNode) oldest.el.parentNode.removeChild(oldest.el);
                }
                _updatePopupMultiState();

            } catch (e) {
                alert("Error rendering figure: " + e.message);
            }
        }

        /* Open the official spec PDF at a figure's page (same 0-indexed → +1
           page convention as citations). No-op on Safari/iOS. */
        function openFigurePdf(f) {
            if (!f) return;
            if (!pdfPageJumpSupported()) return;
            var specId = f.spec
                || (window._specData && window._specData[0] && window._specData[0].id)
                || "base";
            var url = specPdfUrl(specId);
            if (!url) return;
            if (f.pdf_pages && f.pdf_pages.length > 0) {
                var p0 = parseInt(f.pdf_pages[0], 10);
                if (!isNaN(p0)) url += "#page=" + (p0 + 1);
            }
            openInNewTab(url);
        }

        /* Turn inline "Figure N" mentions in the answer into clickable chips.
           Only figures that were retrieved (and so have a known page) are
           linked. Regex-free text walk, mirroring linkifyCitations, to avoid
           backslash-escaping inside this Python-embedded template. */
        function linkifyFigures(root, figures) {
            if (!root || !figures || !figures.length) return;
            var figMap = {};
            figures.forEach(function (f) {
                var k = String(f.figure_number || "").trim();
                if (k) figMap[k] = f;
            });

            var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
            var nodes = [], n;
            while ((n = walker.nextNode())) {
                if (n.parentNode && n.parentNode.closest &&
                    n.parentNode.closest(".cite-chip, .fig-chip, code, pre, a")) continue;
                nodes.push(n);
            }

            function isDigit(ch) { return ch >= "0" && ch <= "9"; }

            nodes.forEach(function (node) {
                var text = node.nodeValue;
                if (text.indexOf("Figure ") === -1) return;
                var frag = document.createDocumentFragment();
                var pos = 0, i = 0, changed = false;
                while (i < text.length) {
                    var at = text.indexOf("Figure ", i);
                    if (at === -1) break;
                    var j = at + 7;                 // just past "Figure "
                    var k = j;
                    while (k < text.length && isDigit(text.charAt(k))) k++;
                    var num = text.slice(j, k);
                    if (num && figMap[num]) {
                        if (at > pos) frag.appendChild(document.createTextNode(text.slice(pos, at)));
                        var span = document.createElement("span");
                        span.className = "cite-chip fig-chip mono";
                        span.setAttribute("data-fig", num);
                        span.setAttribute("tabindex", "0");
                        span.textContent = "Figure " + num;
                        frag.appendChild(span);
                        pos = k;
                        changed = true;
                        i = k;
                    } else {
                        i = at + 7;
                    }
                }
                if (!changed) return;
                if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
                node.parentNode.replaceChild(frag, node);
            });

            root.querySelectorAll(".fig-chip").forEach(function (c) {
                c.addEventListener("mouseenter", function () { setActiveFig(c.getAttribute("data-fig")); });
                c.addEventListener("mouseleave", function () { setActiveFig(null); });
                c.addEventListener("click", function () {
                    showFigPop(c, figMap[c.getAttribute("data-fig")]);
                });
            });
        }

        /* ── sources sidebar ─────────────────────────────────────────────── */
        function renderSourcesSidebar(citations, figures) {
            var list = document.getElementById("citations-list");
            var box = document.getElementById("citations-box");
            var count = document.getElementById("sources-count");
            if (!list || !box) return;
            // Combined source list: section citations + cited figures (figures
            // are real cited sources too, just delivered in a separate payload).
            var sources = [];
            (citations || []).forEach(function (c) { sources.push({ kind: "sec", c: c }); });
            (figures || []).forEach(function (f) { sources.push({ kind: "fig", f: f }); });
            if (!sources.length) { box.classList.add("hidden"); return; }
            list.innerHTML = sources.map(function (s) {
                if (s.kind === "fig") {
                    var fnum = escapeHtml(String(s.f.figure_number || ""));
                    var cap = escapeHtml(String(s.f.caption || ""));
                    return '<button class="src" type="button" data-fig="' + fnum + '">'
                         + '<div class="src-top"><span class="src-sec">Figure ' + fnum + "</span>"
                         + '<span class="src-type">figure</span>'
                         + '<span class="src-dot ok" role="img" title="Verified in retrieved context" aria-label="Verified in retrieved context"></span></div>'
                         + '<div class="src-title">' + cap + "</div></button>";
                }
                var c = s.c;
                var sid = escapeHtml(String(c.section_id || ""));
                var title = escapeHtml(String(c.section_title || ""));
                var type = c.content_type || c.type || "";
                var typeHtml = type ? '<span class="src-type">' + escapeHtml(String(type)) + "</span>" : "";
                var grounded = !c.hallucinated;
                var dotTitle = grounded ? "Verified in retrieved context" : "Referenced but not in retrieved context";
                var dot = '<span class="src-dot ' + (grounded ? "ok" : "warn") + '" role="img" title="' + dotTitle + '" aria-label="' + dotTitle + '"></span>';
                return '<button class="src" type="button" data-sec="' + sid + '">'
                     + '<div class="src-top"><span class="src-sec">&#167;' + sid + "</span>" + typeHtml + dot + "</div>"
                     + '<div class="src-title">' + title + "</div></button>";
            }).join("");
            if (count) count.textContent = String(sources.length);
            box.classList.remove("hidden");
            list.querySelectorAll(".src").forEach(function (el, i) {
                var s = sources[i];
                el.addEventListener("mouseenter", function () {
                    if (s.kind === "fig") setActiveFig(el.getAttribute("data-fig"));
                    else setActiveSec(el.getAttribute("data-sec"));
                });
                el.addEventListener("mouseleave", function () {
                    if (s.kind === "fig") setActiveFig(null);
                    else setActiveSec(null);
                });
                el.addEventListener("click", function () {
                    if (s.kind === "fig") openFigurePdf(s.f);
                    else openCitationPdf(s.c);
                });
            });
        }

        /* ── answer meta row + pipeline summary ──────────────────────────── */
        function _tokensTotal(t) {
            if (!t) return 0;
            if (Array.isArray(t.calls) && t.calls.length) {
                return t.calls.reduce(function (s, c) { return s + (c.prompt || 0) + (c.completion || 0); }, 0);
            }
            return (t.prompt || 0) + (t.completion || 0);
        }
        function _totalCostFromTokens(t) {
            if (!t || !Array.isArray(t.calls) || !t.calls.length) return null;
            return t.calls.reduce(function (s, c) {
                var p = (typeof MODEL_PRICING !== "undefined" && MODEL_PRICING[c.model]) || { in: 0, out: 0 };
                return s + (c.prompt || 0) / 1e6 * p.in + (c.completion || 0) / 1e6 * p.out;
            }, 0);
        }
        function _isGrounded(data) {
            var anyHall = (data.citations || []).some(function (c) { return c.hallucinated; });
            var gapOpen = data.gap_hint && data.gap_hint.needs_followup;
            return !anyHall && !gapOpen;
        }
        function renderAnswerMeta(data) {
            var el = document.getElementById("answer-meta");
            if (!el) return;
            var tok = _tokensTotal(data.tokens_used);
            var html = '<span class="meta-q" title="' + escapeHtml(String(data.query || "")) + '">' + escapeHtml(String(data.query || "")) + "</span>";
            html += _isGrounded(data)
                ? '<span class="badge ok">' + I_check + "grounded</span>"
                : '<span class="badge warn">' + I_warn + "partial</span>";
            if (data.agentic) html += '<span class="badge accent">' + I_robot + "agentic</span>";
            html += '<span class="badge">' + (data.latency_ms / 1000).toFixed(2) + "s</span>";
            if (tok) html += '<span class="badge">' + tok.toLocaleString() + " tok</span>";
            el.innerHTML = html;
        }
        function renderPipeSummary(data) {
            var el = document.getElementById("pipe-summary");
            if (!el) return;
            var trace = data.pipeline_trace || [];
            var parts = [];
            if (trace.length) parts.push("<span><b>" + trace.length + "</b> stages</span>");
            parts.push("<span><b>" + (data.latency_ms / 1000).toFixed(2) + "</b>s</span>");
            var cost = _totalCostFromTokens(data.tokens_used);
            if (cost !== null) parts.push("<span><b>$" + cost.toFixed(4) + "</b></span>");
            el.innerHTML = parts.join("");
        }

        /* ── displayResults override (two-column layout) ─────────────────── */
        window.displayResults = function (data) {
            var empty = document.getElementById("empty-state");
            if (empty) empty.classList.add("hidden");

            var answerEl = document.getElementById("answer-text");
            if (_cancelAnswerStream) { _cancelAnswerStream(); _cancelAnswerStream = null; }
            var answerText = data.answer || "";

            renderAnswerMeta(data);
            renderSourcesSidebar(data.citations, data.figures);
            var lat = document.getElementById("latency"); if (lat) lat.textContent = "";

            window._figures = data.figures || [];
            if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
                _cancelAnswerStream = streamAnswerInto(answerEl, answerText, {
                    onDone: function () {
                        linkifyCitations(answerEl, data.citations, data.figures);
                        linkifyFigures(answerEl, data.figures);
                        styleBlockAttributions(answerEl);
                        markDefinableTerms(answerEl, (data.config && data.config.spec) || window.getSelectedSpec());
                    }
                });
            } else {
                answerEl.textContent = answerText;
            }

            // Stash for a re-render when the (collapsed) disclosure opens, so
            // Mermaid measures against a visible container.
            window._pipeTrace = data.pipeline_trace;
            window._pipeQuery = data.query;
            renderPipelineViz(data.pipeline_trace, data.query);

            var stagesDiv = document.getElementById("pipeline-stages");
            var stageCount = document.getElementById("trace-stage-count");
            if (data.pipeline_trace) {
                stagesDiv.innerHTML = data.pipeline_trace.map(function (stage, idx) { return renderStageCard(stage, idx); }).join("");
                if (stageCount) stageCount.textContent = "(" + data.pipeline_trace.length + " stage" + (data.pipeline_trace.length === 1 ? "" : "s") + ")";
            } else {
                stagesDiv.innerHTML = "";
                if (stageCount) stageCount.textContent = "";
            }

            var isAgentic = !!data.agentic;
            renderModelTable(isAgentic);
            renderModelCost(data.tokens_used, isAgentic);
            renderPipeSummary(data);
            renderSidebar(data);

            // Snapshot for the flag button (POSTed verbatim on flag, no re-run).
            window._lastResponse = data;
            var flagFab = document.getElementById("flag-fab");
            if (flagFab) { flagFab.classList.remove("flagged"); flagFab.hidden = false; }

            document.getElementById("answer-section").classList.remove("hidden");
            document.getElementById("answer-section").classList.remove("answer-stale");
            _notifyDone("specGPT", "Answer ready");
        };

        /* ── Flag answer: FAB + modal wiring ─────────────────────────────── */
        function resetFlagFab() {
            var fab = document.getElementById("flag-fab");
            if (fab) { fab.hidden = true; fab.classList.remove("flagged"); }
            closeFlagModal();
        }
        function openFlagModal() {
            var ov = document.getElementById("flag-modal-overlay");
            if (!ov || !window._lastResponse) return;
            document.getElementById("flag-reason").value = "";
            document.getElementById("flag-modal-err").textContent = "";
            var submit = document.getElementById("flag-submit");
            submit.disabled = false; submit.textContent = "Submit flag";
            ov.hidden = false;
            document.getElementById("flag-reason").focus();
        }
        function closeFlagModal() {
            var ov = document.getElementById("flag-modal-overlay");
            if (ov) ov.hidden = true;
        }
        function showFlagToast() {
            var t = document.getElementById("flag-toast");
            if (!t) return;
            t.hidden = false;
            clearTimeout(t._timer);
            t._timer = setTimeout(function () { t.hidden = true; }, 2400);
        }
        async function submitFlag(reason) {
            var d = window._lastResponse;
            if (!d) return;
            var errEl = document.getElementById("flag-modal-err");
            var submit = document.getElementById("flag-submit");
            errEl.textContent = "";
            submit.disabled = true; submit.textContent = "Submitting…";
            try {
                var res = await fetch("/api/flag-answer", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    credentials: "same-origin",
                    body: JSON.stringify({
                        query: d.query,
                        answer: d.answer,
                        config: d.config,
                        pipeline_trace: d.pipeline_trace || null,
                        citations: d.citations || null,
                        latency_ms: d.latency_ms != null ? d.latency_ms : null,
                        tokens_used: d.tokens_used || null,
                        agentic: !!d.agentic,
                        reason: reason || null,
                    }),
                });
                if (res.ok) {
                    var fab = document.getElementById("flag-fab");
                    if (fab) fab.classList.add("flagged");
                    closeFlagModal();
                    showFlagToast();
                } else {
                    errEl.textContent = "Couldn't submit the flag. Please try again.";
                    submit.disabled = false; submit.textContent = "Submit flag";
                }
            } catch (e) {
                errEl.textContent = "Network error. Couldn't submit the flag.";
                submit.disabled = false; submit.textContent = "Submit flag";
            }
        }
        (function () {
            var fab = document.getElementById("flag-fab");
            if (fab) fab.addEventListener("click", openFlagModal);
            var cancel = document.getElementById("flag-cancel");
            if (cancel) cancel.addEventListener("click", closeFlagModal);
            var submit = document.getElementById("flag-submit");
            if (submit) submit.addEventListener("click", function () {
                submitFlag(document.getElementById("flag-reason").value.trim());
            });
            var ov = document.getElementById("flag-modal-overlay");
            if (ov) ov.addEventListener("click", function (e) { if (e.target === ov) closeFlagModal(); });
            document.addEventListener("keydown", function (e) {
                if (e.key === "Escape" && ov && !ov.hidden) closeFlagModal();
                if (e.key === "Enter" && !e.shiftKey && ov && !ov.hidden) {
                    e.preventDefault();
                    submitFlag(document.getElementById("flag-reason").value.trim());
                }
            });
        })();

        /* ── Dev panel: flagged_answers browser + notes scratchpad ───────── */
        function _devFmtDate(iso) {
            if (!iso) return "";
            var d = new Date(iso);
            if (isNaN(d)) return String(iso);
            return d.toLocaleString();
        }
        function _devRenderMd(text) {
            if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
                try { return DOMPurify.sanitize(marked.parse(text || "")); } catch (e) {}
            }
            return escapeHtml(text || "").replace(/\\n/g, "<br>");
        }
        function _devCitationLine(c) {
            if (!c || typeof c !== "object") return escapeHtml(String(c));
            var bits = [];
            if (c.section_id) bits.push("&#167;" + escapeHtml(String(c.section_id)));
            if (c.section_title) bits.push(escapeHtml(String(c.section_title)));
            if (c.figure_number) bits.push("[" + escapeHtml(String(c.figure_number)) + "]");
            var pages = c.pdf_pages || c.pages;
            if (Array.isArray(pages) && pages.length) bits.push("p." + pages.join(","));
            return bits.length ? bits.join(" &middot; ") : escapeHtml(JSON.stringify(c));
        }
        function _devKv(obj) {
            if (!obj || typeof obj !== "object") return "";
            var keys = Object.keys(obj);
            if (!keys.length) return '<span class="dev-field-val">(none)</span>';
            return '<dl class="dev-kv">' + keys.map(function (k) {
                var v = obj[k];
                if (v && typeof v === "object") v = JSON.stringify(v);
                return "<dt>" + escapeHtml(k) + "</dt><dd>" + escapeHtml(String(v)) + "</dd>";
            }).join("") + "</dl>";
        }
        function _devTokensTotal(t) {
            if (!t) return null;
            if (Array.isArray(t.calls) && t.calls.length) {
                return t.calls.reduce(function (s, c) { return s + (c.prompt || 0) + (c.completion || 0); }, 0);
            }
            return (t.prompt || 0) + (t.completion || 0);
        }
        // Compact list row — the full question/answer/trace live on the detail
        // page so a long question can never overflow the collapsed row.
        function renderFlagRow(f, idx) {
            var sub = f.reason ? escapeHtml(String(f.reason)) : "(no reason given)";
            var meta = '<div class="dev-row-meta">'
                + (f.spec ? '<span class="dev-chip">' + escapeHtml(String(f.spec)) + "</span>" : "")
                + (f.agentic ? '<span class="dev-chip agentic">agentic</span>' : "")
                + '<span class="dev-row-date">' + escapeHtml(_devFmtDate(f.created_at)) + "</span>"
                + "</div>";
            return '<button class="dev-row" type="button" data-row="' + idx + '">'
                + '<div class="dev-row-main">'
                + '<div class="dev-row-q">' + escapeHtml(String(f.query || "(empty query)")) + "</div>"
                + '<div class="dev-row-sub">' + sub + "</div>"
                + "</div>" + meta
                + '<span class="dev-row-arrow"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg></span>'
                + "</button>";
        }
        // Full detail page for one flag — rendered into #dev-flag-detail when a
        // row is opened. Uses a stable flow-host id so the renderer can target it.
        function renderFlagDetail(f, idx) {
            var trace = Array.isArray(f.pipeline_trace) ? f.pipeline_trace : [];
            var traceHtml = trace.length
                ? '<ul class="dev-list">' + trace.map(function (s, i) {
                    var name = s.stage || s.name || ("stage " + (i + 1));
                    var ms = (s.took_ms != null) ? " (" + Math.round(s.took_ms) + "ms)" : "";
                    return '<li><span class="dev-stage-num">' + (i + 1) + ".</span>" + escapeHtml(String(name)) + ms + "</li>";
                }).join("") + "</ul>"
                : '<span class="dev-field-val">(no trace)</span>';

            var cites = Array.isArray(f.citations) ? f.citations : [];
            var citesHtml = cites.length
                ? '<ul class="dev-list">' + cites.map(function (c) { return "<li>" + _devCitationLine(c) + "</li>"; }).join("") + "</ul>"
                : '<span class="dev-field-val">(no citations)</span>';

            var tok = _devTokensTotal(f.tokens_used);
            var timing = '<dl class="dev-kv">'
                + "<dt>latency</dt><dd>" + (f.latency_ms != null ? (f.latency_ms / 1000).toFixed(2) + "s" : "-") + "</dd>"
                + "<dt>tokens</dt><dd>" + (tok != null ? tok.toLocaleString() : "-") + "</dd>"
                + "<dt>model</dt><dd>" + escapeHtml(String(f.llm_model || "-")) + "</dd>"
                + "<dt>flag id</dt><dd>" + escapeHtml(String(f.id)) + "</dd>"
                + "</dl>";

            var metaChips = (f.spec ? '<span class="dev-chip">' + escapeHtml(String(f.spec)) + "</span>" : "")
                + (f.agentic ? '<span class="dev-chip agentic">agentic</span>' : "")
                + '<span class="dev-row-date">' + escapeHtml(_devFmtDate(f.created_at)) + "</span>";

            return '<div class="dev-detail-q">' + escapeHtml(String(f.query || "(empty query)")) + "</div>"
                + '<div class="dev-detail-meta">' + metaChips + "</div>"
                + '<div class="dev-field"><div class="dev-field-label">Answer</div><div class="dev-answer">' + _devRenderMd(f.answer) + "</div></div>"
                + '<div class="dev-field"><div class="dev-field-label">User reason</div><div class="dev-field-val">' + (f.reason ? escapeHtml(String(f.reason)) : "(none)") + "</div></div>"
                + '<div class="dev-field"><div class="dev-field-label">Timing &amp; cost</div>' + timing + "</div>"
                + '<div class="dev-field"><div class="dev-field-label">Flow chart</div>'
                + (trace.length
                    ? '<button class="dev-btn" type="button" data-flow="' + idx + '">Render flow chart</button>'
                      + '<div class="dev-flow-host" id="dev-flow-' + idx + '"></div>'
                    : '<span class="dev-field-val">(no trace to render)</span>')
                + "</div>"
                + '<div class="dev-field"><div class="dev-field-label">Pipeline trace</div>' + traceHtml + "</div>"
                + '<div class="dev-field"><div class="dev-field-label">Citations</div>' + citesHtml + "</div>"
                + '<div class="dev-field"><div class="dev-field-label">Config</div>' + _devKv(f.config) + "</div>"
                + '<div class="dev-field"><button class="dev-btn dev-btn-danger" type="button" data-delflag="' + idx + '">Delete flag</button></div>';
        }
        // Master/detail navigation within the flags pane.
        function showFlagList() {
            var lv = document.getElementById("dev-flags-listview");
            var dv = document.getElementById("dev-flags-detailview");
            if (lv) lv.hidden = false;
            if (dv) dv.hidden = true;
            var body = document.querySelector(".dev-body");
            if (body) body.scrollTop = 0;
        }
        function openFlagDetail(idx) {
            var f = (window._devFlags || [])[idx];
            var host = document.getElementById("dev-flag-detail");
            if (!f || !host) return;
            host.innerHTML = renderFlagDetail(f, idx);
            var flowBtn = host.querySelector("[data-flow]");
            if (flowBtn) flowBtn.addEventListener("click", function () {
                var fh = document.getElementById("dev-flow-" + idx);
                if (!fh) return;
                flowBtn.disabled = true; flowBtn.textContent = "Rendering…";
                renderDevFlow(f.pipeline_trace, f.query, fh).then(function () {
                    flowBtn.style.display = "none";
                }).catch(function () {
                    flowBtn.disabled = false; flowBtn.textContent = "Render flow chart";
                });
            });
            var delBtn = host.querySelector("[data-delflag]");
            if (delBtn) delBtn.addEventListener("click", function () { deleteFlag(f.id, delBtn); });
            var lv = document.getElementById("dev-flags-listview");
            var dv = document.getElementById("dev-flags-detailview");
            if (lv) lv.hidden = true;
            if (dv) dv.hidden = false;
            var body = document.querySelector(".dev-body");
            if (body) body.scrollTop = 0;
        }
        async function loadFlags() {
            var list = document.getElementById("dev-flags-list");
            var count = document.getElementById("dev-flags-count");
            showFlagList();
            list.innerHTML = '<div class="dev-empty">Loading…</div>';
            try {
                var res = await fetch("/api/flags?limit=200", { credentials: "same-origin" });
                if (!res.ok) throw new Error("http " + res.status);
                var data = await res.json();
                var flags = data.flags || [];
                window._devFlags = flags;
                if (count) count.textContent = flags.length + (flags.length === 1 ? " flag" : " flags");
                if (!flags.length) { list.innerHTML = '<div class="dev-empty">No flagged answers yet.</div>'; return; }
                list.innerHTML = flags.map(function (f, i) { return renderFlagRow(f, i); }).join("");
                list.querySelectorAll("[data-row]").forEach(function (btn) {
                    btn.addEventListener("click", function () {
                        openFlagDetail(parseInt(btn.getAttribute("data-row"), 10));
                    });
                });
            } catch (e) {
                list.innerHTML = '<div class="dev-empty">Could not load flags. Is the table created and Supabase reachable?</div>';
                if (count) count.textContent = "";
            }
        }
        async function deleteFlag(id, btn) {
            if (!window.confirm("Delete this flag? This cannot be undone.")) return;
            if (btn) { btn.disabled = true; btn.textContent = "Deleting…"; }
            try {
                var res = await fetch("/api/flags/" + encodeURIComponent(id), {
                    method: "DELETE",
                    credentials: "same-origin",
                });
                if (!res.ok) throw new Error("http " + res.status);
                await loadFlags();
            } catch (e) {
                if (btn) { btn.disabled = false; btn.textContent = "Delete failed, retry"; }
            }
        }
        async function loadDevNotes({ silent = false } = {}) {
            var list = document.getElementById("dev-notes-list");
            if (!silent) list.innerHTML = '<div class="dev-empty">Loading…</div>';
            try {
                var res = await fetch("/api/dev-notes", { credentials: "same-origin" });
                if (!res.ok) throw new Error("http " + res.status);
                var data = await res.json();
                var notes = data.notes || [];
                if (!notes.length) { list.innerHTML = '<div class="dev-empty">No notes yet.</div>'; return; }
                list.innerHTML = notes.map(function (n) {
                    return '<div class="dev-note"><div class="dev-note-head">'
                        + '<span class="dev-note-date">' + escapeHtml(_devFmtDate(n.created_at)) + "</span>"
                        + '<button class="dev-note-del" type="button" data-del="' + n.id + '" aria-label="Delete note" title="Delete note">&#x2715;</button>'
                        + '</div><div class="dev-note-body">' + escapeHtml(String(n.body || "")) + "</div></div>";
                }).join("");
                list.querySelectorAll("[data-del]").forEach(function (btn) {
                    btn.addEventListener("click", function () { deleteDevNote(btn.getAttribute("data-del")); });
                });
            } catch (e) {
                list.innerHTML = '<div class="dev-empty">Could not load notes.</div>';
            }
        }
        async function deleteDevNote(id) {
            if (!window.confirm("Delete this note?")) return;
            try {
                var res = await fetch("/api/dev-notes/" + encodeURIComponent(id), {
                    method: "DELETE",
                    credentials: "same-origin",
                });
                if (!res.ok) throw new Error("http " + res.status);
                await loadDevNotes({ silent: true });
            } catch (e) {
                var err = document.getElementById("dev-note-err");
                if (err) err.textContent = "Could not delete the note.";
            }
        }

        // Self-contained flow-chart renderer for the dev panel. Reuses the main
        // engine's pure builders (splitTraceByIteration, buildMermaidFromTrace)
        // and the global stage popup, but keeps its OWN page state so it can't
        // corrupt the main answer-section viz (_vizPages/_vizPageIdx).
        var _devFlow = { pages: [], idx: 0, host: null };
        async function renderDevFlow(trace, query, host) {
            _devFlow = { pages: [], idx: 0, host: host };
            if (!trace || !trace.length) {
                host.innerHTML = '<div class="viz-empty">No pipeline trace stored for this flag.</div>';
                return;
            }
            if (typeof mermaid === "undefined") {
                host.innerHTML = '<div class="viz-empty">Mermaid failed to load (CDN blocked?). The stage list above still has every step.</div>';
                return;
            }
            // Single split page still needs its `.iterN` suffixes stripped
            // (see renderPipelineViz) - only fall back to the raw trace when
            // there are no iteration suffixes at all.
            var iters = splitTraceByIteration(trace);
            if (iters) {
                iters.forEach(function (sub) { _devFlow.pages.push(buildMermaidFromTrace(sub, query || "")); });
            } else {
                _devFlow.pages.push(buildMermaidFromTrace(trace, query || ""));
            }
            await devFlowGoTo(0);
        }
        async function devFlowGoTo(idx) {
            var f = _devFlow;
            if (!f.pages.length || !f.host) return;
            f.idx = Math.max(0, Math.min(idx, f.pages.length - 1));
            var page = f.pages[f.idx];
            var id = "devviz-" + (++_vizCounter);
            try {
                if (typeof applyMermaidTheme === "function") applyMermaidTheme();
                var out = await mermaid.render(id, page.def);
                f.host.innerHTML = out.svg;
                attachDevNodeClicks(page.nodeMap, f.host);
                attachDevPassNav(f.host);
            } catch (err) {
                f.host.innerHTML = '<div class="viz-empty">Could not render flow: ' + escapeHtml(err.message || String(err)) + "</div>";
            }
        }
        function attachDevNodeClicks(nodeMap, host) {
            var svg = host.querySelector("svg");
            if (!svg) return;
            svg.querySelectorAll("g.node").forEach(function (g) {
                var nodeId = g.dataset.id || "";
                if (!nodeId && g.id) nodeId = g.id.replace(/^flowchart-/, "").replace(/-\\d+$/, "");
                if (!nodeId || !nodeMap[nodeId]) return;
                var stage = nodeMap[nodeId];
                g.style.cursor = "pointer";
                g.addEventListener("click", function (e) {
                    e.stopPropagation();
                    showStagePopup(stage, e.clientX, e.clientY);
                });
            });
        }
        function attachDevPassNav(host) {
            var f = _devFlow;
            if (f.pages.length <= 1) return;
            var nav = document.createElement("div");
            nav.className = "viz-pass-nav";
            nav.innerHTML =
                '<button type="button" data-p="prev" title="Previous pass">\\u2190</button>'
                + '<span class="viz-pass-label">Pass ' + (f.idx + 1) + ' / ' + f.pages.length + "</span>"
                + '<button type="button" data-p="next" title="Next pass">\\u2192</button>';
            var prevB = nav.querySelector('[data-p="prev"]');
            var nextB = nav.querySelector('[data-p="next"]');
            if (prevB) prevB.disabled = f.idx === 0;
            if (nextB) nextB.disabled = f.idx === f.pages.length - 1;
            nav.addEventListener("click", function (e) {
                var b = e.target.closest("button");
                if (!b) return;
                devFlowGoTo(b.dataset.p === "prev" ? f.idx - 1 : f.idx + 1);
            });
            host.appendChild(nav);
        }
        async function addDevNote() {
            var input = document.getElementById("dev-note-input");
            var err = document.getElementById("dev-note-err");
            var btn = document.getElementById("dev-note-add");
            var body = (input.value || "").trim();
            err.textContent = "";
            if (!body) { err.textContent = "Note is empty."; return; }
            btn.disabled = true; btn.textContent = "Saving…";
            try {
                var res = await fetch("/api/dev-notes", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    credentials: "same-origin",
                    body: JSON.stringify({ body: body }),
                });
                if (!res.ok) throw new Error("http " + res.status);
                input.value = "";
                await loadDevNotes({ silent: true });
            } catch (e) {
                err.textContent = "Could not save the note.";
            } finally {
                btn.disabled = false; btn.textContent = "Add note";
            }
        }
        function openDevPanel() {
            var ov = document.getElementById("dev-overlay");
            if (!ov) return;
            ov.hidden = false;
            loadFlags();
            loadDevNotes();
        }
        function closeDevPanel() {
            var ov = document.getElementById("dev-overlay");
            if (ov) ov.hidden = true;
        }
        function setDevTab(tab) {
            document.querySelectorAll(".dev-tab").forEach(function (b) {
                b.classList.toggle("active", b.getAttribute("data-devtab") === tab);
            });
            document.getElementById("dev-pane-flags").hidden = (tab !== "flags");
            document.getElementById("dev-pane-notes").hidden = (tab !== "notes");
            if (tab === "flags") showFlagList();
        }
        (function () {
            var fab = document.getElementById("dev-fab");
            if (fab) fab.addEventListener("click", openDevPanel);
            var close = document.getElementById("dev-close");
            if (close) close.addEventListener("click", closeDevPanel);
            var ov = document.getElementById("dev-overlay");
            if (ov) ov.addEventListener("click", function (e) { if (e.target === ov) closeDevPanel(); });
            document.addEventListener("keydown", function (e) {
                if (e.key === "Escape" && ov && !ov.hidden) closeDevPanel();
            });
            document.querySelectorAll(".dev-tab").forEach(function (b) {
                b.addEventListener("click", function () { setDevTab(b.getAttribute("data-devtab")); });
            });
            var refresh = document.getElementById("dev-refresh");
            if (refresh) refresh.addEventListener("click", loadFlags);
            var flagBack = document.getElementById("dev-flag-back");
            if (flagBack) flagBack.addEventListener("click", showFlagList);
            var addBtn = document.getElementById("dev-note-add");
            if (addBtn) addBtn.addEventListener("click", addDevNote);
            var noteInput = document.getElementById("dev-note-input");
            if (noteInput) noteInput.addEventListener("keydown", function (e) {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); addDevNote(); }
            });
        })();

        /* ── renderSidebar override (gap-hint card) ──────────────────────── */
        window.renderSidebar = function (data) {
            var strip = document.getElementById("agent-strip");
            if (!strip) return;
            if (!data) { strip.className = "agent-strip"; strip.innerHTML = ""; return; }
            strip.classList.add("has-content");
            var latency = '<span class="gap-latency">' + data.latency_ms.toFixed(0) + "ms</span>";

            if (data.agentic) {
                strip.innerHTML = '<div class="gap-card accent"><div class="gap-ico">' + I_robot + "</div>"
                    + '<div class="gap-body"><div class="gap-title">Agentic refinement ran ' + latency + "</div>"
                    + '<div class="gap-note">Gap filling, follow-up retrieval, and regeneration ran automatically.</div></div></div>';
                return;
            }
            var gh = data.gap_hint;
            if (!gh) {
                strip.innerHTML = '<div class="gap-card muted"><div class="gap-ico">' + I_info + "</div>"
                    + '<div class="gap-body"><div class="gap-title">Gap check disabled ' + latency + "</div>"
                    + '<div class="gap-note">Enable Auto Gap Check in config to get refinement suggestions.</div></div></div>';
                return;
            }
            if (!gh.needs_followup) {
                strip.innerHTML = '<div class="gap-card ok"><div class="gap-ico">' + I_check + "</div>"
                    + '<div class="gap-body"><div class="gap-title">Answer looks complete ' + latency + "</div>"
                    + '<div class="gap-note">' + escapeHtml(gh.reason || "The model did not request any additional context.") + "</div></div></div>";
                return;
            }
            var req = gh.requested_resources || {};
            var figs = (req.figures || []).slice(0, 6);
            var flds = (req.fields || []).slice(0, 6);
            var secs = (req.sections || []).slice(0, 4);
            var qs = (gh.queries || []).slice(0, 3);
            function grp(label, items, isSec) {
                if (!items.length) return "";
                return '<div class="detail-group"><span class="detail-label">' + label + '</span><span class="gap-chips">'
                    + items.map(function (x) {
                        return '<span class="gap-chip' + (isSec ? " chip-section" : "") + '">' + (isSec ? "&#167;" : "") + escapeHtml(String(x)) + "</span>";
                    }).join("") + "</span></div>";
            }
            var details = grp("Figures", figs, false) + grp("Fields", flds, false) + grp("Sections", secs, true) + grp("Queries", qs, false);
            strip.innerHTML = '<div class="gap-card warn"><div class="gap-ico">' + I_warn + "</div>"
                + '<div class="gap-body"><div class="gap-title">Model wants more context ' + latency + "</div>"
                + '<div class="gap-note">' + escapeHtml(gh.reason || "The model identified gaps in the retrieved context.") + "</div>"
                + (details ? '<div class="gap-details">' + details + "</div>" : "")
                + '</div><button class="gap-act" id="run-agentic-btn" type="button">Run agentic refinement ' + I_arrow + "</button></div>";

            var btn = document.getElementById("run-agentic-btn");
            if (btn) {
                btn.addEventListener("click", function () {
                    // Enable agentic first so the overlay opens already-enabled,
                    // then open the editable agentic-config overlay BEFORE refining
                    // so the user can dial settings in. "Run" fires this callback.
                    if (!agenticToggle.checked) {
                        agenticToggle.checked = true;
                        agenticToggle.dispatchEvent(new Event("change"));
                    }
                    _agConfirmCallback = function () {
                        btn.disabled = true;
                        btn.textContent = "Running...";
                        runRefine(data.request_id);
                    };
                    openAgenticConfigPopup();
                });
            }
        };

        /* ── empty-state examples ────────────────────────────────────────── */
        (function () {
            var EXAMPLES = [];
            var box = document.getElementById("examples");
            if (!box) return;
            box.innerHTML = EXAMPLES.map(function (ex) {
                return '<button class="ex-chip" type="button">' + escapeHtml(ex) + "</button>";
            }).join("");
            box.querySelectorAll(".ex-chip").forEach(function (b) {
                b.addEventListener("click", function () {
                    var qi = document.getElementById("query-input");
                    qi.value = b.textContent;
                    qi.style.height = "auto";
                    qi.style.height = qi.scrollHeight + "px";
                    hideEmpty();
                    runQuery();
                });
            });
        })();

        function hideEmpty() {
            var e = document.getElementById("empty-state");
            if (e) e.classList.add("hidden");
        }

        /* ── composer focus ring + start-of-query empty hide ─────────────── */
        (function () {
            var c = document.getElementById("composer");
            var i = document.getElementById("query-input");
            if (c && i) {
                i.addEventListener("focus", function () { c.classList.add("focus"); });
                i.addEventListener("blur", function () { c.classList.remove("focus"); });
            }
            var sb = document.getElementById("search-btn");
            if (sb) sb.addEventListener("click", hideEmpty);
            if (i) i.addEventListener("keydown", function (e) { if (e.key === "Enter" && !e.shiftKey) hideEmpty(); });
        })();

        /* ── pipeline disclosure toggle ──────────────────────────────────── */
        (function () {
            var head = document.getElementById("pipe-head");
            var disc = document.getElementById("pipeline-disclosure");
            if (head && disc) head.addEventListener("click", function () {
                var opened = disc.classList.toggle("open");
                if (opened && window._pipeTrace && typeof renderPipelineViz === "function") {
                    // Re-render now that the container is visible and measurable.
                    try { renderPipelineViz(window._pipeTrace, window._pipeQuery); } catch (e) {}
                }
            });
        })();

        /* ── agentic pill toggle ─────────────────────────────────────────── */
        (function () {
            var pill = document.getElementById("agentic-pill");
            var lbl = document.getElementById("agentic-pill-label");
            function sync() {
                var on = agenticToggle.checked;
                if (pill) pill.classList.toggle("on", on);
                if (lbl) lbl.textContent = on ? "Agentic on" : "Agentic";
            }
            if (pill) pill.addEventListener("click", function () {
                agenticToggle.checked = !agenticToggle.checked;
                agenticToggle.dispatchEvent(new Event("change"));
            });
            agenticToggle.addEventListener("change", sync);
            sync();
        })();

        /* ── agentic locks the model picker to the strongest model ─────────────
           Agentic mode always runs on the strong (agentic) model — the backend
           forces it server-side — so the regular model picker is disabled while
           agentic is on and shows the strong model. The previous pick is stashed
           and restored when agentic is turned back off. No change events are
           dispatched, so the advanced agentic-model select is never clobbered. */
        (function _wireAgenticModelLock() {
            function strongModelId() {
                var ag = document.getElementById("config-agentic_model");
                return ag && ag.value ? ag.value : null;
            }
            function apply() {
                var on = agenticToggle.checked;
                var strong = strongModelId();
                ["config-llm_model"].forEach(function (id) {
                    var el = document.getElementById(id);
                    if (!el) return;
                    if (on) {
                        if (el.dataset.prevModel === undefined) el.dataset.prevModel = el.value;
                        if (strong) el.value = strong;
                        el.disabled = true;
                        el.classList.add("locked-agentic");
                        el.title = "Agentic mode always uses the strongest model";
                    } else {
                        el.disabled = false;
                        el.classList.remove("locked-agentic");
                        el.title = "";
                        if (el.dataset.prevModel !== undefined) {
                            el.value = el.dataset.prevModel;
                            delete el.dataset.prevModel;
                        }
                    }
                });
                if (typeof renderCostEstimate === "function") {
                    try { renderCostEstimate(); } catch (e) {}
                }
            }
            agenticToggle.addEventListener("change", apply);
            apply();  // set initial state (handles presets that default agentic on)
        })();

        /* ── close popovers on outside click ─────────────────────────────── */
        (function () {
            document.addEventListener("mousedown", function (e) {
                var panel = document.getElementById("config-panel");
                var toggle = document.getElementById("config-toggle");
                if (panel && panel.classList.contains("open")) {
                    if (!panel.contains(e.target) && toggle && !toggle.contains(e.target)) panel.classList.remove("open");
                }
                var cost = document.getElementById("cost-estimator");
                if (cost && cost.classList.contains("open") && !cost.contains(e.target)) {
                    cost.classList.remove("open");
                    cost.setAttribute("aria-expanded", "false");
                    var t = document.getElementById("cost-toggle");
                    if (t) t.innerHTML = "&#9662;";
                }
            });
        })();

        /* Kick the cost estimate once now that everything is wired. */
        if (typeof renderCostEstimate === "function") { try { renderCostEstimate(); } catch (e) {} }
    </script>

</body>
</html>
"""


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Default to loopback so debug builds aren't exposed to the local network.
    # Override with HOST=0.0.0.0 explicitly when deploying behind a proxy.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    print(f"\n{'='*60}")
    print(f"  specGPT Pipeline Server")
    print(f"{'='*60}")
    print(f"  Listening on http://{host}:{port}")
    print(f"  API: http://{host}:{port}/api/query")
    print(f"  Debug Mode: {DEBUG_PIPELINE}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=host, port=port)
