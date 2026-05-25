"""
Phase 2 - Step 2.5: Web Application (FastAPI Backend)

Exposes the full retrieval + generation pipeline as a web service, gated
by a shared-password login (see src/pipeline/auth.py for the threat model).

Endpoints (auth-gated unless marked public):
  GET  /healthz       — public liveness check (Railway/k8s healthcheck)
  GET  /login         — public; renders the password form
  POST /login         — public; validates password, sets session cookie
  POST /logout        — public; clears session cookie
  GET  /              — gated; serves the web UI (or redirects to /login)
  POST /api/query     — gated; runs the pipeline
  GET  /api/config    — gated; returns default PipelineConfig

Required env vars:
  APP_PASSWORD     — plaintext shared password (hashed at startup, wiped from memory)
  SESSION_SECRET   — ≥16-byte string used as HMAC key for session cookies
  SUPABASE_URL / SUPABASE_KEY / VOYAGE_API_KEY / ANTHROPIC_API_KEY — pipeline backends

Optional env vars:
  DEBUG_PIPELINE   — "1" to include full trace in responses (default: off)
  PORT             — server port (default: 8000)
  HOST             — server host (default: 127.0.0.1)
  COOKIE_SECURE    — "0" to allow non-HTTPS cookies for local dev (default: on)
  LOG_LEVEL        — Python logging level (default: INFO)

Run:
  python -m src.pipeline.app
  Then visit http://localhost:8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid

from fastapi import Cookie, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
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
from src.pipeline.orchestrator import GenerationError, orchestrate, PipelineConfig


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


class QueryResponse(BaseModel):
    """Response from /api/query endpoint."""
    query: str
    answer: str
    citations: list[dict]
    config: dict
    pipeline_trace: list[dict] | None = None
    latency_ms: float


# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="specGPT Pipeline",
    description="NVMe Specification Q&A with Full Pipeline Visibility",
    version="2.0",
)

# Configuration: trace exposes chunk previews, model names, and timings, so
# default to OFF. Set DEBUG_PIPELINE=1 only in development.
DEBUG_PIPELINE = os.getenv("DEBUG_PIPELINE", "0").lower() in ("1", "true", "yes")


# ============================================================================
# Auth helpers + endpoints
# ============================================================================

def _client_ip(request: Request) -> str:
    """Best-effort client IP. Trusts X-Forwarded-For only if explicitly opted in
    via TRUST_PROXY_HEADERS=1 — otherwise the throttle can be bypassed by
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
  <title>specGPT — sign in</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5; color: #333; margin: 0;
            min-height: 100vh; display: flex; align-items: center; justify-content: center; }}
    .card {{ background: white; padding: 32px 28px; border-radius: 8px;
             box-shadow: 0 2px 6px rgba(0,0,0,0.08); width: 320px; }}
    h1 {{ font-size: 22px; margin: 0 0 6px; color: #2c3e50; }}
    p.sub {{ font-size: 13px; color: #666; margin: 0 0 20px; }}
    label {{ display: block; font-size: 12px; font-weight: 600;
             text-transform: uppercase; color: #555; margin-bottom: 6px; }}
    input[type=password] {{ width: 100%; box-sizing: border-box; padding: 10px;
                             border: 2px solid #e0e0e0; border-radius: 4px;
                             font-size: 16px; }}
    input[type=password]:focus {{ outline: none; border-color: #3498db; }}
    button {{ margin-top: 16px; width: 100%; padding: 12px;
              background: #3498db; color: white; border: none;
              border-radius: 4px; font-size: 16px; font-weight: 600;
              cursor: pointer; }}
    button:hover {{ background: #2980b9; }}
    .error {{ background: #fdecea; color: #c0392b;
              padding: 10px 12px; border-radius: 4px;
              border-left: 3px solid #e74c3c;
              font-size: 13px; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <form class="card" method="post" action="/login" autocomplete="off">
    <h1>specGPT</h1>
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
    """Liveness check — intentionally unauthenticated so external healthchecks work.

    Does NOT exercise Supabase/Voyage/Anthropic — those have cost. It just
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

    debug_trace = req.debug and DEBUG_PIPELINE
    request_id = uuid.uuid4().hex[:12]

    start = time.time()
    try:
        # orchestrate() is synchronous and CPU/IO-heavy (Voyage, Supabase,
        # cross-encoder, Anthropic). Run it in a worker thread so we don't
        # block the FastAPI event loop for the duration of the pipeline.
        result = await asyncio.to_thread(
            orchestrate, req.query, config=config, debug=debug_trace,
        )
    except GenerationError as e:
        # Retrieval worked; generation failed. 502 (bad upstream gateway)
        # is more accurate than 500 here, and the trace can still be returned
        # in debug mode so callers can see what *was* retrieved.
        logger.warning("Generation failure [%s]: %s", request_id, e.cause)
        detail: dict = {
            "error": "generation_failed",
            "request_id": request_id,
            "cause_type": type(e.cause).__name__,
        }
        if debug_trace:
            detail["pipeline_trace"] = e.trace
        raise HTTPException(status_code=502, detail=detail)
    except Exception as e:
        logger.exception("Pipeline error [%s]: %s", request_id, e)
        # Don't leak internal error text to the client; surface only an id.
        raise HTTPException(
            status_code=500,
            detail={"error": "pipeline_error", "request_id": request_id},
        )
    latency_ms = (time.time() - start) * 1000

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        citations=result["citations"],
        config=result["config"],
        pipeline_trace=result.get("pipeline_trace") if debug_trace else None,
        latency_ms=latency_ms,
    )


@app.get("/api/config")
async def config_endpoint(_: bool = Depends(require_auth)) -> dict:
    """Return default PipelineConfig."""
    return PipelineConfig().to_dict()


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
    <title>specGPT — NVMe Spec Q&A</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: #2c3e50;
            color: white;
            padding: 30px 0;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
        }

        header h1 {
            font-size: 28px;
            margin-bottom: 5px;
        }

        header p {
            font-size: 14px;
            opacity: 0.9;
        }

        .search-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }

        #query-input {
            flex: 1;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 4px;
            font-size: 16px;
            transition: border-color 0.3s;
        }

        #query-input:focus {
            outline: none;
            border-color: #3498db;
        }

        button {
            padding: 12px 30px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s;
        }

        button:hover {
            background: #2980b9;
        }

        button:active {
            transform: scale(0.98);
        }

        .config-toggle {
            background: #95a5a6;
            font-size: 14px;
            padding: 8px 15px;
        }

        .config-toggle:hover {
            background: #7f8c8d;
        }

        .config-panel {
            display: none;
            background: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin-top: 15px;
            border-left: 4px solid #95a5a6;
        }

        .config-panel.open {
            display: block;
        }

        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }

        .config-item {
            display: flex;
            flex-direction: column;
        }

        .config-item label {
            font-size: 12px;
            font-weight: 600;
            color: #555;
            margin-bottom: 5px;
            text-transform: uppercase;
        }

        .config-item input {
            padding: 8px;
            border: 1px solid #bdc3c7;
            border-radius: 3px;
            font-size: 14px;
        }

        .loading {
            text-align: center;
            padding: 20px;
            color: #3498db;
            font-weight: 600;
        }

        .error {
            background: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }

        .results-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .answer-box {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            border-left: 4px solid #27ae60;
        }

        .answer-box h3 {
            color: #27ae60;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
        }

        .answer-text {
            line-height: 1.6;
            color: #333;
        }

        .citations {
            background: #fef5e7;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #f39c12;
            margin-bottom: 20px;
        }

        .citations h3 {
            color: #f39c12;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
        }

        .citation-item {
            padding: 8px 0;
            border-bottom: 1px solid #fdebd0;
            font-size: 14px;
        }

        .citation-item:last-child {
            border-bottom: none;
        }

        .citation-section {
            color: #d68910;
            font-weight: 600;
        }

        .pipeline-section {
            margin-top: 30px;
        }

        .pipeline-section h2 {
            font-size: 18px;
            margin-bottom: 15px;
            color: #2c3e50;
        }

        .pipeline-stage {
            background: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            margin-bottom: 10px;
            overflow: hidden;
        }

        .stage-header {
            padding: 12px 15px;
            background: #f0f0f0;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
            transition: background 0.2s;
        }

        .stage-header:hover {
            background: #e8e8e8;
        }

        .stage-header.open {
            background: #e3f2fd;
        }

        .stage-name {
            font-weight: 600;
            color: #333;
            flex: 1;
        }

        .stage-time {
            font-size: 12px;
            color: #666;
            margin-right: 10px;
        }

        .stage-toggle {
            color: #666;
            font-size: 18px;
            transition: transform 0.2s;
        }

        .stage-header.open .stage-toggle {
            transform: rotate(180deg);
        }

        .stage-content {
            display: none;
            padding: 15px;
            border-top: 1px solid #e0e0e0;
        }

        .stage-content.open {
            display: block;
        }

        .stage-json {
            background: #f5f5f5;
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
            font-family: "Courier New", monospace;
            font-size: 12px;
            line-height: 1.4;
            color: #333;
        }

        .hidden {
            display: none;
        }

        footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: #666;
            font-size: 12px;
        }

        @media (max-width: 768px) {
            .config-grid {
                grid-template-columns: 1fr;
            }

            .search-box {
                flex-direction: column;
            }

            button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container" style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>specGPT</h1>
                <p>Ask questions about NVMe specifications. See exactly how the system found the answer.</p>
            </div>
            <form method="post" action="/logout" style="margin: 0;">
                <button type="submit" class="config-toggle" style="background:#34495e;">Sign out</button>
            </form>
        </div>
    </header>

    <div class="container">
        <div class="search-section">
            <div class="search-box">
                <input
                    type="text"
                    id="query-input"
                    placeholder="Ask a question... e.g., 'What is bit 7 of CDW10?'"
                    autocomplete="off"
                >
                <button id="search-btn">Search</button>
                <button id="config-toggle" class="config-toggle">⚙️ Config</button>
            </div>

            <div id="config-panel" class="config-panel">
                <strong>Pipeline Configuration</strong>
                <div class="config-grid">
                    <div class="config-item">
                        <label>Vector Top-K</label>
                        <input type="number" id="config-vector_topk" value="10" min="1" max="50">
                    </div>
                    <div class="config-item">
                        <label>tsvector Top-K</label>
                        <input type="number" id="config-tsvector_topk" value="10" min="1" max="50">
                    </div>
                    <div class="config-item">
                        <label>BM25 Top-K</label>
                        <input type="number" id="config-bm25_topk" value="10" min="1" max="50">
                    </div>
                    <div class="config-item">
                        <label>RRF K</label>
                        <input type="number" id="config-rrf_k" value="60" min="10" max="200">
                    </div>
                    <div class="config-item">
                        <label>RRF Output Top-K</label>
                        <input type="number" id="config-rrf_output_topk" value="20" min="5" max="50">
                    </div>
                    <div class="config-item">
                        <label>Final Rerank Top-K</label>
                        <input type="number" id="config-final_rerank_topk" value="7" min="1" max="20">
                    </div>
                    <div class="config-item">
                        <label>Max Sub-Queries</label>
                        <input type="number" id="config-max_subqueries" value="3" min="1" max="5">
                    </div>
                </div>
            </div>
        </div>

        <div id="results" class="hidden">
            <div id="error" class="error hidden"></div>

            <div id="loading" class="loading hidden">
                ⏳ Running pipeline...
            </div>

            <div id="answer-section" class="results-section hidden">
                <div id="latency" style="text-align: right; font-size: 12px; color: #666; margin-bottom: 15px;"></div>

                <div class="answer-box">
                    <h3>Answer</h3>
                    <div id="answer-text" class="answer-text"></div>
                </div>

                <div id="citations-box" class="citations hidden">
                    <h3>Sources Cited</h3>
                    <div id="citations-list"></div>
                </div>

                <div class="pipeline-section">
                    <h2>Pipeline Trace</h2>
                    <div id="pipeline-stages"></div>
                </div>
            </div>
        </div>
    </div>

    <footer>
        <p>Built with FastAPI + Anthropic Claude. Source: <a href="https://github.com/SSchmitt2024/specGPT">SSchmitt2024/specGPT</a></p>
    </footer>

    <script>
        const queryInput = document.getElementById("query-input");
        const searchBtn = document.getElementById("search-btn");
        const configToggle = document.getElementById("config-toggle");
        const configPanel = document.getElementById("config-panel");
        const resultsDiv = document.getElementById("results");
        const loadingDiv = document.getElementById("loading");
        const errorDiv = document.getElementById("error");
        const answerSection = document.getElementById("answer-section");

        // Config panel toggle
        configToggle.addEventListener("click", () => {
            configPanel.classList.toggle("open");
        });

        // Search on Enter key
        queryInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") {
                runQuery();
            }
        });

        // Search button
        searchBtn.addEventListener("click", runQuery);

        async function runQuery() {
            const query = queryInput.value.trim();
            if (!query) return;

            // Collect config
            const config = {
                vector_topk: parseInt(document.getElementById("config-vector_topk").value),
                tsvector_topk: parseInt(document.getElementById("config-tsvector_topk").value),
                bm25_topk: parseInt(document.getElementById("config-bm25_topk").value),
                rrf_k: parseInt(document.getElementById("config-rrf_k").value),
                rrf_output_topk: parseInt(document.getElementById("config-rrf_output_topk").value),
                final_rerank_topk: parseInt(document.getElementById("config-final_rerank_topk").value),
                max_subqueries: parseInt(document.getElementById("config-max_subqueries").value),
            };

            // Show loading
            resultsDiv.classList.remove("hidden");
            loadingDiv.classList.remove("hidden");
            errorDiv.classList.add("hidden");
            answerSection.classList.add("hidden");

            try {
                const response = await fetch("/api/query", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query, config, debug: true }),
                });

                if (response.status === 401) {
                    // Session expired mid-flight — bounce to login.
                    window.location.href = "/login?next=" + encodeURIComponent(window.location.pathname);
                    return;
                }
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(typeof err.detail === "string" ? err.detail : "Request failed");
                }

                const data = await response.json();
                displayResults(data);
            } catch (err) {
                errorDiv.textContent = `Error: ${err.message}`;
                errorDiv.classList.remove("hidden");
            } finally {
                loadingDiv.classList.add("hidden");
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

        function displayResults(data) {
            // Answer
            document.getElementById("answer-text").textContent = data.answer;
            document.getElementById("latency").textContent = `⏱️ ${data.latency_ms.toFixed(0)}ms`;

            // Citations — escape attacker-controlled fields (section_id /
            // section_title flow back from PDF text) before HTML interpolation.
            const citationsList = document.getElementById("citations-list");
            const citationsBox = document.getElementById("citations-box");
            if (data.citations && data.citations.length > 0) {
                citationsList.innerHTML = data.citations
                    .map(c => {
                        const sid = escapeHtml(c.section_id);
                        const title = escapeHtml(c.section_title);
                        const flag = c.hallucinated
                            ? ' <span class="citation-section" title="Cited section not found in retrieved context">⚠ not in context</span>'
                            : "";
                        return `<div class="citation-item"><span class="citation-section">[§${sid}]</span> ${title}${flag}</div>`;
                    })
                    .join("");
                citationsBox.classList.remove("hidden");
            } else {
                citationsBox.classList.add("hidden");
            }

            // Pipeline trace — JSON.stringify is safe content but interpolating
            // it into innerHTML still requires escaping the angle brackets.
            const stagesDiv = document.getElementById("pipeline-stages");
            if (data.pipeline_trace) {
                stagesDiv.innerHTML = data.pipeline_trace
                    .map((stage, idx) => `
                        <div class="pipeline-stage">
                            <div class="stage-header" onclick="toggleStage(this)">
                                <span class="stage-name">${idx + 1}. ${escapeHtml(formatStageName(stage.stage))}</span>
                                <span class="stage-time">${stage.took_ms.toFixed(0)}ms</span>
                                <span class="stage-toggle">▼</span>
                            </div>
                            <div class="stage-content">
                                <div class="stage-json">${escapeHtml(JSON.stringify(stage, null, 2))}</div>
                            </div>
                        </div>
                    `)
                    .join("");
            }

            answerSection.classList.remove("hidden");
        }

        function formatStageName(name) {
            return name
                .replace(/_/g, " ")
                .split(" ")
                .map(w => w.charAt(0).toUpperCase() + w.slice(1))
                .join(" ");
        }

        function toggleStage(header) {
            header.classList.toggle("open");
            header.nextElementSibling.classList.toggle("open");
        }
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
