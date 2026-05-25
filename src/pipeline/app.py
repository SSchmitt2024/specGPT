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
from pathlib import Path


def _load_dotenv(path: str = ".env") -> int:
    """Populate `os.environ` from a KEY=value file. Production env vars win.

    Only meaningful for local dev — Railway/Cloudflare/etc. inject vars
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
            agentic=req.agentic,
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
        tokens_used=result.get("tokens_used"),
    )


@app.get("/api/config")
async def config_endpoint(_: bool = Depends(require_auth)) -> dict:
    """Return default PipelineConfig."""
    return PipelineConfig().to_dict()


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
            "provider": "Local (HuggingFace)",
            "price_per_1m_input": 0.0,
            "price_per_1m_output": None,
            "note": "Runs locally, no API cost",
        },
        "llm": {
            "model": cfg.llm_model,
            "provider": "Anthropic",
            "price_per_1m_input": 3.0,
            "price_per_1m_output": 15.0,
            "note": "Standard queries",
        },
        "agentic_llm": {
            "model": cfg.agentic_model,
            "provider": "Anthropic",
            "price_per_1m_input": 15.0,
            "price_per_1m_output": 75.0,
            "note": "Agentic mode only",
        },
    }


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

        /* ─── Agentic-mode toggle ─────────────────────────────────────── */
        .agentic-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
            padding: 8px 12px;
            background: #f4f6f7;
            border-radius: 4px;
            border-left: 3px solid #8e44ad;
            font-size: 13px;
            color: #555;
        }
        .agentic-row label {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-weight: 600;
            color: #2c3e50;
        }
        .agentic-row input[type="checkbox"] {
            transform: scale(1.2);
            cursor: pointer;
        }
        .agentic-row .agentic-hint {
            color: #7f8c8d;
            font-weight: normal;
            font-size: 12px;
        }
        .agentic-row.active {
            background: #f3e5f5;
            border-left-color: #6c3483;
        }

        /* Sub-panel of agentic knobs, shown only when toggle is on. */
        .agentic-config {
            background: #f3e5f5;
            padding: 12px 14px;
            border-radius: 4px;
            margin-top: 6px;
            border-left: 3px solid #6c3483;
        }
        .agentic-config.hidden {
            display: none;
        }
        .agentic-config .config-item label {
            color: #2c3e50;
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

        /* ─── Markdown rendering inside .answer-text ──────────────────────
           Subset of Claude.ai's style: comfortable spacing, bordered tables
           with a subtle row-zebra, monospace code blocks, soft inline-code
           chips. Scoped to .answer-text so the rest of the page is unaffected. */
        .answer-text h1, .answer-text h2, .answer-text h3, .answer-text h4 {
            color: #2c3e50;
            margin: 1.1em 0 0.4em;
            line-height: 1.3;
        }
        .answer-text h1 { font-size: 1.55em; }
        .answer-text h2 { font-size: 1.35em; }
        .answer-text h3 { font-size: 1.18em; }
        .answer-text h4 { font-size: 1.05em; }
        .answer-text > *:first-child { margin-top: 0; }
        .answer-text > *:last-child  { margin-bottom: 0; }
        .answer-text p { margin: 0.65em 0; }
        .answer-text ul, .answer-text ol {
            margin: 0.65em 0; padding-left: 1.6em;
        }
        .answer-text li { margin: 0.25em 0; }
        .answer-text li > p { margin: 0.25em 0; }
        .answer-text strong { color: #2c3e50; }
        .answer-text em { color: #555; }
        .answer-text blockquote {
            border-left: 3px solid #bdc3c7;
            color: #555;
            margin: 0.8em 0;
            padding: 0.2em 0.9em;
            background: #f4f6f7;
            border-radius: 0 3px 3px 0;
        }
        .answer-text code {
            font-family: "SF Mono", Menlo, Consolas, "Courier New", monospace;
            font-size: 0.9em;
            background: #eef2f6;
            padding: 1px 5px;
            border-radius: 3px;
            color: #2c3e50;
            border: 1px solid #e1e6ec;
        }
        .answer-text pre {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 12px 14px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 0.8em 0;
            line-height: 1.45;
            font-size: 0.88em;
        }
        .answer-text pre code {
            background: transparent;
            border: 0;
            padding: 0;
            color: inherit;
            font-size: 1em;
        }
        .answer-text table {
            border-collapse: collapse;
            margin: 0.9em 0;
            width: 100%;
            font-size: 0.94em;
            background: white;
            border: 1px solid #d0d7de;
            border-radius: 4px;
            overflow: hidden;
        }
        .answer-text th, .answer-text td {
            border: 1px solid #e1e6ec;
            padding: 8px 11px;
            text-align: left;
            vertical-align: top;
        }
        .answer-text th {
            background: #f4f6f7;
            color: #2c3e50;
            font-weight: 600;
        }
        .answer-text tr:nth-child(even) td { background: #fafbfc; }
        .answer-text a {
            color: #2980b9;
            text-decoration: underline;
            text-underline-offset: 2px;
        }
        .answer-text hr {
            border: 0;
            border-top: 1px solid #d0d7de;
            margin: 1.2em 0;
        }

        /* ─── Pipeline visualization (Mermaid DAG) ───────────────────── */
        .viz-section {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 18px 18px 8px;
            margin-top: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .viz-section h2 {
            font-size: 18px;
            color: #2c3e50;
            margin: 0 0 4px;
        }
        .viz-section .viz-sub {
            font-size: 12px;
            color: #888;
            margin-bottom: 14px;
        }
        .viz-container {
            overflow-x: auto;
            background: #fafbfc;
            border: 1px solid #eef2f6;
            border-radius: 6px;
            padding: 16px;
            min-height: 200px;
        }
        .viz-container .mermaid { text-align: center; }
        .viz-container svg { max-width: 100%; height: auto; }
        .viz-empty {
            color: #888;
            text-align: center;
            font-size: 13px;
            padding: 30px;
            font-style: italic;
        }
        /* Mermaid node text — slightly larger for readability */
        .viz-container .nodeLabel, .viz-container .label {
            font-size: 12px !important;
        }

        /* ─── Model info panel ────────────────────────────────────────── */
        .model-panel {
            margin-top: 24px;
            border: 1px solid #dce1e7;
            border-radius: 8px;
            overflow: hidden;
            font-size: 13px;
        }
        .model-panel-header {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #f4f6f7;
            padding: 9px 14px;
            cursor: pointer;
            user-select: none;
            border: none;
            width: 100%;
            text-align: left;
            font-size: 13px;
            font-weight: 600;
            color: #444;
        }
        .model-panel-header:hover { background: #eaecee; }
        .model-panel-badge {
            margin-left: auto;
            background: #eafaf1;
            color: #1e8449;
            font-size: 12px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 10px;
        }
        .model-panel-chevron { font-size: 10px; color: #999; }
        .model-panel-body {
            display: none;
            padding: 14px;
            background: white;
            overflow-x: auto;
        }
        .model-panel-body.open { display: block; }
        .model-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .model-table th {
            text-align: left;
            padding: 5px 8px;
            border-bottom: 2px solid #e0e0e0;
            color: #666;
            font-weight: 700;
            white-space: nowrap;
        }
        .model-table td {
            padding: 5px 8px;
            border-bottom: 1px solid #f2f2f2;
            vertical-align: middle;
        }
        .model-table code {
            background: #f5f5f5;
            padding: 1px 4px;
            border-radius: 3px;
            font-size: 11px;
        }
        .model-row-active td { background: #eaf4ff; font-weight: 600; }
        .model-note { color: #999; font-style: italic; }
        .model-cost-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-items: center;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            font-size: 12px;
            color: #555;
        }
        .model-cost-label { font-weight: 700; color: #333; }
        .model-cost-sep { color: #bbb; }
        .model-cost-total { font-weight: 700; color: #1e8449; font-size: 13px; }
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
                <button id="config-toggle" class="config-toggle">Config</button>
            </div>

            <div id="agentic-row" class="agentic-row">
                <label>
                    <input type="checkbox" id="agentic-toggle">
                    <span>Agentic mode</span>
                </label>
                <span class="agentic-hint">
                    Decomposes the answer, runs follow-up retrieval to fill
                    gaps, then regenerates with Opus + larger context. Slower
                    (~30-60s) and ~10× the cost leave off for routine queries.
                </span>
            </div>

            <div id="agentic-config" class="agentic-config hidden">
                <div class="config-grid">
                    <div class="config-item">
                        <label>Max Follow-ups</label>
                        <input type="number" id="config-agentic_max_followups" value="3" min="0" max="5">
                    </div>
                    <div class="config-item">
                        <label>Agentic Rerank Top-K</label>
                        <input type="number" id="config-agentic_rerank_topk" value="14" min="5" max="25">
                    </div>
                    <div class="config-item">
                        <label>Context Tokens</label>
                        <input type="number" id="config-agentic_max_context_tokens" value="16000" min="4000" max="24000" step="1000">
                    </div>
                    <div class="config-item">
                        <label>Output Tokens</label>
                        <input type="number" id="config-agentic_max_output_tokens" value="2048" min="512" max="4096" step="256">
                    </div>
                    <div class="config-item">
                        <label><input type="checkbox" id="config-agentic_targeted_fetch" checked> Targeted Fetch</label>
                    </div>
                    <div class="config-item">
                        <label><input type="checkbox" id="config-agentic_recursive"> Recursive</label>
                    </div>
                    <div class="config-item">
                        <label>Max Iterations</label>
                        <input type="number" id="config-agentic_max_iterations" value="5" min="1" max="10">
                    </div>
                </div>
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
                Running pipeline…
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

                <div class="viz-section">
                    <h2>Pipeline Flow</h2>
                    <div class="viz-sub">
                        Each color is a stage type. Branches show per-sub-query
                        retrieval (vector / tsvector / BM25); all paths merge
                        through RRF → dedup → rerank → generation.
                    </div>
                    <div id="pipeline-viz" class="viz-container">
                        <div class="viz-empty">Run a query to see the pipeline flow.</div>
                    </div>
                </div>

                <div class="model-panel" id="model-panel">
                    <button class="model-panel-header" onclick="toggleModelPanel()">
                        ⚙ Models &amp; Cost
                        <span class="model-panel-badge" id="model-cost-badge" style="display:none"></span>
                        <span class="model-panel-chevron" id="model-panel-chevron">▼</span>
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
                                <tr><td colspan="6" style="color:#aaa;font-style:italic">Loading…</td></tr>
                            </tbody>
                        </table>
                        <div class="model-cost-row" id="model-cost-row" style="display:none"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Markdown rendering: marked (parser) + DOMPurify (XSS sanitiser).
         LLM output is partially user-influenced via prompt injection, so we
         MUST sanitise the marked-generated HTML before injecting it into the
         DOM. Pinned to specific versions so the URL is effectively immutable. -->
    <script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.5/dist/purify.min.js"></script>

    <!-- Mermaid: renders the pipeline_trace as a downward-facing DAG.
         securityLevel:'strict' so any text we interpolate into node labels
         is encoded; click events disabled. Mermaid's own renderer never
         executes user-supplied HTML. -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.0/dist/mermaid.min.js"></script>

    <script>
        // GitHub-flavored markdown (tables, fenced code, autolinks).
        marked.setOptions({ gfm: true, breaks: false });

        // Mermaid: strict mode so any interpolated label text is encoded by
        // Mermaid itself; we also defensively scrub our own input.
        if (typeof mermaid !== "undefined") {
            mermaid.initialize({
                startOnLoad: false,
                securityLevel: "strict",
                theme: "default",
                flowchart: { curve: "basis", htmlLabels: true, useMaxWidth: true },
            });
        }

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
            if (v === null || v === undefined) return "—";
            if (v === 0) return "free";
            return "$" + v;
        }

        function _fmtCost(dollars) {
            if (dollars === null || dollars === undefined) return "—";
            if (dollars < 0.0001) return "<$0.0001";
            return "$" + dollars.toFixed(4);
        }

        function renderModelTable(isAgentic) {
            if (!_modelsData) return;
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

        function renderModelCost(tokensUsed, isAgentic) {
            if (!tokensUsed || !_modelsData) return;
            const llm = isAgentic ? _modelsData.agentic_llm : _modelsData.llm;
            if (!llm) return;
            const inCost  = (tokensUsed.prompt     / 1e6) * llm.price_per_1m_input;
            const outCost = (tokensUsed.completion  / 1e6) * llm.price_per_1m_output;
            const total   = inCost + outCost;

            const badge = document.getElementById("model-cost-badge");
            badge.textContent = _fmtCost(total) + " / query";
            badge.style.display = "";

            const row = document.getElementById("model-cost-row");
            row.style.display = "flex";
            row.innerHTML = `
                <span class="model-cost-label">Last query:</span>
                <span>${tokensUsed.prompt.toLocaleString()} in → ${_fmtCost(inCost)}</span>
                <span class="model-cost-sep">+</span>
                <span>${tokensUsed.completion.toLocaleString()} out → ${_fmtCost(outCost)}</span>
                <span class="model-cost-sep">=</span>
                <span class="model-cost-total">${_fmtCost(total)}</span>
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

        function buildMermaidFromTrace(trace, query) {
            const stages = {};
            for (const s of trace) stages[s.stage] = s;

            const L = ["flowchart TD"];
            L.push(`  Q["Query<br/><i>${_vizText(query, 60)}</i>"]:::input`);

            const qp = stages.query_processor;
            if (!qp) {
                L.push("  classDef input fill:#34495e,color:#fff,stroke:#2c3e50,stroke-width:2px");
                return L.join("\\n");
            }

            const qpType = _vizText(qp.output.type, 20);
            const qpEnts = (qp.output.entities || []).length;
            const qpSubs = (qp.output.sub_queries || []).length;
            L.push(`  QP["Query Processor<br/>type=<b>${qpType}</b><br/>${qpEnts} entities, ${qpSubs} sub-queries<br/>${_ms(qp)}"]:::stage_qp`);
            L.push("  Q --> QP");

            // Structured lookup — side branch that merges back into dedup
            const sl = stages.structured_lookup;
            let slActive = false;
            if (sl) {
                if (sl.output.skipped) {
                    L.push(`  SL["Structured Lookup<br/><i>skipped</i><br/>${_vizText(sl.output.reason, 40)}"]:::stage_skipped`);
                } else {
                    slActive = true;
                    const found = sl.output.found;
                    const conf = _vizText(sl.output.confidence, 12);
                    const flds = sl.output.field_count || 0;
                    const tbls = sl.output.table_count || 0;
                    L.push(`  SL["Structured Lookup<br/>found=<b>${found}</b> conf=${conf}<br/>${flds} fields · ${tbls} tables<br/>${_ms(sl)}"]:::stage_struct`);
                }
                L.push("  QP --> SL");
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
                L.push(`  SQ${i}{{"Sub-query ${i}<br/><i>${_vizText(sqText, 60)}</i>"}}:::stage_subq`);
                L.push(`  QP --> SQ${i}`);

                if (v) {
                    L.push(`  V${i}["Vector (Voyage)<br/>${v.output.count || 0} hits<br/>${_ms(v)}"]:::stage_vector`);
                    L.push(`  SQ${i} --> V${i}`);
                    if (rrf) L.push(`  V${i} --> RRF`);
                }
                if (t) {
                    L.push(`  T${i}["tsvector (Postgres)<br/>${t.output.count || 0} hits<br/>${_ms(t)}"]:::stage_tsv`);
                    L.push(`  SQ${i} --> T${i}`);
                    if (rrf) L.push(`  T${i} --> RRF`);
                }
                if (b) {
                    L.push(`  B${i}["BM25 (Okapi)<br/>${b.output.count || 0} hits<br/>${_ms(b)}"]:::stage_bm25`);
                    L.push(`  SQ${i} --> B${i}`);
                    if (rrf) L.push(`  B${i} --> RRF`);
                }
            }

            if (rrf) {
                L.push(`  RRF["RRF Merge<br/>${rrf.output.count || 0} merged<br/>${_ms(rrf)}"]:::stage_rrf`);
            }

            const dd = stages.result_dedup;
            if (dd) {
                L.push(`  DEDUP["Dedup<br/>${dd.output.deduped_count || 0} chunks<br/>${_ms(dd)}"]:::stage_dedup`);
                if (rrf) L.push("  RRF --> DEDUP");
                if (slActive) L.push("  SL --> DEDUP");
            }

            const rr = stages.final_rerank;
            if (rr) {
                L.push(`  RR["Cross-encoder Rerank<br/>top ${rr.output.count || 0}<br/>${_ms(rr)}"]:::stage_rerank`);
                if (dd) L.push("  DEDUP --> RR");
            }

            const gen = stages.generation;
            if (gen) {
                const cits = (gen.output.citation_count !== undefined) ? gen.output.citation_count : "?";
                const ans = gen.output.answer_length || 0;
                L.push(`  GEN["Generation (Claude)<br/>${ans} chars · ${cits} citations<br/>${_ms(gen)}"]:::stage_gen`);
                if (rr) L.push("  RR --> GEN");
            }

            // ─── Agentic refinement branch (only present when agentic=true) ───
            const gap = stages["agentic.gap_analysis"];
            const tfetch = stages["agentic.targeted_fetch"];
            const ag_rr = stages["agentic.rerank"];
            const ag_gen = stages["agentic.regenerate"];
            let agAnswerNode = "GEN"; // node whose output is the final answer
            if (gap) {
                const needs = gap.output && gap.output.needs_followup;
                const reason = _vizText((gap.output && gap.output.reason) || "", 60);
                L.push(`  GAP{"Gap Analysis<br/>needs_followup=<b>${!!needs}</b><br/><i>${reason}</i><br/>${_ms(gap)}"}:::stage_gap`);
                if (gen) L.push("  GEN --> GAP");

                if (needs) {
                    // (a) Targeted resource fetch — direct table/field lookup
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
                        L.push(`  TFETCH[/"Targeted Fetch<br/>requested: ${reqSummary}<br/>fetched: <b>${got}</b> chunks<br/>${_ms(tfetch)}"/]:::stage_tfetch`);
                        L.push("  GAP --> TFETCH");
                        if (ag_rr) L.push("  TFETCH --> RR2");
                    }

                    // (b) Per-followup natural-language retrieval branches
                    const fqIds = new Set();
                    for (const s of trace) {
                        const m = s.stage.match(/^agentic\\.followup_search_q(\\d+)$/);
                        if (m) fqIds.add(parseInt(m[1], 10));
                    }
                    for (const i of [...fqIds].sort((a, b) => a - b)) {
                        const fq = stages[`agentic.followup_search_q${i}`];
                        const qText = (fq && fq.input && fq.input.query) || `gap-q${i}`;
                        L.push(`  FQ${i}{{"Follow-up ${i}<br/><i>${_vizText(qText, 60)}</i><br/>${(fq && fq.output && fq.output.chunk_count) || 0} chunks<br/>${_ms(fq)}"}}:::stage_followup`);
                        L.push(`  GAP --> FQ${i}`);
                        if (ag_rr) L.push(`  FQ${i} --> RR2`);
                    }
                    if (ag_rr) {
                        L.push(`  RR2["Agentic Rerank<br/>top ${(ag_rr.output && ag_rr.output.count) || 0}<br/>+${(ag_rr.input && ag_rr.input.added_by_followups) || 0} new chunks<br/>${_ms(ag_rr)}"]:::stage_rerank`);
                    }
                    if (ag_gen) {
                        const c2 = (ag_gen.output && ag_gen.output.citation_count !== undefined) ? ag_gen.output.citation_count : "?";
                        const a2 = (ag_gen.output && ag_gen.output.answer_length) || 0;
                        const errType = ag_gen.output && ag_gen.output.error_type;
                        const label = errType
                            ? `Agentic Regenerate<br/><i>${_vizText(errType, 30)}</i> — fell back<br/>${_ms(ag_gen)}`
                            : `Agentic Regenerate (Opus)<br/>${a2} chars · ${c2} citations<br/>${_ms(ag_gen)}`;
                        L.push(`  GEN2["${label}"]:::stage_agen`);
                        if (ag_rr) L.push("  RR2 --> GEN2");
                        if (!errType) agAnswerNode = "GEN2";
                    }
                }
            }

            L.push('  ANS(["Final Answer"]):::output');
            L.push(`  ${agAnswerNode} --> ANS`);

            // Color palette — distinct per stage type, readable on light bg
            L.push("  classDef input    fill:#34495e,color:#fff,stroke:#2c3e50,stroke-width:2px");
            L.push("  classDef output   fill:#27ae60,color:#fff,stroke:#229954,stroke-width:2px");
            L.push("  classDef stage_qp     fill:#9b59b6,color:#fff,stroke:#7d3c98,stroke-width:1px");
            L.push("  classDef stage_struct fill:#16a085,color:#fff,stroke:#117a65,stroke-width:1px");
            L.push("  classDef stage_skipped fill:#ecf0f1,color:#7f8c8d,stroke:#bdc3c7,stroke-width:1px,stroke-dasharray:4 3");
            L.push("  classDef stage_subq   fill:#2980b9,color:#fff,stroke:#1f618d,stroke-width:1px");
            L.push("  classDef stage_vector fill:#3498db,color:#fff");
            L.push("  classDef stage_tsv    fill:#5dade2,color:#fff");
            L.push("  classDef stage_bm25   fill:#85c1e2,color:#1b2631");
            L.push("  classDef stage_rrf    fill:#e67e22,color:#fff,stroke:#ba6817,stroke-width:1px");
            L.push("  classDef stage_dedup  fill:#d35400,color:#fff");
            L.push("  classDef stage_rerank fill:#e74c3c,color:#fff,stroke:#922b21,stroke-width:1px");
            L.push("  classDef stage_gen    fill:#c0392b,color:#fff,stroke:#641e16,stroke-width:1px");
            // Agentic-branch colors — purple family to set them apart from the
            // normal-path warm colors and tie back to the toggle's accent.
            L.push("  classDef stage_gap      fill:#8e44ad,color:#fff,stroke:#6c3483,stroke-width:1px");
            L.push("  classDef stage_followup fill:#af7ac5,color:#fff,stroke:#7d3c98,stroke-width:1px");
            L.push("  classDef stage_agen     fill:#4a235a,color:#fff,stroke:#1b4f72,stroke-width:2px");
            // Targeted-fetch — a different family (teal) to visually distinguish
            // "direct table lookup" from "natural-language search".
            L.push("  classDef stage_tfetch   fill:#117a65,color:#fff,stroke:#0e6251,stroke-width:1px");

            return L.join("\\n");
        }

        let _vizCounter = 0;
        async function renderPipelineViz(trace, query) {
            const host = document.getElementById("pipeline-viz");
            if (!trace || !trace.length) {
                host.innerHTML = '<div class="viz-empty">No pipeline trace returned — set <code>DEBUG_PIPELINE=1</code> on the server to enable.</div>';
                return;
            }
            if (typeof mermaid === "undefined") {
                host.innerHTML = '<div class="viz-empty">Mermaid failed to load (CDN blocked?). The expanded trace above still has every stage.</div>';
                return;
            }
            const id = `viz-${++_vizCounter}`;
            const def = buildMermaidFromTrace(trace, query);
            try {
                const { svg } = await mermaid.render(id, def);
                host.innerHTML = svg;
            } catch (err) {
                console.error("mermaid render failed:", err, def);
                host.innerHTML = `<div class="viz-empty">Could not render flow: ${escapeHtml(err.message || String(err))}</div>`;
            }
        }

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
        const agenticRow = document.getElementById("agentic-row");
        const agenticConfig = document.getElementById("agentic-config");
        agenticToggle.addEventListener("change", () => {
            agenticRow.classList.toggle("active", agenticToggle.checked);
            agenticConfig.classList.toggle("hidden", !agenticToggle.checked);
        });

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
                agentic_max_followups: parseInt(document.getElementById("config-agentic_max_followups").value),
                agentic_rerank_topk: parseInt(document.getElementById("config-agentic_rerank_topk").value),
                agentic_max_context_tokens: parseInt(document.getElementById("config-agentic_max_context_tokens").value),
                agentic_max_output_tokens: parseInt(document.getElementById("config-agentic_max_output_tokens").value),
                agentic_targeted_fetch: document.getElementById("config-agentic_targeted_fetch").checked,
                agentic_recursive: document.getElementById("config-agentic_recursive").checked,
                agentic_max_iterations: parseInt(document.getElementById("config-agentic_max_iterations").value),
            };

            // Show loading
            resultsDiv.classList.remove("hidden");
            loadingDiv.classList.remove("hidden");
            errorDiv.classList.add("hidden");
            answerSection.classList.add("hidden");

            const agentic = agenticToggle.checked;
            if (agentic) {
                loadingDiv.innerHTML = "Running pipeline (agentic mode this can take 30-60s)…";
            } else {
                loadingDiv.textContent = "Running pipeline…";
            }

            try {
                const response = await fetch("/api/query", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query, config, debug: true, agentic }),
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
            // Render the answer through marked → DOMPurify so markdown tables,
            // headers, code blocks, and lists display the way Claude.ai does.
            // textContent fallback if either lib failed to load (network issue,
            // CDN block, offline) so the user still sees the raw answer.
            const answerEl = document.getElementById("answer-text");
            if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
                answerEl.innerHTML = renderMarkdown(data.answer || "");
            } else {
                answerEl.textContent = data.answer || "";
            }
            document.getElementById("latency").textContent = `${data.latency_ms.toFixed(0)}ms`;

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
                            ? ' <span class="citation-section" title="Cited section not found in retrieved context">(not in context)</span>'
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
            } else {
                stagesDiv.innerHTML = '';
            }

            // Pipeline visualization (Mermaid DAG)
            renderPipelineViz(data.pipeline_trace, data.query);

            // Model panel — update active row + cost
            const isAgentic = !!data.agentic;
            renderModelTable(isAgentic);
            renderModelCost(data.tokens_used, isAgentic);

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
