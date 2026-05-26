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
import threading
import time
import uuid
from collections import OrderedDict
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
    agentic: bool = False
    # Present when agentic mode is off and `auto_gap_check` ran. The UI
    # uses this to surface a one-click "run agentic refinement" prompt
    # when the model would have requested more context.
    gap_hint: dict | None = None
    # Opaque handle the UI passes to /api/refine to resume from the
    # already-computed first-pass state (no Stages 1–4 redo).
    request_id: str | None = None


class RefineRequest(BaseModel):
    """Request body for /api/refine — resumes the prior /api/query call by
    request_id, runs only Stage 5 (gap analysis + targeted fetch + follow-up
    retrieval + re-rerank + Opus regen) against the cached first-pass state.
    """
    request_id: str
    config: dict | None = None
    debug: bool = True


# ============================================================================
# Refine cache — in-process LRU mapping request_id → first-pass state.
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
  <title>specGPT - sign in</title>
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

    # Cache first-pass state so /api/refine can resume without redoing
    # Stages 1–4. Only meaningful when the request landed in non-agentic
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

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        citations=result["citations"],
        config=result["config"],
        pipeline_trace=result.get("pipeline_trace") if debug_trace else None,
        latency_ms=latency_ms,
        tokens_used=result.get("tokens_used"),
        agentic=bool(result.get("agentic")),
        gap_hint=result.get("gap_hint"),
        request_id=request_id if not req.agentic else None,
    )


@app.post("/api/refine", response_model=QueryResponse)
async def refine_endpoint(req: RefineRequest, _: bool = Depends(require_auth)) -> QueryResponse:
    """Resume a prior /api/query by request_id and run the agentic
    refinement against the cached first-pass state — no Stages 1–4 redo.
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
        detail: dict = {
            "error": "generation_failed",
            "request_id": req.request_id,
            "cause_type": type(e.cause).__name__,
        }
        if debug_trace:
            detail["pipeline_trace"] = e.trace
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

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        citations=result["citations"],
        config=result["config"],
        pipeline_trace=result.get("pipeline_trace") if debug_trace else None,
        latency_ms=latency_ms,
        tokens_used=result.get("tokens_used"),
        agentic=True,
        gap_hint=None,
        request_id=req.request_id,
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
    <title>specGPT - NVMe Spec Q&A</title>
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
            padding: 12px 16px;
        }

        header {
            background: #2c3e50;
            color: white;
            padding: 14px 0 12px;
            margin-bottom: 14px;
            border-bottom: 2px solid #3498db;
        }

        header h1 {
            font-size: 20px;
            margin-bottom: 2px;
            line-height: 1.2;
        }

        header p {
            font-size: 12.5px;
            opacity: 0.85;
            line-height: 1.3;
        }

        .search-section {
            background: white;
            border-radius: 6px;
            padding: 12px 14px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }

        .search-box {
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }

        #query-input {
            flex: 1;
            padding: 9px 11px;
            border: 1.5px solid #e0e0e0;
            border-radius: 4px;
            font-size: 14.5px;
            transition: border-color 0.2s;
        }

        #query-input:focus {
            outline: none;
            border-color: #3498db;
        }

        button {
            padding: 9px 22px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }

        button:hover {
            background: #2980b9;
        }

        button:active {
            transform: scale(0.98);
        }

        .config-toggle {
            background: #95a5a6;
            font-size: 13px;
            padding: 6px 12px;
        }

        .config-toggle:hover {
            background: #7f8c8d;
        }

        /* ─── Agentic-mode toggle ─────────────────────────────────────── */
        .agentic-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 8px;
            padding: 6px 10px;
            background: #f4f6f7;
            border-radius: 4px;
            border-left: 3px solid #8e44ad;
            font-size: 12.5px;
            color: #555;
        }
        .agentic-row label {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            font-weight: 600;
            color: #2c3e50;
        }
        .agentic-row input[type="checkbox"] {
            transform: scale(1.1);
            cursor: pointer;
        }
        .agentic-row .agentic-hint {
            color: #7f8c8d;
            font-weight: normal;
            font-size: 11.5px;
        }
        .agentic-row.active {
            background: #f3e5f5;
            border-left-color: #6c3483;
        }

        /* Sub-panel of agentic knobs, shown only when toggle is on. */
        .agentic-config {
            background: #f3e5f5;
            padding: 8px 12px;
            border-radius: 4px;
            margin-top: 4px;
            border-left: 3px solid #6c3483;
        }
        .agentic-config.hidden {
            display: none;
        }
        .agentic-config .config-item label {
            color: #2c3e50;
        }

        /* ─── Cost estimator (live updates on any config change) ──────── */
        .cost-estimator {
            margin-top: 10px;
            background: #ffffff;
            border: 1px solid #e1e6ec;
            border-radius: 6px;
            font-size: 13px;
            overflow: hidden;
        }
        .cost-summary {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            cursor: pointer;
            user-select: none;
            background: #fafbfc;
            transition: background 0.15s;
        }
        .cost-summary:hover { background: #f4f6f7; }
        .cost-icon {
            font-size: 13px;
            opacity: 0.7;
        }
        .cost-label {
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
            font-weight: 600;
        }
        .cost-total {
            font-family: "SF Mono", Menlo, Consolas, monospace;
            font-weight: 700;
            color: #1e8449;
            font-size: 13px;
        }
        .cost-total.cost-warn { color: #b7791f; }
        .cost-total.cost-high { color: #b91c1c; }
        .cost-context {
            color: #64748b;
            font-size: 11.5px;
        }
        .cost-toggle {
            margin-left: auto;
            color: #94a3b8;
            font-size: 10.5px;
            transition: transform 0.2s;
        }
        .cost-estimator.open .cost-toggle { transform: rotate(180deg); }
        .cost-breakdown {
            display: none;
            padding: 4px 10px 10px;
            border-top: 1px solid #f1f5f9;
        }
        .cost-estimator.open .cost-breakdown { display: block; }
        .cost-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 4px 0;
            border-bottom: 1px dashed #f1f5f9;
            font-size: 12px;
        }
        .cost-row:last-child { border-bottom: 0; }
        .cost-row-name { color: #334155; }
        .cost-row-name small {
            color: #94a3b8;
            margin-left: 6px;
            font-size: 11.5px;
        }
        .cost-row-value {
            font-family: "SF Mono", Menlo, Consolas, monospace;
            color: #1f2d3d;
            font-weight: 600;
        }
        .cost-row.cost-row-total {
            margin-top: 4px;
            padding-top: 8px;
            border-top: 1px solid #e1e6ec;
            border-bottom: 0;
        }
        .cost-row.cost-row-total .cost-row-value { color: #1e8449; }
        .cost-disclaimer {
            margin-top: 8px;
            color: #94a3b8;
            font-size: 11px;
            font-style: italic;
            line-height: 1.45;
        }

        /* ─── Agent activity strip (lives inside the header banner) ──── */
        .agent-strip {
            margin-top: 10px;
            padding: 7px 12px 7px 14px;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-left: 3px solid #a855f7;
            border-radius: 5px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 8px 12px;
            color: #e2e8f0;
            font-size: 12.5px;
            line-height: 1.35;
            min-height: 36px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
        }
        .agent-strip-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #f1f5f9;
            font-weight: 700;
            margin-right: 6px;
            padding-right: 10px;
            border-right: 1px solid rgba(255, 255, 255, 0.18);
        }
        .agent-strip-empty {
            color: #cbd5e1;
            font-style: italic;
            font-size: 13px;
        }
        .agent-strip-state {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: #cbd5e1;
            background: rgba(255, 255, 255, 0.06);
            padding: 3px 10px;
            border-radius: 11px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .agent-strip-state .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #94a3b8;
            display: inline-block;
        }
        .agent-strip-state.state-ok    .dot { background: #34d399; }
        .agent-strip-state.state-warn  .dot { background: #fbbf24; }
        .agent-strip-state.state-agent .dot { background: #c084fc; }
        .agent-strip-state.state-error .dot { background: #f87171; }
        .agent-strip-state b { color: #f1f5f9; font-weight: 600; }
        .agent-strip-state .strip-latency {
            font-family: "SF Mono", Menlo, Consolas, monospace;
            color: #cbd5e1;
        }
        .agent-strip-reason {
            flex: 1;
            min-width: 200px;
            color: #cbd5e1;
            font-size: 12.5px;
        }
        .agent-strip-actions { margin-left: auto; }
        .agent-strip-chips {
            display: inline-flex;
            flex-wrap: wrap;
            gap: 4px;
            align-items: center;
        }
        .agent-strip-chip {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #e2e8f0;
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 10px;
        }
        .agent-strip-chip.chip-section { color: #cbd5e1; }
        .agent-strip-details {
            flex-basis: 100%;
            margin-top: 4px;
            padding-top: 8px;
            border-top: 1px dashed rgba(255, 255, 255, 0.12);
            display: flex;
            flex-wrap: wrap;
            gap: 8px 16px;
            font-size: 12px;
            color: #cbd5e1;
        }
        .agent-strip-details .detail-group {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .agent-strip-details .detail-label {
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: #94a3b8;
            font-weight: 600;
        }
        .run-agentic-btn {
            padding: 6px 14px;
            background: #8b5cf6;
            color: #fff;
            font-size: 12.5px;
            font-weight: 600;
            border: 1px solid #7c3aed;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.15s;
        }
        .run-agentic-btn:hover { background: #7c3aed; }
        .run-agentic-btn:disabled {
            background: #475569;
            border-color: #475569;
            color: #cbd5e1;
            cursor: not-allowed;
        }

        .config-panel {
            display: none;
            background: #ecf0f1;
            padding: 10px 12px;
            border-radius: 4px;
            margin-top: 8px;
            border-left: 3px solid #95a5a6;
        }

        .config-panel.open {
            display: block;
        }

        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 8px 10px;
            margin-top: 6px;
        }

        .config-item {
            display: flex;
            flex-direction: column;
        }

        .config-item label {
            font-size: 10.5px;
            font-weight: 600;
            color: #555;
            margin-bottom: 3px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        .config-item input,
        .config-item select {
            padding: 5px 7px;
            border: 1px solid #bdc3c7;
            border-radius: 3px;
            font-size: 13px;
            background: white;
            color: #2c3e50;
            font-family: inherit;
        }
        .config-item select { cursor: pointer; }
        .config-item.config-item-wide { grid-column: span 2; }

        .loading {
            background: white;
            border: 1px solid #e1e6ec;
            border-radius: 6px;
            padding: 14px 16px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            display: flex;
            align-items: center;
            gap: 14px;
            color: #1f2d3d;
            font-size: 13.5px;
        }
        .loading-spinner {
            width: 22px;
            height: 22px;
            border: 2.5px solid #e0e7ff;
            border-top-color: #3498db;
            border-right-color: #3498db;
            border-radius: 50%;
            flex-shrink: 0;
            animation: loading-spin 0.8s linear infinite;
        }
        .loading-body {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .loading-title {
            font-weight: 600;
            color: #1f2d3d;
            line-height: 1.3;
        }
        .loading-meta {
            font-size: 11.5px;
            color: #64748b;
            font-family: "SF Mono", Menlo, Consolas, monospace;
            letter-spacing: 0.2px;
        }
        .loading-meta .loading-dots::after {
            content: "";
            display: inline-block;
            width: 14px;
            text-align: left;
            animation: loading-dots 1.4s steps(4, end) infinite;
        }
        .loading-cancel {
            padding: 6px 14px;
            background: white;
            color: #b91c1c;
            border: 1px solid #fecaca;
            border-radius: 4px;
            font-size: 12.5px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s, border-color 0.15s;
            flex-shrink: 0;
        }
        .loading-cancel:hover {
            background: #fee2e2;
            border-color: #fca5a5;
        }
        .loading-cancel:active { transform: scale(0.97); }

        @keyframes loading-spin {
            to { transform: rotate(360deg); }
        }
        @keyframes loading-dots {
            0%   { content: ""; }
            25%  { content: "."; }
            50%  { content: ".."; }
            75%  { content: "..."; }
            100% { content: ""; }
        }

        button:disabled {
            opacity: 0.55;
            cursor: not-allowed;
            transform: none !important;
        }

        .error {
            background: #e74c3c;
            color: white;
            padding: 10px 12px;
            border-radius: 4px;
            margin-bottom: 12px;
            font-size: 13px;
        }

        .results-section {
            background: white;
            border-radius: 6px;
            padding: 14px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }

        .answer-box {
            background: #ecf0f1;
            padding: 10px 12px;
            border-radius: 4px;
            margin-bottom: 12px;
            border-left: 3px solid #27ae60;
        }

        .answer-box h3 {
            color: #27ae60;
            margin-bottom: 6px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }

        .answer-text {
            line-height: 1.55;
            color: #333;
        }

        .citations {
            background: #fef5e7;
            padding: 10px 12px;
            border-radius: 4px;
            border-left: 3px solid #f39c12;
            margin-bottom: 12px;
        }

        .citations h3 {
            color: #f39c12;
            margin-bottom: 6px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }

        .citation-item {
            padding: 5px 0;
            border-bottom: 1px solid #fdebd0;
            font-size: 13px;
        }

        .citation-item:last-child {
            border-bottom: none;
        }

        .citation-section {
            color: #d68910;
            font-weight: 600;
        }

        .pipeline-section {
            margin-top: 18px;
        }

        .pipeline-section h2 {
            font-size: 15px;
            margin-bottom: 8px;
            color: #2c3e50;
        }

        .pipeline-stage {
            background: #fff;
            border: 1px solid #e1e6ec;
            border-radius: 6px;
            margin-bottom: 8px;
            overflow: hidden;
            transition: box-shadow 0.15s, border-color 0.15s;
        }
        .pipeline-stage:hover {
            border-color: #cdd5df;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        }
        .pipeline-stage.stage-group-agentic {
            border-left: 3px solid #a855f7;
        }
        .pipeline-stage.stage-group-normal {
            border-left: 3px solid #3b82f6;
        }

        .stage-header {
            padding: 10px 14px;
            background: #fafbfc;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            user-select: none;
            transition: background 0.15s;
        }
        .stage-header:hover { background: #f4f6f7; }
        .stage-header.open { background: #f1f5f9; }

        .stage-title-block {
            flex: 1;
            min-width: 0;
        }
        .stage-name {
            font-weight: 600;
            color: #1f2d3d;
            font-size: 14px;
            display: block;
            line-height: 1.35;
        }
        .stage-index {
            color: #94a3b8;
            font-weight: 500;
            margin-right: 4px;
        }
        .stage-subtitle {
            display: block;
            font-size: 12px;
            color: #64748b;
            margin-top: 2px;
            font-weight: 400;
        }

        .stage-time {
            font-family: "SF Mono", Menlo, Consolas, monospace;
            font-size: 12px;
            color: #475569;
            background: #eef2f6;
            padding: 2px 8px;
            border-radius: 10px;
            white-space: nowrap;
            flex-shrink: 0;
        }
        .stage-time.stage-time-slow { background: #fef3c7; color: #92400e; }

        .stage-toggle {
            color: #94a3b8;
            font-size: 14px;
            transition: transform 0.2s;
            flex-shrink: 0;
        }
        .stage-header.open .stage-toggle { transform: rotate(180deg); }

        .stage-content {
            display: none;
            padding: 14px 16px 16px;
            border-top: 1px solid #eef2f6;
            background: #fdfdfe;
        }
        .stage-content.open { display: block; }

        .stage-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 12px;
        }
        .stage-chip {
            font-size: 12px;
            background: #eef2f6;
            color: #334155;
            border: 1px solid #e1e6ec;
            padding: 3px 9px;
            border-radius: 12px;
            font-weight: 500;
        }
        .stage-chip.stage-chip-error {
            background: #fee2e2;
            color: #991b1b;
            border-color: #fecaca;
        }
        .stage-chip.stage-chip-warn {
            background: #fef3c7;
            color: #92400e;
            border-color: #fde68a;
        }
        .stage-chip.stage-chip-ok {
            background: #dcfce7;
            color: #166534;
            border-color: #bbf7d0;
        }
        .stage-chip.stage-chip-info {
            background: #dbeafe;
            color: #1e40af;
            border-color: #bfdbfe;
        }
        .stage-chip.stage-chip-skipped {
            background: #f3f4f6;
            color: #6b7280;
            border-color: #e5e7eb;
            font-style: italic;
        }

        .stage-kv {
            display: grid;
            grid-template-columns: 140px 1fr;
            gap: 12px;
            padding: 8px 0;
            border-top: 1px solid #f1f5f9;
            font-size: 13px;
            align-items: start;
        }
        .stage-kv:first-of-type { border-top: 0; padding-top: 0; }
        .stage-kv-label {
            color: #64748b;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            padding-top: 2px;
        }
        .stage-kv-value {
            color: #1f2d3d;
            min-width: 0;
            word-break: break-word;
        }
        .stage-kv-value code {
            background: #f1f5f9;
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 12.5px;
            border: 1px solid #e2e8f0;
        }

        .stage-list {
            margin: 0;
            padding-left: 18px;
            color: #1f2d3d;
        }
        .stage-list li { margin: 3px 0; }

        .stage-tag {
            display: inline-block;
            background: #eef2f6;
            color: #334155;
            font-size: 12px;
            padding: 2px 8px;
            border-radius: 4px;
            margin: 2px 4px 2px 0;
            border: 1px solid #e1e6ec;
        }
        .stage-tag-kind {
            color: #94a3b8;
            font-size: 11px;
            margin-left: 4px;
        }

        .stage-mono {
            font-family: "SF Mono", Menlo, Consolas, monospace;
            font-size: 12.5px;
        }
        .stage-meta { color: #64748b; }

        .stage-hits {
            width: 100%;
            border-collapse: collapse;
            font-size: 12.5px;
            background: white;
            border: 1px solid #e1e6ec;
            border-radius: 4px;
            overflow: hidden;
        }
        .stage-hits thead th {
            background: #f4f6f7;
            color: #475569;
            text-align: left;
            font-weight: 600;
            font-size: 11.5px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            padding: 6px 10px;
            border-bottom: 1px solid #e1e6ec;
        }
        .stage-hits tbody td {
            padding: 6px 10px;
            border-top: 1px solid #f1f5f9;
            vertical-align: top;
        }
        .stage-hits tbody tr:first-child td { border-top: 0; }

        .stage-raw {
            margin-top: 14px;
            border-top: 1px dashed #e1e6ec;
            padding-top: 10px;
        }
        .stage-raw summary {
            cursor: pointer;
            font-size: 12px;
            color: #64748b;
            user-select: none;
            font-weight: 500;
        }
        .stage-raw summary:hover { color: #1f2d3d; }
        .stage-raw[open] summary { margin-bottom: 8px; }
        .stage-json {
            background: #1e293b;
            color: #e2e8f0;
            padding: 12px 14px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: "SF Mono", Menlo, Consolas, monospace;
            font-size: 11.5px;
            line-height: 1.5;
            margin: 0;
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
            border: 1px solid #e1e6ec;
            border-radius: 8px;
            padding: 12px 14px 10px;
            margin-top: 18px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }
        .viz-section h2 {
            font-size: 15px;
            color: #1f2d3d;
            margin: 0 0 2px;
            font-weight: 600;
        }
        .viz-section .viz-sub {
            font-size: 12px;
            color: #64748b;
            margin-bottom: 10px;
            line-height: 1.4;
        }
        .viz-container {
            overflow-x: auto;
            background:
                radial-gradient(circle at 1px 1px, #e2e8f0 1px, transparent 0);
            background-size: 22px 22px;
            background-color: #fafbfc;
            border: 1px solid #eef2f6;
            border-radius: 8px;
            padding: 24px 18px;
            min-height: 240px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .viz-container .mermaid { text-align: center; width: 100%; }
        .viz-container svg { max-width: 100%; height: auto; display: block; margin: 0 auto; }
        /* Mermaid node typography */
        .viz-container .nodeLabel,
        .viz-container .label,
        .viz-container foreignObject div {
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
            font-size: 12.5px !important;
            line-height: 1.45 !important;
        }
        .viz-container .nodeLabel b { font-weight: 600; }
        .viz-container .nodeLabel .vz-sub {
            display: block;
            font-size: 11px;
            opacity: 0.78;
            margin-top: 2px;
            font-weight: 400;
        }
        .viz-container .nodeLabel .vz-time {
            display: block;
            font-size: 10.5px;
            margin-top: 3px;
            opacity: 0.7;
            font-family: "SF Mono", Menlo, Consolas, monospace;
        }
        /* Subtle drop shadows for nodes */
        .viz-container .node rect,
        .viz-container .node polygon,
        .viz-container .node path {
            filter: drop-shadow(0 1px 2px rgba(15, 23, 42, 0.08));
        }
        /* Edges: thinner, softer color */
        .viz-container .edgePath path {
            stroke-width: 1.4px !important;
            stroke: #94a3b8 !important;
        }
        .viz-container .arrowheadPath,
        .viz-container marker path {
            fill: #94a3b8 !important;
            stroke: #94a3b8 !important;
        }

        .viz-empty {
            color: #94a3b8;
            text-align: center;
            font-size: 13px;
            padding: 30px;
            font-style: italic;
            width: 100%;
        }

        .viz-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 10px 18px;
            margin-top: 14px;
            padding: 11px 14px;
            background: #fafbfc;
            border: 1px solid #eef2f6;
            border-radius: 6px;
            font-size: 12px;
        }
        .viz-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: #475569;
        }
        .viz-legend-swatch {
            width: 12px;
            height: 12px;
            border-radius: 3px;
            border: 1px solid rgba(15, 23, 42, 0.12);
            flex-shrink: 0;
        }

        /* ─── Model info panel ────────────────────────────────────────── */
        .model-panel {
            margin-top: 14px;
            border: 1px solid #dce1e7;
            border-radius: 6px;
            overflow: hidden;
            font-size: 12.5px;
        }
        .model-panel-header {
            display: flex;
            align-items: center;
            gap: 6px;
            background: #f4f6f7;
            padding: 6px 12px;
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
        <div class="container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>specGPT</h1>
                    <p>Ask questions about NVMe specifications. See exactly how the system found the answer.</p>
                </div>
                <form method="post" action="/logout" style="margin: 0;">
                    <button type="submit" class="config-toggle" style="background:#34495e;">Sign out</button>
                </form>
            </div>
            <div id="agent-strip" class="agent-strip" aria-label="Agent activity">
                <span class="agent-strip-label">Agent</span>
                <span class="agent-strip-empty">Run a query to see gap hints and one-click follow-ups here.</span>
            </div>
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
                    <div class="config-item config-item-wide">
                        <label>Agentic LLM</label>
                        <select id="config-agentic_model">
                            <option value="claude-sonnet-4-5">Claude Sonnet 4.5</option>
                            <option value="claude-sonnet-4-6">Claude Sonnet 4.6</option>
                            <option value="claude-opus-4-7" selected>Claude Opus 4.7 (default)</option>
                        </select>
                    </div>
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
                    <div class="config-item config-item-wide">
                        <label>Regular LLM</label>
                        <select id="config-llm_model">
                            <option value="claude-haiku-4-5-20251001">Claude Haiku 4.5 (fastest, cheapest)</option>
                            <option value="claude-sonnet-4-5" selected>Claude Sonnet 4.5 (default)</option>
                            <option value="claude-sonnet-4-6">Claude Sonnet 4.6 (newer Sonnet)</option>
                            <option value="claude-opus-4-7">Claude Opus 4.7 (most capable)</option>
                        </select>
                    </div>
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
                    <div class="config-item">
                        <label><input type="checkbox" id="config-auto_gap_check" checked> Auto Gap Check</label>
                    </div>
                </div>
            </div>

            <div id="cost-estimator" class="cost-estimator" onclick="toggleCostBreakdown(event)" role="button" aria-expanded="false" tabindex="0">
                <div class="cost-summary">
                    <span class="cost-label">Est. cost / query</span>
                    <span class="cost-total" id="cost-total">$0.00</span>
                    <span class="cost-context" id="cost-context"></span>
                    <span class="cost-toggle" id="cost-toggle">▼</span>
                </div>
                <div class="cost-breakdown" id="cost-breakdown"></div>
            </div>
        </div>

        <div id="results" class="hidden">
            <div id="error" class="error hidden"></div>

            <div id="loading" class="loading hidden" role="status" aria-live="polite">
                <div class="loading-spinner" aria-hidden="true"></div>
                <div class="loading-body">
                    <div class="loading-title" id="loading-title">Running pipeline</div>
                    <div class="loading-meta">
                        <span id="loading-elapsed">0.0s elapsed</span>
                        <span class="loading-dots"></span>
                    </div>
                </div>
                <button type="button" class="loading-cancel" id="loading-cancel">Cancel</button>
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
                        Each color is a stage family. Branches show per-sub-query
                        retrieval (semantic · keyword · BM25); all paths merge
                        through rank fusion → dedup → rerank → generation.
                    </div>
                    <div id="pipeline-viz" class="viz-container">
                        <div class="viz-empty">
                            Run a query to see the pipeline flow.
                        </div>
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

                <div class="model-panel" id="model-panel">
                    <button class="model-panel-header" onclick="toggleModelPanel()">
                        Models &amp; Cost
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
                theme: "base",
                themeVariables: {
                    fontFamily: "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
                    fontSize: "13px",
                    primaryColor: "#fafbfc",
                    primaryTextColor: "#1f2d3d",
                    primaryBorderColor: "#cbd5e1",
                    lineColor: "#94a3b8",
                    textColor: "#1f2d3d",
                    nodeBorder: "#cbd5e1",
                    mainBkg: "#fafbfc",
                    clusterBkg: "#f8fafc",
                    clusterBorder: "#e2e8f0",
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

        // Per-1M pricing for each selectable Claude model. Kept client-side so
        // the model panel can update the cost calc as soon as the user picks
        // a different model — no extra round-trip needed. Numbers track
        // Anthropic's public pricing pages at the time of writing.
        const MODEL_PRICING = {
            "claude-haiku-4-5-20251001":  {in: 1.0,  out: 5.0},
            "claude-sonnet-4-5":           {in: 3.0,  out: 15.0},
            "claude-sonnet-4-6":           {in: 3.0,  out: 15.0},
            "claude-opus-4-7":             {in: 15.0, out: 75.0},
        };

        // Overlay the model selectors onto `_modelsData` so the model panel +
        // cost calc reflect whatever the user picked. No-op until both the
        // /api/models response and the selectors are in the DOM.
        function _applySelectedModels() {
            if (!_modelsData) return;
            const llmEl = document.getElementById("config-llm_model");
            const agEl  = document.getElementById("config-agentic_model");
            if (llmEl && _modelsData.llm) {
                const id = llmEl.value;
                const p  = MODEL_PRICING[id];
                _modelsData.llm.model = id;
                if (p) {
                    _modelsData.llm.price_per_1m_input  = p.in;
                    _modelsData.llm.price_per_1m_output = p.out;
                }
            }
            if (agEl && _modelsData.agentic_llm) {
                const id = agEl.value;
                const p  = MODEL_PRICING[id];
                _modelsData.agentic_llm.model = id;
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

        // Re-render the model panel whenever the user picks a different model
        // so the table + cost-per-query badge reflect the live selection.
        ["config-llm_model", "config-agentic_model"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener("change", () => renderModelTable(_lastIsAgentic));
        });

        // ─── Cost estimator ───────────────────────────────────────────────
        // Rough, deliberately optimistic-side estimate so the user sees the
        // approximate cost of the *next* query before running it. Updates
        // live on every config change. Assumes the LLM call dominates cost;
        // embedding + reranker + structured lookup are essentially free.
        //
        // Token-budget assumptions (calibrated against typical NVMe queries):
        //   • System prompt:    ~900 tokens (load-bearing instructions)
        //   • User query:       ~60 tokens
        //   • Avg chunk:        ~450 tokens (text + section title + header)
        //   • Gap-analysis IO:  ~2200 in / ~400 out
        //   • Targeted-fetch:   ~1200 in / ~250 out
        // Numbers are coarse but consistent — the goal is "is this $0.01 or
        // $0.50?", not three-decimal precision.
        const COST_ASSUMPTIONS = {
            sys_tokens: 900,
            query_tokens: 60,
            avg_chunk_tokens: 450,
            gap_in: 2200, gap_out: 400,
            tfetch_in: 1200, tfetch_out: 250,
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
                llm_model:                  str("config-llm_model", "claude-sonnet-4-5"),
                agentic_model:              str("config-agentic_model", "claude-opus-4-7"),
                final_rerank_topk:          num("config-final_rerank_topk", 7),
                auto_gap_check:             chk("config-auto_gap_check"),
                agentic_rerank_topk:        num("config-agentic_rerank_topk", 14),
                agentic_max_context_tokens: num("config-agentic_max_context_tokens", 16000),
                agentic_max_output_tokens:  num("config-agentic_max_output_tokens", 2048),
                agentic_max_followups:      num("config-agentic_max_followups", 3),
                agentic_targeted_fetch:     chk("config-agentic_targeted_fetch"),
                agentic_recursive:          chk("config-agentic_recursive"),
                agentic_max_iterations:     num("config-agentic_max_iterations", 5),
            };
        }

        function estimateCost(cfg) {
            const A = COST_ASSUMPTIONS;
            const regPrice = _modelPrice(cfg.llm_model,     {in: 3,  out: 15});
            const agPrice  = _modelPrice(cfg.agentic_model, {in: 15, out: 75});
            const rows = [];

            // Embedding the query.
            const embIn = cfg.query_tokens || A.query_tokens;
            const embCost = (embIn / 1e6) * A.embedding_price_per_1m;
            rows.push({
                name: "Query embedding",
                sub: `Voyage · ~${embIn} tok`,
                value: embCost,
            });

            // First-pass generation (always runs).
            const normalCtxBudget = 4000; // matches PipelineConfig.llm_max_context_tokens default
            const normalCtxTok = Math.min(cfg.final_rerank_topk * A.avg_chunk_tokens, normalCtxBudget);
            const normalIn  = A.sys_tokens + A.query_tokens + normalCtxTok;
            const normalOut = 1024; // matches PipelineConfig.llm_max_output_tokens default
            const normalCost = _llmCallCost(normalIn, normalOut, regPrice);
            rows.push({
                name: "Generate (regular)",
                sub: `${cfg.llm_model} · ${cfg.final_rerank_topk} chunks → ~${normalIn.toLocaleString()} in / ~${normalOut} out`,
                value: normalCost,
            });

            // Optional auto-gap-check (regular mode only — agentic loop has its own).
            if (cfg.auto_gap_check && !cfg.agentic) {
                const gapCost = _llmCallCost(A.gap_in, A.gap_out, regPrice);
                rows.push({
                    name: "Auto gap check",
                    sub: `~${A.gap_in.toLocaleString()} in / ~${A.gap_out} out`,
                    value: gapCost,
                });
            }

            // Agentic loop: gap-analysis + optional targeted-fetch + regenerate
            // each iteration. Recursive multiplies by max_iterations (worst case).
            if (cfg.agentic) {
                const iters = cfg.agentic_recursive ? Math.max(1, cfg.agentic_max_iterations) : 1;
                const agCtxTok = Math.min(cfg.agentic_rerank_topk * A.avg_chunk_tokens, cfg.agentic_max_context_tokens);
                const agIn  = A.sys_tokens + A.query_tokens + agCtxTok;
                const agOut = cfg.agentic_max_output_tokens;

                const gapCostOne   = _llmCallCost(A.gap_in, A.gap_out, regPrice);
                const tfetchCostOne = cfg.agentic_targeted_fetch ? _llmCallCost(A.tfetch_in, A.tfetch_out, regPrice) : 0;
                const regenCostOne = _llmCallCost(agIn, agOut, agPrice);
                const perIter = gapCostOne + tfetchCostOne + regenCostOne;
                const agTotal = perIter * iters;

                rows.push({
                    name: "Agentic gap analysis",
                    sub: `${iters}× ~${A.gap_in.toLocaleString()} in / ~${A.gap_out} out`,
                    value: gapCostOne * iters,
                });
                if (cfg.agentic_targeted_fetch) {
                    rows.push({
                        name: "Targeted-fetch parse",
                        sub: `${iters}× ~${A.tfetch_in.toLocaleString()} in / ~${A.tfetch_out} out`,
                        value: tfetchCostOne * iters,
                    });
                }
                rows.push({
                    name: "Regenerate (agentic)",
                    sub: `${cfg.agentic_model} · ${iters}× ${cfg.agentic_rerank_topk} chunks → ~${agIn.toLocaleString()} in / ~${agOut} out`,
                    value: regenCostOne * iters,
                });
                if (iters > 1) {
                    rows.push({
                        name: "Iterations",
                        sub: `recursive · up to ${iters} passes`,
                        value: null,
                    });
                }
            }

            const total = rows.reduce((s, r) => s + (typeof r.value === "number" ? r.value : 0), 0);
            return { rows, total, cfg };
        }

        function renderCostEstimate() {
            const cfg = _readCostInputs();
            const est = estimateCost(cfg);

            const totalEl = document.getElementById("cost-total");
            const ctxEl   = document.getElementById("cost-context");
            const breakEl = document.getElementById("cost-breakdown");
            if (!totalEl || !ctxEl || !breakEl) return;

            totalEl.textContent = "~" + _fmtCostShort(est.total);
            totalEl.classList.remove("cost-warn", "cost-high");
            if (est.total >= 1.0)      totalEl.classList.add("cost-high");
            else if (est.total >= 0.10) totalEl.classList.add("cost-warn");

            const ctxParts = [];
            ctxParts.push(cfg.agentic ? "Agentic mode" : "Regular mode");
            ctxParts.push(cfg.agentic ? cfg.agentic_model : cfg.llm_model);
            if (cfg.agentic && cfg.agentic_recursive) ctxParts.push(`up to ${cfg.agentic_max_iterations}× iter`);
            ctxEl.textContent = " · " + ctxParts.join(" · ");

            breakEl.innerHTML = est.rows.map(r => `
                <div class="cost-row">
                    <div class="cost-row-name">${escapeHtml(r.name)}<small>${escapeHtml(r.sub)}</small></div>
                    <div class="cost-row-value">${typeof r.value === "number" ? _fmtCostShort(r.value) : ""}</div>
                </div>
            `).join("") + `
                <div class="cost-row cost-row-total">
                    <div class="cost-row-name"><b>Total</b></div>
                    <div class="cost-row-value">${_fmtCostShort(est.total)}</div>
                </div>
                <div class="cost-disclaimer">
                    Rough estimate. Assumes ~${COST_ASSUMPTIONS.avg_chunk_tokens} tok per chunk and a ~${COST_ASSUMPTIONS.sys_tokens}-token system prompt. Real usage depends on query complexity and chunk size, so expect about a 30% variance. Embedding/rerank costs are negligible.
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

        // Compose a node label with title (bold) + optional subtitle (italic)
        // + time. Keeps the visual rhythm consistent across all node types.
        function _label(title, subtitle, stage) {
            const parts = [`<b>${title}</b>`];
            if (subtitle) parts.push(`<i>${subtitle}</i>`);
            const t = _ms(stage);
            if (t) parts.push(t);
            return parts.join("<br/>");
        }

        function buildMermaidFromTrace(trace, query) {
            // Recursive agentic stages get an `.iterN` suffix — collapse to the
            // base name so a single visualization still shows the agentic loop
            // (using the latest iteration's data).
            const stages = {};
            for (const s of trace) {
                stages[s.stage] = s;
                const base = s.stage.replace(/\\.iter\\d+$/, "");
                if (base !== s.stage) stages[base] = s;
            }

            const L = ["flowchart TD"];
            L.push(`  Q["${_label("Query", _vizText(query, 60), null)}"]:::input`);

            // Refine-mode trace: /api/refine reused a prior /api/query's
            // first-pass state, so Stages 1–4 didn't run. Emit a "Resume"
            // marker so the diagram still has a visible upstream node feeding
            // into the agentic branch below. Without this we bail with just
            // the Query node and the user sees an empty canvas.
            const refineSeed = stages["refine.seed"];
            const qp = stages.query_processor;
            if (!qp && !refineSeed) {
                L.push("  classDef input fill:#1e293b,color:#fff,stroke:#0f172a,stroke-width:2px,rx:8,ry:8");
                return L.join("\\n");
            }
            if (refineSeed && !qp) {
                const ddCount = (refineSeed.output && refineSeed.output.deduplicated_count) || 0;
                const ctxCount = (refineSeed.output && refineSeed.output.context_chunk_count) || 0;
                L.push(`  RESUME[/"${_label("Resume from cache", ddCount + " pooled chunks · " + ctxCount + " in prior context", refineSeed)}"/]:::stage_resume`);
                L.push("  Q --> RESUME");
            }
            if (qp) {
                const qpType = _vizText(qp.output.type, 20);
                const qpEnts = (qp.output.entities || []).length;
                const qpSubs = (qp.output.sub_queries || []).length;
                const qpSub = `${qpType ? "type: " + qpType : ""}${qpEnts ? " · " + qpEnts + " entit" + (qpEnts===1?"y":"ies") : ""}${qpSubs ? " · " + qpSubs + " sub-quer" + (qpSubs===1?"y":"ies") : ""}`.replace(/^ · /, "");
                L.push(`  QP["${_label("Understand query", qpSub, qp)}"]:::stage_qp`);
                L.push("  Q --> QP");
            }

            // Structured lookup — side branch that merges back into dedup
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
                L.push(`  SQ${i}{{"${_label("Sub-query " + (i+1), _vizText(sqText, 60), null)}"}}:::stage_subq`);
                L.push(`  QP --> SQ${i}`);

                if (v) {
                    L.push(`  V${i}["${_label("Semantic search", (v.output.count || 0) + " hits · Voyage", v)}"]:::stage_vector`);
                    L.push(`  SQ${i} --> V${i}`);
                    if (rrf) L.push(`  V${i} --> RRF`);
                }
                if (t) {
                    L.push(`  T${i}["${_label("Keyword search", (t.output.count || 0) + " hits · tsvector", t)}"]:::stage_tsv`);
                    L.push(`  SQ${i} --> T${i}`);
                    if (rrf) L.push(`  T${i} --> RRF`);
                }
                if (b) {
                    L.push(`  B${i}["${_label("BM25 search", (b.output.count || 0) + " hits · Okapi", b)}"]:::stage_bm25`);
                    L.push(`  SQ${i} --> B${i}`);
                    if (rrf) L.push(`  B${i} --> RRF`);
                }
            }

            if (rrf) {
                L.push(`  RRF["${_label("Fuse results", (rrf.output.count || 0) + " merged · RRF", rrf)}"]:::stage_rrf`);
            }

            const dd = stages.result_dedup;
            if (dd) {
                L.push(`  DEDUP["${_label("Deduplicate", (dd.output.deduped_count || 0) + " unique chunks", dd)}"]:::stage_dedup`);
                if (rrf) L.push("  RRF --> DEDUP");
                if (slActive) L.push("  SL --> DEDUP");
            }

            const rr = stages.final_rerank;
            if (rr) {
                L.push(`  RR["${_label("Rerank", "top " + (rr.output.count || 0) + " · cross-encoder", rr)}"]:::stage_rerank`);
                if (dd) L.push("  DEDUP --> RR");
            }

            const gen = stages.generation;
            if (gen) {
                const cits = (gen.output.citation_count !== undefined) ? gen.output.citation_count : 0;
                const ans = gen.output.answer_length || 0;
                L.push(`  GEN["${_label("Generate answer", ans.toLocaleString() + " chars · " + cits + " citation" + (cits===1?"":"s") + " · Claude", gen)}"]:::stage_gen`);
                if (rr) L.push("  RR --> GEN");
            }

            // ─── Agentic refinement branch (only present when agentic=true) ───
            const gap = stages["agentic.gap_analysis"];
            const tfetch = stages["agentic.targeted_fetch"];
            const ag_rr = stages["agentic.rerank"];
            const ag_gen = stages["agentic.regenerate"];
            // In refine mode there's no GEN node, so the final-answer arrow
            // falls back to RESUME until the agentic regen succeeds and
            // promotes itself to GEN2.
            let agAnswerNode = gen ? "GEN" : (refineSeed ? "RESUME" : "GEN");
            if (gap) {
                const needs = gap.output && gap.output.needs_followup;
                const reason = _vizText((gap.output && gap.output.reason) || "", 60);
                const gapSub = needs ? ("needs follow-up: " + reason) : "answer covers the question";
                L.push(`  GAP{"${_label("Gap analysis", gapSub, gap)}"}:::stage_gap`);
                // Normal path: first-pass Sonnet → gap. Refine path: cached
                // first-pass (via RESUME) → gap (no GEN node to wire from).
                if (gen) L.push("  GEN --> GAP");
                else if (refineSeed) L.push("  RESUME --> GAP");

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
                        L.push(`  TFETCH[/"${_label("Targeted fetch", "asked: " + reqSummary + " · got: " + got, tfetch)}"/]:::stage_tfetch`);
                        L.push("  GAP --> TFETCH");
                        if (ag_rr) L.push("  TFETCH --> RR2");
                    }

                    // (b) Per-followup natural-language retrieval branches.
                    // Each follow-up gets its own subgraph showing the full
                    // decompose → vector/keyword/bm25 per sub-query → mini-RRF
                    // path. Sub-stages are namespaced as
                    // `agentic.followup_q{fi}.hybrid_search.*` server-side
                    // so they don't collide with the main query's stages.
                    const fqIds = new Set();
                    for (const s of trace) {
                        const m = s.stage.match(/^agentic\\.followup_search_q(\\d+)(\\.iter\\d+)?$/);
                        if (m) fqIds.add(parseInt(m[1], 10));
                    }
                    for (const i of [...fqIds].sort((a, b) => a - b)) {
                        const fq = stages[`agentic.followup_search_q${i}`];
                        const decomp = stages[`agentic.followup_decomp_q${i}`];
                        const qText = (fq && fq.input && fq.input.query) || `gap-q${i}`;
                        const chunks = (fq && fq.output && fq.output.chunk_count) || 0;

                        // Collect this follow-up's sub-query indices from the
                        // namespaced hybrid_search stages.
                        const nsPrefix = `agentic.followup_q${i}.`;
                        const fqSubIds = new Set();
                        for (const s of trace) {
                            // Strip the .iterN suffix if present, then match.
                            const baseStage = s.stage.replace(/\\.iter\\d+$/, "");
                            if (!baseStage.startsWith(nsPrefix)) continue;
                            const m2 = baseStage.slice(nsPrefix.length).match(/^hybrid_search\\.\\w+_q(\\d+)$/);
                            if (m2) fqSubIds.add(parseInt(m2[1], 10));
                        }
                        const fqSorted = [...fqSubIds].sort((a, b) => a - b);
                        const fqRrf = stages[`${nsPrefix}hybrid_search.rrf_merge`];

                        // Open a subgraph for this follow-up.
                        L.push(`  subgraph FQGRP${i}["Follow-up ${i+1}: ${_vizText(qText, 56)} · ${chunks} chunks"]`);
                        // Follow-up header node (clickable target for the GAP edge).
                        const decompSub = decomp && decomp.output && decomp.output.sub_queries
                            ? `${decomp.output.sub_queries.length} sub-quer${decomp.output.sub_queries.length===1?"y":"ies"}`
                            : "verbatim";
                        L.push(`    FQ${i}{{"${_label("Decompose", decompSub, decomp)}"}}:::stage_followup`);

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
                            }
                            if (t) {
                                L.push(`    FQ${i}T${j}["${_label("Keyword", (t.output.count || 0) + " hits", t)}"]:::stage_tsv`);
                                L.push(`    FQ${i}SQ${j} --> FQ${i}T${j}`);
                                if (fqRrf) L.push(`    FQ${i}T${j} --> FQ${i}RRF`);
                            }
                            if (b) {
                                L.push(`    FQ${i}B${j}["${_label("BM25", (b.output.count || 0) + " hits", b)}"]:::stage_bm25`);
                                L.push(`    FQ${i}SQ${j} --> FQ${i}B${j}`);
                                if (fqRrf) L.push(`    FQ${i}B${j} --> FQ${i}RRF`);
                            }
                        }
                        if (fqRrf) {
                            L.push(`    FQ${i}RRF["${_label("Fuse", (fqRrf.output.count || 0) + " merged", fqRrf)}"]:::stage_rrf`);
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
                        if (!errType) agAnswerNode = "GEN2";
                    }
                }
            }

            L.push('  ANS(["<b>Final answer</b>"]):::output');
            L.push(`  ${agAnswerNode} --> ANS`);

            // ─── Color palette ───────────────────────────────────────────────
            // Soft pastel fills paired with darker strokes and dark text — the
            // pre-search "thinking" stages run cool (blues/greens), retrieval
            // sweeps through warm hues, and the agentic loop uses purples to
            // signal "second pass". Rounded corners + 1.4px strokes give the
            // diagram a softer, more modern look than flat saturated boxes.
            L.push("  classDef input    fill:#1e293b,color:#fff,stroke:#0f172a,stroke-width:2px,rx:9,ry:9");
            L.push("  classDef output   fill:#10b981,color:#fff,stroke:#047857,stroke-width:2px,rx:10,ry:10");
            L.push("  classDef stage_qp     fill:#ede9fe,color:#4c1d95,stroke:#a78bfa,stroke-width:1.4px,rx:7,ry:7");
            L.push("  classDef stage_struct fill:#d1fae5,color:#065f46,stroke:#34d399,stroke-width:1.4px,rx:7,ry:7");
            L.push("  classDef stage_skipped fill:#f3f4f6,color:#6b7280,stroke:#cbd5e1,stroke-width:1.2px,stroke-dasharray:4 3,rx:7,ry:7");
            L.push("  classDef stage_subq   fill:#e0f2fe,color:#0c4a6e,stroke:#7dd3fc,stroke-width:1.4px");
            L.push("  classDef stage_vector fill:#dbeafe,color:#1e3a8a,stroke:#93c5fd,stroke-width:1.2px,rx:6,ry:6");
            L.push("  classDef stage_tsv    fill:#cffafe,color:#155e75,stroke:#67e8f9,stroke-width:1.2px,rx:6,ry:6");
            L.push("  classDef stage_bm25   fill:#ccfbf1,color:#115e59,stroke:#5eead4,stroke-width:1.2px,rx:6,ry:6");
            L.push("  classDef stage_rrf    fill:#fef3c7,color:#854d0e,stroke:#fbbf24,stroke-width:1.4px,rx:7,ry:7");
            L.push("  classDef stage_dedup  fill:#fed7aa,color:#7c2d12,stroke:#fb923c,stroke-width:1.4px,rx:7,ry:7");
            L.push("  classDef stage_rerank fill:#fecaca,color:#7f1d1d,stroke:#fca5a5,stroke-width:1.4px,rx:7,ry:7");
            L.push("  classDef stage_gen    fill:#1e40af,color:#fff,stroke:#1e3a8a,stroke-width:2px,rx:8,ry:8");
            // Refine fast-path marker — slate, distinct from any other stage.
            L.push("  classDef stage_resume   fill:#e2e8f0,color:#0f172a,stroke:#64748b,stroke-width:1.6px,stroke-dasharray:5 3,rx:7,ry:7");
            // Agentic branch — purples set it apart from the main path.
            L.push("  classDef stage_gap      fill:#f3e8ff,color:#6b21a8,stroke:#c084fc,stroke-width:1.4px");
            L.push("  classDef stage_followup fill:#e9d5ff,color:#581c87,stroke:#a855f7,stroke-width:1.4px");
            L.push("  classDef stage_agen     fill:#581c87,color:#fff,stroke:#3b0764,stroke-width:2px,rx:8,ry:8");
            // Targeted fetch — teal family to distinguish "direct lookup"
            // from "natural-language search".
            L.push("  classDef stage_tfetch   fill:#ccfbf1,color:#115e59,stroke:#14b8a6,stroke-width:1.4px,rx:7,ry:7");

            return L.join("\\n");
        }

        let _vizCounter = 0;
        async function renderPipelineViz(trace, query) {
            const host = document.getElementById("pipeline-viz");
            const legend = document.getElementById("pipeline-legend");
            if (legend) legend.style.display = "none";
            if (!trace || !trace.length) {
                host.innerHTML = '<div class="viz-empty">No pipeline trace returned. Set <code>DEBUG_PIPELINE=1</code> on the server to enable.</div>';
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
                if (legend) legend.style.display = "";
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
            if (searchBtn) searchBtn.disabled = false;
            _activeAbort = null;
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

            // Collect config
            const config = {
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
            };

            // Show loading
            resultsDiv.classList.remove("hidden");
            errorDiv.classList.add("hidden");
            answerSection.classList.add("hidden");

            const agentic = agenticToggle.checked;
            const title = agentic
                ? "Running agentic pipeline (this can take 30-60s)"
                : "Running pipeline";
            _startLoading(title);

            _activeAbort = new AbortController();
            try {
                const response = await fetch("/api/query", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query, config, debug: true, agentic }),
                    signal: _activeAbort.signal,
                });

                if (response.status === 401) {
                    // Session expired mid-flight; bounce to login.
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

            // Pipeline trace — render as human-readable cards (title + subtitle
            // + chips + key/value rows) with the raw JSON tucked behind a
            // "Show raw JSON" details toggle for power users.
            const stagesDiv = document.getElementById("pipeline-stages");
            if (data.pipeline_trace) {
                stagesDiv.innerHTML = data.pipeline_trace
                    .map((stage, idx) => renderStageCard(stage, idx))
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

            // Sidebar — surface gap_hint / agentic state.
            renderSidebar(data);

            answerSection.classList.remove("hidden");
        }

        // ─── Stage name → human-friendly display ──────────────────────────
        // Maps the internal stage identifier to {title, subtitle, group}. Sub-
        // query and iteration suffixes are stripped before lookup so e.g.
        // `hybrid_search.vector_search_q2.iter1` maps to the "Semantic search"
        // entry with " (sub-query 3)" and " · pass 2" annotations.
        const STAGE_INFO = {
            "query_processor":              {t: "Understand the question",       s: "Decompose into sub-queries and extract entities",        g: "normal"},
            "structured_lookup":            {t: "Structured lookup",             s: "Direct hit against named fields, tables, or figures",     g: "normal"},
            "hybrid_search.vector_search":  {t: "Semantic search",               s: "Embeddings via Voyage (vector similarity)",               g: "normal"},
            "hybrid_search.tsvector_search":{t: "Keyword search",                s: "Postgres full-text tsvector match",                       g: "normal"},
            "hybrid_search.bm25_search":    {t: "BM25 search",                   s: "Classic Okapi BM25 ranking over the corpus",              g: "normal"},
            "hybrid_search.rrf_merge":      {t: "Fuse search branches",          s: "Reciprocal Rank Fusion across semantic + keyword + BM25", g: "normal"},
            "hybrid_search.total":          {t: "Hybrid search (total)",         s: "End-to-end time across all retrieval branches",           g: "normal"},
            "result_dedup":                 {t: "Deduplicate combined pool",     s: "Merge structured + hybrid chunks, drop duplicates",        g: "normal"},
            "final_rerank":                 {t: "Rerank (cross-encoder)",        s: "Score each chunk against the query",                       g: "normal"},
            "generation":                   {t: "Generate answer",               s: "Synthesize the answer with citations (Claude)",            g: "normal"},
            "agentic.gap_analysis":         {t: "Agentic gap analysis",          s: "Does the answer fully cover the question?",                g: "agentic"},
            "agentic.targeted_fetch":       {t: "Targeted fetch",                s: "Pull figures, fields, or sections the model named",        g: "agentic"},
            "agentic.followup_search":      {t: "Follow-up search",              s: "Retrieve more chunks for a remaining gap",                 g: "agentic"},
            "agentic.rerank":               {t: "Rerank expanded pool",          s: "Rescore everything collected so far",                      g: "agentic"},
            "agentic.regenerate":           {t: "Regenerate answer",             s: "Synthesize the final answer with a larger context",        g: "agentic"},
            "agentic.cap_reached":          {t: "Iteration cap reached",         s: "Agentic loop stopped at its max-iterations setting",       g: "agentic"},
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
        function renderSidebar(data) {
            const strip = document.getElementById("agent-strip");
            if (!strip) return;

            const label = `<span class="agent-strip-label">Agent</span>`;

            // No query yet — keep the empty hint.
            if (!data) {
                strip.innerHTML = label + `<span class="agent-strip-empty">Run a query to see gap hints and one-click follow-ups here.</span>`;
                return;
            }

            const latency = `<span class="strip-latency">${data.latency_ms.toFixed(0)}ms</span>`;

            // Agentic mode: the loop already fetched gaps — just confirm.
            if (data.agentic) {
                strip.innerHTML = label + `
                    <span class="agent-strip-state state-agent"><span class="dot"></span><b>Agentic refinement ran</b></span>
                    <span class="agent-strip-reason">Gap-filling, follow-up retrieval, and Opus regeneration ran automatically.</span>
                    ${latency}
                `;
                return;
            }

            const gh = data.gap_hint;
            if (!gh) {
                strip.innerHTML = label + `
                    <span class="agent-strip-state"><span class="dot"></span><b>Gap check skipped</b></span>
                    <span class="agent-strip-reason">Auto gap check is disabled in config.</span>
                    ${latency}
                `;
                return;
            }

            if (!gh.needs_followup) {
                strip.innerHTML = label + `
                    <span class="agent-strip-state state-ok"><span class="dot"></span><b>Answer looks complete</b></span>
                    <span class="agent-strip-reason">${escapeHtml(gh.reason || "The model didn't request any additional context.")}</span>
                    ${latency}
                `;
                return;
            }

            // Has gaps — render the offer in the strip, with optional details
            // line below (chips for figures/fields/sections, follow-up queries).
            const req = gh.requested_resources || {};
            const figs = (req.figures  || []).slice(0, 6);
            const flds = (req.fields   || []).slice(0, 6);
            const secs = (req.sections || []).slice(0, 4);
            const qs   = (gh.queries   || []).slice(0, 3);

            const detailParts = [];
            if (figs.length) detailParts.push(`<span class="detail-group"><span class="detail-label">Figures</span><span class="agent-strip-chips">${figs.map(f => `<span class="agent-strip-chip">${escapeHtml(f)}</span>`).join("")}</span></span>`);
            if (flds.length) detailParts.push(`<span class="detail-group"><span class="detail-label">Fields</span><span class="agent-strip-chips">${flds.map(f => `<span class="agent-strip-chip">${escapeHtml(f)}</span>`).join("")}</span></span>`);
            if (secs.length) detailParts.push(`<span class="detail-group"><span class="detail-label">Sections</span><span class="agent-strip-chips">${secs.map(s => `<span class="agent-strip-chip chip-section">§${escapeHtml(s)}</span>`).join("")}</span></span>`);
            if (qs.length)   detailParts.push(`<span class="detail-group"><span class="detail-label">Queries</span><span class="agent-strip-chips">${qs.map(q => `<span class="agent-strip-chip">${escapeHtml(q)}</span>`).join("")}</span></span>`);
            const details = detailParts.length
                ? `<div class="agent-strip-details">${detailParts.join("")}</div>`
                : "";

            strip.innerHTML = label + `
                <span class="agent-strip-state state-warn"><span class="dot"></span><b>Model wants more context</b></span>
                <span class="agent-strip-reason">${escapeHtml(gh.reason || "The model identified gaps in the retrieved context.")}</span>
                ${latency}
                <span class="agent-strip-actions">
                    <button class="run-agentic-btn" id="run-agentic-btn">Run agentic refinement</button>
                </span>
                ${details}
            `;

            const btn = document.getElementById("run-agentic-btn");
            if (btn) {
                btn.addEventListener("click", () => {
                    btn.disabled = true;
                    btn.textContent = "Running…";
                    // Reflect agentic state in the toggle (purely cosmetic — the
                    // refine endpoint runs Stage 5 regardless).
                    if (!agenticToggle.checked) {
                        agenticToggle.checked = true;
                        agenticToggle.dispatchEvent(new Event("change"));
                    }
                    runRefine(data.request_id);
                });
            }
        }

        async function runRefine(requestId) {
            if (!requestId) {
                errorDiv.textContent = "Error: no request_id available for refine";
                errorDiv.classList.remove("hidden");
                return;
            }
            // Reuse the same config payload the user has set, so agentic_*
            // tweaks (max_followups, rerank_topk, recursive, etc.) apply.
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
                auto_gap_check: document.getElementById("config-auto_gap_check").checked,
            };

            errorDiv.classList.add("hidden");
            answerSection.classList.add("hidden");
            _startLoading("Refining answer (agentic Opus regen, 20-60s)");

            _activeAbort = new AbortController();
            try {
                const response = await fetch("/api/refine", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ request_id: requestId, config, debug: true }),
                    signal: _activeAbort.signal,
                });
                if (response.status === 401) {
                    window.location.href = "/login?next=" + encodeURIComponent(window.location.pathname);
                    return;
                }
                if (response.status === 404) {
                    // Cache evicted (older session or restart). Fall back to a
                    // full re-run so the user always has a working path. Hand
                    // off to runQuery which manages its own loading lifecycle.
                    _stopLoading();
                    runQuery();
                    return;
                }
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(typeof err.detail === "string" ? err.detail : "Refine failed");
                }
                const data = await response.json();
                displayResults(data);
            } catch (err) {
                if (err.name === "AbortError") return;
                errorDiv.textContent = `Error: ${err.message}`;
                errorDiv.classList.remove("hidden");
            } finally {
                _stopLoading();
            }
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
