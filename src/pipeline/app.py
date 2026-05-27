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
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #ffffff; --bg-soft: #fafaf9; --bg-muted: #f5f5f4;
      --border: #e7e5e4; --border-strong: #d6d3d1;
      --text: #1c1917; --text-muted: #57534e; --text-subtle: #78716c; --text-faint: #a8a29e;
      --accent: #1c1917; --danger: #b91c1c;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      --font-mono: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
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
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        /* ─── Minimalist internal-tool theme ────────────────────────────
           Adapted from the Claude Design handoff bundle: Linear/Vercel/Stripe
           dashboard aesthetic — single near-black accent on stone backgrounds,
           hairline borders, Inter UI + JetBrains Mono for technical data. All
           existing class names and HTML structure preserved. */
        :root {
            --bg: #ffffff;
            --bg-soft: #fafaf9;
            --bg-muted: #f5f5f4;
            --border: #e7e5e4;
            --border-strong: #d6d3d1;
            --text: #1c1917;
            --text-muted: #57534e;
            --text-subtle: #78716c;
            --text-faint: #a8a29e;
            --accent: #1c1917;
            --accent-soft: #f5f5f4;
            --danger: #b91c1c;
            --warn: #a16207;
            --ok: #15803d;
            --indigo: #4f46e5;
            --radius: 6px;
            --radius-sm: 4px;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: var(--font-sans);
            font-feature-settings: "cv11", "ss01";
            background: var(--bg-soft);
            color: var(--text);
            font-size: 14px;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            letter-spacing: -0.005em;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 0 20px;
        }

        /* ─── Topbar / header ─────────────────────────────────────────── */
        header {
            background: var(--bg);
            color: var(--text);
            border-bottom: 1px solid var(--border);
            padding: 14px 0 12px;
            margin-bottom: 18px;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        header h1 {
            font-size: 14px;
            font-weight: 600;
            letter-spacing: -0.01em;
            line-height: 1.3;
            margin: 0;
            color: var(--text);
        }
        header p {
            font-size: 12px;
            color: var(--text-subtle);
            line-height: 1.4;
            margin-top: 2px;
            opacity: 1;
        }

        /* ─── Composer / search section ───────────────────────────────── */
        .search-section {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: var(--shadow-sm);
            padding: 12px 14px;
            margin-bottom: 14px;
        }
        .search-box {
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }
        #query-input {
            flex: 1;
            padding: 9px 11px;
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            font-size: 14px;
            background: var(--bg);
            color: var(--text);
            font-family: inherit;
            letter-spacing: -0.005em;
            transition: border-color 0.12s, box-shadow 0.12s;
        }
        #query-input::placeholder { color: var(--text-faint); }
        #query-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(28,25,23,0.08);
        }

        button {
            height: 36px;
            padding: 0 14px;
            background: var(--accent);
            color: #fff;
            border: 1px solid var(--accent);
            border-radius: var(--radius-sm);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            font-family: inherit;
            letter-spacing: -0.005em;
            transition: background 0.12s, border-color 0.12s, color 0.12s;
            white-space: nowrap;
        }
        button:hover { background: #000; border-color: #000; }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }

        .config-toggle {
            background: var(--bg);
            color: var(--text-muted);
            border: 1px solid var(--border);
            font-weight: 500;
            font-size: 12.5px;
            padding: 0 12px;
            height: 36px;
        }
        .config-toggle:hover {
            background: var(--bg-muted);
            border-color: var(--border-strong);
            color: var(--text);
        }

        /* ─── Agentic toggle row ──────────────────────────────────────── */
        .agentic-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 8px;
            padding: 8px 12px;
            background: var(--bg-soft);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            font-size: 12.5px;
            color: var(--text-subtle);
        }
        .agentic-row label {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-weight: 500;
            color: var(--text);
        }
        .agentic-row input[type="checkbox"] {
            transform: scale(1.05);
            cursor: pointer;
            accent-color: var(--accent);
        }
        .agentic-row .agentic-hint {
            color: var(--text-subtle);
            font-weight: 400;
            font-size: 12px;
        }
        .agentic-row.active {
            background: var(--bg-muted);
            border-color: var(--border-strong);
        }

        .agentic-config {
            background: var(--bg-soft);
            padding: 12px 14px;
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            margin-top: 6px;
        }
        .agentic-config.hidden { display: none; }
        .agentic-config .config-item label { color: var(--text-subtle); }

        /* ─── Cost estimator ──────────────────────────────────────────── */
        .cost-estimator {
            margin-top: 10px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            font-size: 13px;
            overflow: hidden;
        }
        .cost-summary {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            cursor: pointer;
            user-select: none;
            background: var(--bg);
            transition: background 0.12s;
        }
        .cost-summary:hover { background: var(--bg-soft); }
        .cost-icon { display: none; }
        .cost-label {
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-faint);
            font-weight: 600;
        }
        .cost-total {
            font-family: var(--font-mono);
            font-weight: 500;
            color: var(--text);
            font-size: 13px;
        }
        .cost-total.cost-warn { color: var(--warn); }
        .cost-total.cost-high { color: var(--danger); }
        .cost-context {
            color: var(--text-subtle);
            font-size: 11.5px;
        }
        .cost-toggle {
            margin-left: auto;
            color: var(--text-faint);
            font-size: 10px;
            transition: transform 0.15s;
        }
        .cost-estimator.open .cost-toggle { transform: rotate(180deg); }
        .cost-breakdown {
            display: none;
            padding: 6px 12px 12px;
            border-top: 1px solid var(--border);
            background: var(--bg-soft);
        }
        .cost-estimator.open .cost-breakdown { display: block; }
        .cost-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 5px 0;
            border-bottom: 1px dashed var(--border);
            font-size: 12.5px;
        }
        .cost-row:last-child { border-bottom: 0; }
        .cost-row-name { color: var(--text-muted); }
        .cost-row-name small {
            color: var(--text-faint);
            margin-left: 8px;
            font-size: 11.5px;
        }
        .cost-row-value {
            font-family: var(--font-mono);
            color: var(--text);
            font-weight: 500;
            font-size: 12px;
        }
        .cost-row.cost-row-total {
            margin-top: 5px;
            padding-top: 9px;
            border-top: 1px solid var(--border);
            border-bottom: 0;
        }
        .cost-row.cost-row-total .cost-row-value { color: var(--text); font-weight: 600; }
        .cost-row.cost-row-total .cost-row-name b { color: var(--text); }
        .cost-disclaimer {
            margin-top: 10px;
            color: var(--text-faint);
            font-size: 11.5px;
            font-style: normal;
            line-height: 1.5;
        }

        /* ─── Agent activity strip (now sits in the light header) ─────── */
        .agent-strip {
            margin-top: 12px;
            padding: 8px 12px;
            background: var(--bg-soft);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 8px 14px;
            color: var(--text-muted);
            font-size: 12.5px;
            line-height: 1.4;
            min-height: 36px;
        }
        .agent-strip-label {
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-faint);
            font-weight: 600;
            margin-right: 4px;
            padding-right: 12px;
            border-right: 1px solid var(--border);
        }
        .agent-strip-empty {
            color: var(--text-subtle);
            font-style: normal;
            font-size: 12.5px;
        }
        .agent-strip-state {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            font-size: 12.5px;
            color: var(--text-muted);
            background: var(--bg);
            padding: 3px 10px;
            border-radius: 11px;
            border: 1px solid var(--border);
        }
        .agent-strip-state .dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            background: var(--text-faint);
            display: inline-block;
        }
        .agent-strip-state.state-ok    .dot { background: var(--ok); }
        .agent-strip-state.state-warn  .dot { background: var(--warn); }
        .agent-strip-state.state-agent .dot { background: var(--indigo); }
        .agent-strip-state.state-error .dot { background: var(--danger); }
        .agent-strip-state b { color: var(--text); font-weight: 600; }
        .agent-strip-state .strip-latency {
            font-family: var(--font-mono);
            color: var(--text-subtle);
        }
        .agent-strip-reason {
            flex: 1;
            min-width: 200px;
            color: var(--text-subtle);
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
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 11px;
            padding: 2px 7px;
            border-radius: 10px;
            font-family: var(--font-mono);
        }
        .agent-strip-chip.chip-section { color: var(--text-muted); }
        .agent-strip-details {
            flex-basis: 100%;
            margin-top: 6px;
            padding-top: 8px;
            border-top: 1px dashed var(--border);
            display: flex;
            flex-wrap: wrap;
            gap: 8px 16px;
            font-size: 12px;
            color: var(--text-subtle);
        }
        .agent-strip-details .detail-group {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .agent-strip-details .detail-label {
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-faint);
            font-weight: 600;
        }
        .run-agentic-btn {
            height: 28px;
            padding: 0 12px;
            background: var(--accent);
            color: #fff;
            font-size: 12.5px;
            font-weight: 500;
            border: 1px solid var(--accent);
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-family: inherit;
            letter-spacing: -0.005em;
        }
        .run-agentic-btn:hover { background: #000; border-color: #000; }
        .run-agentic-btn:disabled {
            background: var(--bg-muted);
            border-color: var(--border);
            color: var(--text-faint);
            cursor: not-allowed;
        }

        /* ─── Config panel ────────────────────────────────────────────── */
        .config-panel {
            display: none;
            background: var(--bg-soft);
            padding: 12px 14px;
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            margin-top: 8px;
        }
        .config-panel.open { display: block; }
        .config-panel strong {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-faint);
            font-weight: 600;
            display: block;
            margin-bottom: 10px;
        }

        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px 14px;
            margin-top: 4px;
        }
        .config-item {
            display: flex;
            flex-direction: column;
        }
        .config-item label {
            font-size: 10.5px;
            font-weight: 500;
            color: var(--text-subtle);
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .config-item input,
        .config-item select {
            padding: 6px 9px;
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            font-size: 12.5px;
            background: var(--bg);
            color: var(--text);
            font-family: var(--font-mono);
        }
        .config-item select { font-family: var(--font-sans); cursor: pointer; }
        .config-item input:focus,
        .config-item select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(28,25,23,0.08);
        }
        .config-item input[type="checkbox"] {
            accent-color: var(--accent);
            width: auto;
            margin-right: 6px;
        }
        .config-item.config-item-wide { grid-column: span 2; }

        /* ─── Loading row ─────────────────────────────────────────────── */
        .loading {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 14px;
            box-shadow: var(--shadow-sm);
            display: flex;
            align-items: center;
            gap: 14px;
            color: var(--text);
            font-size: 13.5px;
        }
        .loading-spinner {
            width: 18px; height: 18px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            flex-shrink: 0;
            animation: loading-spin 0.7s linear infinite;
        }
        .loading-body {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .loading-title {
            font-weight: 500;
            color: var(--text);
            line-height: 1.3;
            font-size: 13.5px;
        }
        .loading-meta {
            font-size: 11.5px;
            color: var(--text-subtle);
            font-family: var(--font-mono);
            letter-spacing: 0;
        }
        .loading-meta .loading-dots::after {
            content: "";
            display: inline-block;
            width: 14px;
            text-align: left;
            animation: loading-dots 1.4s steps(4, end) infinite;
        }
        .loading-cancel {
            height: 28px;
            padding: 0 12px;
            background: var(--bg);
            color: var(--danger);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            font-size: 12.5px;
            font-weight: 500;
            cursor: pointer;
            flex-shrink: 0;
            font-family: inherit;
        }
        .loading-cancel:hover {
            background: #fef2f2;
            border-color: #fecaca;
        }
        .loading-cancel:active { transform: scale(0.97); }

        @keyframes loading-spin { to { transform: rotate(360deg); } }
        @keyframes loading-dots {
            0%   { content: ""; }
            25%  { content: "."; }
            50%  { content: ".."; }
            75%  { content: "..."; }
            100% { content: ""; }
        }

        /* ─── Error banner ────────────────────────────────────────────── */
        .error {
            background: #fef2f2;
            color: var(--danger);
            padding: 10px 14px;
            border: 1px solid #fecaca;
            border-radius: var(--radius-sm);
            margin-bottom: 14px;
            font-size: 13px;
        }

        /* ─── Results / answer card ───────────────────────────────────── */
        .results-section {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0;
            box-shadow: var(--shadow-sm);
            overflow: hidden;
            margin-bottom: 0;
        }
        #latency {
            color: var(--text-subtle) !important;
            font-family: var(--font-mono);
            font-size: 11.5px !important;
            padding: 12px 22px 0 !important;
            margin: 0 !important;
        }

        .answer-box {
            background: var(--bg);
            padding: 8px 22px 22px;
            margin-bottom: 0;
            border-left: none;
            border-radius: 0;
        }
        .answer-box h3 {
            color: var(--text-faint);
            margin-bottom: 8px;
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }
        .answer-text {
            line-height: 1.65;
            color: var(--text);
            font-size: 14.5px;
        }

        .citations {
            background: var(--bg-soft);
            padding: 14px 22px 16px;
            border-top: 1px solid var(--border);
            border-left: none;
            border-radius: 0;
            margin-bottom: 0;
        }
        .citations h3 {
            color: var(--text-faint);
            margin-bottom: 10px;
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }
        .citation-item {
            padding: 6px 0;
            border-top: 1px solid var(--border);
            font-size: 13px;
            color: var(--text-muted);
        }
        .citation-item:first-of-type {
            border-top: 0;
            padding-top: 0;
        }
        .citation-item:last-child {
            border-bottom: none;
        }
        .citation-section {
            color: var(--text);
            font-weight: 500;
            font-family: var(--font-mono);
            font-size: 12px;
            margin-right: 6px;
        }

        /* ─── Pipeline trace cards ────────────────────────────────────── */
        .pipeline-section { margin-top: 16px; }
        .pipeline-section h2 {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text);
            letter-spacing: -0.005em;
        }

        .pipeline-stage {
            background: var(--bg);
            border-top: 1px solid var(--border);
            border-radius: 0;
            border-left: 0;
            border-right: 0;
            border-bottom: 0;
            margin-bottom: 0;
            overflow: hidden;
            transition: background 0.12s;
        }
        .pipeline-stage:first-child { border-top: 0; }
        .pipeline-stage:hover { background: var(--bg); }
        .pipeline-stage.stage-group-agentic,
        .pipeline-stage.stage-group-normal {
            border-left: 0;
        }

        .stage-header {
            padding: 9px 14px;
            background: var(--bg);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            user-select: none;
            transition: background 0.12s;
        }
        .stage-header:hover { background: var(--bg-soft); }
        .stage-header.open { background: var(--bg-soft); }

        .stage-title-block { flex: 1; min-width: 0; }
        .stage-name {
            font-weight: 500;
            color: var(--text);
            font-size: 13px;
            display: block;
            line-height: 1.35;
        }
        .stage-index {
            color: var(--text-faint);
            font-weight: 500;
            margin-right: 6px;
            font-family: var(--font-mono);
            font-size: 11.5px;
        }
        .stage-subtitle {
            display: block;
            font-size: 12px;
            color: var(--text-subtle);
            margin-top: 2px;
            font-weight: 400;
        }

        .stage-time {
            font-family: var(--font-mono);
            font-size: 11.5px;
            color: var(--text-subtle);
            background: transparent;
            padding: 0;
            border-radius: 0;
            white-space: nowrap;
            flex-shrink: 0;
        }
        .stage-time.stage-time-slow { color: var(--warn); background: transparent; }

        .stage-toggle {
            color: var(--text-faint);
            font-size: 10px;
            transition: transform 0.15s;
            flex-shrink: 0;
        }
        .stage-header.open .stage-toggle { transform: rotate(180deg); }

        .stage-content {
            display: none;
            padding: 12px 18px 16px 42px;
            border-top: 1px solid var(--border);
            background: var(--bg-soft);
        }
        .stage-content.open { display: block; }

        .stage-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 12px;
        }
        .stage-chip {
            font-size: 11.5px;
            background: var(--bg);
            color: var(--text-muted);
            border: 1px solid var(--border);
            padding: 2px 8px;
            border-radius: 10px;
            font-weight: 400;
            font-family: var(--font-mono);
        }
        .stage-chip.stage-chip-error {
            background: #fef2f2;
            color: var(--danger);
            border-color: #fecaca;
        }
        .stage-chip.stage-chip-warn {
            background: #fefce8;
            color: var(--warn);
            border-color: #fde68a;
        }
        .stage-chip.stage-chip-ok {
            background: #f0fdf4;
            color: var(--ok);
            border-color: #bbf7d0;
        }
        .stage-chip.stage-chip-info {
            background: #eef2ff;
            color: #3730a3;
            border-color: #c7d2fe;
        }
        .stage-chip.stage-chip-skipped {
            background: var(--bg-muted);
            color: var(--text-subtle);
            border-color: var(--border);
            font-style: normal;
        }

        .stage-kv {
            display: grid;
            grid-template-columns: 140px 1fr;
            gap: 14px;
            padding: 6px 0;
            border-top: 1px solid var(--border);
            font-size: 12.5px;
            align-items: start;
        }
        .stage-kv:first-of-type { border-top: 0; padding-top: 0; }
        .stage-kv-label {
            color: var(--text-subtle);
            font-weight: 500;
            font-size: 11.5px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding-top: 2px;
        }
        .stage-kv-value {
            color: var(--text);
            min-width: 0;
            word-break: break-word;
        }
        .stage-kv-value code {
            background: var(--bg-muted);
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 12px;
            border: 1px solid var(--border);
            font-family: var(--font-mono);
            color: var(--text);
        }

        .stage-list { margin: 0; padding-left: 18px; color: var(--text); }
        .stage-list li { margin: 3px 0; }

        .stage-tag {
            display: inline-block;
            background: var(--bg);
            color: var(--text-muted);
            font-size: 11.5px;
            padding: 2px 8px;
            border-radius: 10px;
            margin: 2px 4px 2px 0;
            border: 1px solid var(--border);
            font-family: var(--font-mono);
        }
        .stage-tag-kind {
            color: var(--text-faint);
            font-size: 11px;
            margin-left: 4px;
        }

        .stage-mono { font-family: var(--font-mono); font-size: 12px; }
        .stage-meta { color: var(--text-subtle); }

        .stage-hits {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            overflow: hidden;
        }
        .stage-hits thead th {
            background: var(--bg-muted);
            color: var(--text-subtle);
            text-align: left;
            font-weight: 500;
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 6px 10px;
            border-bottom: 1px solid var(--border);
        }
        .stage-hits tbody td {
            padding: 6px 10px;
            border-top: 1px solid var(--border);
            vertical-align: top;
            font-family: var(--font-mono);
            font-size: 11.5px;
            color: var(--text);
        }
        .stage-hits tbody tr:first-child td { border-top: 0; }

        .stage-raw {
            margin-top: 14px;
            border-top: 1px dashed var(--border);
            padding-top: 10px;
        }
        .stage-raw summary {
            cursor: pointer;
            font-size: 11.5px;
            color: var(--text-subtle);
            user-select: none;
            font-weight: 500;
        }
        .stage-raw summary:hover { color: var(--text); }
        .stage-raw[open] summary { margin-bottom: 8px; }
        .stage-json {
            background: #0c0c0c;
            color: #e5e5e5;
            padding: 12px 14px;
            border-radius: var(--radius-sm);
            overflow-x: auto;
            font-family: var(--font-mono);
            font-size: 11.5px;
            line-height: 1.5;
            margin: 0;
        }

        .hidden { display: none !important; }

        footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: var(--text-faint);
            font-size: 11.5px;
        }

        @media (max-width: 768px) {
            .config-grid { grid-template-columns: 1fr; }
            .search-box { flex-direction: column; }
            button { width: 100%; }
        }

        /* ─── Markdown rendering inside .answer-text ──────────────────── */
        .answer-text h1, .answer-text h2, .answer-text h3, .answer-text h4 {
            color: var(--text);
            margin: 1em 0 0.4em;
            line-height: 1.3;
            font-weight: 600;
            letter-spacing: -0.01em;
        }
        .answer-text h1 { font-size: 1.4em; }
        .answer-text h2 { font-size: 1.2em; }
        .answer-text h3 { font-size: 1.05em; }
        .answer-text h4 { font-size: 1em; }
        .answer-text > *:first-child { margin-top: 0; }
        .answer-text > *:last-child  { margin-bottom: 0; }
        .answer-text p { margin: 0.7em 0; }
        .answer-text ul, .answer-text ol { margin: 0.7em 0; padding-left: 1.4em; }
        .answer-text li { margin: 0.25em 0; }
        .answer-text li > p { margin: 0.25em 0; }
        .answer-text strong { color: var(--text); font-weight: 600; }
        .answer-text em { color: var(--text-muted); }
        .answer-text blockquote {
            border-left: 2px solid var(--border-strong);
            color: var(--text-muted);
            margin: 0.8em 0;
            padding: 0.2em 0.9em;
            background: var(--bg-soft);
            border-radius: 0 3px 3px 0;
        }
        .answer-text code {
            font-family: var(--font-mono);
            font-size: 0.88em;
            background: var(--bg-muted);
            padding: 1px 5px;
            border-radius: 3px;
            color: var(--text);
            border: 1px solid var(--border);
        }
        .answer-text pre {
            background: #0c0c0c;
            color: #e5e5e5;
            padding: 12px 14px;
            border-radius: var(--radius-sm);
            overflow-x: auto;
            margin: 0.8em 0;
            line-height: 1.45;
            font-size: 0.88em;
            font-family: var(--font-mono);
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
            font-size: 0.92em;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            overflow: hidden;
        }
        .answer-text th, .answer-text td {
            border-bottom: 1px solid var(--border);
            border-left: 0;
            border-right: 0;
            padding: 8px 11px;
            text-align: left;
            vertical-align: top;
        }
        .answer-text th {
            background: var(--bg-muted);
            color: var(--text-muted);
            font-weight: 600;
        }
        .answer-text tr:last-child td { border-bottom: 0; }
        .answer-text tr:nth-child(even) td { background: var(--bg-soft); }
        .answer-text a {
            color: var(--text);
            text-decoration: underline;
            text-underline-offset: 2px;
            text-decoration-color: var(--border-strong);
        }
        .answer-text a:hover { text-decoration-color: var(--text); }
        .answer-text hr {
            border: 0;
            border-top: 1px solid var(--border);
            margin: 1.2em 0;
        }

        /* ─── Pipeline visualization (Mermaid DAG) ────────────────────── */
        .viz-section {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 16px 12px;
            margin: 14px 0;
            box-shadow: var(--shadow-sm);
        }
        .viz-section h2 {
            font-size: 13px;
            color: var(--text);
            margin: 0 0 2px;
            font-weight: 600;
            letter-spacing: -0.005em;
        }
        .viz-section .viz-sub {
            font-size: 12px;
            color: var(--text-subtle);
            margin-bottom: 12px;
            line-height: 1.45;
        }
        .viz-container {
            overflow-x: auto;
            background:
                radial-gradient(circle at 1px 1px, var(--border) 1px, transparent 0);
            background-size: 22px 22px;
            background-color: var(--bg-soft);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 24px 18px;
            min-height: 220px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .viz-container .mermaid { text-align: center; width: 100%; }
        .viz-container svg { max-width: 100%; height: auto; display: block; margin: 0 auto; }
        .viz-container .nodeLabel,
        .viz-container .label,
        .viz-container foreignObject div {
            font-family: var(--font-sans) !important;
            font-size: 12px !important;
            line-height: 1.4 !important;
            letter-spacing: -0.005em;
        }
        .viz-container .nodeLabel b { font-weight: 600; }
        .viz-container .nodeLabel .vz-sub {
            display: block;
            font-size: 11px;
            opacity: 0.75;
            margin-top: 2px;
            font-weight: 400;
        }
        .viz-container .nodeLabel .vz-time {
            display: block;
            font-size: 10.5px;
            margin-top: 3px;
            opacity: 0.65;
            font-family: var(--font-mono);
        }
        .viz-container .node rect,
        .viz-container .node polygon,
        .viz-container .node path {
            filter: drop-shadow(0 1px 1.5px rgba(0,0,0,0.04));
        }
        .viz-container .edgePath path {
            stroke-width: 1.2px !important;
            stroke: var(--text-faint) !important;
        }
        .viz-container .arrowheadPath,
        .viz-container marker path {
            fill: var(--text-faint) !important;
            stroke: var(--text-faint) !important;
        }

        .viz-empty {
            color: var(--text-faint);
            text-align: center;
            font-size: 12.5px;
            padding: 28px;
            font-style: normal;
            width: 100%;
        }

        .viz-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 8px 16px;
            margin-top: 14px;
            padding: 10px 12px;
            background: var(--bg-soft);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            font-size: 11.5px;
        }
        .viz-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: var(--text-muted);
        }
        .viz-legend-swatch {
            width: 10px; height: 10px;
            border-radius: 3px;
            border: 1px solid rgba(0,0,0,0.06);
            flex-shrink: 0;
        }

        .viz-header-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 4px;
        }
        .viz-nav {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-shrink: 0;
        }
        .viz-nav-btn {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            width: 28px;
            height: 28px;
            font-size: 13px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            padding: 0;
        }
        .viz-nav-btn:hover:not(:disabled) {
            background: var(--bg-muted);
            border-color: var(--border-strong);
            color: var(--text);
        }
        .viz-nav-btn:disabled { opacity: 0.35; cursor: default; }
        .viz-nav-label {
            font-size: 11.5px;
            color: var(--text-subtle);
            min-width: 64px;
            text-align: center;
            font-variant-numeric: tabular-nums;
            font-family: var(--font-mono);
        }

        /* ─── Collapsed trace accordion ───────────────────────────────── */
        .trace-details { margin: 14px 0; }
        .trace-summary {
            cursor: pointer;
            list-style: none;
            padding: 10px 14px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 8px;
            user-select: none;
            transition: background 0.12s;
            letter-spacing: -0.005em;
        }
        .trace-summary:hover { background: var(--bg-soft); }
        .trace-summary::-webkit-details-marker { display: none; }
        .trace-summary::marker { display: none; }
        .trace-details[open] .trace-summary {
            border-radius: 8px 8px 0 0;
            border-bottom-color: var(--border);
        }
        .trace-summary-chevron {
            margin-left: auto;
            font-size: 10px;
            color: var(--text-faint);
            transition: transform 0.15s;
        }
        .trace-details[open] .trace-summary-chevron { transform: rotate(180deg); }
        .trace-count {
            font-size: 12px;
            font-weight: 400;
            color: var(--text-subtle);
            font-family: var(--font-mono);
        }
        .trace-details #pipeline-stages {
            border: 1px solid var(--border);
            border-top: none;
            border-radius: 0 0 8px 8px;
            background: var(--bg);
            padding: 0;
            overflow: hidden;
        }

        /* ─── Draggable stage detail popup ────────────────────────────── */
        .stage-popup {
            position: fixed;
            z-index: 9999;
            top: 80px;
            left: 50%;
            transform: translateX(-50%);
            width: 480px;
            max-width: 92vw;
            max-height: 80vh;
            background: var(--bg);
            border: 1px solid var(--border-strong);
            border-radius: 8px;
            box-shadow: 0 12px 32px rgba(0,0,0,0.12);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .stage-popup-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            background: var(--bg-soft);
            border-bottom: 1px solid var(--border);
            cursor: grab;
            user-select: none;
            flex-shrink: 0;
        }
        .stage-popup-header:active { cursor: grabbing; }
        .stage-popup-title {
            font-weight: 600;
            font-size: 13px;
            color: var(--text);
            letter-spacing: -0.005em;
        }
        .stage-popup-close {
            background: none;
            border: none;
            font-size: 14px;
            cursor: pointer;
            color: var(--text-subtle);
            line-height: 1;
            padding: 2px 6px;
            border-radius: 3px;
            height: auto;
        }
        .stage-popup-close:hover {
            background: var(--bg-muted);
            color: var(--text);
        }
        .stage-popup-body {
            overflow-y: auto;
            padding: 12px 14px;
            flex: 1;
            font-size: 13px;
        }
        .viz-container svg g.node { cursor: pointer; }
        .viz-container svg g.node:hover rect,
        .viz-container svg g.node:hover polygon,
        .viz-container svg g.node:hover path,
        .viz-container svg g.node:hover circle,
        .viz-container svg g.node:hover ellipse {
            filter: brightness(0.96) drop-shadow(0 2px 4px rgba(0,0,0,0.08));
        }

        /* ─── Model info panel ────────────────────────────────────────── */
        .model-panel {
            margin: 14px 0;
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            font-size: 12.5px;
            background: var(--bg);
        }
        .model-panel-header {
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--bg);
            padding: 10px 14px;
            cursor: pointer;
            user-select: none;
            border: none;
            width: 100%;
            text-align: left;
            font-size: 13px;
            font-weight: 600;
            color: var(--text);
            height: auto;
            letter-spacing: -0.005em;
            border-radius: 0;
        }
        .model-panel-header:hover { background: var(--bg-soft); }
        .model-panel-badge {
            margin-left: auto;
            background: var(--bg-muted);
            color: var(--text-muted);
            font-size: 11.5px;
            font-weight: 500;
            padding: 2px 8px;
            border-radius: 10px;
            font-family: var(--font-mono);
            border: 1px solid var(--border);
        }
        .model-panel-chevron { font-size: 10px; color: var(--text-faint); }
        .model-panel-body {
            display: none;
            padding: 0;
            background: var(--bg);
            overflow-x: auto;
            border-top: 1px solid var(--border);
        }
        .model-panel-body.open { display: block; }
        .model-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .model-table th {
            text-align: left;
            padding: 8px 14px;
            border-bottom: 1px solid var(--border);
            color: var(--text-subtle);
            font-weight: 500;
            white-space: nowrap;
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: var(--bg);
        }
        .model-table td {
            padding: 8px 14px;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
            color: var(--text);
        }
        .model-table tr:last-child td { border-bottom: 0; }
        .model-table code {
            background: var(--bg-muted);
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 11px;
            border: 1px solid var(--border);
            font-family: var(--font-mono);
        }
        .model-row-active td {
            background: var(--bg-muted);
            font-weight: 500;
        }
        .model-row-active td:first-child { position: relative; }
        .model-row-active td:first-child::before {
            content: '';
            position: absolute;
            left: 0; top: 6px; bottom: 6px;
            width: 2px;
            background: var(--accent);
        }
        .model-note { color: var(--text-faint); font-style: normal; font-size: 11.5px; }
        .model-cost-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-items: center;
            margin: 0;
            padding: 10px 14px;
            border-top: 1px solid var(--border);
            background: var(--bg-soft);
            font-size: 12px;
            color: var(--text-muted);
        }
        .model-cost-label { font-weight: 600; color: var(--text); }
        .model-cost-sep { color: var(--text-faint); }
        .model-cost-total {
            font-weight: 600;
            color: var(--text);
            font-size: 12.5px;
            font-family: var(--font-mono);
        }
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
                    <button type="submit" class="config-toggle">Sign out</button>
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
                            <option value="deepthought">DeepThought (free)</option>
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
                            <option value="deepthought">DeepThought (free)</option>
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
                <div id="latency" style="text-align: right;"></div>

                <div class="answer-box">
                    <h3>Answer</h3>
                    <div id="answer-text" class="answer-text"></div>
                </div>

                <div id="citations-box" class="citations hidden">
                    <h3>Sources Cited</h3>
                    <div id="citations-list"></div>
                </div>

                <div class="viz-section">
                    <div class="viz-header-row">
                        <div>
                            <h2>Pipeline Flow</h2>
                            <div class="viz-sub">
                                Each color is a stage family. Branches show per-sub-query
                                retrieval (semantic · keyword · BM25); all paths merge
                                through rank fusion → dedup → rerank → generation.
                                Click any node to inspect its data.
                            </div>
                        </div>
                        <div id="viz-nav" class="viz-nav" style="display:none">
                            <button id="viz-nav-prev" class="viz-nav-btn" onclick="vizPrev()" title="Previous pass">&#8592;</button>
                            <span id="viz-nav-label" class="viz-nav-label">Pass 1 / 1</span>
                            <button id="viz-nav-next" class="viz-nav-btn" onclick="vizNext()" title="Next pass">&#8594;</button>
                        </div>
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

                <details id="trace-details" class="trace-details">
                    <summary class="trace-summary">
                        Pipeline Trace
                        <span id="trace-stage-count" class="trace-count"></span>
                        <span class="trace-summary-chevron">▼</span>
                    </summary>
                    <div id="pipeline-stages"></div>
                </details>

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

    <!-- Draggable stage-detail popup (hidden until a flowchart node is clicked) -->
    <div id="stage-popup" class="stage-popup" style="display:none" role="dialog" aria-modal="true">
        <div class="stage-popup-header" id="stage-popup-drag-handle">
            <span id="stage-popup-title"></span>
            <button class="stage-popup-close" onclick="closeStagePopup()" title="Close">&#x2715;</button>
        </div>
        <div class="stage-popup-body" id="stage-popup-body"></div>
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
                    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
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
            "deepthought":                 {in: 0.0,  out: 0.0},
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

            let maxIter = -1;
            for (const s of trace) {
                const m = s.stage.match(ITER_RE);
                if (m) maxIter = Math.max(maxIter, parseInt(m[1], 10));
            }
            if (maxIter < 0) return null;

            const result = [];
            for (let i = 0; i <= maxIter; i++) {
                if (i === 0) {
                    // Pass 1: full base pipeline + iter0 agentic stages.
                    result.push(trace.filter(s => {
                        const m = s.stage.match(ITER_RE);
                        if (!m) return true; // base stage
                        return parseInt(m[1], 10) === 0;
                    }).map(s => ({...s, stage: stripIter(s.stage)})));
                } else {
                    // Pass 2+: only this iteration's agentic stages. The
                    // base pipeline didn't re-run — the chart starts from
                    // what gap analysis requested and shows the follow-up
                    // sub-pipeline (decompose → hybrid search → rerank →
                    // regenerate).
                    result.push(trace.filter(s => {
                        const m = s.stage.match(ITER_RE);
                        return m && parseInt(m[1], 10) === i;
                    }).map(s => ({...s, stage: stripIter(s.stage)})));
                }
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
            // first-pass state, so Stages 1–4 didn't run. Emit a "Resume"
            // marker so the diagram still has a visible upstream node feeding
            // into the agentic branch below. Without this we bail with just
            // the Query node and the user sees an empty canvas.
            if (!qp && !refineSeed && !gapEarly) {
                L.push("  classDef input fill:#1c1917,color:#fff,stroke:#1c1917,stroke-width:1.5px,rx:6,ry:6");
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
                L.push(`  RR["${_label("Rerank", "top " + (rr.output.count || 0) + " · cross-encoder", rr)}"]:::stage_rerank`);
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
            // pass 2+) the answer node is just GEN2 (nothing upstream).
            let agAnswerNode = gen ? "GEN" : (refineSeed ? "RESUME" : "GEN2");
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
                        const qText = (fq && fq.input && fq.input.query) || `gap-q${i}`;
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

                        // Open a subgraph for this follow-up.
                        L.push(`  subgraph FQGRP${i}["Follow-up ${i+1}: ${_vizText(qText, 56)} · ${chunks} chunks"]`);
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
                        if (!errType) agAnswerNode = "GEN2";
                        nodeMap["GEN2"] = ag_gen;
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
            // Minimalist palette: monochrome anchors (input/generate),
            // light tints to keep stage families distinguishable but quiet.
            L.push("  classDef input    fill:#1c1917,color:#fff,stroke:#1c1917,stroke-width:1.5px,rx:6,ry:6");
            L.push("  classDef output   fill:#ffffff,color:#15803d,stroke:#15803d,stroke-width:1.5px,rx:8,ry:8");
            L.push("  classDef stage_qp     fill:#fafaf9,color:#1c1917,stroke:#d6d3d1,stroke-width:1px,rx:5,ry:5");
            L.push("  classDef stage_struct fill:#f0fdf4,color:#166534,stroke:#bbf7d0,stroke-width:1px,rx:5,ry:5");
            L.push("  classDef stage_skipped fill:#fafaf9,color:#a8a29e,stroke:#e7e5e4,stroke-width:1px,stroke-dasharray:3 3,rx:5,ry:5");
            L.push("  classDef stage_subq   fill:#fafaf9,color:#57534e,stroke:#d6d3d1,stroke-width:1px");
            L.push("  classDef stage_vector fill:#eff6ff,color:#1e3a8a,stroke:#bfdbfe,stroke-width:1px,rx:5,ry:5");
            L.push("  classDef stage_tsv    fill:#ecfeff,color:#155e75,stroke:#a5f3fc,stroke-width:1px,rx:5,ry:5");
            L.push("  classDef stage_bm25   fill:#f0fdfa,color:#115e59,stroke:#99f6e4,stroke-width:1px,rx:5,ry:5");
            L.push("  classDef stage_rrf    fill:#fefce8,color:#854d0e,stroke:#fde68a,stroke-width:1px,rx:5,ry:5");
            L.push("  classDef stage_dedup  fill:#fff7ed,color:#9a3412,stroke:#fed7aa,stroke-width:1px,rx:5,ry:5");
            L.push("  classDef stage_rerank fill:#fef2f2,color:#991b1b,stroke:#fecaca,stroke-width:1px,rx:5,ry:5");
            L.push("  classDef stage_gen    fill:#1c1917,color:#fff,stroke:#1c1917,stroke-width:1.5px,rx:6,ry:6");
            // Refine fast-path marker — dashed stone, distinct from other stages.
            L.push("  classDef stage_resume   fill:#f5f5f4,color:#57534e,stroke:#a8a29e,stroke-width:1px,stroke-dasharray:4 3,rx:5,ry:5");
            // Agentic branch — indigo tint sets it apart from the main path.
            L.push("  classDef stage_gap      fill:#eef2ff,color:#3730a3,stroke:#c7d2fe,stroke-width:1px");
            L.push("  classDef stage_followup fill:#eef2ff,color:#3730a3,stroke:#c7d2fe,stroke-width:1px");
            L.push("  classDef stage_agen     fill:#312e81,color:#fff,stroke:#312e81,stroke-width:1.5px,rx:6,ry:6");
            // Targeted fetch — teal family for "direct lookup".
            L.push("  classDef stage_tfetch   fill:#f0fdfa,color:#115e59,stroke:#99f6e4,stroke-width:1px,rx:5,ry:5");

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
                const {svg} = await mermaid.render(id, page.def);
                host.innerHTML = svg;
                if (legend) legend.style.display = "";
                attachNodeClickListeners(page.nodeMap);
            } catch (err) {
                console.error("mermaid render failed:", err, page.def);
                host.innerHTML = `<div class="viz-empty">Could not render flow: ${escapeHtml(err.message || String(err))}</div>`;
            }

            // Update navigation controls.
            const isMulti = _vizPages.length > 1;
            if (nav) nav.style.display = isMulti ? "" : "none";
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

            const iters = splitTraceByIteration(trace);
            if (iters && iters.length > 1) {
                for (const sub of iters) {
                    _vizPages.push(buildMermaidFromTrace(sub, _vizQuery));
                }
            } else {
                _vizPages.push(buildMermaidFromTrace(trace, _vizQuery));
            }

            await vizGoTo(0);
        }

        // ─── Draggable stage-detail popup ─────────────────────────────────
        function showStagePopup(stage, clickX, clickY) {
            const popup = document.getElementById("stage-popup");
            if (!popup) return;
            const display = formatStageDisplay(stage.stage);
            document.getElementById("stage-popup-title").textContent = display.title;
            document.getElementById("stage-popup-body").innerHTML = renderStageBody(stage);

            // Position near the click, clamped inside the viewport.
            popup.style.transform = "none";
            popup.style.display = "flex";
            const w = popup.offsetWidth || 480;
            const h = popup.offsetHeight || 300;
            popup.style.left = Math.max(8, Math.min(clickX, window.innerWidth  - w - 8)) + "px";
            popup.style.top  = Math.max(8, Math.min(clickY + 14, window.innerHeight - h - 8)) + "px";
        }

        function closeStagePopup() {
            const popup = document.getElementById("stage-popup");
            if (popup) popup.style.display = "none";
        }

        // Wire up drag on popup header once the DOM is ready.
        document.addEventListener("DOMContentLoaded", () => {
            const handle = document.getElementById("stage-popup-drag-handle");
            const popup  = document.getElementById("stage-popup");
            if (!handle || !popup) return;
            let ox = 0, oy = 0;
            function onMove(e) {
                const cx = e.touches ? e.touches[0].clientX : e.clientX;
                const cy = e.touches ? e.touches[0].clientY : e.clientY;
                popup.style.left = Math.max(0, cx - ox) + "px";
                popup.style.top  = Math.max(0, cy - oy) + "px";
            }
            function onUp() {
                document.removeEventListener("mousemove", onMove);
                document.removeEventListener("mouseup",   onUp);
                document.removeEventListener("touchmove", onMove);
                document.removeEventListener("touchend",  onUp);
            }
            handle.addEventListener("mousedown", e => {
                const r = popup.getBoundingClientRect();
                ox = e.clientX - r.left;
                oy = e.clientY - r.top;
                document.addEventListener("mousemove", onMove);
                document.addEventListener("mouseup",   onUp);
                e.preventDefault();
            });
            handle.addEventListener("touchstart", e => {
                const r = popup.getBoundingClientRect();
                ox = e.touches[0].clientX - r.left;
                oy = e.touches[0].clientY - r.top;
                document.addEventListener("touchmove", onMove, {passive: false});
                document.addEventListener("touchend",  onUp);
                e.preventDefault();
            }, {passive: false});
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

            // Pipeline visualization (Mermaid DAG) — rendered first so the
            // flowchart appears above the collapsed trace accordion.
            renderPipelineViz(data.pipeline_trace, data.query);

            // Pipeline trace — rendered into the collapsed <details> accordion
            // below the flowchart. Each card shows title + subtitle + chips +
            // key/value rows; raw JSON is behind a "Show raw JSON" toggle.
            const stagesDiv = document.getElementById("pipeline-stages");
            const stageCount = document.getElementById("trace-stage-count");
            if (data.pipeline_trace) {
                stagesDiv.innerHTML = data.pipeline_trace
                    .map((stage, idx) => renderStageCard(stage, idx))
                    .join("");
                if (stageCount) stageCount.textContent = `(${data.pipeline_trace.length} stage${data.pipeline_trace.length === 1 ? "" : "s"})`;
            } else {
                stagesDiv.innerHTML = '';
                if (stageCount) stageCount.textContent = '';
            }

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
