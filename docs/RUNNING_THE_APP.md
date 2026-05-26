# Running the Web App

The FastAPI web app (§2.5) is now complete. You can ask questions about the NVMe spec and see the full retrieval pipeline in action.

## Prerequisites

1. **Environment variables** — copy `.env.example` to `.env` and fill in values. At a minimum you need:
   ```
   APP_PASSWORD=pick-something-strong
   SESSION_SECRET=<32+ random bytes; generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`>
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-key
   VOYAGE_API_KEY=your-voyage-key
   ANTHROPIC_API_KEY=your-anthropic-key
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your-gemini-key     # or OPENAI_API_KEY if LLM_PROVIDER=openai
   ```

   The app **refuses to start** if `APP_PASSWORD` or `SESSION_SECRET` are missing — that's intentional, the gate is required.

2. **Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data must be indexed** (Phase 1 output in Supabase):
   - Chunks indexed with embeddings
   - Field index loaded (`scripts/load_lookup_data.py`)
   - Tables available in data/

## Running the Server

```bash
python -m src.pipeline.app
```

Output:
```
============================================================
  specGPT Pipeline Server
============================================================
  Listening on http://127.0.0.1:8000
  API: http://127.0.0.1:8000/api/query
  Debug Mode: True
============================================================
```

Then open your browser to **http://localhost:8000**. You'll be redirected to a password form on first visit; after signing in, the session cookie is good for 30 days (or until you `/logout`, or until you rotate `SESSION_SECRET`).

### Authenticated routes

All real routes require the session cookie. Only these are public:

| Route | Purpose |
|-------|---------|
| `GET /healthz` | Liveness check for Railway / k8s healthchecks |
| `GET /login`   | Renders the password form |
| `POST /login`  | Validates password, sets session cookie |
| `GET/POST /logout` | Clears the session cookie |

Failed logins are rate-limited per source IP (1s → 60s sliding-window backoff after a few rapid misses).

## Web Interface

### Query Input
- Type any question: "What is bit 7 of CDW10?", "How is the SQ organized?", etc.
- Press Enter or click "Search"
- The app will run the full pipeline and show the answer

### Configuration Panel
Click the **⚙️ Config** button to tune parameters:
- **Vector Top-K**: 1-50 (default 10) — how many vector search results
- **BM25 Top-K**: 1-50 (default 10) — how many keyword search results
- **RRF K**: 10-200 (default 60) — RRF merge aggressiveness
- **RRF Output Top-K**: 5-50 (default 20) — candidates before reranking
- **Final Rerank Top-K**: 1-20 (default 7) — final answer context size
- **Max Sub-Queries**: 1-5 (default 3) — query decomposition limit

Change any value and search again. The config is logged in the pipeline trace so you can see the impact.

### Results Display

**Answer**: The generated response from Claude Sonnet, using only the retrieved context.

**Sources Cited**: Sections of the spec that were cited in the answer. Formatted as `[§5.2.1] Title`.

**Pipeline Trace**: Expandable stages showing:
- Query Processor (classification, entity extraction, decomposition)
- Structured Lookup (if lookup query)
- Vector Search (per sub-query)
- BM25 Search (per sub-query)
- RRF Merge (combining results)
- Final Rerank (cross-encoder scoring)
- Generation (Sonnet call)

Click any stage to expand and see:
- Input: what was passed to that stage
- Output: what came out (result summaries, counts)
- Timing: how long it took in milliseconds

## API Endpoint

For programmatic use:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is bit 7 of CDW10?",
    "config": {
      "vector_topk": 15,
      "bm25_topk": 15,
      "rrf_k": 45,
      "rrf_output_topk": 25,
      "final_rerank_topk": 10,
      "max_subqueries": 3
    },
    "debug": true
  }'
```

Response:
```json
{
  "query": "What is bit 7 of CDW10?",
  "answer": "Bits 7:4 of CDW10 represent...",
  "citations": [
    {
      "section_id": "5.2.1",
      "section_title": "Set Features Command",
      "content_type": "prose"
    }
  ],
  "config": {...},
  "pipeline_trace": [...],
  "latency_ms": 1234
}
```

## Environment Variables

See `.env.example` for the canonical list. The frequently-touched ones:

- **APP_PASSWORD** (required): Shared password for the login gate. Hashed at startup, wiped from env. App refuses to boot if unset.
- **SESSION_SECRET** (required, ≥16 bytes): HMAC key for session cookies. Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Rotating it invalidates every session immediately.
- **DEBUG_PIPELINE** (default: "0"): Set to "1" to include the full pipeline trace in `/api/query` responses. Off by default because the trace exposes chunk previews, model names, and timings.
- **PORT** (default: "8000"): Server port.
- **HOST** (default: "127.0.0.1"): Server host. Use "0.0.0.0" only inside a container behind a reverse proxy.
- **COOKIE_SECURE** (default: "1"): Cookies are only sent over HTTPS. Set to "0" for local HTTP testing, never in production.
- **TRUST_PROXY_HEADERS** (default: "0"): Set to "1" if you're behind a proxy that strips spoofed `X-Forwarded-For` (Railway, Cloudflare). Otherwise the login throttle uses the direct socket peer.

## Common Issues

### 500 Error: "Pipeline error: ..."
- Check that Supabase is accessible (SUPABASE_URL and SUPABASE_KEY)
- Check that spec chunks are indexed in Supabase
- Check that VOYAGE_API_KEY is valid

### No results in pipeline trace
- Set `DEBUG_PIPELINE=1` in environment
- Make sure request includes `"debug": true`

### Models not found / slow first request
- Cross-encoder model (~80MB) downloads on first rerank call
- Sonnet is called directly via Anthropic API
- First request may be slower due to model loading

## Testing the Pipeline

### Quick Manual Test

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Submission Queue?", "debug": true}'
```

Expected: A detailed answer from the spec with citations and full pipeline trace.

### Test with Different Configs

```python
import requests

query = "What is bit 7 of CDW10?"

# Test 1: Default config
response1 = requests.post("http://localhost:8000/api/query", json={"query": query})
print(f"Default: {response1.json()['latency_ms']:.0f}ms")

# Test 2: More candidates
response2 = requests.post("http://localhost:8000/api/query", json={
    "query": query,
    "config": {"vector_topk": 20, "bm25_topk": 20}
})
print(f"High recall: {response2.json()['latency_ms']:.0f}ms")

# Test 3: Tighter RRF
response3 = requests.post("http://localhost:8000/api/query", json={
    "query": query,
    "config": {"rrf_k": 30}
})
print(f"Sharp RRF: {response3.json()['latency_ms']:.0f}ms")
```

## Next Steps

Now that the app is running, you can:
1. **Run the eval set** (§2.6): Measure quality on test questions
2. **Tune parameters**: Use the config panel to optimize for your eval metrics
3. **Iterate**: Run eval → tune → repeat until satisfied
