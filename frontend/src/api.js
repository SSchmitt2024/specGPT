/**
 * API client for the specGPT backend.
 *
 * HOOKUP GUIDE:
 * ─────────────
 * When the FastAPI backend (src/pipeline/app.py) is ready:
 *
 * 1. Start the backend:
 *      cd <project_root>
 *      uvicorn src.pipeline.app:app --reload --port 8000
 *
 * 2. Start the frontend:
 *      cd frontend
 *      npm run dev
 *
 * 3. The frontend dev server (Vite, port 5173) proxies /api/* to localhost:8000.
 *    See vite.config.js for the proxy config.
 *
 * 4. The backend POST /api/query expects:
 *      { "query": "your question here" }
 *
 *    And returns:
 *      {
 *        "answer": "...",
 *        "citations": ["3.1.2", "5.27.3"],
 *        "confidence": "HIGH",
 *        "sources": [
 *          {
 *            "chunk_id": "...",
 *            "section_id": "...",
 *            "section_title": "...",
 *            "content_type": "prose",
 *            "text_raw": "...",
 *            "pdf_pages": [142]
 *          }
 *        ]
 *      }
 */

const API_BASE = '/api'

export async function querySpec(question) {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: question }),
  })

  if (!res.ok) {
    const err = await res.text()
    throw new Error(`Query failed (${res.status}): ${err}`)
  }

  return res.json()
}
