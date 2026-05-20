"""
Phase 2 - Step 2.5: Web Application (FastAPI Backend)

Exposes the full retrieval + generation pipeline as a web service.

Endpoints:
  POST /api/query — run pipeline, return answer + pipeline trace
  GET / — serve web frontend
  GET /api/config — get default config

Environment:
  DEBUG_PIPELINE — set to "1" to include full trace in responses (default: on)
  PORT — server port (default: 8000)
  HOST — server host (default: 127.0.0.1)

Run:
  python -m src.pipeline.app
  Then visit http://localhost:8000

Architecture:
  - FastAPI backend orchestrates the pipeline
  - Frontend is vanilla HTML/CSS/JS for visualization
  - Config passed from frontend to backend
  - Pipeline trace returned for debugging
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline.orchestrator import orchestrate, PipelineConfig


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

# Configuration
DEBUG_PIPELINE = os.getenv("DEBUG_PIPELINE", "1").lower() in ("1", "true", "yes")


# ============================================================================
# Endpoints
# ============================================================================

@app.post("/api/query")
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """
    Run the full retrieval + generation pipeline.

    Returns answer with citations and optional pipeline trace for debugging.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    # Parse config from request, use defaults if not provided
    config_dict = req.config or {}
    config = PipelineConfig(**config_dict)

    start = time.time()
    try:
        result = orchestrate(req.query, config=config, debug=req.debug and DEBUG_PIPELINE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    latency_ms = (time.time() - start) * 1000

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        citations=result["citations"],
        config=result["config"],
        pipeline_trace=result.get("pipeline_trace") if req.debug else None,
        latency_ms=latency_ms,
    )


@app.get("/api/config")
async def config_endpoint() -> dict:
    """Return default PipelineConfig."""
    return PipelineConfig().to_dict()


@app.get("/", response_class=HTMLResponse)
async def frontend() -> str:
    """Serve the web frontend."""
    return FRONTEND_HTML


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
        <div class="container">
            <h1>specGPT</h1>
            <p>Ask questions about NVMe specifications. See exactly how the system found the answer.</p>
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

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || "Request failed");
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

        function displayResults(data) {
            // Answer
            document.getElementById("answer-text").textContent = data.answer;
            document.getElementById("latency").textContent = `⏱️ ${data.latency_ms.toFixed(0)}ms`;

            // Citations
            const citationsList = document.getElementById("citations-list");
            const citationsBox = document.getElementById("citations-box");
            if (data.citations && data.citations.length > 0) {
                citationsList.innerHTML = data.citations
                    .map(c => `<div class="citation-item"><span class="citation-section">[§${c.section_id}]</span> ${c.section_title}`)
                    .join("");
                citationsBox.classList.remove("hidden");
            } else {
                citationsBox.classList.add("hidden");
            }

            // Pipeline trace
            const stagesDiv = document.getElementById("pipeline-stages");
            if (data.pipeline_trace) {
                stagesDiv.innerHTML = data.pipeline_trace
                    .map((stage, idx) => `
                        <div class="pipeline-stage">
                            <div class="stage-header" onclick="toggleStage(this)">
                                <span class="stage-name">${idx + 1}. ${formatStageName(stage.stage)}</span>
                                <span class="stage-time">${stage.took_ms.toFixed(0)}ms</span>
                                <span class="stage-toggle">▼</span>
                            </div>
                            <div class="stage-content">
                                <div class="stage-json">${JSON.stringify(stage, null, 2)}</div>
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

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(f"\n{'='*60}")
    print(f"  specGPT Pipeline Server")
    print(f"{'='*60}")
    print(f"  Listening on http://{host}:{port}")
    print(f"  API: http://{host}:{port}/api/query")
    print(f"  Debug Mode: {DEBUG_PIPELINE}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=host, port=port)
