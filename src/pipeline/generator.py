"""
Phase 2 - Step 2.4: Generation

Takes retrieved context chunks and generates a cited answer using Claude Sonnet.

Pipeline:
  1. Context Assembly: trim large tables, respect 3-5k token budget
  2. Prompt Assembly: system instructions + context + query
  3. Sonnet Call: generate answer with strict system prompt
  4. Citation Extraction: parse answer for section references

Output: (answer, citations) where citations are {"text": quote, "source": section_id}.

CLI:
  python -m src.pipeline.generator "What is bit 7 of CDW10?" context.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from anthropic import Anthropic, BadRequestError
except ImportError:
    print("Missing dependency: pip install anthropic")
    sys.exit(1)


# Model defaults
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MAX_CONTEXT_TOKENS = 4000
DEFAULT_SYSTEM_PROMPT = """You are an expert on NVMe specifications. Answer the user's question using ONLY the provided context.

RULES:
1. Answer only using information from the provided context sections.
2. Cite the section number for every claim (e.g., "per Section 5.2.1").
3. For bit/field definitions, include the exact offset and size if available.
4. If the context does not contain the answer, explicitly state what information is missing.
5. Never speculate, infer beyond the spec, or hallucinate details.
6. If multiple sections address the question, synthesize them clearly and cite all relevant sections.
7. Keep answers concise but complete.

CONTEXT:
{}

USER QUESTION:
{}"""


@dataclass
class GenerationResult:
    """Result from generate() with answer, citations, and metadata."""
    answer: str
    citations: list[dict] = field(default_factory=list)
    context_used: list[dict] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    tokens_used: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0})

    def to_dict(self) -> dict:
        return asdict(self)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (Claude average)."""
    return max(1, len(text) // 4)


def _trim_table_chunk(chunk: dict, max_tokens: int = 1000) -> dict:
    """Trim large table chunks to fit token budget."""
    if chunk.get("content_type") != "table":
        return chunk

    text = chunk.get("text_raw", "")
    tokens = _estimate_tokens(text)

    if tokens <= max_tokens:
        return chunk

    # For large tables, keep header + first N rows
    lines = text.split("\n")
    header_line_count = 2  # Usually table title + column headers
    kept_lines = lines[:header_line_count]
    current_tokens = _estimate_tokens("\n".join(kept_lines))

    for line in lines[header_line_count:]:
        line_tokens = _estimate_tokens(line)
        if current_tokens + line_tokens > max_tokens:
            kept_lines.append("... (table truncated) ...")
            break
        kept_lines.append(line)
        current_tokens += line_tokens

    trimmed_text = "\n".join(kept_lines)
    return {**chunk, "text_raw": trimmed_text}


def assemble_context(
    query: str,
    context_chunks: list[dict],
    *,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
) -> tuple[str, list[dict]]:
    """
    Assemble context from chunks, respecting token budget.

    Large tables are trimmed; chunks are included in order until budget is hit.

    Args:
        query: the user's question (for context setting).
        context_chunks: retrieved/ranked chunks from retrieval stage.
        max_context_tokens: token budget for context.

    Returns:
        (formatted_context, used_chunks) where formatted_context is ready for
        Sonnet and used_chunks is metadata about what was included.
    """
    used_chunks: list[dict] = []
    context_lines: list[str] = []
    total_tokens = 0

    for chunk in context_chunks:
        # Trim large tables
        trimmed = _trim_table_chunk(chunk, max_tokens=max_context_tokens // 3)
        chunk_text = trimmed.get("text_raw", "")
        chunk_tokens = _estimate_tokens(chunk_text)

        # Check budget
        if total_tokens + chunk_tokens > max_context_tokens:
            break

        section_id = chunk.get("section_id", "unknown")
        section_title = chunk.get("section_title", "")
        content_type = chunk.get("content_type", "prose")

        # Format: [Section X.Y.Z] Title (type)
        header = f"[Section {section_id}] {section_title}"
        if content_type == "table":
            header += " (table)"

        context_lines.append(header)
        context_lines.append(chunk_text)
        context_lines.append("")  # blank line between sections

        used_chunks.append({
            "id": chunk.get("id"),
            "section_id": section_id,
            "section_title": section_title,
            "content_type": content_type,
            "tokens_used": chunk_tokens,
        })

        total_tokens += chunk_tokens

    formatted_context = "\n".join(context_lines).strip()
    return formatted_context, used_chunks


def _extract_citations(answer: str, context_chunks: list[dict]) -> list[dict]:
    """
    Extract section citations from the answer.

    Looks for patterns like "Section 5.2.1", "per Section X.Y.Z", etc.
    Matches them against context_chunks to find source sections.
    """
    citations: list[dict] = []
    seen_sections: set = set()

    # Pattern: Section X.Y.Z (up to 3 digits, optional decimals)
    section_pattern = r"Section\s+([\d.]+)"
    for match in re.finditer(section_pattern, answer, re.IGNORECASE):
        section_id = match.group(1)
        if section_id in seen_sections:
            continue
        seen_sections.add(section_id)

        # Find matching context chunk
        for chunk in context_chunks:
            if chunk.get("section_id") == section_id:
                citations.append({
                    "section_id": section_id,
                    "section_title": chunk.get("section_title", ""),
                    "content_type": chunk.get("content_type", "prose"),
                })
                break

    return citations


def generate(
    query: str,
    context_chunks: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    system_prompt: str | None = None,
) -> tuple[str, list[dict]]:
    """
    Generate an answer using Claude Sonnet from retrieved context.

    Args:
        query: the user's question.
        context_chunks: ranked chunks from retrieval (sorted by relevance).
        model: Claude model to use (default: claude-3-5-sonnet-20241022).
        max_context_tokens: token budget for context assembly.
        system_prompt: custom system prompt (default: spec-focused).

    Returns:
        (answer, citations) where answer is the generated text and citations
        are [{"section_id": "5.2.1", "section_title": "...", ...}].

    Raises:
        ValueError: if context_chunks is empty.
        BadRequestError: if the API call fails.
    """
    if not context_chunks:
        raise ValueError("No context chunks provided")

    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # Step 1: Assemble context
    context_text, used_chunks = assemble_context(
        query,
        context_chunks,
        max_context_tokens=max_context_tokens,
    )

    # Step 2: Format full prompt
    full_system_prompt = system_prompt.format(context_text, "")  # {context} placeholder
    user_message = query

    # Step 3: Call Sonnet
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=full_system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text if response.content else ""
    tokens_used = {
        "prompt": response.usage.input_tokens,
        "completion": response.usage.output_tokens,
    }

    # Step 4: Extract citations
    citations = _extract_citations(answer, used_chunks)

    return answer, citations, used_chunks, tokens_used


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate answer from context.")
    parser.add_argument("query", help="user query")
    parser.add_argument(
        "context_json",
        type=Path,
        help="JSON file with context chunks (list of dicts)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    parser.add_argument("--json", action="store_true", help="output as JSON")
    args = parser.parse_args(argv)

    with open(args.context_json, encoding="utf-8") as f:
        context_chunks = json.load(f)

    try:
        answer, citations, used_chunks, tokens_used = generate(
            args.query,
            context_chunks,
            model=args.model,
            max_context_tokens=args.max_context_tokens,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    result = GenerationResult(
        answer=answer,
        citations=citations,
        context_used=used_chunks,
        model=args.model,
        tokens_used=tokens_used,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("ANSWER:")
        print(answer)
        print("\nCITATIONS:")
        for c in citations:
            print(f"  [{c['section_id']}] {c['section_title']}")

    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
