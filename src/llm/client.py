"""
Thin wrapper around an LLM provider (Gemini or OpenAI).

Everything the rest of the codebase needs to talk to an LLM goes through
`generate()`. Provider, model, retry behavior, and rate limiting are all
contained here — swap provider by setting LLM_PROVIDER.

Provider selection:
  LLM_PROVIDER=gemini   (default)   -> Gemini 2.5 Flash via google-genai
  LLM_PROVIDER=openai               -> GPT-4o-mini via openai SDK

Required env vars:
  GEMINI_API_KEY  (if provider=gemini)
  OPENAI_API_KEY  (if provider=openai)

Optional:
  LLM_MODEL       override the per-provider default model
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Config

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower().strip()

_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}
DEFAULT_MODEL = os.environ.get("LLM_MODEL") or _DEFAULT_MODELS.get(PROVIDER, "gemini-2.5-flash")

# Conservative pacing between calls. Paid tiers can handle much more, but we
# keep this modest so we don't burst-trip any per-minute limits.
#   gemini free tier: ~9 RPM
#   openai tier-1:    500 RPM   (so 0.2s is plenty)
_PACING = {
    "gemini": 6.5,
    "openai": 0.25,
}
MIN_SECONDS_BETWEEN_CALLS = _PACING.get(PROVIDER, 6.5)

MAX_RETRIES = 5
BACKOFF_BASE = 5.0  # seconds


# ---------------------------------------------------------------------------
# Result type

@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int | None
    output_tokens: int | None


# ---------------------------------------------------------------------------
# Shared pacing

_last_call_ts: float = 0.0


def _pace() -> None:
    global _last_call_ts
    now = time.monotonic()
    elapsed = now - _last_call_ts
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_ts = time.monotonic()


# ---------------------------------------------------------------------------
# Gemini backend

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai  # lazy import

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a key at https://aistudio.google.com/apikey"
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _gemini_generate(
    prompt: str,
    *,
    model: str,
    system: str | None,
    temperature: float,
    max_output_tokens: int,
    json_mode: bool,
    disable_thinking: bool,
) -> LLMResult:
    from google.genai import errors as genai_errors
    from google.genai import types

    client = _get_gemini_client()

    cfg_kwargs: dict = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if system is not None:
        cfg_kwargs["system_instruction"] = system
    if json_mode:
        cfg_kwargs["response_mime_type"] = "application/json"
    if disable_thinking:
        try:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:  # noqa: BLE001
            pass

    config = types.GenerateContentConfig(**cfg_kwargs)

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        _pace()
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            usage = getattr(resp, "usage_metadata", None)
            return LLMResult(
                text=resp.text or "",
                model=model,
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
            )
        except genai_errors.ClientError as e:
            last_exc = e
            if getattr(e, "code", None) == 429:
                delay = _retry_delay_from_error(e) or (BACKOFF_BASE * (attempt + 1))
                print(f"  [gemini rate-limited, sleeping {delay:.1f}s]")
                time.sleep(delay)
                continue
            raise
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(BACKOFF_BASE * (attempt + 1))

    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# OpenAI backend

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI  # lazy import

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Get a key at https://platform.openai.com/api-keys"
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _openai_generate(
    prompt: str,
    *,
    model: str,
    system: str | None,
    temperature: float,
    max_output_tokens: int,
    json_mode: bool,
) -> LLMResult:
    from openai import RateLimitError, APIError  # lazy import

    client = _get_openai_client()

    messages: list[dict] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        _pace()
        try:
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            text = choice.message.content or ""
            usage = resp.usage
            return LLMResult(
                text=text,
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            )
        except RateLimitError as e:
            last_exc = e
            delay = BACKOFF_BASE * (attempt + 1)
            print(f"  [openai rate-limited, sleeping {delay:.1f}s]")
            time.sleep(delay)
        except APIError as e:
            last_exc = e
            time.sleep(BACKOFF_BASE * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(BACKOFF_BASE * (attempt + 1))

    raise RuntimeError(f"OpenAI call failed after {MAX_RETRIES} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Public API

def generate(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    json_mode: bool = False,
    disable_thinking: bool = True,
) -> LLMResult:
    """
    Synchronous text generation. Blocks on rate-limit, retries on transient
    errors. Dispatches to the provider selected by LLM_PROVIDER.

    disable_thinking applies only to Gemini 2.5 models (ignored elsewhere).
    """
    if PROVIDER == "openai":
        return _openai_generate(
            prompt,
            model=model,
            system=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_mode=json_mode,
        )
    # default: gemini
    return _gemini_generate(
        prompt,
        model=model,
        system=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        json_mode=json_mode,
        disable_thinking=disable_thinking,
    )


def generate_json(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    disable_thinking: bool = True,
) -> tuple[dict | list, LLMResult]:
    """Request JSON-mode and parse. Returns (parsed_object, raw_result)."""
    result = generate(
        prompt,
        model=model,
        system=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        json_mode=True,
        disable_thinking=disable_thinking,
    )
    text = result.text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned non-JSON despite json_mode:\n{result.text[:500]}"
        ) from e
    return parsed, result


# ---------------------------------------------------------------------------
# Helpers

_RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


def _retry_delay_from_error(e: Exception) -> float | None:
    msg = str(e)
    m = _RETRY_DELAY_RE.search(msg)
    if m:
        try:
            return float(m.group(1)) + 1.0
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# CLI smoke test

if __name__ == "__main__":
    print(f"provider={PROVIDER}  model={DEFAULT_MODEL}")
    r = generate("Reply with the single word OK.")
    print(f"tokens_in={r.prompt_tokens} tokens_out={r.output_tokens}")
    print(f"reply: {r.text!r}")
