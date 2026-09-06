"""OpenRouter LLM client: cached, retrying, audited, opt-in.

Design contract (matches the project thesis):
  * OFF by default — without OPENROUTER_API_KEY nothing in the pipeline calls
    a model and behaviour is byte-identical to the deterministic build.
  * Every response is cached to disk keyed by sha256(model + messages), so
    re-runs are reproducible and the free tier's rate limits are respected.
  * Every call (or cache hit) is recorded in the hash-chained audit log.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


class LLMError(RuntimeError):
    """Raised when the LLM is unavailable, fails, or answers unusably."""


def llm_enabled() -> bool:
    """True only when an OpenRouter API key is configured."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _model_chain(explicit: str | None = None) -> list[str]:
    """Models to try in order: explicit arg > LLM_MODEL_CHAIN > LLM_MODEL > default.
    Free-model capacity on OpenRouter is bursty, so a chain mirrors the
    multi-source retrieval philosophy: rotate instead of fail."""
    if explicit:
        return [explicit]
    chain = os.environ.get("LLM_MODEL_CHAIN") or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    return [m.strip() for m in chain.split(",") if m.strip()] or [DEFAULT_MODEL]


def _cache_dir() -> str:
    d = os.environ.get("LLM_CACHE_DIR", "llm-cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path(key: str) -> str:
    d = _cache_dir()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, key + ".json")


_LAST_CALL = [0.0]  # shared pacing clock across threads


def _post(body: dict, timeout: float = 60.0) -> dict:
    # free-tier pacing: never start calls faster than LLM_MIN_INTERVAL seconds
    # apart, and back off generously (5/15/30s) when a shared-capacity 429 hits
    min_interval = float(os.environ.get("LLM_MIN_INTERVAL", "3"))
    wait = _LAST_CALL[0] + min_interval - time.time()
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(
        OPENROUTER_URL, method="POST",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/raselahmed1337/medai-audit-agent",
            "X-Title": "MedAI Research Agent",
        })
    last: Exception | None = None
    backoff = (5, 15, 30)
    for attempt in range(len(backoff) + 1):
        _LAST_CALL[0] = time.time()
        try:
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503) and attempt < len(backoff):
                time.sleep(backoff[attempt])
                continue
            raise LLMError(f"OpenRouter HTTP {e.code}") from e
        except Exception as e:  # network errors are not retried (fail fast)
            raise LLMError(repr(e)) from e
    raise LLMError(f"OpenRouter failed after retries: {last!r}")


def chat(messages: list[dict], *, model: str | None = None, json_mode: bool = False,
         max_tokens: int = 900, audit=None) -> str:
    """Send a chat completion; returns the assistant text.

    Responses are cached on disk (replay = identical + free). Audit records
    llm_call on network calls and llm_cache on cache hits. If the primary
    model exhausts its retries on rate limits, the model chain rotates."""
    if not llm_enabled():
        raise LLMError("OPENROUTER_API_KEY not configured")
    last_err: LLMError | None = None
    for m in _model_chain(model):
        try:
            return _chat_one(messages, m, json_mode=json_mode,
                             max_tokens=max_tokens, audit=audit)
        except LLMError as e:
            last_err = e
            if "HTTP 4" in str(e) and "429" not in str(e):
                raise  # bad request / auth: other models won't help
    raise last_err or LLMError("no models configured")


def _chat_one(messages: list[dict], model: str, json_mode: bool = False,
              max_tokens: int = 900, audit=None) -> str:
    key_payload = json.dumps({"model": model, "messages": messages,
                              "json_mode": json_mode}, sort_keys=True)
    digest = hashlib.sha256(key_payload.encode()).hexdigest()
    cpath = _cache_path(digest)
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            cached = json.load(f)
        if audit:
            audit.record("llm_cache", model=model, prompt_sha=digest[:12])
        return cached["content"]

    body: dict = {"model": model, "messages": messages,
                  "temperature": 0, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    t0 = time.time()
    data = _post(body)
    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump({"messages": messages, "content": content, "model": model,
                   "ts": time.time()}, f)
    if audit:
        audit.record("llm_call", model=model, prompt_sha=digest[:12],
                     cached=False, tokens_in=usage.get("prompt_tokens"),
                     tokens_out=usage.get("completion_tokens"),
                     latency_ms=int((time.time() - t0) * 1000))
    return content


def chat_json(messages: list[dict], *, model: str | None = None,
              max_tokens: int = 900, audit=None) -> dict:
    """chat() + strict JSON parsing with one repair round-trip."""
    raw = chat(messages, model=model, json_mode=True, max_tokens=max_tokens, audit=audit)
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    repaired = chat(messages + [
        {"role": "assistant", "content": raw},
        {"role": "user", "content": "Return ONLY the valid JSON object. No prose, no code fences."},
    ], model=model, json_mode=True, max_tokens=max_tokens, audit=audit)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", repaired.strip(), flags=re.S).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMError(f"model returned invalid JSON after repair: {raw[:120]!r}") from e
