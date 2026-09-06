"""LLM-powered node helpers — each sits BEHIND a deterministic contract.

Every helper returns contract-conformant data or raises llm.LLMError; the
calling node falls back to the deterministic path on any failure. Nothing an
LLM produces reaches the pipeline without passing the same validators the
rule-based path uses.
"""
from __future__ import annotations

from medai import llm
from medai.tools.extraction import build_effect

_KNOWN_MEASURES = {"rr", "or", "hr", "md"}


def llm_screen_papers(query: str, papers: list[dict], *, model: str | None = None,
                      audit=None) -> tuple[list[dict], list[dict]]:
    """Topical relevance screening. Returns (kept, excluded) where excluded
    records carry an `exclusion_reason`. Batched 10 records per call.
    Raises llm.LLMError on failure — caller falls back to keyword screening."""
    kept: list[dict] = []
    excluded: list[dict] = []
    for start in range(0, len(papers), 10):
        batch = papers[start:start + 10]
        listing = "\n".join(
            f'[{p["id"]}] {p.get("title", "")} :: {p.get("abstract", "")[:500]}'
            for p in batch)
        data = llm.chat_json([
            {"role": "system",
             "content": ("You are a systematic-review screening assistant. Judge ONLY "
                         "topical relevance to the research question. Be inclusive for "
                         "on-topic records and exclude clearly unrelated ones.")},
            {"role": "user",
             "content": (f'Research question: "{query}"\n'
                         f'Screen each record. Return ONLY JSON: '
                         f'{{"include": [record ids], "exclude": [{{"id": ..., "reason": "..."}}]}}'
                         f'\n\nRecords:\n{listing}')},
        ], model=model, max_tokens=1200, audit=audit)
        inc = {str(x) for x in (data.get("include") or [])}
        exc = {str(x.get("id")): str(x.get("reason", "not relevant"))
               for x in (data.get("exclude") or []) if isinstance(x, dict)}
        for p in batch:
            pid = str(p["id"])
            if pid in exc:
                excluded.append({**p, "exclusion_reason": exc[pid]})
            elif pid in inc or not (inc or exc):
                kept.append(p)  # unparsable verdict -> inclusive default
            else:
                excluded.append({**p, "exclusion_reason": "not relevant"})
    return kept, excluded


def llm_extract_effects(paper: dict, query: str, *, model: str | None = None,
                        audit=None) -> list[dict]:
    """Extract effect estimates from one abstract. Returns validated,
    provenance-bound records (same shape as the rule-based extractor) or
    raises llm.LLMError. The span must appear VERBATIM in the abstract and
    every record passes the same numeric validation as the regex path."""
    messages = [
        {"role": "system",
         "content": ("You are a systematic-review data-extraction assistant. Extract "
                     "only what the text literally reports; never invent values. The "
                     "`span` field must be copied verbatim from the abstract.")},
        {"role": "user",
         "content": (f'Research question: "{query}"\n'
                     f'Abstract (verbatim):\n"""\n{paper.get("abstract", "")}\n"""\n'
                     f'Extract every effect estimate with a 95% confidence interval. '
                     f'measure is one of "rr", "or", "hr", "md". Return ONLY JSON: '
                     f'{{"effects": [{{"measure": "...", "point": 0.0, "lo": 0.0, '
                     f'"hi": 0.0, "span": "<verbatim sentence>"}}]}}. '
                     f'If there are none return {{"effects": []}}.')},
    ]
    data = llm.chat_json(messages, model=model, max_tokens=800, audit=audit)
    out = []
    for e in (data.get("effects") or []):
        if not isinstance(e, dict):
            continue
        measure = str(e.get("measure", "")).lower()
        span = str(e.get("span", ""))
        if measure not in _KNOWN_MEASURES:
            continue
        if span and span not in paper.get("abstract", ""):
            continue  # span must be verbatim — fabricated spans are rejected
        try:
            rec = build_effect(paper["id"], span, measure,
                               float(e["point"]), float(e["lo"]), float(e["hi"]))
        except (KeyError, TypeError, ValueError):
            continue
        if rec:
            rec["extracted_by"] = "llm"
            out.append(rec)
    return out


def llm_rewrite_abstract(sections: list[dict], query: str, *,
                         model: str | None = None, audit=None) -> str:
    """Rewrite the structured abstract as flowing academic prose. The caller
    must verify that every key statistic survives (numbers-preserving check);
    this function only produces candidate text."""
    src = "\n\n".join(f"{s['label']}: {s['text']}" for s in sections)
    messages = [
        {"role": "system",
         "content": ("You are an academic writer for medical systematic reviews. Rewrite "
                     "the given structured abstract as one flowing, formal paragraph in "
                     "past tense for methods/results. CRITICAL: preserve every number, "
                     "confidence interval, I² value and [n] reference marker exactly as "
                     "given. Do not add claims, interpretations or citations that are not "
                     "present in the source text.")},
        {"role": "user", "content": f"Structured abstract:\n\n{src}"},
    ]
    return llm.chat(messages, model=model, max_tokens=900, audit=audit)
