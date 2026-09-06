"""Evidence extraction: find effect estimates with CIs in abstracts.

Each extracted effect is bound to its paper id and a verbatim span, forming
the units of the provenance ledger.

Two passes per sentence:
  1. strict: a named measure (risk/odds/hazard ratio, (standardized) mean
     difference, between-group difference, Cohen's d) adjacent to the point
     estimate and its 95% CI;
  2. CI-anchored harvest: for remaining "95% CI" occurrences, take the nearest
     preceding number as the point estimate — provided only punctuation
     separates it from the CI — and classify the measure from the sentence
     (full phrases, or uppercase abbreviations like HR/OR/RR/SMD/WMD).
     Sentences whose measure cannot be identified are skipped, never guessed.
"""
from __future__ import annotations

import re
from math import log

# pass 1: measure immediately governing the estimate
_STRICT_EFFECT_RE = re.compile(
    r"(?P<measure>risk\s+ratio|odds\s+ratio|hazard\s+ratio|"
    r"(?:standardi[sz]ed\s+)?mean\s+difference|between[-\s]group\s+difference|"
    r"cohen'?s?\s+d)"
    r"\D{0,20}?(?P<point>-?\d+\.?\d*)\s*[,;]?\s*\(?\s*"
    r"95\s*%?\s*CI\s*[,;:]?\s*"
    r"(?P<lo>-?\d+\.?\d*)\s*(?:to|–|—|-)\s*(?P<hi>-?\d+\.?\d*)",
    re.IGNORECASE)

# pass 2: anchor on the CI itself
_CI_RE = re.compile(
    r"95\s*%?\s*CI\s*[,;:]?\s*"
    r"(?P<lo>-?\d+\.?\d*)\s*(?:to|–|—|-)\s*(?P<hi>-?\d+\.?\d*)",
    re.IGNORECASE)

_POINT_AT_END = re.compile(r"(?P<point>-?\d+\.?\d*)\s*$")

# measure classification for pass 2; abbreviations are matched
# case-sensitively so ordinary words like "or" never qualify
_MEASURE_TOKENS = [
    (re.compile(r"hazard\s+ratio", re.I), "hr"),
    (re.compile(r"\ba?HR\b"), "hr"),
    (re.compile(r"odds\s+ratio", re.I), "or"),
    (re.compile(r"\ba?OR\b"), "or"),
    (re.compile(r"risk\s+ratio", re.I), "rr"),
    (re.compile(r"\bRRs?\b"), "rr"),
    (re.compile(r"(?:standardi[sz]ed\s+)?mean\s+difference|"
                r"between[-\s]group\s+difference|cohen'?s?\s+d", re.I), "md"),
    (re.compile(r"\b(?:S|W)?MD\b"), "md"),
]


def _strict_code(measure: str) -> str:
    m = measure.lower()
    if "hazard" in m:
        return "hr"
    if "odds" in m:
        return "or"
    if "risk" in m:
        return "rr"
    return "md"


def _measure_before(text: str) -> str | None:
    """Code of the measure token occurring last in `text`, or None."""
    best_end, best_code = -1, None
    for pat, code in _MEASURE_TOKENS:
        last = None
        for last in pat.finditer(text):
            pass
        if last and last.end() > best_end:
            best_end, best_code = last.end(), code
    return best_code


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _build(paper_id: str, span: str, measure: str,
           point: float, lo: float, hi: float) -> dict | None:
    if hi == lo:
        return None  # degenerate CI carries no information
    if lo > point or hi < point:
        return None  # internally inconsistent CI
    if measure != "md" and lo <= 0:
        return None  # ratio measures must be positive; mean differences may be negative
    rec = dict(paper_id=paper_id, span=span, measure=measure,
               point=point, lo=lo, hi=hi)
    if measure == "md":
        rec["y"], rec["se"] = point, (hi - lo) / (2 * 1.96)
    else:  # log scale for ratio measures
        rec["y"], rec["se"] = log(point), (log(hi) - log(lo)) / (2 * 1.96)
    return rec


def build_effect(paper_id: str, span: str, measure: str,
                 point: float, lo: float, hi: float) -> dict | None:
    """Public constructor shared by the rule-based and LLM extraction paths:
    validates and assembles one provenance-bound effect record (or None)."""
    return _build(paper_id, span, measure, point, lo, hi)


def extract_effects(paper: dict) -> list[dict]:
    """Extract effect estimates from one paper's title+abstract.

    Returns records: {paper_id, span, measure, point, lo, hi, y, se} where
    y = log point estimate (or raw point for mean differences) and
    se derived from the 95% CI on the same scale.
    """
    out: list[dict] = []
    text = paper.get("title", "") + " " + paper.get("abstract", "")
    for sent in split_sentences(text):
        taken: list[tuple[int, int]] = []
        for m in _STRICT_EFFECT_RE.finditer(sent):
            rec = _build(paper["id"], sent, _strict_code(m.group("measure")),
                         float(m.group("point")), float(m.group("lo")), float(m.group("hi")))
            if rec:
                out.append(rec)
                taken.append(m.span())
        for m in _CI_RE.finditer(sent):
            if any(a <= m.start() < b for a, b in taken):
                continue  # already covered by a strict match
            prefix = sent[:m.start()]
            j = len(prefix)
            while j > 0 and not prefix[j - 1].isalnum():
                j -= 1  # only punctuation/space may separate point from CI
            num = _POINT_AT_END.search(prefix[:j])
            if not num:
                continue
            code = _measure_before(prefix[:num.start("point")])
            if code is None:
                continue  # unknown measure (e.g. "MIE"): never guess
            rec = _build(paper["id"], sent, code, float(num.group("point")),
                         float(m.group("lo")), float(m.group("hi")))
            if rec:
                out.append(rec)
    return out


def extract_evidence(papers: list[dict]) -> list[dict]:
    """Extract from a list of papers. One sentence may carry several effects."""
    effects: list[dict] = []
    for p in papers:
        effects.extend(extract_effects(p))
    return effects


def valid_evidence(result) -> bool:
    return (isinstance(result, list)
            and all(isinstance(e, dict) and "paper_id" in e and "y" in e and "se" in e
                    and isinstance(e["se"], (int, float)) and e["se"] > 0 for e in result))
