"""Deterministic synthesis: combined abstract + review outline.

Runs as a graph node after citation verification. Everything here is
rule-based and auditable — the LLM-backed writer can later replace this
node behind the same output contract. Reference numbers [n] correspond to
the position of a paper in the run's retrieved-paper list (same numbering
the console renders).
"""
from __future__ import annotations

import time

_MEASURE_LABEL = {"hr": "HR", "or": "OR", "rr": "RR", "md": "MD"}
_DB_NAME = {"pubmed": "PubMed", "scopus": "Scopus", "openalex": "OpenAlex",
            "arxiv": "arXiv", "semantic_scholar": "Semantic Scholar",
            "ieee": "IEEE Xplore", "corpus": "the built-in corpus",
            "fixture": "the built-in corpus"}


def _fmt_p(p: float) -> str:
    return "p < 0.001" if p <= 0.001 else f"p = {p:.3f}"


def _fmt_date(ts: float) -> str:
    return time.strftime("%B %d, %Y", time.localtime(ts))


def _fmt_effect(e: dict) -> str:
    label = _MEASURE_LABEL.get(e["measure"], str(e["measure"]).upper())
    f = lambda v: round(v * 1000) / 1000  # noqa: E731
    return f"{label} {f(e['point'])} (95% CI {f(e['lo'])} to {f(e['hi'])})"


def relevance_rank(papers: list[dict], evidence: list[dict], query: str) -> list[dict]:
    """Transparent relevance score: contributed an effect (+3 per paper),
    plus query-term overlap with title+abstract (+0.5 per term, capped)."""
    terms = [t for t in query.lower().split() if len(t) > 2]
    eff_ids = {e["paper_id"] for e in evidence}
    scored = []
    for i, p in enumerate(papers, 1):
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        overlap = sum(1 for t in terms if t in text)
        scored.append({
            "ref": i, "id": p["id"], "title": p.get("title", ""),
            "has_effect": p["id"] in eff_ids,
            "score": round(3.0 * (p["id"] in eff_ids) + 0.5 * min(overlap, 6), 2),
        })
    scored.sort(key=lambda d: (-d["score"], d["ref"]))
    return scored


def _direction(analysis: dict) -> str | None:
    if analysis.get("effect_measure") == "ratio":
        null = 1.0
        word = "below the null (fewer events under the indexed condition)" \
            if analysis["pooled_point"] < null else "above the null"
    else:
        null = 0.0
        word = "positive" if analysis["pooled_point"] > null else "negative"
    return f"The pooled point estimate lies {word} (null = {null:g})."


def _heterogeneity_word(i2: float) -> str:
    if i2 < 25: return "low"
    if i2 < 50: return "moderate"
    if i2 < 75: return "substantial"
    return "very high"


def build_synthesis(query: str, papers: list[dict], evidence: list[dict],
                    analysis: dict | None, sources_used: list[str] | None = None,
                    prisma: dict | None = None, created: float | None = None) -> dict:
    """Compose the combined abstract, review outline, and relevance ranking."""
    ranked = relevance_rank(papers, evidence, query)
    ref_of = {p["id"]: i + 1 for i, p in enumerate(papers)}
    srcs = sorted({p.get("source", "fixture") for p in papers}) or \
        sorted(set(sources_used or []))

    # ---- combined abstract (journal structured format) ----------------------
    eff_by_paper: dict[str, list[dict]] = {}
    for e in evidence:
        eff_by_paper.setdefault(e["paper_id"], []).append(e)

    identified_n = sum((prisma or {}).get("identified", {}).values()) or len(papers)
    screened_n = (prisma or {}).get("screened", len(papers))
    included_n = (prisma or {}).get("included", len(papers))
    db_names = ", ".join(_DB_NAME.get(s, s) for s in
                         sorted({p.get("source", "fixture") for p in papers})) or         ", ".join(_DB_NAME.get(x, x) for x in (sources_used or []))
    date = _fmt_date(created or time.time())
    search_verb = "was searched" if "," not in db_names else "were searched"
    study_word = "study" if included_n == 1 else "studies"

    background = (f"Primary evidence relevant to “{query}” is indexed across multiple "
                  f"bibliographic databases and reported in heterogeneous effect metrics, "
                  f"which hinders direct comparison of individual study results.")
    objective = (f"We conducted a PRISMA-style rapid systematic review to identify, screen "
                 f"and quantitatively synthesise studies relevant to “{query}”.")
    methods = (f"{db_names} {search_verb} on {date}. Records were deduplicated "
               f"({identified_n} identified; {screened_n} screened after de-duplication) and "
               f"{included_n} {study_word} met keyword-based eligibility criteria. Effect estimates "
               f"(RR, OR, HR or MD with 95% CIs) were extracted with sentence-level provenance "
               f"and pooled by inverse-variance weighting; a random-effects (DerSimonian–Laird) "
               f"model was prespecified for I² ≥ 50%. The pooling step was performed only after "
               f"explicit human approval.")
    if analysis:
        p_str = _fmt_p(analysis.get("p_value", 0))
        het_word = _heterogeneity_word(analysis.get("I2", 0))
        results = (f"The searches identified {identified_n} records; {screened_n} were screened "
                   f"and {included_n} met the eligibility criteria. {len(eff_by_paper)} {'study' if len(eff_by_paper) == 1 else 'studies'} contributed "
                   f"{len(evidence)} {'estimate' if len(evidence) == 1 else 'estimates'} ("
                   + "; ".join(
                       f"{_fmt_effect(e)} [{ref_of[e['paper_id']]}]"
                       for e in sorted(evidence, key=lambda e: ref_of.get(e["paper_id"], 99))
                       if e["paper_id"] in ref_of)
                   + f"). The pooled estimate was {analysis['pooled_point']} "
                   f"(95% CI {analysis['ci_lo']}–{analysis['ci_hi']}; {p_str}), with "
                   f"{het_word} heterogeneity (I² = {analysis['I2']}%; "
                   f"τ² = {analysis.get('tau2', 0)}).")
        if analysis["effect_measure"] == "ratio":
            direction = ("an inverse summary association (pooled ratio below 1.0)"
                         if analysis["pooled_point"] < 1 else
                         "a positive summary association (pooled ratio above 1.0)")
        else:
            direction = ("a positive summary difference" if analysis["pooled_point"] > 0
                         else "a negative summary difference")
        conclusions = (f"The available evidence suggests {direction} for outcomes related to "
                       f"“{query}”. Given {het_word} heterogeneity, abstract-level data and "
                       f"an automated single-screen review, these findings should be "
                       f"interpreted as exploratory and confirmed in a fully manual "
                       f"systematic review.")
    else:
        results = (f"Although {len(papers)} records were included, none reported effect "
                   f"estimates in a poolable metric; a narrative summary of the included "
                   f"records is provided instead of quantitative synthesis.")
        conclusions = (f"No poolable quantitative evidence was identified for “{query}” in "
                       f"the searched databases. Re-scoping the question toward comparative "
                       f"studies, or extending extraction to further effect metrics, may be "
                       f"necessary before quantitative synthesis is possible.")

    methods = methods[0].upper() + methods[1:]  # sentence-case the opening

    abstract_sections = [
        {"label": "Background", "text": background},
        {"label": "Objective", "text": objective},
        {"label": "Methods", "text": methods},
        {"label": "Results", "text": results},
        {"label": "Conclusions", "text": conclusions},
    ]
    abstract = " ".join(f"{sec['label']}: {sec['text']}" for sec in abstract_sections)

    # ---- review outline ----------------------------------------------------
    # section 1 = every paper that contributed an effect estimate, ranked by
    # relevance; section 3 = pure context papers (no poolable effects)
    contributing = [r for r in ranked if r["has_effect"]]
    context = [r for r in ranked if not r["has_effect"]]
    outline = [{
        "heading": "1. Most relevant papers (contributing effect estimates)",
        "bullets": [
            f"[{r['ref']}] {r['title']} — "
            + "; ".join(_fmt_effect(e) for e in eff_by_paper.get(r["id"], []))
            for r in contributing
        ],
    }]
    if analysis:
        bullets = [
            f"Pooled estimate: {analysis['pooled_point']} (95% CI {analysis['ci_lo']} to "
            f"{analysis['ci_hi']}), {analysis['model']} model, k={analysis['k']}",
            f"Heterogeneity: I²={analysis['I2']}%, τ²={analysis.get('tau2', 0)}, Q={analysis.get('Q')}",
        ]
        fe = analysis.get("fixed_effect")
        if fe:
            bullets.append("Sensitivity anchor: fixed-effect "
                           f"{fe['point']} {tuple(fe['ci'])}")
        outline.append({"heading": "2. Quantitative synthesis", "bullets": bullets})
    if context:
        outline.append({
            "heading": "3. Context and background papers (no poolable effects)",
            "bullets": [f"[{r['ref']}] {r['title']}" for r in context],
        })
    outline.append({
        "heading": "4. Limitations and next steps",
        "bullets": [
            "Abstract-level data only; full-text review may change conclusions",
            (f"Substantial heterogeneity (I²={analysis['I2']}%) — interpret the pooled "
             "estimate as exploratory" if analysis and analysis.get("I2", 0) >= 50
             else "Consider subgroup or meta-regression if heterogeneity grows"),
            f"{len(papers) - len(eff_by_paper)} of {len(papers)} papers lacked poolable "
            "effects; targeted re-search could close the gap",
        ],
    })

    return {"abstract": abstract, "abstract_sections": abstract_sections,
            "outline": outline, "most_relevant": ranked[:5]}
