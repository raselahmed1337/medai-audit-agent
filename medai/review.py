"""PRISMA-style rapid review builder.

Composes a complete short-review manuscript from the run's own artifacts:
structured abstract, PRISMA flow numbers, methods (with the exact executed
search strategies), results, discussion, limitations, and references.

Scope honesty is part of the contract: the review self-identifies as an
automated rapid review (single automated screener, abstract-level data,
no formal risk-of-bias assessment) — never as a fully compliant systematic
review. Deterministic; an LLM writer can later replace this node behind the
same output contract.
"""
from __future__ import annotations

import time

_MEASURE_LABEL = {"hr": "HR", "or": "OR", "rr": "RR", "md": "MD"}
_SOURCE_NAMES = {"pubmed": "PubMed", "openalex": "OpenAlex", "arxiv": "arXiv",
                 "semantic_scholar": "Semantic Scholar", "corpus": "built-in corpus",
                 "fixture": "built-in corpus"}


def _fmt_effect(e: dict) -> str:
    label = _MEASURE_LABEL.get(e["measure"], str(e["measure"]).upper())
    f = lambda v: round(v * 1000) / 1000  # noqa: E731
    return f"{label} {f(e['point'])} (95% CI {f(e['lo'])} to {f(e['hi'])})"


def _fmt_date(ts: float) -> str:
    return time.strftime("%B %d, %Y", time.localtime(ts))


def build_review(query: str, papers: list[dict], evidence: list[dict],
                 analysis: dict | None, prisma: dict, sources_failed: list[str],
                 search_strategy: dict, created: float) -> dict:
    ref_of = {p["id"]: i + 1 for i, p in enumerate(papers)}
    eff_by_paper: dict[str, list[dict]] = {}
    for e in evidence:
        eff_by_paper.setdefault(e["paper_id"], []).append(e)

    identified = {(_SOURCE_NAMES.get(k, k)): v for k, v in (prisma.get("identified") or {}).items()}
    screened = prisma.get("screened", len(papers))
    duplicates = prisma.get("duplicates_removed", 0)
    excluded_screen = prisma.get("excluded_screen", 0)
    failed = [_SOURCE_NAMES.get(s, s) for s in sources_failed]

    # ---- PRISMA flow -------------------------------------------------------
    flow = {
        "identified": [{"database": db, "n": n} for db, n in identified.items()],
        "unavailable": failed,
        "duplicates_removed": duplicates,
        "screened": screened,
        "excluded": excluded_screen,
        "included": len(papers),
        "with_effects": len(eff_by_paper),
        "effect_estimates": len(evidence),
    }

    # ---- structured abstract ----------------------------------------------
    pool_sentence = ""
    if analysis:
        model_desc = ("random-effects (DerSimonian–Laird)"
                      if analysis["model"] == "random-effects" else "fixed-effect")
        pool_sentence = (f"Pooling with the {model_desc} model gave "
                         f"{analysis['pooled_point']} (95% CI {analysis['ci_lo']} to "
                         f"{analysis['ci_hi']}; k={analysis['k']}, I²={analysis['I2']}%).")
    abstract = {
        "background": (f"Primary evidence relevant to “{query}” is indexed across multiple "
                       "bibliographic databases and reported in heterogeneous effect metrics, "
                       "which hinders direct comparison of individual study results."),
        "objective": (f"To conduct a PRISMA-style rapid systematic review that identifies, "
                      f"screens and quantitatively synthesises studies relevant to "
                      f"“{query}”, with provenance-tracked extraction and human-approved "
                      f"statistical synthesis."),
        "methods": (f"Following PRISMA reporting structure, we searched PubMed, OpenAlex, "
                    f"arXiv and Semantic Scholar on {_fmt_date(created)} (search strings in "
                    f"Methods). Records were deduplicated and screened by an automated "
                    f"rule-based screener; effect estimates (RR/OR/HR/MD with 95% CIs) were "
                    f"extracted with provenance to source abstracts. A human approver gated "
                    f"the statistical synthesis."),
        "results": (f"Of {screened} screened records, {len(papers)} were included; "
                    f"{len(eff_by_paper)} reported extractable effect estimates "
                    f"({len(evidence)} estimates in total). " + pool_sentence),
        "conclusions": (f"Automated, provenance-tracked synthesis can produce a reproducible "
                        f"rapid evidence summary for “{query}”, but conclusions remain "
                        f"exploratory pending full-text review and risk-of-bias assessment."),
    }

    # ---- methods ------------------------------------------------------------
    methods = {
        "search_strategy": [
            {"database": db, "query": q} for db, q in search_strategy.items()
        ],
        "eligibility": [
            "Included: records whose title or abstract mentions at least one topic term "
            "derived from the research question (automated keyword screen).",
            "Included: records with a machine-readable abstract.",
            "Excluded: records failing the keyword screen; records without abstracts.",
            "Preprints (arXiv) are retained and flagged, as they are primary evidence for "
            "computer-science aspects of the question.",
        ],
        "extraction": ("Effect estimates (risk ratios, odds ratios, hazard ratios, mean "
                       "differences with 95% CIs) were extracted by a rule-based parser and "
                       "bound to the source record and verbatim sentence; unrecognisable "
                       "metrics were skipped rather than guessed."),
        "synthesis": ("Inverse-variance pooling with automatic model selection: "
                      "random-effects (DerSimonian–Laird) when I² ≥ 50%, otherwise "
                      "fixed-effect. The synthesis step required explicit human approval "
                      "before execution."),
    }

    # ---- results ------------------------------------------------------------
    characteristics = []
    for i, p in enumerate(papers, 1):
        effs = "; ".join(_fmt_effect(e) for e in eff_by_paper.get(p["id"], []))
        characteristics.append({
            "ref": i, "id": p["id"], "title": p.get("title", ""),
            "source": _SOURCE_NAMES.get(p.get("source", ""), p.get("source", "")),
            "effect": effs or None,
        })
    from medai.synthesis import build_highlights
    results = {
        "prisma": flow,
        "study_characteristics": characteristics,
        "key_points": build_highlights(papers, evidence),
        "synthesis": (analysis or None),
        "narrative": (None if analysis else
                      "No pooled estimate was computed: none of the included abstracts "
                      "reported effect estimates in a poolable metric. A narrative summary "
                      "of the included records is provided in the references."),
    }

    # ---- discussion / limitations -------------------------------------------
    hetero = analysis.get("I2") if analysis else None
    discussion = [
        (f"Across {len(papers)} included records, {len(eff_by_paper)} contributed "
         f"quantitative evidence. "
         + (f"The pooled estimate was {analysis['pooled_point']} "
            f"(95% CI {analysis['ci_lo']} to {analysis['ci_hi']}) under the "
            f"{analysis['model']} model." if analysis else
            "The evidence base describes the landscape of the question without yielding "
            "a common effect metric.")),
        ("Relevance-ranked, the most informative records were: "
         + ", ".join(f"[{ref_of[p['id']]}] {p.get('title', '')}"
                     for p in papers if p["id"] in eff_by_paper)[:400]),
        ("Heterogeneity was "
         + (f"substantial (I²={hetero}%); the pooled estimate should be read as "
            "exploratory and hypothesis-generating." if analysis and hetero and hetero >= 50 else
            (f"moderate (I²={hetero}%)." if analysis else
             "not quantifiable without a common effect metric."))),
    ]
    limitations = [
        "Automated single-screen review: records were screened by keyword rules, not by "
        "two independent human reviewers.",
        "Abstract-level data only; full texts were not retrieved, and results reported "
        "outside abstracts were missed.",
        "No formal risk-of-bias assessment was performed.",
        "arXiv records are preprints and not peer-reviewed.",
        "Rule-based extraction recognises RR/OR/HR/MD with 95% CIs; other metrics "
        "(e.g. AUC-type discrimination measures) are deliberately not pooled.",
        "The search reflects a single snapshot; no update or grey-literature search "
        "beyond the configured databases was performed.",
    ]
    conclusion = (f"For “{query}”, automated retrieval and provenance-tracked synthesis "
                  f"produced a reproducible rapid evidence summary from {len(papers)} "
                  f"records. " + (f"The pooled association was {analysis['pooled_point']} "
                                  f"(95% CI {analysis['ci_lo']} to {analysis['ci_hi']}); "
                                  "confirmation requires a fully manual systematic review."
                                  if analysis else
                                  "A poolable evidence base may require re-scoping the "
                                  "question toward comparative studies."))

    title = f"“{query}”: a PRISMA-style rapid systematic review with automated meta-analysis"

    references = [{
        "n": ref_of[p["id"]], "id": p["id"], "title": p.get("title", ""),
        "source": _SOURCE_NAMES.get(p.get("source", ""), p.get("source", "")),
    } for p in papers]

    return {
        "title": title,
        "generated_at": _fmt_date(created),
        "abstract": abstract,
        "prisma": flow,
        "methods": methods,
        "results": results,
        "discussion": discussion,
        "limitations": limitations,
        "conclusion": conclusion,
        "references": references,
    }
