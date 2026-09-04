"""A2 — ReAct-style single-agent loop baseline.

One controller loop calling the same tools through a guard that only
*surfaces* errors back to the controller (crash failures retried once);
no output-schema validation (silent malformed output passes), no structural
approval gate, no citation verification.
"""
from __future__ import annotations

from medai.audit import AuditLog
from medai.tools.analysis import meta_analysis_fixed
from medai.tools.extraction import extract_evidence
from medai.tools.guard import ToolError, ToolGuard
from medai.tools.retrieval import infer_topic_keywords, screen_papers, search_papers
from experiments.runresult import RunResult


def run(query: str, failure_rate: float, seed: int) -> RunResult:
    res = RunResult(architecture="A2_react", query=query, status="ok")
    audit = AuditLog()
    # ReAct-style guard: crash errors surfaced to controller, retry once;
    # no validator -> silent (malformed) failures go undetected.
    g_search = ToolGuard(search_papers, "search_papers", validator=None,
                         failure_rate=failure_rate, fault_type="crash",
                         max_retries=1, seed=seed, audit=audit)
    g_extract = ToolGuard(extract_evidence, "extract_evidence", validator=None,
                          failure_rate=failure_rate, fault_type="crash",
                          max_retries=1, seed=seed + 1, audit=audit)
    g_analyze = ToolGuard(meta_analysis_fixed, "run_meta_analysis", validator=None,
                          failure_rate=failure_rate, fault_type="crash",
                          max_retries=1, seed=seed + 2, audit=audit)

    # controller loop ("thought -> action -> observation")
    try:
        papers = screen_papers(g_search.call(query), infer_topic_keywords(query))
        effects = g_extract.call(papers)
        # silent-failure injection representative of unvalidated output
        analysis = g_analyze.call(effects)
        import random
        rng = random.Random(seed + 3)
        if rng.random() < failure_rate and rng.random() < 0.5:
            analysis = dict(analysis, pooled_point=None)  # undetected corruption
            audit.record("silent_fault", tool="run_meta_analysis", detected=False)
    except ToolError as e:
        res.status = f"failed:{str(e).split(':')[0]}"
        _finish(res, audit)
        return res

    res.analysis_ran = True
    res.pooled_point = analysis.get("pooled_point")
    res.ci = (analysis.get("ci_lo"), analysis.get("ci_hi"))
    res.citations = analysis.get("cited_papers", [])
    audit.report(f"pooled={res.pooled_point}", res.citations)
    _finish(res, audit)
    return res


def _finish(res: RunResult, audit: AuditLog):
    res.audit_ok = audit.verify_chain()
    res.report = f"pooled={res.pooled_point} status={res.status}"
    res.guard_stats = {}
