"""A1 — naive pipeline baseline.

No guard (so injected failures raise straight through or pass silently),
no human approval, no citation verification.
"""
from __future__ import annotations

from medai.audit import AuditLog
from medai.tools.analysis import meta_analysis
from medai.tools.extraction import extract_evidence
from medai.tools.retrieval import infer_topic_keywords, screen_papers, search_papers
from experiments.runresult import RunResult


def run(query: str, failure_rate: float, seed: int) -> RunResult:
    res = RunResult(architecture="A1_naive", query=query, status="ok")
    audit = AuditLog()
    # emulate the same fault draw sequence by consuming equivalent randomness
    # through unguarded calls guarded only for injection (no detection):
    if failure_rate > 0:
        import random
        rng = random.Random(seed)
        # A1 has no retries; one injected fault kills the run (crash) or
        # silently corrupts it (malformed output flows into the report)
        try:
            papers = screen_papers(search_papers(query), infer_topic_keywords(query))
            if rng.random() < failure_rate and rng.random() < 0.5:
                papers = [p for p in papers if p is None] or [None]
            audit.tool_call("search_papers", {}, status="ok")
        except Exception as e:
            res.status = "failed:search_papers"
            audit.tool_failure("search_papers", {}, repr(e), "unhandled", 1)
            _finish(res, audit)
            return res
        try:
            effects = extract_evidence(papers)
            if rng.random() < failure_rate and rng.random() < 0.5:
                effects = [e for e in effects if e.get("se") is None] or [dict(paper_id="?", y=0, se=0)]
        except Exception as e:
            res.status = "failed:extract_evidence"
            audit.tool_failure("extract_evidence", {}, repr(e), "unhandled", 1)
            _finish(res, audit)
            return res
        try:
            analysis = meta_analysis(effects)
            if rng.random() < failure_rate and rng.random() < 0.5:
                analysis = dict(analysis, pooled_point=None)
        except Exception as e:
            res.status = "failed:run_meta_analysis"
            audit.tool_failure("run_meta_analysis", {}, repr(e), "unhandled", 1)
            _finish(res, audit)
            return res
    else:
        papers = screen_papers(search_papers(query), infer_topic_keywords(query))
        audit.tool_call("search_papers", {})
        effects = extract_evidence(papers)
        analysis = meta_analysis(effects)

    res.analysis_ran = True
    res.pooled_point = analysis.get("pooled_point")
    res.ci = (analysis.get("ci_lo"), analysis.get("ci_hi"))
    # naive report cites whatever the analysis lists, unverified
    res.citations = analysis.get("cited_papers", [])
    audit.report(f"pooled={res.pooled_point}", res.citations)
    _finish(res, audit)
    return res


def _finish(res: RunResult, audit: AuditLog):
    res.audit_ok = audit.verify_chain()
    res.report = f"pooled={res.pooled_point} status={res.status}"
    res.guard_stats = {"injected_or_not_detected": True} if res.status.startswith("failed") else {}
