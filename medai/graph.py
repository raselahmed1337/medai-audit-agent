"""Flagship architecture (A3): LangGraph supervisor with

  plan ──Send fan-out──▶ [search per source] ──▶ merge ──▶ extract
       ──▶ [human approval gate] ──▶ analyze ──▶ verify ──▶ report

Properties that distinguish it from the baselines (experiments/):
  * source retrieval fans out as parallel graph nodes (Send API) — one
    guarded tool per source, latency = slowest source, and per-source
    audit events;
  * structural human-in-the-loop approval before any sensitive analysis tool;
  * tool failure detection + retry + graceful escalation (never silent);
  * provenance-enforced report: every citation must exist in the ledger;
  * meta-analysis with automatic fixed/random-effects model selection.
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from medai import approvals
from medai.audit import AuditLog
from medai import llm
from medai.llm_nodes import (llm_extract_effects,  # noqa: E402
                               llm_rewrite_abstract, llm_screen_papers)
from medai.review import build_review
from medai.synthesis import build_synthesis
from medai.tools.analysis import meta_analysis, valid_analysis
from medai.tools.extraction import extract_evidence, valid_evidence
from medai.tools.guard import ToolError, ToolGuard
from medai.tools.retrieval import (DEFAULT_LIVE_SOURCES, _arxiv_search,  # noqa: E402
                                   _fixture_search, _ieee_search,
                                   _openalex_search, _pubmed_search,
                                   _scopus_search, _semantic_scholar_search,
                                   available_sources, infer_topic_keywords,
                                   merge_sources, pubmed_effective_query,
                                   screen_papers, valid_paper_list)


class ResearchState(TypedDict):
    query: str
    max_results: int
    live: bool
    # accumulated artifacts
    papers: list[dict]
    evidence: list[dict]
    analysis: dict | None
    report: str
    citations: list[str]
    synthesis: dict | None
    review: dict | None
    llm: dict | None
    prisma: dict | None
    status: str  # ok | declined | no_evidence | no_extractable_evidence | failed:<tool>
    audit_events: Annotated[list[dict], operator.add]
    # parallel retrieval fan-out
    source_results: Annotated[list[dict], operator.add]  # [{source, papers}]
    sources_failed: Annotated[list[str], operator.add]


def build_toolchain(failure_rate: float = 0.0, fault_type: str = "mixed",
                    seed: int = 0, audit: AuditLog | None = None) -> dict:
    """Guarded tools shared by all architectures (same fault model). One guard
    per retrieval source so the fan-out nodes fail/retry independently."""
    def guarded(name, fn, validator, seed_off):
        return ToolGuard(fn, name, validator=validator, failure_rate=failure_rate,
                         fault_type=fault_type, seed=seed + seed_off, audit=audit)
    return {
        "search_corpus": guarded("search_corpus", _fixture_search, valid_paper_list, 0),
        "search_pubmed": guarded("search_pubmed", _pubmed_search, valid_paper_list, 1),
        "search_openalex": guarded("search_openalex", _openalex_search, valid_paper_list, 2),
        "search_arxiv": guarded("search_arxiv", _arxiv_search, valid_paper_list, 3),
        "search_scopus": guarded("search_scopus", _scopus_search, valid_paper_list, 4),
        "search_ieee": guarded("search_ieee", _ieee_search, valid_paper_list, 5),
        "search_semantic_scholar": guarded("search_semantic_scholar",
                                           _semantic_scholar_search, valid_paper_list, 6),
        "extract_evidence": guarded("extract_evidence", extract_evidence, valid_evidence, 7),
        "run_meta_analysis": guarded("run_meta_analysis", meta_analysis, valid_analysis, 8),
    }


# --------------------------------------------------------------------------
# nodes

def _plan(state: ResearchState, tools) -> list[Send]:
    """Choose sources and fan out one parallel search node per source."""
    audit = tools["search_corpus"].audit
    live = bool(state.get("live", False))
    sources = list(available_sources()) if live else ["corpus"]
    audit.record("retrieval_mode", live=live, sources=sources,
                 pubmed_query=pubmed_effective_query(state["query"]) if live else None)
    return [Send("search_source", {"source": s, "query": state["query"],
                                   "max_results": state.get("max_results", 10),
                                   "live": live, "status": "ok"})
            for s in sources]


def _search_source(state: ResearchState, tools) -> dict:
    """One guarded search against a single source (executes in parallel)."""
    src = state["source"]
    guard = tools[f"search_{src}"]
    try:
        papers = guard.call(state["query"], state.get("max_results", 10))
    except ToolError as e:
        return {"sources_failed": [src],
                "source_results": [{"source": src, "papers": []}],
                "audit_events": [guard.audit.record(
                    "node_failure", node=f"search_{src}", error=str(e))]}
    return {"source_results": [{"source": src, "papers": papers}]}


def _merge(state: ResearchState, tools) -> dict:
    """Dedupe + interleave the parallel results, then apply the PICOS screen.
    Records PRISMA flow counts (identification -> screening -> inclusion)."""
    audit = tools["search_corpus"].audit
    results = state.get("source_results", [])
    failed = set(state.get("sources_failed", []))
    attempted = {r["source"] for r in results} | failed
    identified = {r["source"]: len(r["papers"]) for r in results}
    raw_total = sum(identified.values())
    order = [s for s in ["corpus", "pubmed", "scopus", "openalex", "arxiv",
                         "semantic_scholar", "ieee"]
             if s in {r["source"] for r in results}]
    merged_raw = merge_sources([r["papers"] for r in results if r["papers"]],
                               state.get("max_results", 10), priority=order)
    duplicates_removed = raw_total - len(merged_raw)
    llm_flags = state.get("llm") or {}
    if bool(llm_flags.get("screen")) and llm.llm_enabled():
        try:
            papers, llm_excluded = llm_screen_papers(
                state["query"], merged_raw, audit=audit)
            # sanity bound: excluding nearly everything for an on-topic query
            # signals poor model judgment — recall is protected over the LLM's
            # exclusion verdict, and the keyword screen takes over (audited)
            excl_ratio = len(llm_excluded) / max(1, len(merged_raw))
            if len(merged_raw) >= 4 and excl_ratio > 0.8:
                audit.record("llm_fallback", node="screen",
                             reason=f"implausible exclusion rate {excl_ratio:.0%}")
                papers = screen_papers(merged_raw, infer_topic_keywords(state["query"]))
            else:
                audit.record("llm_screening", kept=len(papers), excluded=len(llm_excluded),
                             reasons=[{"id": x["id"], "reason": x.get("exclusion_reason", "")}
                                      for x in llm_excluded][:20])
        except (llm.LLMError, ToolError) as e:
            audit.record("llm_fallback", node="screen", error=str(e))
            papers = screen_papers(merged_raw, infer_topic_keywords(state["query"]))
    else:
        papers = screen_papers(merged_raw, infer_topic_keywords(state["query"]))
    if not papers:
        # all sources failed or nothing matched -> corpus fallback (offline-safe)
        try:
            papers = screen_papers(tools["search_corpus"].call(state["query"],
                                   state.get("max_results", 10)),
                                   infer_topic_keywords(state["query"]))
            if papers:
                audit.record("retrieval_fallback", reason="no results from sources")
                identified["corpus"] = len(papers)
        except ToolError:
            pass
    if not papers:
        audit.record("no_evidence", query=state["query"])
        return {"status": "no_evidence"}
    prisma = {
        "identified": identified,
        "duplicates_removed": duplicates_removed,
        "screened": len(merged_raw),
        "excluded_screen": len(merged_raw) - len(papers),
        "included": len(papers),
    }
    audit.record("retrieval_summary", sources_ok=sorted(attempted - failed),
                 sources_failed=sorted(failed), papers_after_screen=len(papers))
    return {"papers": papers, "prisma": prisma}


def _extract(state: ResearchState, tools) -> dict:
    if state.get("status", "").startswith("failed"):
        return {}  # propagate upstream failure; router sends us to report_failed
    audit = tools["extract_evidence"].audit
    papers = state.get("papers", [])
    llm_flags = state.get("llm") or {}
    use_llm = bool(llm_flags.get("extract")) and llm.llm_enabled()
    evidence: list[dict] = []
    fallback: list[dict] = []
    llm_studies = 0
    if use_llm:
        # LLM path per paper; any failure (rate limit, invalid JSON, fabricated
        # span, out-of-range numbers) falls back to the rule-based extractor
        for p in papers:
            try:
                recs = llm_extract_effects(p, state["query"], audit=audit)
                evidence.extend(recs)
                llm_studies += bool(recs)
            except (llm.LLMError, ToolError, ValueError) as e:
                audit.record("llm_fallback", node="extract", paper=p["id"], error=str(e))
                fallback.append(p)
        if llm_studies:
            audit.record("llm_extraction", studies_extracted=llm_studies,
                         papers_fallback=len(fallback))
    else:
        fallback = papers
    if fallback:
        try:
            evidence.extend(tools["extract_evidence"].call(fallback))
        except ToolError as e:
            return {"status": "failed:extract_evidence",
                    "audit_events": [audit.record("node_failure", node="extract", error=str(e))]}
    if not evidence:
        # papers retrieved, but no parseable effect estimates: there is nothing
        # to synthesize, so the approval gate must never be reached
        audit.record("no_extractable_evidence", query=state["query"],
                     papers=len(state.get("papers", [])))
        return {"status": "no_extractable_evidence"}
    return {"evidence": evidence}


def _approval_gate(state: ResearchState, audit: AuditLog) -> Command[Literal["analyze", "report_declined"]]:
    """HITL gate: pauses execution until a human decides. Side effects are
    recorded only *after* the interrupt returns, so resumes never duplicate."""
    payload = {
        "action": "run_meta_analysis",
        "why": approvals.describe("run_meta_analysis"),
        "query": state["query"],
        "papers": len(state.get("papers", [])),
        "effects": len(state.get("evidence", [])),
    }
    decision = interrupt(payload)  # {"approved": bool, "approver": str?}
    audit.approval_decision("run_meta_analysis", decision, decision.get("approver", "human"))
    if decision.get("approved"):
        return Command(goto="analyze")
    return Command(goto="report_declined")


def _analyze(state: ResearchState, tools) -> dict:
    audit = tools["run_meta_analysis"].audit
    try:
        analysis = tools["run_meta_analysis"].call(state["evidence"])
    except ToolError as e:
        return {"status": "failed:run_meta_analysis",
                "audit_events": [audit.record("node_failure", node="analyze", error=str(e))]}
    return {"analysis": analysis}


def _verify_citations(state: ResearchState, audit: AuditLog) -> dict:
    """Provenance enforcement: drop any citation not backed by the ledger."""
    ledger_ids = {e["paper_id"] for e in state.get("evidence", [])}
    retrieved = {p["id"] for p in state.get("papers", [])}
    proposed = set(state.get("analysis", {}).get("cited_papers", []))
    supported_set = proposed & ledger_ids & retrieved
    supported, rejected = sorted(supported_set), sorted(proposed - supported_set)
    audit.record("citation_verification", proposed=sorted(proposed),
                 supported=supported, rejected=rejected)
    return {"citations": supported,
            "analysis": {**state["analysis"], "cited_papers": supported}}


def _synthesize(state: ResearchState, audit: AuditLog) -> dict:
    """Deterministic combined abstract + review outline from the run's own
    artifacts (audited; the LLM writer can later replace this node)."""
    analysis = state.get("analysis") or {}
    if not analysis:
        return {}
    sources_used = sorted({p.get("source", "fixture") for p in state.get("papers", [])})
    synthesis = build_synthesis(state["query"], state.get("papers", []),
                                state.get("evidence", []), analysis, sources_used,
                                prisma=state.get("prisma"), created=time.time())
    llm_flags = state.get("llm") or {}
    if llm_flags.get("write") and llm.llm_enabled():
        try:
            rewritten = llm_rewrite_abstract(synthesis["abstract_sections"],
                                             state["query"], audit=audit)
            required = [str(analysis["pooled_point"]), str(analysis["ci_lo"]),
                        str(analysis["ci_hi"]), f"{analysis['I2']}%"]
            if rewritten and all(x in rewritten for x in required):
                synthesis["abstract"] = rewritten
                synthesis["written_by"] = "llm"
                audit.record("llm_writer", status="accepted")
            else:
                audit.record("llm_fallback", node="synthesize",
                             reason="key statistics missing from LLM draft")
        except (llm.LLMError, ToolError) as e:
            audit.record("llm_fallback", node="synthesize", error=str(e))
    audit.record("synthesis", abstract_chars=len(synthesis["abstract"]),
                 outline_sections=len(synthesis["outline"]),
                 top_papers=[r["id"] for r in synthesis["most_relevant"][:3]])
    return {"synthesis": synthesis}


def _write_review(state: ResearchState, audit: AuditLog) -> dict:
    """PRISMA-style rapid review manuscript from the run's own artifacts."""
    if not state.get("papers"):
        return {}
    search_strategy = {}
    if state.get("live"):
        srcs = set(available_sources())
        if "pubmed" in srcs:
            search_strategy["PubMed"] = pubmed_effective_query(state["query"])
        if "scopus" in srcs:
            search_strategy["Scopus"] = state["query"]
        if "openalex" in srcs:
            search_strategy["OpenAlex"] = state["query"]
        if "arxiv" in srcs:
            search_strategy["arXiv"] = " AND ".join(
                f"all:{t}" for t in state["query"].lower().split() if len(t) > 1)
        if "semantic_scholar" in srcs:
            search_strategy["Semantic Scholar"] = state["query"]
        if "ieee" in srcs:
            search_strategy["IEEE Xplore"] = state["query"]
    else:
        search_strategy["built-in corpus"] = state["query"]
    review = build_review(state["query"], state.get("papers", []),
                          state.get("evidence", []), state.get("analysis"),
                          state.get("prisma") or {},
                          sources_failed=state.get("sources_failed", []),
                          search_strategy=search_strategy,
                          created=time.time())
    audit.record("review_written", title=review["title"],
                 references=len(review["references"]))
    return {"review": review}


def _report(state: ResearchState, audit: AuditLog) -> dict:
    a = state.get("analysis") or {}
    text = (f"Pooled estimate for '{state['query']}': {a.get('pooled_point')} "
            f"(95% CI {a.get('ci_lo')} to {a.get('ci_hi')}; k={a.get('k')}, "
            f"I2={a.get('I2')}%, model={a.get('model')}, p={a.get('p_value')}). "
            f"Cited: {', '.join(state.get('citations', []))}.")
    audit.report(text, state.get("citations", []))
    return {"report": text, "status": "ok"}


def _report_declined(state: ResearchState, audit: AuditLog) -> dict:
    return {"report": "Analysis declined by human approver; no results produced.",
            "citations": [], "status": "declined"}


def _report_failed(state: ResearchState, audit: AuditLog) -> dict:
    return {"report": f"Run aborted: {state.get('status', 'failed')}. See audit log.",
            "citations": [], }


def _report_no_evidence(state: ResearchState, audit: AuditLog) -> dict:
    msg = (f"No relevant papers found for '{state['query']}'. "
           "Corpus domains: aspirin/myocardial infarction, statins/LDL-cholesterol, "
           "SSRIs/depression — rephrase with those keywords or enable live "
           "retrieval (--live).")
    return {"report": msg, "citations": [], "status": "no_evidence"}


def _report_no_extractable(state: ResearchState, audit: AuditLog) -> dict:
    n = len(state.get("papers", []))
    msg = (f"Retrieved {n} papers for '{state['query']}', but none reported effect "
           "estimates the analyzer can pool (it reads risk ratios, odds ratios, "
           "hazard ratios, or mean differences with 95% CIs from abstracts). "
           "Try queries targeting randomized trial results, or screen the "
           "retrieved papers in the audit log.")
    return {"report": msg, "citations": [], "status": "no_extractable_evidence"}


# --------------------------------------------------------------------------

def build_supervisor_graph(failure_rate: float = 0.0, fault_type: str = "mixed",
                           seed: int = 0, audit: AuditLog | None = None,
                           checkpointer=None):
    """Build the supervisor graph.

    checkpointer: pass a shared durable saver (e.g. PostgresSaver) for
    long-lived deployments; defaults to a fresh InMemorySaver (tests/demo).
    """
    audit = audit or AuditLog()
    tools = build_toolchain(failure_rate, fault_type, seed, audit)

    def plan(s):
        return _plan(s, tools)

    def search_source(s):
        return _search_source(s, tools)

    def merge(s):
        return _merge(s, tools)

    def extract(s):
        return _extract(s, tools)

    def approval_gate(s):
        return _approval_gate(s, audit)

    def analyze(s):
        return _analyze(s, tools)

    def verify(s):
        return _verify_citations(s, audit)

    def synthesize(s):
        return _synthesize(s, audit)

    def write_review(s):
        return _write_review(s, audit)

    def report(s):
        return _report(s, audit)

    def report_declined(s):
        return _report_declined(s, audit)

    def report_failed(s):
        return _report_failed(s, audit)

    def report_no_evidence(s):
        return _report_no_evidence(s, audit)

    def report_no_extractable(s):
        return _report_no_extractable(s, audit)

    def after_merge(s) -> Literal["extract", "report_no_evidence", "report_failed"]:
        st = s.get("status", "ok")
        if st.startswith("failed"):
            return "report_failed"
        if st == "no_evidence":
            return "report_no_evidence"
        return "extract"

    def after_extract(s) -> Literal["approval_gate", "report_no_extractable", "report_failed"]:
        st = s["status"]
        if st.startswith("failed"):
            return "report_failed"
        if st == "no_extractable_evidence":
            return "report_no_extractable"
        return "approval_gate"

    def after_analyze(s) -> Literal["verify", "report_failed"]:
        return "report_failed" if s["status"].startswith("failed") else "verify"

    builder = (
        StateGraph(ResearchState)
        .add_node("search_source", search_source)
        .add_node("merge", merge)
        .add_node("extract", extract)
        .add_node("approval_gate", approval_gate)
        .add_node("analyze", analyze)
        .add_node("verify", verify)
        .add_node("synthesize", synthesize)
        .add_node("write_review", write_review)
        .add_node("report", report)
        .add_node("report_declined", report_declined)
        .add_node("report_failed", report_failed)
        .add_node("report_no_evidence", report_no_evidence)
        .add_node("report_no_extractable", report_no_extractable)
        .add_conditional_edges(START, plan)           # fan-out: returns [Send(...)]
        .add_edge("search_source", "merge")           # LangGraph joins the branches
        .add_conditional_edges("merge", after_merge)
        .add_conditional_edges("extract", after_extract)
        .add_conditional_edges("analyze", after_analyze)
        .add_edge("verify", "synthesize")
        .add_edge("synthesize", "write_review")
        .add_edge("write_review", "report")
        .add_edge("report", END)
        .add_edge("report_declined", END)
        .add_edge("report_failed", END)
        .add_edge("report_no_evidence", END)
        .add_edge("report_no_extractable", END)
    )
    return builder.compile(checkpointer=checkpointer or InMemorySaver()), audit, tools
