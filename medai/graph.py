"""Flagship architecture (A3): LangGraph supervisor with

  plan -> retrieve -> extract -> [human approval gate] -> analyze -> verify -> report

Properties that distinguish it from the baselines (experiments/):
  * structural human-in-the-loop approval before any sensitive analysis tool
  * tool failure detection + retry + graceful escalation (never silent)
  * provenance-enforced report: every citation must exist in the ledger
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from medai import approvals
from medai.audit import AuditLog
from medai.tools.analysis import meta_analysis_fixed, valid_analysis
from medai.tools.extraction import extract_evidence, valid_evidence
from medai.tools.guard import ToolError, ToolGuard
from medai.tools.retrieval import (DEFAULT_LIVE_SOURCES, infer_topic_keywords,  # noqa: E402
                                   pubmed_effective_query, screen_papers,
                                   search_papers, valid_paper_list)


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
    status: str  # "ok" | "declined" | "failed:<tool>"
    audit_events: Annotated[list[dict], operator.add]


def build_toolchain(failure_rate: float = 0.0, fault_type: str = "mixed",
                    seed: int = 0, audit: AuditLog | None = None) -> dict:
    """Guarded tools shared by all architectures (same fault model)."""
    return {
        "search_papers": ToolGuard(search_papers, "search_papers", validator=valid_paper_list,
                                   failure_rate=failure_rate, fault_type=fault_type, seed=seed, audit=audit),
        "extract_evidence": ToolGuard(extract_evidence, "extract_evidence", validator=valid_evidence,
                                      failure_rate=failure_rate, fault_type=fault_type, seed=seed + 1, audit=audit),
        "run_meta_analysis": ToolGuard(meta_analysis_fixed, "run_meta_analysis", validator=valid_analysis,
                                       failure_rate=failure_rate, fault_type=fault_type, seed=seed + 2, audit=audit),
    }


# --------------------------------------------------------------------------
# nodes

def _retrieve(state: ResearchState, tools) -> dict:
    audit = tools["search_papers"].audit
    live = bool(state.get("live", False))
    audit.record("retrieval_mode", live=live,
                 sources=list(DEFAULT_LIVE_SOURCES) if live else ["fixture"],
                 pubmed_query=pubmed_effective_query(state["query"]) if live else None)
    try:
        papers = tools["search_papers"].call(state["query"], state.get("max_results", 10),
                                             state.get("live", False))
        papers = screen_papers(papers, infer_topic_keywords(state["query"]))
    except ToolError as e:
        return {"status": "failed:search_papers",
                "audit_events": [audit.record("node_failure", node="retrieve", error=str(e))]}
    if not papers:
        audit.record("no_evidence", query=state["query"])
        return {"status": "no_evidence"}
    return {"papers": papers}


def _extract(state: ResearchState, tools) -> dict:
    if state.get("status", "").startswith("failed"):
        return {}  # propagate upstream failure; router sends us to report_failed
    audit = tools["extract_evidence"].audit
    try:
        evidence = tools["extract_evidence"].call(state["papers"])
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


def _report(state: ResearchState, audit: AuditLog) -> dict:
    a = state.get("analysis") or {}
    text = (f"Pooled estimate for '{state['query']}': {a.get('pooled_point')} "
            f"(95% CI {a.get('ci_lo')} to {a.get('ci_hi')}; k={a.get('k')}, "
            f"I2={a.get('I2')}%, p={a.get('p_value')}). Cited: {', '.join(state.get('citations', []))}.")
    audit.report(text, state.get("citations", []))
    return {"report": text, "status": "ok"}


def _report_declined(state: ResearchState, audit: AuditLog) -> dict:
    return {"report": "Analysis declined by human approver; no results produced.",
            "citations": [], "status": "declined"}


def _report_no_extractable(state: ResearchState, audit: AuditLog) -> dict:
    n = len(state.get("papers", []))
    msg = (f"Retrieved {n} papers for '{state['query']}', but none reported effect "
           "estimates the analyzer can pool (it reads risk ratios, odds ratios, "
           "hazard ratios, or mean differences with 95% CIs from abstracts). "
           "Try queries targeting randomized trial results, or screen the "
           "retrieved papers in the audit log.")
    return {"report": msg, "citations": [], "status": "no_extractable_evidence"}


def _report_failed(state: ResearchState, audit: AuditLog) -> dict:
    return {"report": f"Run aborted: {state.get('status', 'failed')}. See audit log.",
            "citations": [], }


def _report_no_evidence(state: ResearchState, audit: AuditLog) -> dict:
    msg = (f"No relevant papers found for '{state['query']}'. "
           "Corpus domains: aspirin/myocardial infarction, statins/LDL-cholesterol, "
           "SSRIs/depression — rephrase with those keywords or enable live "
           "retrieval (--live).")
    return {"report": msg, "citations": [], "status": "no_evidence"}


# --------------------------------------------------------------------------

def build_supervisor_graph(failure_rate: float = 0.0, fault_type: str = "mixed",
                           seed: int = 0, audit: AuditLog | None = None):
    audit = audit or AuditLog()
    tools = build_toolchain(failure_rate, fault_type, seed, audit)

    def retrieve(s):
        return _retrieve(s, tools)

    def extract(s):
        return _extract(s, tools)

    def approval_gate(s):
        return _approval_gate(s, audit)

    def analyze(s):
        return _analyze(s, tools)

    def verify(s):
        return _verify_citations(s, audit)

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

    def after_retrieve(s) -> Literal["extract", "report_no_evidence", "report_failed"]:
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
        .add_node("retrieve", retrieve)
        .add_node("extract", extract)
        .add_node("approval_gate", approval_gate)
        .add_node("analyze", analyze)
        .add_node("verify", verify)
        .add_node("report", report)
        .add_node("report_declined", report_declined)
        .add_node("report_failed", report_failed)
        .add_node("report_no_evidence", report_no_evidence)
        .add_node("report_no_extractable", report_no_extractable)
        .add_edge(START, "retrieve")
        .add_conditional_edges("retrieve", after_retrieve)
        .add_conditional_edges("extract", after_extract)
        .add_conditional_edges("analyze", after_analyze)
        .add_edge("verify", "report")
        .add_edge("report", END)
        .add_edge("report_declined", END)
        .add_edge("report_failed", END)
    )
    return builder.compile(checkpointer=InMemorySaver()), audit, tools
