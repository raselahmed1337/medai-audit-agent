"""A3 — run the flagship supervisor graph with an approver.

The approver is a callable(payload) -> {"approved": bool, "approver": str}.
In evaluation an auto-approver stands in for the human; in interactive use
(experiments.human_approver) the payload is printed and decided at the
terminal — the graph itself pauses via LangGraph interrupt() either way.
"""
from __future__ import annotations

from langgraph.types import Command

from medai.graph import build_supervisor_graph
from experiments.runresult import RunResult


def auto_approver(payload: dict) -> dict:
    return {"approved": True, "approver": "auto"}


def human_approver(payload: dict) -> dict:
    print("\n=== APPROVAL REQUESTED ===")
    for k, v in payload.items():
        print(f"  {k}: {v}")
    ans = input("Approve this analysis? [y/N] ").strip().lower()
    return {"approved": ans == "y", "approver": "terminal-user"}


def run(query: str, failure_rate: float, seed: int, approver=None) -> RunResult:
    approver = approver or auto_approver
    graph, audit, tools = build_supervisor_graph(failure_rate=failure_rate, seed=seed)
    res = RunResult(architecture="A3_supervisor", query=query, status="ok")

    config = {"configurable": {"thread_id": res.run_id}}
    state = graph.invoke({"query": query, "max_results": 10, "status": "ok"}, config)

    # drive approval interrupts to completion (possibly several sensitive steps)
    while "__interrupt__" in state and state["__interrupt__"]:
        interrupts = state["__interrupt__"]
        resume = {i.id: approver(i.value) for i in interrupts}
        res.approval_requested = True
        state = graph.invoke(Command(resume=resume), config)
        if res.approval_requested and len(state.get("__interrupt__", []) or []) == 0:
            break

    final = state
    res.status = final.get("status", "failed:unknown")
    analysis = final.get("analysis") or {}
    res.analysis_ran = bool(analysis) and not res.status.startswith("failed")
    res.pooled_point = analysis.get("pooled_point")
    res.ci = (analysis.get("ci_lo"), analysis.get("ci_hi"))
    res.citations = final.get("citations", [])
    res.report = final.get("report", "")
    res.audit_ok = audit.verify_chain()
    res.guard_stats = {name: vars(g.stats) for name, g in tools.items()}
    if audit.events_of("approval_requested"):
        res.approval_requested = True
    return res
