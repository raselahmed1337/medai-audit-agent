"""Three architectures under comparison, sharing the same guarded tools.

A1 naive_pipeline: straight-line calls, no failure detection (no guard), no
   approval gate, no provenance enforcement — reports whatever it computed.
A2 react_loop: ReAct-style loop over the same tools; tool errors are surfaced
   back to the controller, which may retry once (crash recovery only, no
   output-schema checking, no approval gate).
A3 supervisor: medai.graph.build_supervisor_graph — structural approval gate,
   failure detection with retries + escalation, citation verification.

All three return the same RunResult record for the evaluation harness.
"""
from __future__ import annotations

import uuid
from experiments import a1_naive, a2_react, a3_supervisor
from experiments.runresult import RunResult


def run_a1(query: str, failure_rate: float, seed: int, auto_approve=True) -> RunResult:
    return a1_naive.run(query, failure_rate, seed)


def run_a2(query: str, failure_rate: float, seed: int, auto_approve=True) -> RunResult:
    return a2_react.run(query, failure_rate, seed)


def run_a3(query: str, failure_rate: float, seed: int,
           approver=None, auto_approve=True) -> RunResult:
    return a3_supervisor.run(query, failure_rate, seed, approver=approver)


ARCHITECTURES = {"A1_naive": run_a1, "A2_react": run_a2, "A3_supervisor": run_a3}
