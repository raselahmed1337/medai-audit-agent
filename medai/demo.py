"""Interactive demo: run the supervisor agent on a query with a real
terminal approval gate and a persisted, tamper-evident audit log.

Usage:  python -m medai.demo "aspirin myocardial infarction risk"
"""
from __future__ import annotations

import json
import os
import sys

from experiments.a3_supervisor import human_approver
from medai.audit import AuditLog
from medai.graph import build_supervisor_graph
from langgraph.types import Command


def main(query: str = "aspirin myocardial infarction risk", live: bool = False):
    os.makedirs("results", exist_ok=True)
    audit_path = "results/audit-demo.jsonl"
    if os.path.exists(audit_path):
        os.remove(audit_path)
    audit = AuditLog(path=audit_path)
    graph, audit, tools = build_supervisor_graph(failure_rate=0.0, audit=audit)
    config = {"configurable": {"thread_id": "demo"}}

    print(f"Query: {query}")
    if live:
        from medai.tools.retrieval import DEFAULT_LIVE_SOURCES
        print(f"Retrieval mode: live ({' + '.join(DEFAULT_LIVE_SOURCES)})")
    else:
        print("Retrieval mode: offline corpus")
    print("Running supervisor graph...")
    state = graph.invoke({"query": query, "max_results": 10, "status": "ok", "live": live}, config)
    while state.get("__interrupt__"):
        resume = {i.id: human_approver(i.value) for i in state["__interrupt__"]}
        state = graph.invoke(Command(resume=resume), config)

    print("\n=== FINAL REPORT ===")
    print(state.get("report", ""))
    print(f"\nstatus={state.get('status')}  citations={state.get('citations')}")
    print(f"audit chain intact: {audit.verify_chain()}")
    print(f"audit log: {audit_path} ({len(audit.events)} events)")

    # tamper-evidence demonstration
    if "--tamper" in sys.argv and os.path.exists(audit_path):
        with open(audit_path) as f:
            lines = f.readlines()
        rec = json.loads(lines[0]); rec["args"] = {"edited": True}
        lines[0] = json.dumps(rec) + "\n"
        with open(audit_path, "w") as f:
            f.writelines(lines)
        reloaded = AuditLog(path=None)
        with open(audit_path) as f:
            reloaded.events = [json.loads(l) for l in f]
        print(f"after tampering with event 0, chain verifies: "
              f"{_chain_check(reloaded)} (expected False)")


def _chain_check(log: AuditLog) -> bool:
    import hashlib
    prev = "GENESIS"
    for i, e in enumerate(log.events):
        body = {k: v for k, v in e.items() if k != "hash"}
        if e["seq"] != i or e["prev_hash"] != prev:
            return False
        if hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != e["hash"]:
            return False
        prev = e["hash"]
    return True


if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    query = positional[0] if positional else "aspirin myocardial infarction risk"
    main(query, live="--live" in sys.argv)
