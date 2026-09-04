"""Web service + approval console for the MedAI research agent.

Runs are created over HTTP, execute in background threads, and *pause* at the
human-approval gate (LangGraph interrupt). The console (medai/web/index.html)
polls run state and resumes paused runs with an Approve/Decline decision.

    .venv/bin/python -m medai.server --port 8000   # then open the printed URL
"""
from __future__ import annotations

import argparse
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langgraph.types import Command
from pydantic import BaseModel

from medai.audit import AuditLog
from medai.graph import build_supervisor_graph

WEB_DIR = Path(__file__).parent / "web"
ACTIVE_STATUSES = {"running", "pending_approval", "resuming"}

app = FastAPI(title="MedAI Research Agent", version="1.0.0")

RUNS: dict[str, "RunHandle"] = {}


@dataclass
class RunHandle:
    run_id: str
    query: str
    live: bool
    failure_rate: float
    graph: object
    audit: AuditLog
    tools: dict
    status: str = "running"
    pending: list = field(default_factory=list)   # [{id, payload}] while paused
    state: dict = field(default_factory=dict)     # final state after completion
    error: str | None = None
    created: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


class RunRequest(BaseModel):
    query: str
    live: bool = False
    failure_rate: float = 0.0


class ApprovalBody(BaseModel):
    approved: bool
    approver: str = "web-reviewer"


def _advance(handle: RunHandle, resume=None) -> None:
    """Drive the graph one segment: until it pauses on an interrupt or ends."""
    with handle.lock:
        config = {"configurable": {"thread_id": handle.run_id}}
        try:
            if resume is None:
                state = handle.graph.invoke(
                    {"query": handle.query, "max_results": 10, "status": "ok",
                     "live": handle.live}, config)
            else:
                state = handle.graph.invoke(Command(resume=resume), config)
        except Exception as e:  # never let a worker thread die silently
            handle.status, handle.error = "error", repr(e)
            handle.audit.record("server_error", error=repr(e))
            return
        if state.get("__interrupt__"):
            handle.pending = [{"id": i.id, "payload": i.value}
                              for i in state["__interrupt__"]]
            handle.status = "pending_approval"
        else:
            handle.pending = []
            handle.state = dict(state)
            handle.status = state.get("status", "ok")


def _spawn(handle: RunHandle, resume=None) -> None:
    threading.Thread(target=_advance, args=(handle, resume), daemon=True).start()


@app.post("/runs")
def create_run(req: RunRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    run_id = uuid.uuid4().hex[:10]
    audit = AuditLog()
    audit.record("run_created", query=req.query, live=req.live,
                 failure_rate=req.failure_rate)
    graph, audit, tools = build_supervisor_graph(
        failure_rate=req.failure_rate,
        seed=zlib.crc32(run_id.encode()),
        audit=audit)
    handle = RunHandle(run_id=run_id, query=req.query, live=req.live,
                       failure_rate=req.failure_rate, graph=graph,
                       audit=audit, tools=tools)
    RUNS[run_id] = handle
    _spawn(handle)
    return {"run_id": run_id, "status": "running"}


@app.get("/runs")
def list_runs():
    return [
        {"run_id": h.run_id, "query": h.query, "status": h.status,
         "live": h.live, "created": h.created}
        for h in sorted(RUNS.values(), key=lambda h: -h.created)
    ]


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    h = RUNS.get(run_id)
    if h is None:
        raise HTTPException(status_code=404, detail="unknown run")
    return {
        "run_id": h.run_id, "query": h.query, "live": h.live,
        "failure_rate": h.failure_rate, "status": h.status,
        "pending": h.pending,
        "report": h.state.get("report", ""),
        "citations": h.state.get("citations", []),
        "papers": [{"id": p["id"], "title": p.get("title", ""),
                    "source": p.get("source", "fixture")}
                   for p in h.state.get("papers", [])],
        "audit_ok": h.audit.verify_chain(),
        "events": h.audit.events,
        "error": h.error,
        "created": h.created,
    }


@app.post("/runs/{run_id}/approval")
def approve(run_id: str, body: ApprovalBody):
    h = RUNS.get(run_id)
    if h is None:
        raise HTTPException(status_code=404, detail="unknown run")
    if h.status != "pending_approval":
        raise HTTPException(status_code=409,
                            detail=f"run is '{h.status}', not pending_approval")
    resume = {p["id"]: {"approved": body.approved, "approver": body.approver}
              for p in h.pending}
    h.status = "resuming"
    _spawn(h, resume)
    return {"run_id": run_id, "status": "resuming"}


@app.delete("/runs/{run_id}")
def delete_run(run_id: str):
    if RUNS.pop(run_id, None) is None:
        raise HTTPException(status_code=404, detail="unknown run")
    return {"deleted": run_id}


@app.get("/health")
def health():
    """Health check for platform monitors (Render, HF Spaces, uptime pings)."""
    return {"status": "ok", "runs": len(RUNS)}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="MedAI approval console server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
