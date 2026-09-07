"""Web service + approval console for the MedAI research agent.

Runs are created over HTTP, execute in background threads, and *pause* at the
human-approval gate (LangGraph interrupt). The console (medai/web/index.html)
polls run state and resumes paused runs with an Approve/Decline decision.

Long-lived deployments: set DATABASE_URL (Postgres) and AUDIT_DIR (mounted
volume) to make checkpoints, paused approvals, run registry, and audit trails
survive process restarts. Without them everything is in-memory (demo mode).

    .venv/bin/python -m medai.server --port 8000   # then open the printed URL
"""
from __future__ import annotations

import argparse
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from langgraph.types import Command
from pydantic import BaseModel

from medai import checkpointing
from medai.audit import AuditLog
from medai.graph import build_supervisor_graph

WEB_DIR = Path(__file__).parent / "web"
ACTIVE_STATUSES = {"running", "pending_approval", "resuming"}


@asynccontextmanager
async def lifespan(_app):
    ensure_persistence()  # init DB + rebuild runs before serving traffic
    yield


app = FastAPI(title="MedAI Research Agent", version="1.1.0", lifespan=lifespan)

RUNS: dict[str, "RunHandle"] = {}
_PERSISTENCE_READY = False


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
    llm: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


class RunRequest(BaseModel):
    query: str
    live: bool = False
    failure_rate: float = 0.0
    llm: dict | None = None  # {"screen": bool, "extract": bool, "write": bool}


class ApprovalBody(BaseModel):
    approved: bool
    approver: str = "web-reviewer"


# ---------------- service hardening ----------------

def require_token(request: Request) -> None:
    """Optional bearer-token auth: set MEDAI_API_TOKEN to lock mutating
    endpoints; leave unset for open local/demo use."""
    token = os.environ.get("MEDAI_API_TOKEN")
    if not token:
        return
    if request.headers.get("Authorization", "") != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="invalid or missing API token")


_MAX_ACTIVE = int(os.environ.get("MAX_CONCURRENT_RUNS", "8"))
_active_runs = 0
_active_lock = threading.Lock()


def _admit() -> None:
    """Admission control: bound concurrently *executing* runs. Runs paused at
    the approval gate do not hold a slot — humans may take hours."""
    global _active_runs
    with _active_lock:
        if _active_runs >= _MAX_ACTIVE:
            raise HTTPException(status_code=429,
                                detail=f"server busy: {int(_MAX_ACTIVE)} runs already executing")
        _active_runs += 1


def _release() -> None:
    global _active_runs
    with _active_lock:
        _active_runs = max(0, _active_runs - 1)


# ---------------- durability ----------------

def ensure_persistence() -> None:
    """Init durable layers and rebuild run handles after a process restart.

    Marks ready only on success, so a transient DB error at startup makes the
    next request retry instead of silently running without durability.
    """
    global _PERSISTENCE_READY
    if _PERSISTENCE_READY:
        return
    if checkpointing._db_url() is None:  # demo mode: nothing to init
        _PERSISTENCE_READY = True
        return
    last_err = None
    for attempt in range(3):  # tolerate the DB still settling at container start
        try:
            checkpointing.get_checkpointer()
            for row in checkpointing.registry_list():
                if row["run_id"] in RUNS:
                    continue
                _rebuild_handle(row)
            _PERSISTENCE_READY = True
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise HTTPException(status_code=503,
                        detail=f"database unavailable: {last_err!r}")


def _rebuild_handle(row: dict) -> None:
    """Recreate a RunHandle from the registry + durable checkpoint + audit file."""
    run_id = row["run_id"]
    audit_path = checkpointing.audit_path(run_id)
    audit = AuditLog.load(audit_path) if audit_path else AuditLog()
    graph, audit, tools = build_supervisor_graph(
        failure_rate=row["failure_rate"], seed=checkpointing.run_seed(run_id),
        audit=audit, checkpointer=checkpointing.get_checkpointer())
    handle = RunHandle(run_id=run_id, query=row["query"], live=row["live"],
                       failure_rate=row["failure_rate"], graph=graph,
                       audit=audit, tools=tools, created=row["created"])

    config = {"configurable": {"thread_id": run_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception as e:  # checkpoint unreadable
        handle.status, handle.error = "failed:server_restart", repr(e)
        RUNS[run_id] = handle
        return
    interrupts = [i for t in snapshot.tasks for i in getattr(t, "interrupts", [])]
    if interrupts:
        handle.pending = [{"id": i.id, "payload": i.value} for i in interrupts]
        handle.status = "pending_approval"   # the human can now decide, even post-restart
    elif snapshot.values:
        handle.state = dict(snapshot.values)
        handle.status = snapshot.values.get("status", "ok")
    else:
        handle.status = "failed:server_restart"
        handle.error = "run was mid-flight when the server restarted"
    RUNS[run_id] = handle


# ---------------- execution ----------------

def _advance(handle: RunHandle, resume=None) -> None:
    """Drive the graph one segment: until it pauses on an interrupt or ends."""
    _admit()
    try:
        with handle.lock:
            config = {"configurable": {"thread_id": handle.run_id}}
            try:
                if resume is None:
                    state = handle.graph.invoke(
                        {"query": handle.query, "max_results": 10, "status": "ok",
                         "live": handle.live, "llm": handle.llm}, config)
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
    finally:
        _release()  # runs paused at the gate release their execution slot


def _spawn(handle: RunHandle, resume=None) -> None:
    threading.Thread(target=_advance, args=(handle, resume), daemon=True).start()


# ---------------- API ----------------

@app.post("/runs", dependencies=[Depends(require_token)])
def create_run(req: RunRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    ensure_persistence()
    run_id = uuid.uuid4().hex[:10]
    audit_path = checkpointing.audit_path(run_id)
    audit = AuditLog(path=audit_path)
    audit.record("run_created", query=req.query, live=req.live,
                 failure_rate=req.failure_rate)
    graph, audit, tools = build_supervisor_graph(
        failure_rate=req.failure_rate,
        seed=checkpointing.run_seed(run_id),
        audit=audit,
        checkpointer=checkpointing.get_checkpointer())
    handle = RunHandle(run_id=run_id, query=req.query, live=req.live,
                       failure_rate=req.failure_rate, graph=graph,
                       audit=audit, tools=tools, llm=req.llm or {})
    RUNS[run_id] = handle
    checkpointing.registry_insert(run_id, req.query, req.live, req.failure_rate,
                                  handle.created)
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
        "llm": h.llm,
        "pending": h.pending,
        "report": h.state.get("report", ""),
        "citations": h.state.get("citations", []),
        "papers": [{"id": p["id"], "title": p.get("title", ""),
                    "source": p.get("source", "fixture"),
                    "url": p.get("url") or None,
                    "abstract_snippet": (p.get("abstract", "") or "")[:240]}
                   for p in h.state.get("papers", [])],
        "evidence": [{"paper_id": e["paper_id"], "measure": e["measure"],
                      "point": e["point"], "lo": e["lo"], "hi": e["hi"]}
                     for e in h.state.get("evidence", [])],
        "analysis": h.state.get("analysis"),
        "synthesis": h.state.get("synthesis"),
        "review": h.state.get("review"),
        "audit_ok": h.audit.verify_chain(),
        "events": h.audit.events,
        "error": h.error,
        "created": h.created,
    }


@app.post("/runs/{run_id}/approval", dependencies=[Depends(require_token)])
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


@app.delete("/runs/{run_id}", dependencies=[Depends(require_token)])
def delete_run(run_id: str):
    if RUNS.pop(run_id, None) is None:
        raise HTTPException(status_code=404, detail="unknown run")
    checkpointing.registry_delete(run_id)
    audit_path = checkpointing.audit_path(run_id)
    if audit_path and os.path.exists(audit_path):
        os.remove(audit_path)  # deletion also removes the persisted audit trail
    return {"deleted": run_id}


@app.get("/health")
def health():
    """Health check for platform monitors (Render, HF Spaces, uptime pings)."""
    return {"status": "ok", "runs": len(RUNS),
            "durable": checkpointing._db_url() is not None}


@app.get("/")
def index():
    # no-store: the console is served fresh on every load, so UI updates and
    # rebuilds can never be shadowed by a stale browser cache
    return FileResponse(WEB_DIR / "index.html",
                        headers={"Cache-Control": "no-store, must-revalidate"})


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="MedAI approval console server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
