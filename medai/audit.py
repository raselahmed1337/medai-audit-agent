"""Hash-chained, append-only audit log.

Every event carries the SHA-256 of the previous event's canonical JSON, so any
after-the-fact edit or deletion breaks the chain and is detectable via
`verify_chain`.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class AuditLog:
    def __init__(self, path: str | None = None, run_id: str | None = None):
        self.path = path
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.events: list[dict] = []
        self._prev_hash = "GENESIS"

    # -- writing ---------------------------------------------------------
    def record(self, kind: str, **fields) -> dict:
        event = {
            "run_id": self.run_id,
            "seq": len(self.events),
            "ts": time.time(),
            "kind": kind,
            "prev_hash": self._prev_hash,
            **fields,
        }
        event["hash"] = hashlib.sha256(_canonical(event).encode()).hexdigest()
        self._prev_hash = event["hash"]
        self.events.append(event)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        return event

    # convenience emitters
    def tool_call(self, tool: str, args: dict, result=None, status: str = "ok", **extra):
        return self.record("tool_call", tool=tool, args=args, status=status,
                           result=result, **extra)

    def tool_failure(self, tool: str, args: dict, error: str, classification: str, attempt: int, **extra):
        return self.record("tool_failure", tool=tool, args=args, error=str(error),
                           classification=classification, attempt=attempt, **extra)

    def approval_requested(self, action: str, payload):
        return self.record("approval_requested", action=action, payload=payload)

    def approval_decision(self, action: str, decision, approver: str = "human"):
        return self.record("approval_decision", action=action, decision=decision,
                           approver=approver)

    def report(self, summary: str, citations: list[str]):
        return self.record("report", summary=summary, citations=citations)

    # -- verification ------------------------------------------------------
    def verify_chain(self) -> bool:
        prev = "GENESIS"
        for i, e in enumerate(self.events):
            if e["seq"] != i or e["prev_hash"] != prev:
                return False
            body = {k: v for k, v in e.items() if k != "hash"}
            if hashlib.sha256(_canonical(body).encode()).hexdigest() != e["hash"]:
                return False
            prev = e["hash"]
        return True

    def events_of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind]
