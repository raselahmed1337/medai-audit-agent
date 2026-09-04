from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class RunResult:
    architecture: str
    query: str
    status: str                      # ok | declined | failed:<tool>
    report: str = ""
    citations: list[str] = field(default_factory=list)
    pooled_point: float | None = None
    ci: tuple | None = None
    analysis_ran: bool = False       # did the sensitive tool execute?
    approval_requested: bool = False # was a human approval asked for first?
    audit_ok: bool = True            # hash chain verifies
    guard_stats: dict = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
