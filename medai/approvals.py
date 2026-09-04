"""Sensitive-action policy: which tools require explicit human approval."""
from __future__ import annotations

SENSITIVE_ACTIONS = {
    "run_meta_analysis": "statistical synthesis of medical evidence (results may inform care)",
    "publish_report": "emitting a final research report with citations",
}


def requires_approval(tool_name: str) -> bool:
    return tool_name in SENSITIVE_ACTIONS


def describe(tool_name: str) -> str:
    return SENSITIVE_ACTIONS.get(tool_name, "")
