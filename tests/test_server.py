"""End-to-end tests for the web service + approval console API."""
import time

import pytest
from fastapi.testclient import TestClient

from medai.server import RUNS, app

client = TestClient(app)


def _poll(run_id: str):
    """Wait until the run leaves an active status; fail on timeout."""
    for _ in range(200):
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] not in ("running", "pending_approval", "resuming"):
            return body
        time.sleep(0.05)
    pytest.fail(f"run {run_id} never reached a terminal status")


def _wait_pending(run_id: str):
    for _ in range(200):
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] == "pending_approval":
            return body
        if body["status"] not in ("running",):
            pytest.fail(f"run went to {body['status']} instead of pending_approval")
        time.sleep(0.05)
    pytest.fail("run never reached pending_approval")


def test_full_approval_flow():
    r = client.post("/runs", json={"query": "aspirin myocardial infarction risk"})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert run_id in [x["run_id"] for x in client.get("/runs").json()]

    body = _wait_pending(run_id)
    payload = body["pending"][0]["payload"]
    assert payload["action"] == "run_meta_analysis"
    assert payload["papers"] > 0 and payload["effects"] > 0

    r = client.post(f"/runs/{run_id}/approval",
                    json={"approved": True, "approver": "web-test"})
    assert r.status_code == 200

    body = _poll(run_id)
    assert body["status"] == "ok"
    assert "Pooled estimate" in body["report"]
    assert body["citations"]
    assert body["audit_ok"] is True
    # the human decision must be in the audit trail
    decisions = [e for e in body["events"] if e["kind"] == "approval_decision"]
    assert decisions and decisions[-1]["decision"]["approver"] == "web-test"


def test_declined_approval_blocks_sensitive_tool():
    run_id = client.post("/runs", json={"query": "statin LDL cholesterol"}).json()["run_id"]
    _wait_pending(run_id)
    client.post(f"/runs/{run_id}/approval", json={"approved": False})
    body = _poll(run_id)
    assert body["status"] == "declined"
    sensitive = [e for e in body["events"]
                 if e["kind"] == "tool_call" and e.get("tool") == "run_meta_analysis"]
    assert sensitive == []  # the sensitive tool provably never ran
    assert body["audit_ok"] is True


def test_out_of_domain_run_reports_no_evidence():
    run_id = client.post("/runs", json={"query": "quantum astrology"}).json()["run_id"]
    body = _poll(run_id)
    assert body["status"] == "no_evidence"
    assert "No relevant papers" in body["report"]


def test_unknown_run_and_conflicting_approvals():
    assert client.get("/runs/nope").status_code == 404
    assert client.post("/runs/nope/approval",
                       json={"approved": True}).status_code == 404
    assert client.post("/runs", json={"query": "   "}).status_code == 400

    run_id = client.post("/runs", json={"query": "aspirin MI"}).json()["run_id"]
    _wait_pending(run_id)
    client.post(f"/runs/{run_id}/approval", json={"approved": True})
    # a second decision while the run is no longer waiting must be rejected
    code = None
    for _ in range(100):
        r = client.post(f"/runs/{run_id}/approval", json={"approved": True})
        if r.status_code == 409:
            code = 409
            break
        time.sleep(0.05)
    assert code == 409
    _poll(run_id)  # still finishes cleanly


def test_delete_run():
    run_id = client.post("/runs", json={"query": "quantum astrology"}).json()["run_id"]
    _poll(run_id)
    assert run_id in [x["run_id"] for x in client.get("/runs").json()]
    r = client.delete(f"/runs/{run_id}")
    assert r.status_code == 200
    assert client.get(f"/runs/{run_id}").status_code == 404
    assert run_id not in [x["run_id"] for x in client.get("/runs").json()]
    assert client.delete(f"/runs/{run_id}").status_code == 404


def test_console_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Approval Console" in r.text
