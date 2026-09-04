# Auditable Medical-AI Research Agent

A LangGraph-based medical research agent with **provenance tracking, a hash-chained
audit log, human-in-the-loop approval for sensitive actions, tool-failure
detection/recovery, and an experimental evaluation harness comparing three
architectures**. Runs fully offline and deterministically (no API keys needed);
retrieval can optionally fall back to live PubMed.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # unit + API tests (network smoke test auto-skips offline)
.venv/bin/python -m medai.server              # web approval console -> http://127.0.0.1:8000
.venv/bin/python -m medai.demo                # CLI run with terminal approval gate (offline corpus)
.venv/bin/python -m medai.demo --live         # CLI, retrieving real papers in real time
.venv/bin/python -m medai.demo --tamper       # also demonstrates audit tamper-evidence
.venv/bin/python eval/run_experiments.py 5    # 540-run factorial experiment -> results/
```

## Web approval console

`python -m medai.server` serves an HTTP API plus a browser console:

- **Start runs** with a query, live-retrieval toggle, and a simulated
  tool-failure rate (for live robustness demos).
- **Approve or decline** paused runs — the browser shows the approval
  payload (action, evidence counts) and the approver's name is written
  into the audit log. A declined approval provably blocks the sensitive
  analysis tool.
- **Inspect results**: report, citation chips (PMID/DOI links), retrieved
  papers with source tags, and the full hash-chained audit log with a
  live chain-integrity badge and `.jsonl` download.
- **Delete runs** from the sidebar or the run header (two-step confirm).

API: `POST /runs` → `GET /runs/{id}` (poll; `status: pending_approval`) →
`POST /runs/{id}/approval` → poll to completion.

## Deployment (free options)

The app is a long-running FastAPI process holding run state in memory, so it
needs a container/VM host (not serverless). A `Dockerfile` is included; the
container listens on `0.0.0.0:$PORT` (Render's `$PORT`, HF's `app_port`, or 8000).

- **Hugging Face Spaces (Docker)** — free, permanent URL, sleeps after ~48h idle.
  Create a Space (SDK: Docker), push this repo, and add this frontmatter to the
  Space's `README.md`: `sdk: docker`, `app_port: 8000`.
- **Render (free)** — sleeps after ~15 min idle (~30–60s cold start).
  `render.yaml` is a ready Blueprint; health check is `/health`.
- **Oracle Cloud Always-Free VM** — always-on, no sleep; run the container plus
  a Postgres for durable state. Needs a credit card and DIY setup.

Caveats for public URLs: run state is in-memory (restarts clear it) and there
is no auth — add a reverse-proxy token before sharing broadly. A free
keep-alive ping every 10 min prevents sleep on Render/HF.

Local container test: `docker build -t medai . && docker run -p 8000:8000 medai`

In live mode, papers come from PubMed, OpenAlex, and Semantic Scholar with
DOI-based deduplication; citations are real source IDs (`PMID:…`, `DOI:…`)
traced through the provenance ledger and audit log. Semantic Scholar works
best with a free API key (`export S2_API_KEY=…`); without one it may be
rate-limited and the run continues on the remaining sources. Queries whose
abstracts contain no parseable effect estimates fail gracefully rather than
fabricate results; out-of-domain offline queries return a distinct
`no_evidence` outcome (never an approval gate). Experiments always run on
the offline corpus for reproducibility.

## System (`medai/`)

| Module | Responsibility |
|---|---|
| `graph.py` | Flagship supervisor graph: `retrieve → screen → extract → **approval gate (interrupt)** → analyze → verify citations → report`, with failure-recovery edges. |
| `audit.py` | Append-only JSONL audit log; each event carries SHA-256 of the previous event → tamper-evident (`verify_chain`). |
| `tools/retrieval.py` | **Multi-source live retrieval: PubMed + OpenAlex + Semantic Scholar** (Google Scholar has no API and forbids automated access, so scholarly-web coverage comes from OpenAlex, the open ~250M-work index). Round-robin merge with DOI-based dedup; any failing source degrades gracefully. PubMed restricted to study designs reporting poolable effects; effective queries recorded in the audit log. Offline fixture-corpus keyword search with synonym/concept expansion remains the deterministic default. |
| `tools/extraction.py` | Effect extraction (RR/OR/**HR**/MD with 95% CI, incl. `95% CI, x–y` and negative MD formats seen in real PubMed abstracts) bound to `paper_id` + verbatim span = provenance units. |
| `tools/analysis.py` | Inverse-variance fixed-effects meta-analysis, heterogeneity (Q, I²) — a **sensitive action**. |
| `tools/guard.py` | Failure injection + detection: crash faults (retried, then escalated) and malformed-output faults (schema-validated, retried, escalated). Deterministic per-seed PRNG. |
| `approvals.py` | Policy declaring `run_meta_analysis` and `publish_report` sensitive. |

Human approval is structural: the graph *pauses* at the gate via LangGraph
`interrupt()` and cannot reach the analysis tool without a resume decision —
a declined approval provably blocks the sensitive tool (see
`test_supervisor_declined_blocks_sensitive_tool`). Citation verification drops
any citation not backed by both the provenance ledger and the retrieved set.

## Architecture comparison (`experiments/`)

- **A1 naive pipeline** — straight-line tool calls; no failure detection, no approval, no citation verification.
- **A2 ReAct-style loop** — guarded tool loop surfacing errors to a controller (crash retries only); no output validation, no structural approval gate.
- **A3 supervisor graph** (flagship) — structural approval gate, schema-validated failure detection with retry/escalation, provenance-enforced citations.

All three share the *same* tools, relevance screen, fault model, and audit format,
so measured differences isolate orchestration, not component quality.

## Evaluation (`eval/`)

12 tasks × 3 architectures × failure rates {0, 0.2, 0.4} × 5 seeds = **540 runs**.
Gold answers (citations + pooled effect) are computed independently from
gold-relevant corpus labels. Metrics: task success (pooled estimate within 15%
of gold), citation precision/recall/F1, fabrication count, approval violations,
audit integrity. `results/table.md` reports means with 95% bootstrap CIs.

## Experimental findings (540 runs)

1. **Clean-room parity**: with no injected failures, all architectures reach
   success = 1.0 and citation F1 = 1.0 — differences only emerge under stress,
   as intended.
2. **Failure robustness**: at 40% injected failure, the supervisor retains
   0.80 task success vs 0.47 (naive) and 0.42 (ReAct loop) — structural
   detection + retry converts most crashes and silent corruptions into
   successful recoveries instead of wrong reports.
3. **Safety**: A3 is the only architecture with **zero approval violations**
   (baselines executed the sensitive analysis without approval in 100% of
   successful runs at every failure rate).
4. **Auditability**: the hash chain verified after every run in all
   architectures; only A3 additionally logs approval requests/decisions,
   citation verification, and per-attempt failure classifications.
5. **Fabrication**: citation verification reduced fabricated citations to 0;
   precision failures in baselines came from unverified pooling of off-topic
   retrieval hits.

## Reproducibility

All randomness derives from `zlib.crc32(arch|rate|task|rep)` seeds; the fault
model is deterministic per seed, so every row in `results/results.json` is
exactly reproducible.
