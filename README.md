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

Service hardening (both optional, env-driven):
- `MEDAI_API_TOKEN=<token>` — mutating endpoints (`POST /runs`, approval,
  `DELETE`) then require `Authorization: Bearer <token>`; reads stay open.
- `MAX_CONCURRENT_RUNS=<n>` (default 8) — bounds concurrently *executing*
  runs; excess requests get `429`. Runs paused at the approval gate do not
  hold a slot.

## Long-term deployment (durable)

`docker-compose.yml` runs the app + Postgres with `restart: unless-stopped`:
checkpoints, **paused approvals**, the run registry, and per-run audit JSONL
files all survive crashes, restarts, and redeploys. On any VM (free options:
Oracle Cloud Always-Free; cheap: Hetzner/DO ~$4/mo):

```bash
git clone https://github.com/raselahmed1337/medai-audit-agent && cd medai-audit-agent
docker compose up -d --build        # app on :8000, data in the pgdata volume
docker compose logs -f app          # watch it
git pull && docker compose up -d --build   # update in place
```

Durability is env-driven: with `DATABASE_URL` set the server uses a shared
PostgresSaver and a `runs` registry table, and writes each run's audit trail
to `AUDIT_DIR/audit-<run_id>.jsonl` (mounted volume). After a restart the
console rebuilds every run — completed runs show their full report; runs that
were waiting at the approval gate are still pending and can be decided as if
nothing happened. Without `DATABASE_URL` everything is in-memory (demo mode).

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

In live mode, papers come from PubMed, OpenAlex, arXiv and Semantic Scholar by
default; **Scopus and IEEE Xplore join the fan-out when API keys are set**
(`export SCOPUS_API_KEY=…` / `IEEE_API_KEY=…` — both free with registration;
docker-compose passes them through). Results are DOI-deduplicated; citations
are real source IDs (`PMID:…`, `DOI:…`, `ARXIV:…`) traced through the
provenance ledger and audit log. Semantic Scholar works best with a free API
key (`S2_API_KEY=…`); rate-limited sources degrade gracefully. Queries whose
abstracts contain no parseable effect estimates fail gracefully rather than
fabricate results; out-of-domain offline queries return a distinct
`no_evidence` outcome (never an approval gate). Experiments always run on
the offline corpus for reproducibility.

## System (`medai/`)

| Module | Responsibility |
|---|---|
| `graph.py` | Flagship supervisor graph: `plan ──Send fan-out──▶ [search per source] ──▶ merge/screen ──▶ extract ──▶ **approval gate (interrupt)** ──▶ analyze ──▶ verify citations ──▶ report`. Source searches execute as **parallel graph nodes** (one guarded tool per source, per-source audit events); merge dedupes by DOI in priority order and applies the PICOS screen; failure-recovery edges route to honest-failure reports. |
| `audit.py` | Append-only JSONL audit log; each event carries SHA-256 of the previous event → tamper-evident (`verify_chain`). |
| `tools/retrieval.py` | **Multi-source live retrieval: PubMed, Scopus, OpenAlex, arXiv, Semantic Scholar, IEEE Xplore** (Google Scholar has no API and forbids automated access, so scholarly-web coverage comes from OpenAlex, the open ~250M-work index). Round-robin merge with DOI-based dedup; any failing source degrades gracefully; Scopus and IEEE are **key-gated** (`SCOPUS_API_KEY` / `IEEE_API_KEY`) and join the fan-out only when configured. PubMed restricted to study designs reporting poolable effects; effective queries recorded in the audit log. Offline fixture-corpus keyword search with synonym/concept expansion remains the deterministic default. |
| `tools/extraction.py` | Effect extraction (RR/OR/**HR**/MD with 95% CI, incl. `95% CI, x–y` and negative MD formats seen in real PubMed abstracts) bound to `paper_id` + verbatim span = provenance units. |
| `tools/analysis.py` | Meta-analysis with **automatic model selection**: inverse-variance fixed-effects vs DerSimonian-Laird random-effects (primary when I² ≥ 50%); both estimates, tau², Q and I² always reported. A **sensitive action**. |
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
