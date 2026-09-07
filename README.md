# Auditable Medical-AI Research Agent

A **LangGraph-based medical research agent** with provenance tracking, hash-chained audit logging, human-in-the-loop approval for sensitive actions, tool-failure detection/recovery, and experimental evaluation comparing three architectures.

**Key features:**
- ✅ Runs fully offline and deterministically (no API keys needed)
- ✅ Optional live retrieval from PubMed, Scopus, OpenAlex, arXiv, Semantic Scholar, IEEE Xplore
- ✅ Tamper-evident audit log with SHA-256 chain verification
- ✅ Structural approval gate (sensitive actions cannot execute without human sign-off)
- ✅ Automatic failure detection and recovery with schema validation
- ✅ Citation verification and de-duplication
- ✅ Meta-analysis with automatic model selection (fixed-effects vs random-effects)

---

## Quick Start

### 1. Setup environment
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### 2. Run tests
```bash
.venv/bin/python -m pytest tests/ -q          # unit + API tests
```

### 3. Try the demo
```bash
# CLI with offline corpus (no internet required)
.venv/bin/python -m medai.demo

# CLI with live PubMed retrieval
.venv/bin/python -m medai.demo --live

# CLI demonstrating audit tamper-evidence
.venv/bin/python -m medai.demo --tamper

# Run full factorial experiment (540 runs, ~5-10 min)
.venv/bin/python eval/run_experiments.py 5
```

### 4. Launch web console
```bash
.venv/bin/python -m medai.server              # http://127.0.0.1:8000
```

---

## Web Approval Console

A browser-based console for managing research runs with human-in-the-loop approval.

**Features:**
- **Start runs** — Submit a query with options for live retrieval and simulated failure injection
- **Approve or decline** — Review the approval payload (action, evidence counts); approver identity is logged in the audit trail
- **Inspect results** — View:
  - Full research report with recommendations
  - Citation chips with clickable PMID/DOI links
  - Retrieved papers tagged with their source
  - Hash-chained audit log with live chain-integrity badge
  - Export audit trail as `.jsonl`
- **Manage runs** — Delete runs with two-step confirmation

**API Flow:**
```
POST /runs (start)
  ↓
GET /runs/{id} (poll until pending_approval)
  ↓
POST /runs/{id}/approval (approve/decline)
  ↓
GET /runs/{id} (poll to completion)
```

### Service Hardening (Optional)
Set via environment variables:
- `MEDAI_API_TOKEN=<token>` — Require token auth for mutating endpoints (`POST /runs`, approvals, `DELETE`); reads stay open
- `MAX_CONCURRENT_RUNS=<n>` (default 8) — Limit concurrent executing runs; queued/paused runs don't consume slots; excess requests get `429`

---

## Deployment

### Development (In-Memory)
```bash
python -m medai.server              # Runs on http://127.0.0.1:8000
```
State is in-memory; all data clears on restart.

### Docker Deployment

**Single container (stateless):**
```bash
docker build -t medai . && docker run -p 8000:8000 medai
```

**Production with PostgreSQL (durable state):**
```bash
git clone https://github.com/raselahmed1337/medai-audit-agent && cd medai-audit-agent
docker compose up -d --build        # app on :8000, data in pgdata volume
docker compose logs -f app          # watch logs
git pull && docker compose up -d --build   # update in place
```

With `docker-compose.yml`, checkpoints, approvals, run registry, and audit logs survive crashes and redeploys.

### Durability Configuration
- **With `DATABASE_URL` env var:** PostgreSQL-backed; checkpoints, paused approvals, and audit logs are persisted
- **Without `DATABASE_URL`:** Everything in-memory (demo mode only)

### Free Cloud Hosting Options

| Platform | Cost | Duration | Config |
|---|---|---|---|
| **Hugging Face Spaces (Docker)** | Free | Permanent URL, ~48h idle sleep | Push repo + set `sdk: docker`, `app_port: 8000` in README frontmatter |
| **Render (free tier)** | Free | ~15 min idle sleep (~30–60s cold start) | Use included `render.yaml` Blueprint; health check at `/health` |
| **Oracle Cloud Always-Free VM** | Free | Always-on (no sleep) | DIY: run container + Postgres for state persistence |

**Note on public URLs:** State is in-memory unless `DATABASE_URL` is set. Add reverse-proxy auth before sharing. Use a keep-alive ping every 10 min to prevent sleep on Render/HF.

### Live Retrieval Sources
Papers come from multiple sources (offline corpus + optional live APIs):
- **Free, no key required:** PubMed, OpenAlex, arXiv, Semantic Scholar (basic)
- **Free with registration:** Scopus, IEEE Xplore  
- **Recommended:** Set `S2_API_KEY` for better Semantic Scholar performance

Export keys:
```bash
export SCOPUS_API_KEY=...
export IEEE_API_KEY=...
export S2_API_KEY=...
```

Or pass them to `docker-compose` via `.env` or command-line.

Results are DOI-deduplicated; citations use real source IDs (`PMID:…`, `DOI:…`, `ARXIV:…`).

---

## System Architecture

### Core Modules

| Module | Responsibility |
|---|---|
| `graph.py` | **Supervisor graph** orchestrating the research pipeline: `plan` → `search` (per source) → `merge/screen` → `extract` → **approval gate** → `analyze` → `publish` |
| `audit.py` | Append-only JSONL audit log with SHA-256 chain verification; tamper-evident via `verify_chain()` |
| `tools/retrieval.py` | **Multi-source live retrieval** from PubMed, Scopus, OpenAlex, arXiv, Semantic Scholar, IEEE Xplore |
| `tools/extraction.py` | Effect extraction (RR/OR/HR/MD with 95% CI); binds findings to paper_id + verbatim span for provenance |
| `tools/analysis.py` | **Automatic meta-analysis** with model selection: inverse-variance fixed-effects vs DerSimonian-Laird random-effects (I² ≥ 50% triggers random-effects) |
| `tools/guard.py` | Failure injection + detection: crash faults (retry → escalate) and malformed output faults (schema-validate → retry → escalate); deterministic per seed |
| `approvals.py` | Policy declaring `run_meta_analysis` and `publish_report` as sensitive (requires approval) |

### Approval Gate (Structural)
The LangGraph supervisor **pauses at the approval gate** via `interrupt()` — the analysis tool cannot be reached without a resume decision. A declined approval provably blocks the sensitive tool.

### Citation Verification
Only citations backed by both the provenance ledger **and** the retrieved set are included, eliminating fabricated citations.

---

## Architecture Comparison

Evaluated three orchestration approaches on the same tools, relevance screen, and fault model:

| Architecture | Approach | Failure Detection | Output Validation | Approval Gate | Audit Logging |
|---|---|---|---|---|---|
| **A1: Naive Pipeline** | Straight-line tool calls | ❌ None | ❌ None | ❌ None | ✅ Basic |
| **A2: ReAct Loop** | Agent loop with error surfacing | ✅ Crash retries | ❌ None | ❌ None | ✅ Basic |
| **A3: Supervisor Graph** (Flagship) | Structured graph with gates | ✅ Crash + schema detection | ✅ Schema-validated | ✅ Yes | ✅ Full + approval audit |

Measured differences isolate **orchestration**, not component quality.

---

## Evaluation & Results

**Experimental setup:** 12 tasks × 3 architectures × failure rates {0, 0.2, 0.4} × 5 seeds = **540 runs**

Gold answers (citations + pooled effect) computed independently. Metrics:
- Task success (pooled estimate within 15% of gold)
- Citation precision/recall/F1
- Fabrication count
- Approval violations
- Audit chain integrity

**See `results/table.md` for full results with 95% bootstrap CIs.**

### Key Findings

1. **Clean-room parity** — With no injected failures, all architectures reach success = 1.0 and citation F1 = 1.0 (differences emerge under stress only)

2. **Failure robustness** — At 40% injected failure:
   - **A3 (Supervisor):** 0.80 task success
   - **A2 (ReAct):** 0.42 task success
   - **A1 (Naive):** 0.47 task success
   
   Structural detection + retry converts crashes and silent corruptions into successful recoveries.

3. **Safety** — **A3 only:** zero approval violations across all runs. Baselines executed sensitive analysis without approval in 100% of successful runs.

4. **Auditability** — A3 logs approval requests/decisions, citation verification, and per-attempt failure classifications; all architectures verify the hash chain.

5. **Fabrication** — Citation verification reduced fabricated citations to 0 (baselines had precision failures from unverified off-topic hits).

---

## Reproducibility

All randomness derives from `zlib.crc32(arch|rate|task|rep)` seeds. The fault model is deterministic per seed, so every row in `results/results.json` is **exactly reproducible**.

---

## Repository Structure

```
medai-audit-agent/
├── medai/
│   ├── graph.py              # Supervisor graph
│   ├── audit.py              # Audit log + verification
│   ├── approvals.py          # Approval policies
│   ├── server.py             # FastAPI console
│   ├── demo.py               # CLI demo
│   ├── tools/
│   │   ├── retrieval.py       # Multi-source search
│   │   ├── extraction.py      # Effect extraction
│   │   ├── analysis.py        # Meta-analysis
│   │   └── guard.py           # Failure detection
│   └── ...
├── eval/
│   ├── run_experiments.py    # Experiment orchestrator
│   ├── tasks/                # 12 evaluation tasks
│   └── results/              # Experiment outputs
├── tests/
│   └── ...                   # Unit + integration tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml        # PostgreSQL + app stack
├── render.yaml               # Render.com blueprint
└── README.md
```

---

## Contributing

Contributions welcome! Areas of interest:
- Additional retrieval sources
- Enhanced failure recovery strategies
- UI/UX improvements
- Performance optimization for large-scale deployments

---

## License

See LICENSE file for details.
