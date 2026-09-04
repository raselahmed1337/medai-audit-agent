import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medai.audit import AuditLog
from medai.corpus import FIXTURE_CORPUS, corpus_by_id, gold_relevant
from medai.graph import build_supervisor_graph
from medai.tools.analysis import meta_analysis_fixed, valid_analysis
from medai.tools.extraction import extract_evidence, extract_effects
from medai.tools.guard import ToolError, ToolGuard
from medai.tools.retrieval import search_papers, valid_paper_list


# ---------------- audit ----------------
def test_audit_chain_verifies_and_detects_tampering():
    log = AuditLog()
    log.tool_call("t", {})
    log.approval_decision("a", {"approved": True})
    assert log.verify_chain()
    log.events[0]["args"] = {"tampered": True}  # edit after the fact
    assert not log.verify_chain()


def test_audit_detects_deletion():
    log = AuditLog()
    for i in range(5):
        log.tool_call("t", {"i": i})
    del log.events[2]
    assert not log.verify_chain()


# ---------------- retrieval / extraction ----------------
def test_retrieval_returns_relevant_papers():
    res = search_papers("aspirin myocardial infarction")
    assert valid_paper_list(res)
    ids = {p["id"] for p in res}
    assert {"P001", "P002"} <= ids


def test_search_covers_abstracts_not_only_titles():
    # "physicians" appears only in P001's abstract; "gastroduodenal" only in P002's
    for word, pid in [("physicians", "P001"), ("gastroduodenal", "P002")]:
        res = search_papers(f"{word} study", 16)
        assert pid in {p["id"] for p in res}


def test_query_expansion_bridges_synonyms():
    # no abstract contains "heart attack", but expansion maps it to
    # myocardial infarction terms, so the aspirin-MI papers are found
    res = search_papers("heart attack aspirin", 10)
    assert {"P001", "P002"} <= {p["id"] for p in res}


def test_keyword_order_and_extra_words_do_not_matter():
    a = {p["id"] for p in search_papers("aspirin myocardial infarction risk", 4)}
    b = {p["id"] for p in search_papers("risk low cardiovascular aspirin events infarction", 4)}
    assert a == b


def test_pubmed_efetch_parses_real_abstracts(monkeypatch):
    import urllib.request
    import medai.tools.retrieval as R

    esearch = b'{"esearchresult":{"idlist":["111","222"]}}'
    efetch = (b'<PubmedArticleSet><PubmedArticle><MedlineCitation>'
              b'<PMID>111</PMID><Article><ArticleTitle>Aspirin after MI</ArticleTitle>'
              b'<Abstract><AbstractText Label="BACKGROUND">Risk ratio 0.6, 95% CI 0.5 to 0.8.</AbstractText>'
              b'<AbstractText Label="METHODS">We randomized 5000 patients.</AbstractText></Abstract>'
              b'</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>')

    calls = []

    class FakeResp:
        def __init__(self, payload): self.payload = payload
        def read(self): return self.payload

    def fake_urlopen(url, timeout=0):
        calls.append(url)
        return FakeResp(efetch if "efetch" in url else esearch)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    papers = R._pubmed_search("aspirin", 2)
    assert [p["id"] for p in papers] == ["PMID:111"]
    assert "Risk ratio 0.6" in papers[0]["abstract"]
    assert "randomized 5000" in papers[0]["abstract"]  # both CI sections joined
    assert any("efetch" in u for u in calls)  # abstracts actually fetched
    # live retrieval is restricted to study designs that report poolable effects
    esearch_call = next(u for u in calls if "efetch" not in u)
    assert "hasabstract" in esearch_call
    assert "randomized+controlled+trial%5Bpt%5D" in esearch_call


def test_pubmed_live_search_returns_real_abstracts():
    # network smoke test; skipped automatically when offline
    import urllib.request
    try:
        urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov", timeout=5)
    except Exception:
        import pytest
        pytest.skip("network unavailable")
    papers = search_papers("aspirin myocardial infarction randomized", 3, live=True)
    assert len(papers) == 3
    assert all(p["id"].startswith("PMID:") for p in papers)
    assert all(len(p["abstract"]) > 50 for p in papers), "efetch abstracts missing"


def test_extraction_binds_provenance():
    p = corpus_by_id()["P001"]
    eff = extract_effects(p)
    assert len(eff) == 1
    e = eff[0]
    assert e["paper_id"] == "P001"
    assert e["measure"] == "rr" and abs(e["point"] - 0.56) < 1e-9
    assert "risk ratio" in e["span"].lower() and e["se"] > 0


def test_extraction_handles_real_pubmed_formats():
    # formats observed in live PubMed abstracts (July 2026 sampling)
    cases = [
        # abbreviated measure, outcome name between measure and estimate
        ("RRs for low versus high education were: AD 1.80 (95% CI: 1.43-2.27); "
         "non-AD dementias, 1.32 (95% CI: 0.92-1.88).", "rr", 2),
        # Cohen's d
        ("Effects were maintained at 9 months (Cohen's d = 0.94, 95% CI 0.71 to 1.17).",
         "md", 1),
        # HR with comma after CI
        ("Hazard ratio 0.72 (95% CI, 0.60-0.85) was observed.", "hr", 1),
        # negative mean difference
        ("Mean difference -0.5 (95% CI -0.8 to -0.2) mmol/L.", "md", 1),
    ]
    for abstract, measure, n in cases:
        eff = extract_effects(dict(id="X", title="", abstract=abstract))
        assert len(eff) == n, abstract
        assert all(e["measure"] == measure for e in eff), abstract


def test_extraction_skips_unidentifiable_measures():
    # "MIE" is a measure the extractor does not know: skip, never guess
    eff = extract_effects(dict(id="X", title="", abstract=(
        "Residents were more likely to experience pleasure (MIE: 0.038; "
        "95% CI: 0.001 to 0.075).")))
    assert eff == []
    # a number near a CI with no measure at all is not evidence
    eff = extract_effects(dict(id="Y", title="", abstract=(
        "Improvement was seen at 24 months (95% CI 0.4 to 0.9).")))
    assert eff == []


def test_extraction_rejects_inconsistent_ci():
    bad = dict(id="X", title="", abstract="risk ratio 0.5, 95% CI 0.9 to 0.2.")
    assert extract_effects(bad) == []


# ---------------- analysis ----------------
def test_meta_analysis_hand_computed():
    # two identical-strength effects, hand-checkable pooled estimate:
    # y_i = ln(0.56), se from CI; single-effect pooling must return the input
    e = extract_effects(corpus_by_id()["P001"])[0]
    r = meta_analysis_fixed([e])
    assert abs(r["pooled_point"] - 0.56) < 1e-9
    # log-symmetric CI reconstruction: close to the published CI but not exact
    assert abs(r["ci_lo"] - 0.45) < 0.01 and abs(r["ci_hi"] - 0.70) < 0.01
    assert r["k"] == 1 and r["I2"] == 0.0 and valid_analysis(r)


def test_meta_analysis_two_effects_between_inputs():
    a = extract_effects(corpus_by_id()["P001"])[0]
    b = extract_effects(corpus_by_id()["P002"])[0]
    r = meta_analysis_fixed([a, b])
    assert r["k"] == 2
    lo = min(a["point"], b["point"]) - 1e-6
    hi = max(a["point"], b["point"]) + 1e-6
    assert lo <= r["pooled_point"] <= hi


def test_meta_analysis_rejects_bad_input():
    import pytest
    with pytest.raises(ValueError):
        meta_analysis_fixed([])


# ---------------- guard / failure handling ----------------
def test_guard_detects_malformed_output_and_recovers():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        return [None] if calls["n"] == 1 else [{"id": "P001", "title": "t", "abstract": "a"}]

    g = ToolGuard(flaky, "flaky", validator=valid_paper_list,
                  failure_rate=0.0, max_retries=2)
    res = g.call()
    assert valid_paper_list(res) and calls["n"] == 2
    assert g.stats.detected == 1 and g.stats.retries == 1


def test_guard_crash_retry_then_fatal():
    def always_crash():
        raise TimeoutError("down")

    g = ToolGuard(always_crash, "crashy", failure_rate=0.0, max_retries=1)
    import pytest
    with pytest.raises(ToolError) as exc:
        g.call()
    assert exc.value.classification == "fatal"


def test_guard_injected_crash_recovers():
    ok = lambda: [{"id": "P001", "title": "t", "abstract": "a"}]  # noqa: E731
    g = ToolGuard(ok, "inj", validator=valid_paper_list,
                  failure_rate=1.0, fault_type="crash", max_retries=3, seed=1)
    # p=1.0 crash every attempt -> still fatal; retry budget exhausts
    import pytest
    with pytest.raises(ToolError):
        g.call()
    assert g.stats.injected >= 2


# ---------------- multi-source live retrieval ----------------
def test_openalex_abstract_reconstruction():
    # OpenAlex stores abstracts as an inverted index {word: [positions]}
    from medai.tools.retrieval import _openalex_abstract
    inv = {"Dementia": [0], "risk": [1], "increases": [2], "with": [3], "age": [4]}
    assert _openalex_abstract(inv) == "Dementia risk increases with age"
    assert _openalex_abstract(None) == ""


def test_multi_source_merge_dedupes_by_doi(monkeypatch):
    import medai.tools.retrieval as R
    monkeypatch.setattr(R, "_pubmed_search", lambda q, n: [
        dict(id="PMID:1", title="Same work", abstract="a", doi="10.1/x", source="pubmed")])
    monkeypatch.setattr(R, "_openalex_search", lambda q, n: [
        dict(id="DOI:10.1/x", title="Same work", abstract="a", doi="10.1/x", source="openalex"),
        dict(id="DOI:10.2/y", title="Other work", abstract="b", doi="10.2/y", source="openalex")])
    monkeypatch.setattr(R, "_semantic_scholar_search", lambda q, n: [
        dict(id="S2:z", title="Third work", abstract="c", doi="", source="semantic_scholar")])
    res = R.search_papers("query", 10, live=True)
    ids = [p["id"] for p in res]
    # PubMed copy wins over the OpenAlex duplicate of the same DOI
    assert "PMID:1" in ids and "DOI:10.1/x" not in ids
    assert "DOI:10.2/y" in ids and "S2:z" in ids
    assert all("source" in p for p in res) and len(res) == 3


def test_source_failure_degrades_gracefully(monkeypatch):
    import medai.tools.retrieval as R

    def boom(*a, **k):
        raise TimeoutError("source down")

    monkeypatch.setattr(R, "_pubmed_search", boom)
    monkeypatch.setattr(R, "_openalex_search", boom)
    monkeypatch.setattr(R, "_semantic_scholar_search",
                        lambda q, n: [dict(id="S2:1", title="t", abstract="a",
                                           doi="", source="semantic_scholar")])
    res = R.search_papers("query", 10, live=True)
    assert [p["id"] for p in res] == ["S2:1"]


def test_all_sources_failing_falls_back_to_corpus(monkeypatch):
    import medai.tools.retrieval as R

    def boom(*a, **k):
        raise TimeoutError("offline")

    for src in ("_pubmed_search", "_openalex_search", "_semantic_scholar_search"):
        monkeypatch.setattr(R, src, boom)
    res = R.search_papers("aspirin myocardial infarction", 5, live=True)
    assert {"P001", "P002"} <= {p["id"] for p in res}


def test_live_scholarly_search_returns_multi_source_papers():
    # network smoke test; skipped automatically when offline
    import urllib.request
    try:
        urllib.request.urlopen("https://api.openalex.org/works?per-page=1", timeout=8)
    except Exception:
        import pytest
        pytest.skip("network unavailable")
    papers = search_papers("machine learning clinical diagnosis", 10, live=True)
    sources = {p["source"] for p in papers}
    assert len(papers) >= 5
    assert "openalex" in sources or "semantic_scholar" in sources, sources
    assert all(p["id"] and p["title"] and p["abstract"] for p in papers)


# ---------------- supervisor graph / HITL ----------------
def _drive(graph, query="aspirin myocardial infarction risk", approve=True):
    from langgraph.types import Command
    cfg = {"configurable": {"thread_id": "t1"}}
    state = graph.invoke({"query": query, "max_results": 10, "status": "ok"}, cfg)
    if state.get("__interrupt__"):
        resume = {i.id: {"approved": approve, "approver": "test"} for i in state["__interrupt__"]}
        state = graph.invoke(Command(resume=resume), cfg)
    return state


def test_supervisor_out_of_domain_reports_no_evidence():
    graph, audit, tools = build_supervisor_graph()
    cfg = {"configurable": {"thread_id": "t-noev"}}
    state = graph.invoke({"query": "quantum astrology healing", "max_results": 10,
                          "status": "ok"}, cfg)
    assert state["status"] == "no_evidence"
    assert "No relevant papers" in state["report"]
    assert audit.events_of("no_evidence")
    # the sensitive tool must never have run, and no approval was requested
    assert tools["run_meta_analysis"].stats.calls == 0
    assert not audit.events_of("approval_decision")


def test_supervisor_zero_effects_never_reaches_approval_gate(monkeypatch):
    # live-mode scenario: papers retrieved, but no abstract contains a
    # parseable effect estimate -> nothing to synthesize, no approval asked.
    # Retrieval is stubbed so the test is hermetic (no network dependency).
    import medai.graph as G
    fake_papers = [
        dict(id="L1", title="Trial of drug X",
             abstract="We randomized 500 patients. Outcomes improved at 12 months."),
        dict(id="L2", title="Cohort study of marker Y",
             abstract="Participants were followed for five years. No events occurred."),
    ]
    monkeypatch.setattr(G, "search_papers", lambda *a, **k: [dict(p) for p in fake_papers])
    graph, audit, tools = build_supervisor_graph()
    cfg = {"configurable": {"thread_id": "t-noeff"}}
    state = graph.invoke({"query": "neurodegenerative disorder", "max_results": 10,
                          "status": "ok", "live": True}, cfg)
    assert state["status"] == "no_extractable_evidence"
    assert "none reported effect estimates" in state["report"]
    assert tools["run_meta_analysis"].stats.calls == 0
    assert not audit.events_of("approval_decision")
    assert audit.events_of("no_extractable_evidence")


def test_supervisor_happy_path():
    graph, audit, _ = build_supervisor_graph()
    state = _drive(graph)
    assert state["status"] == "ok"
    assert state["analysis"]["k"] >= 1
    assert state["citations"]
    assert set(state["citations"]) <= set(state["analysis"]["cited_papers"])
    assert audit.verify_chain()
    decisions = audit.events_of("approval_decision")
    assert decisions and decisions[0]["decision"]["approved"] is True


def test_supervisor_declined_blocks_sensitive_tool():
    graph, audit, tools = build_supervisor_graph()
    state = _drive(graph, approve=False)
    assert state["status"] == "declined"
    # the sensitive analysis tool must never have executed
    assert tools["run_meta_analysis"].stats.calls == 0
    assert state.get("analysis") is None


def test_supervisor_catches_fabricated_citations():
    graph, audit, _ = build_supervisor_graph()
    state = _drive(graph, query="aspirin MI")
    # simulate a fabricated citation sneaking in before verification:
    # re-run verify logic directly
    from medai.graph import _verify_citations
    fake = {**state, "analysis": {**state["analysis"], "cited_papers": ["P001", "FAKE01"]}}
    out = _verify_citations(fake, audit)
    assert "FAKE01" not in out["analysis"]["cited_papers"]
    assert audit.events_of("citation_verification")


def test_supervisor_survives_injected_failures():
    graph, audit, _ = build_supervisor_graph(failure_rate=0.3, seed=42)
    state = _drive(graph)
    # either completed ok, or failed *gracefully* with an audit trail intact
    assert state["status"] == "ok" or state["status"].startswith("failed")
    assert state["status"] != "ok" or state["analysis"]["k"] >= 1
    assert audit.verify_chain()


# ---------------- eval harness ----------------
def test_gold_answers_are_consistent():
    from eval.tasks import TASKS, gold_answer
    for t in TASKS:
        g = gold_answer(t)
        assert g["gold_citations"], t["task_id"]
        assert 0 < g["gold_pooled"] < 10, (t["task_id"], g["gold_pooled"])
        assert g["gold_ci"][0] < g["gold_pooled"] < g["gold_ci"][1]


def test_supervisor_zero_failure_scores_well_on_one_task():
    from experiments.architectures import ARCHITECTURES
    from eval.metrics import citation_metrics, task_success
    from eval.tasks import TASKS, gold_answer
    task = TASKS[0]
    gold = gold_answer(task)
    res = ARCHITECTURES["A3_supervisor"](task["query"], 0.0, 123)
    assert res.status == "ok"
    assert res.approval_requested and res.analysis_ran
    assert task_success(res, gold) == 1.0
    cm = citation_metrics(res.citations, gold["gold_citations"])
    assert cm["citation_precision"] == 1.0 and cm["citation_recall"] == 1.0
    assert res.audit_ok
