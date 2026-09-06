"""Hermetic tests for the LLM layer: mocked OpenRouter transport, cache
behaviour, contract validation, and deterministic fallback. No network."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAKE_PAPER = dict(id="P1", title="Trial of X", source="pubmed",
                  abstract="Drug X reduced outcomes (risk ratio 0.56, 95% CI 0.45 to 0.70). "
                           "The trial randomized 500 patients.")


def _completion(content):
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}}


def _with_transport(monkeypatch, handler):
    """Patch medai.llm transport with a handler(req_body_dict) -> response dict."""
    import medai.llm as L
    calls = []

    class FakeResp:
        def __init__(self, payload): self.payload = payload
        def read(self): return json.dumps(self.payload).encode()

    def fake_urlopen(req, timeout=0):
        body = json.loads(req.data.decode())
        calls.append(body)
        resp = handler(body, calls)
        if isinstance(resp, Exception):
            raise resp
        return FakeResp(resp)

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return L, calls


def _with_key(monkeypatch, tmp_path):
    import medai.llm as L
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(L, "_cache_dir", lambda: str(tmp_path / "llm-cache"))
    return L


def test_llm_disabled_without_key(monkeypatch, tmp_path):
    import medai.llm as L
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert not L.llm_enabled()
    try:
        L.chat([{"role": "user", "content": "hi"}])
        raise AssertionError("should have raised")
    except L.LLMError as e:
        assert "not configured" in str(e)


def test_chat_caches_and_audits(monkeypatch, tmp_path):
    L, calls = _with_transport(monkeypatch, lambda body, calls: _completion("hello"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(L, "_cache_dir", lambda: str(tmp_path / "llm-cache"))
    log = []
    class Audit:
        def record(self, kind, **kw): log.append({"kind": kind, **kw})
    a = Audit()
    r1 = L.chat([{"role": "user", "content": "q"}], audit=a)
    r2 = L.chat([{"role": "user", "content": "q"}], audit=a)  # served from cache
    assert r1 == r2 == "hello"
    assert len(calls) == 1, "second call must be a cache hit"
    kinds = [x["kind"] for x in log]
    assert kinds.count("llm_call") == 1 and kinds.count("llm_cache") == 1
    assert (tmp_path / "llm-cache").exists()


def test_chat_retries_on_429(monkeypatch, tmp_path):
    state = {"n": 0}
    def handler(body, calls):
        state["n"] += 1
        if state["n"] == 1:
            import urllib.error
            return urllib.error.HTTPError("u", 429, "rate limited", None, None)
        return _completion("ok")
    L, calls = _with_transport(monkeypatch, handler)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(L, "_cache_dir", lambda: str(tmp_path / "llm-cache"))
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    assert L.chat([{"role": "user", "content": "q"}]) == "ok"
    assert state["n"] == 2


def test_chat_json_parses_and_repairs(monkeypatch, tmp_path):
    L, _ = _with_transport(monkeypatch, lambda body, calls: _completion("```json\n{\"a\": 1}\n```"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(L, "_cache_dir", lambda: str(tmp_path / "llm-cache"))
    assert L.chat_json([{"role": "user", "content": "q"}]) == {"a": 1}


def test_llm_extraction_validates_spans_and_numbers(monkeypatch, tmp_path):
    L, _ = _with_transport(monkeypatch, lambda body, calls: _completion(json.dumps({
        "effects": [
            {"measure": "rr", "point": 0.56, "lo": 0.45, "hi": 0.70,
             "span": "risk ratio 0.56, 95% CI 0.45 to 0.70"},
            {"measure": "rr", "point": 0.10, "lo": 0.2, "hi": 0.05,
             "span": "fabricated numbers"},                       # lo>hi -> rejected
            {"measure": "rr", "point": 0.30, "lo": 0.1, "hi": 0.6,
             "span": "this span is not in the abstract at all"},  # not verbatim -> rejected
            {"measure": "auc", "point": 0.9, "lo": 0.8, "hi": 0.95,
             "span": "AUC 0.9"},                                  # unknown metric -> rejected
        ]})))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(L, "_cache_dir", lambda: str(tmp_path / "llm-cache"))
    from medai.llm_nodes import llm_extract_effects
    effs = llm_extract_effects(FAKE_PAPER, "drug x outcomes")
    assert len(effs) == 1
    assert effs[0]["measure"] == "rr" and effs[0]["point"] == 0.56
    assert effs[0]["extracted_by"] == "llm"
    assert effs[0]["paper_id"] == "P1" and effs[0]["se"] > 0


def test_llm_screening_partitions(monkeypatch, tmp_path):
    L, _ = _with_transport(monkeypatch, lambda body, calls: _completion(json.dumps(
        {"include": ["P1"], "exclude": [{"id": "P2", "reason": "off topic"}]})))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(L, "_cache_dir", lambda: str(tmp_path / "llm-cache"))
    from medai.llm_nodes import llm_screen_papers
    papers = [dict(id="P1", title="on topic", abstract="aspirin trial"),
              dict(id="P2", title="off topic", abstract="soil chemistry")]
    kept, excluded = llm_screen_papers("aspirin outcomes", papers)
    assert [p["id"] for p in kept] == ["P1"]
    assert excluded[0]["id"] == "P2" and "off topic" in excluded[0]["exclusion_reason"]


def test_llm_writer_requires_all_statistics(monkeypatch, tmp_path):
    L, _ = _with_transport(monkeypatch, lambda body, calls: _completion(
        "The pooled estimate was 0.6 (95% CI 0.5–0.7), with I² = 60%."))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(L, "_cache_dir", lambda: str(tmp_path / "llm-cache"))
    from medai.llm_nodes import llm_rewrite_abstract
    sections = [{"label": "Results",
                 "text": "The pooled estimate was 0.6 (95% CI 0.5–0.7; p = 0.016), I² = 60%."}]
    text = llm_rewrite_abstract(sections, "drug x")
    assert "0.6" in text and "I² = 60%" in text


def test_graph_llm_extraction_with_deterministic_fallback(monkeypatch, tmp_path):
    # Two stubbed papers: P1 is LLM-extracted (valid verbatim span), P2 makes
    # the LLM fail so the rule-based extractor must rescue it.
    import medai.graph as G
    import medai.llm as L
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "llm-cache"))

    fake_papers = [
        dict(id="P1", title="Trial of drug X",
             abstract="We randomized 500 patients. Drug X reduced outcomes "
                      "(risk ratio 0.56, 95% CI 0.45 to 0.70)."),
        dict(id="P2", title="Cohort of marker Y",
             abstract="Follow-up showed harm (odds ratio 1.50, 95% CI 1.10 to 2.05)."),
    ]
    for adapter in ("_pubmed_search", "_openalex_search", "_arxiv_search",
                    "_semantic_scholar_search", "_fixture_search"):
        monkeypatch.setattr(G, adapter, lambda *a, **k: [dict(p) for p in fake_papers])

    class FakeResp:
        def __init__(self, payload): self.payload = payload
        def read(self): return json.dumps(self.payload).encode()

    def fake_transport(req, timeout=0):
        body = json.loads(req.data.decode())
        user = body["messages"][-1]["content"]
        if "risk ratio 0.56" in user:      # P1: LLM extracts successfully
            content = json.dumps({"effects": [
                {"measure": "rr", "point": 0.60, "lo": 0.50, "hi": 0.72,
                 "span": "risk ratio 0.56, 95% CI 0.45 to 0.70"}]})
        elif "odds ratio 1.50" in user:    # P2: LLM unavailable -> regex fallback
            raise L.LLMError("OpenRouter down for this paper")
        else:
            content = json.dumps({"effects": []})
        return FakeResp(_completion(content))

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_transport)

    graph, audit, tools = G.build_supervisor_graph()
    from langgraph.types import Command
    cfg = {"configurable": {"thread_id": "llm-e2e"}}
    state = graph.invoke({"query": "drug x outcomes", "max_results": 10,
                          "status": "ok", "llm": {"extract": True}}, cfg)
    if state.get("__interrupt__"):
        resume = {i.id: {"approved": True, "approver": "t"} for i in state["__interrupt__"]}
        state = graph.invoke(Command(resume=resume), cfg)
    assert state["status"] == "ok"
    extracted_by = [e.get("extracted_by", "rules") for e in state["evidence"]]
    assert "llm" in extracted_by and "rules" in extracted_by
    assert audit.verify_chain()


def test_graph_falls_back_to_rules_when_llm_fails(monkeypatch, tmp_path):
    import medai.graph as G
    import medai.llm as L
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "llm-cache2"))
    monkeypatch.setattr(L, "_cache_dir", lambda: str(tmp_path / "llm-cache2"))

    def boom(req, timeout=0):
        raise L.LLMError("OpenRouter down")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", boom)

    graph, audit, tools = G.build_supervisor_graph()
    from langgraph.types import Command
    cfg = {"configurable": {"thread_id": "llm-fallback"}}
    state = graph.invoke({"query": "aspirin myocardial infarction risk", "max_results": 10,
                          "status": "ok", "llm": {"extract": True, "screen": True}}, cfg)
    if state.get("__interrupt__"):
        resume = {i.id: {"approved": True, "approver": "t"} for i in state["__interrupt__"]}
        state = graph.invoke(Command(resume=resume), cfg)
    assert state["status"] == "ok", "deterministic path must rescue the run"
    assert len(state["evidence"]) >= 4
    assert audit.verify_chain()
    assert any(e["kind"] == "llm_fallback" for e in audit.events)


def test_screening_sanity_bound_blocks_mass_exclusion(monkeypatch, tmp_path):
    # a model that excludes EVERY record fails the sanity bound: the keyword
    # screen must take over so recall is protected
    import medai.graph as G
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "llm-cache3"))
    import medai.llm as L
    monkeypatch.setattr(L, "_cache_dir", lambda: str(tmp_path / "llm-cache3"))

    papers = [dict(id="P1", title="Statin trial", abstract="statin outcomes",
                   source="pubmed"),
              dict(id="P2", title="Statin cohort", abstract="statin follow-up",
                   source="pubmed"),
              dict(id="P3", title="Statin rct", abstract="statin randomized",
                   source="pubmed"),
              dict(id="P4", title="Statin meta", abstract="statin overview",
                   source="pubmed")]
    for adapter in ("_pubmed_search", "_openalex_search", "_arxiv_search",
                    "_semantic_scholar_search", "_fixture_search"):
        monkeypatch.setattr(G, adapter, lambda *a, **k: [dict(p) for p in papers])

    class FakeResp:
        def __init__(self, payload): self.payload = payload
        def read(self): return json.dumps(self.payload).encode()

    def fake_transport(req, timeout=0):
        body = json.loads(req.data.decode())
        if "screen" in body["messages"][-1]["content"] or len(body["messages"]) == 2:
            excl = [{"id": f"P{i+1}", "reason": "not relevant"} for i in range(4)]
            return FakeResp(_completion(json.dumps({"include": [], "exclude": excl})))
        return FakeResp(_completion(json.dumps({"effects": []})))

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_transport)

    graph, audit, tools = G.build_supervisor_graph()
    from langgraph.types import Command
    cfg = {"configurable": {"thread_id": "llm-sanity"}}
    state = graph.invoke({"query": "statin outcomes", "max_results": 10,
                          "status": "ok", "llm": {"screen": True}}, cfg)
    # sanity bound caught the mass exclusion: papers survive via keyword screen
    assert len(state.get("papers", [])) >= 3
    fb = [e for e in audit.events if e["kind"] == "llm_fallback" and e.get("node") == "screen"]
    assert fb and "implausible exclusion rate" in fb[0].get("reason", "")
    assert audit.verify_chain()
