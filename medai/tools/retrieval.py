"""Paper retrieval: offline fixture corpus (default) or live PubMed.

Live mode uses NCBI E-utilities (free, no key needed below 3 req/s):
  esearch (find PMIDs, relevance-sorted) -> efetch (full title + abstract XML)
so live papers carry real abstract text that the evidence extractor can parse.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from medai.corpus import FIXTURE_CORPUS, TOPIC_KEYWORDS

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENALEX = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API = "https://export.arxiv.org/api/query"
SCOPUS_API = "https://api.elsevier.com/content/search/scopus"
IEEE_API = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
SERPER_SCHOLAR = "https://google.serper.dev/scholar"  # licensed Google Scholar access
OPENALEX_MAILTO = "medai-agent@example.org"  # OpenAlex polite pool

# Live sources, queried in order. Google Scholar itself has no API and forbids
# automated access (CAPTCHA/ToS), so scholarly-web coverage comes from OpenAlex
# (~250M works), Semantic Scholar, and arXiv (preprints). Scopus and IEEE
# Xplore are key-gated: they join the fan-out only when SCOPUS_API_KEY /
# IEEE_API_KEY are configured.
DEFAULT_LIVE_SOURCES = ("pubmed", "scopus", "openalex", "google_scholar",
                        "arxiv", "semantic_scholar", "ieee")
KEYED_SOURCES = {"scopus": "SCOPUS_API_KEY", "ieee": "IEEE_API_KEY",
                 "google_scholar": "SERPER_API_KEY"}


def available_sources() -> tuple[str, ...]:
    """Live sources that can actually run right now — key-gated ones are
    included only when their API key is configured."""
    out = []
    for src in DEFAULT_LIVE_SOURCES:
        env = KEYED_SOURCES.get(src)
        if env and not os.environ.get(env):
            continue
        out.append(src)
    return tuple(out)

# Live retrieval targets study designs that report poolable effects: without
# this, broad topical queries surface narrative reviews and commentary whose
# abstracts contain no effect estimates. Meta-analyses are deliberately
# excluded — this agent pools primary studies, not other syntheses.
PUBMED_STUDY_FILTER = (' AND (randomized controlled trial[pt] OR clinical trial[pt] '
                       'OR observational study[pt] OR comparative study[pt])')
PUBMED_HASABSTRACT = ' AND hasabstract[filter]'


def pubmed_effective_query(query: str) -> str:
    """The PubMed term actually executed (recorded in the audit log)."""
    return f"({query}){PUBMED_HASABSTRACT}{PUBMED_STUDY_FILTER}"

# Concept groups for keyword expansion: a query term matching any member of a
# group also matches the other members at reduced weight, so "heart attack"
# retrieves papers that only say "myocardial infarction". Phrase forms are
# counted as substrings.
CONCEPT_GROUPS = [
    {"myocardial infarction", "heart attack", "infarction", "myocardial"},
    {"statin", "ldl", "cholesterol", "lipid",
     "atorvastatin", "rosuvastatin", "simvastatin", "ezetimibe"},
    {"ssri", "depression", "depressive", "antidepressant",
     "sertraline", "fluoxetine", "escitalopram", "paroxetine"},
]
SYNONYM_WEIGHT = 0.5


def expand_terms(tokens: list[str]) -> list[tuple[str, float]]:
    """Return (term, weight) pairs: original tokens (and adjacent bigrams) at
    weight 1.0, concept-group synonyms at SYNONYM_WEIGHT."""
    units = [(t, 1.0) for t in tokens]
    units += [(f"{a} {b}", 1.0) for a, b in zip(tokens, tokens[1:])]
    out: dict[str, float] = {}
    for term, w in units:
        out[term] = max(out.get(term, 0.0), w)
        for group in CONCEPT_GROUPS:
            if term in group:
                for syn in group:
                    out[syn] = max(out.get(syn, 0.0), SYNONYM_WEIGHT)
    return sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))


SOURCE_PRIORITY = {"corpus": 0, "pubmed": 1, "scopus": 2, "openalex": 3,
                   "google_scholar": 4, "arxiv": 5, "semantic_scholar": 6, "ieee": 7}


def merge_sources(paper_lists: list[list[dict]], cap: int,
                  priority: list[str] | None = None) -> list[dict]:
    """Deduplicate (DOI first-occurrence wins) and round-robin interleave
    source result lists up to `cap`. Lists are processed in priority order
    (earlier sources keep their copy of duplicates) so the merge is
    deterministic regardless of completion order."""
    order = priority or [f"src{i}" for i in range(len(paper_lists))]
    seen: set[str] = set()
    deduped: list[list[dict]] = []
    for _src, lst in sorted(zip(order, paper_lists), key=lambda t: SOURCE_PRIORITY.get(t[0], 99)):
        fresh = []
        for p in lst:
            key = _dedup_key(p)
            if key not in seen:
                seen.add(key)
                fresh.append(p)
        deduped.append(fresh)
    merged: list[dict] = []
    for batch in itertools.zip_longest(*deduped):
        for p in batch:
            if p is not None and len(merged) < cap:
                merged.append(p)
    return merged


def search_papers(query: str, max_results: int = 10, live: bool = False,
                  sources: tuple[str, ...] | None = None) -> list[dict]:
    """Return papers [{id, title, abstract, source}] best matching `query`.

    Offline mode does keyword scoring (with synonym expansion) over the
    fixture corpus — deterministic and reproducible. Live mode queries the
    selected sources (PubMed + OpenAlex + Semantic Scholar by default),
    merges results with DOI-based deduplication, and degrades gracefully
    when any single source fails. (The supervisor graph fans these sources
    out as parallel nodes; this sequential version is used by the baselines
    and CLI.)
    """
    if live:
        failures = 0
        results: list[list[dict]] = []
        order: list[str] = []
        for src in (sources or available_sources()):
            try:
                if src == "pubmed":
                    papers = _pubmed_search(query, max_results)
                elif src == "scopus":
                    papers = _scopus_search(query, max_results)
                elif src == "openalex":
                    papers = _openalex_search(query, max_results)
                elif src == "google_scholar":
                    papers = _google_scholar_search(query, max_results)
                elif src == "arxiv":
                    papers = _arxiv_search(query, max_results)
                elif src == "semantic_scholar":
                    papers = _semantic_scholar_search(query, max_results)
                elif src == "ieee":
                    papers = _ieee_search(query, max_results)
                else:
                    continue
            except Exception:
                failures += 1
                continue  # one failing source never breaks the run
            if papers:
                results.append(papers)
                order.append(src)
        if results:
            return merge_sources(results, max_results, priority=order)
        if failures == 0:
            return []  # genuinely no results anywhere
        # every source failed (e.g. offline) -> fall back to the corpus
    return _fixture_search(query, max_results)


def _dedup_key(paper: dict) -> str:
    doi = (paper.get("doi") or "").lower().removeprefix("https://doi.org/")
    return f"doi:{doi}" if doi else f"title:{paper.get('title', '').lower()[:80]}"


def _get_json(url: str, headers: dict | None = None, retries: int = 1) -> dict:
    """GET JSON with one polite retry on HTTP 429 (rate limiting)."""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            return json.load(urllib.request.urlopen(req, timeout=15))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(2.0)
                continue
            raise


def _openalex_search(query: str, max_results: int) -> list[dict]:
    params = urllib.parse.urlencode({
        "search": query, "per-page": max_results,
        # "article" is OpenAlex's current type value (journal-article is retired)
        "filter": "has_abstract:true,type:article",
        "mailto": OPENALEX_MAILTO,
    })
    data = _get_json(f"{OPENALEX}?{params}")
    out = []
    for w in data.get("results", []):
        doi = (w.get("doi") or "").removeprefix("https://doi.org/").lower()
        pid = f"DOI:{doi}" if doi else f"OAW:{(w.get('id') or '').rsplit('/', 1)[-1]}"
        out.append(dict(id=pid, title=w.get("display_name") or "",
                        abstract=_openalex_abstract(w.get("abstract_inverted_index")),
                        doi=doi, source="openalex"))
    return out


def _openalex_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def _semantic_scholar_search(query: str, max_results: int) -> list[dict]:
    params = urllib.parse.urlencode({
        "query": query, "limit": max_results,
        "fields": "title,abstract,externalIds,year",
    })
    # anonymous access is strictly rate-limited; a free S2 API key lifts it
    api_key = os.environ.get("S2_API_KEY")
    headers = {"x-api-key": api_key} if api_key else None
    data = _get_json(f"{SEMANTIC_SCHOLAR}?{params}", headers=headers)
    out = []
    for w in data.get("data", []):
        if not w.get("abstract"):
            continue
        ext = w.get("externalIds") or {}
        doi = (ext.get("DOI") or "").lower()
        pid = f"DOI:{doi}" if doi else f"S2:{w.get('paperId', '')}"
        out.append(dict(id=pid, title=w.get("title") or "",
                        abstract=w["abstract"], doi=doi,
                        source="semantic_scholar"))
    return out


def _arxiv_search(query: str, max_results: int) -> list[dict]:
    """arXiv Atom API: AND-join the query terms over the all-field, sorted by
    relevance. arXiv's politeness guideline is ~1 request / 3s — we issue one
    call per run, which respects it."""
    tokens = [t for t in query.lower().split() if len(t) > 1]
    if not tokens:
        return []
    params = urllib.parse.urlencode({
        "search_query": " AND ".join(f"all:{t}" for t in tokens),
        "max_results": max_results, "sortBy": "relevance",
    })
    xml_bytes = urllib.request.urlopen(f"{ARXIV_API}?{params}", timeout=15).read()
    return _parse_arxiv_xml(xml_bytes)


_ATOM = "{http://www.w3.org/2005/Atom}"


def _parse_arxiv_xml(xml_bytes: bytes) -> list[dict]:
    out = []
    for entry in ET.fromstring(xml_bytes).iter(f"{_ATOM}entry"):
        raw_id = (entry.findtext(f"{_ATOM}id") or "").rsplit("/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", raw_id)  # strip version suffix
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        abstract = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())
        if not arxiv_id or not abstract:
            continue
        doi_el = entry.find(f"{_ATOM}doi")
        if doi_el is None:  # arXiv puts its own doi in the arxiv namespace
            doi_el = entry.find("{http://arxiv.org/schemas/atom}doi")
        doi = (doi_el.text or "").lower() if doi_el is not None else ""
        out.append(dict(id=f"ARXIV:{arxiv_id}", title=title, abstract=abstract,
                        doi=doi, source="arxiv"))
    return out


def _strip_html(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", text or "").split()).strip()


def _scopus_search(query: str, max_results: int) -> list[dict]:
    """Scopus (Elsevier) search API. Requires SCOPUS_API_KEY — the source is
    only dispatched when the key is configured (see available_sources)."""
    key = os.environ.get("SCOPUS_API_KEY")
    if not key:
        raise RuntimeError("SCOPUS_API_KEY not configured")
    params = urllib.parse.urlencode({
        "query": query, "count": max_results,
        "field": "dc:title,dc:description,prism:doi,dc:identifier",
    })
    data = _get_json(f"{SCOPUS_API}?{params}", headers={"X-ELS-APIKey": key})
    entries = (data.get("search-results") or {}).get("entry") or []
    out = []
    for e in entries:
        if isinstance(e, dict) and "error" in e:
            continue
        abstract = _strip_html(e.get("dc:description"))
        if not abstract:
            continue
        doi = (e.get("prism:doi") or "").lower()
        scopus_id = (e.get("dc:identifier") or "").split(":")[-1]
        pid = f"DOI:{doi}" if doi else f"SCOPUS:{scopus_id}"
        out.append(dict(id=pid, title=_strip_html(e.get("dc:title")),
                        abstract=abstract, doi=doi, source="scopus"))
    return out


def _ieee_search(query: str, max_results: int) -> list[dict]:
    """IEEE Xplore metadata API. Requires IEEE_API_KEY (free registration);
    only dispatched when the key is configured."""
    key = os.environ.get("IEEE_API_KEY")
    if not key:
        raise RuntimeError("IEEE_API_KEY not configured")
    params = urllib.parse.urlencode({
        "apikey": key, "querytext": query,
        "max_records": max_results, "format": "json",
    })
    data = _get_json(f"{IEEE_API}?{params}")
    out = []
    for a in (data.get("articles") or []):
        abstract = _strip_html(a.get("abstract"))
        if not abstract:
            continue
        doi = (a.get("doi") or "").lower()
        article_number = str(a.get("article_number") or "")
        html_url = a.get("html_url") or ""
        pid = f"DOI:{doi}" if doi else (f"IEEE:{article_number}" if article_number
                                        else "IEEE:" + html_url.rsplit("/", 1)[-1])
        out.append(dict(id=pid, title=_strip_html(a.get("title")),
                        abstract=abstract, doi=doi, source="ieee"))
    return out


def _google_scholar_search(query: str, max_results: int) -> list[dict]:
    """Google Scholar results via the licensed Serper.dev API (SERPER_API_KEY;
    free one-time allowance at signup). NOTE: Scholar serves titles + short
    snippets, not full abstracts — this source is strongest for discovery,
    coverage and reading links; effect extraction rarely applies to snippets."""
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY not configured")
    body = json.dumps({"q": query, "num": min(max_results, 20)}).encode()
    req = urllib.request.Request(SERPER_SCHOLAR, data=body, method="POST",
        headers={"X-API-KEY": key, "Content-Type": "application/json"})
    data = json.load(urllib.request.urlopen(req, timeout=15))
    out = []
    for r in (data.get("organic") or [])[:max_results]:
        link = r.get("link") or ""
        title = _strip_html(r.get("title"))
        snippet = _strip_html(r.get("snippet"))
        if not title or not link:
            continue
        m = re.search(r"doi\.org/(10\.\S+)", link)
        doi = m.group(1).lower() if m else ""
        gs_id = "GS:" + hashlib.md5(link.encode()).hexdigest()[:10]
        pid = f"DOI:{doi}" if doi else gs_id
        out.append(dict(id=pid, title=title, abstract=snippet or title,
                        doi=doi, source="google_scholar", url=link))
    return out


def _fixture_search(query: str, max_results: int) -> list[dict]:
    tokens = [t for t in query.lower().replace(":", " ").split() if len(t) > 2]
    terms = expand_terms(tokens)
    scored = [(_fixture_score(p, terms), p) for p in FIXTURE_CORPUS]
    scored = [sp for sp in sorted(scored, key=lambda x: -x[0]) if sp[0] > 0]
    return [dict(id=p["id"], title=p["title"], abstract=p["abstract"], score=s)
            for s, p in scored[:max_results]]


def _fixture_score(paper: dict, terms: list[tuple[str, float]]) -> float:
    text = (paper["title"] + " " + paper["abstract"]).lower()
    return sum(w * text.count(t) for t, w in terms)


def _pubmed_search(query: str, max_results: int) -> list[dict]:
    params = urllib.parse.urlencode({"db": "pubmed", "term": pubmed_effective_query(query),
                                     "retmax": max_results, "retmode": "json",
                                     "sort": "relevance"})
    ids = json.load(urllib.request.urlopen(f"{EUTILS}/esearch.fcgi?{params}", timeout=15))
    id_list = ids["esearchresult"].get("idlist", [])
    if not id_list:
        return []
    # efetch returns full records (title + structured abstract) as XML
    params = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(id_list),
                                     "rettype": "abstract", "retmode": "xml"})
    xml_bytes = urllib.request.urlopen(f"{EUTILS}/efetch.fcgi?{params}", timeout=15).read()
    return _parse_pubmed_xml(xml_bytes)


def _parse_pubmed_xml(xml_bytes: bytes) -> list[dict]:
    """Parse efetch XML into papers, joining all AbstractText sections."""
    out = []
    for art in ET.fromstring(xml_bytes).iter("PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""
        abstract = " ".join("".join(a.itertext()) for a in art.iter("AbstractText")).strip()
        doi_el = art.find(".//ArticleId[@IdType='doi']")
        doi = (doi_el.text or "").lower() if doi_el is not None else ""
        out.append(dict(id=f"PMID:{pmid}", title=title.strip(), abstract=abstract,
                        doi=doi, source="pubmed"))
    return out


def screen_papers(papers: list[dict], keywords: list[str]) -> list[dict]:
    """PICOS-style relevance screen: keep papers whose title+abstract mention
    at least one topic keyword. Shared by all architectures."""
    if not keywords:
        return papers
    out = []
    for p in papers:
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        if any(k in text for k in keywords):
            out.append(p)
    return out


def infer_topic_keywords(query: str) -> list[str]:
    """Deterministic planner: pick the topic whose keywords appear in the query."""
    q = query.lower()
    best, best_hits = None, 0
    for topic, kws in TOPIC_KEYWORDS.items():
        hits = sum(1 for k in kws if k in q)
        if hits > best_hits:
            best, best_hits = topic, hits
    return TOPIC_KEYWORDS.get(best, [])


def valid_paper_list(result) -> bool:
    """Validator for the guard: a well-formed (possibly empty) paper list.
    Legitimate no-hit runs are valid; corrupted items (None, wrong types,
    missing fields) are silent failures."""
    return (isinstance(result, list)
            and all(isinstance(p, dict) and isinstance(p.get("id"), str)
                    and isinstance(p.get("title"), str) and isinstance(p.get("abstract"), str)
                    for p in result))
