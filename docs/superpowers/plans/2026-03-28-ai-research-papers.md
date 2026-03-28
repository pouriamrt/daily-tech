# AI Research Papers Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a curated daily AI research papers section (arXiv + HuggingFace Daily Papers) to the Daily Tech Intelligence report, with LLM two-pass ranking and methodology-focused structured summaries.

**Architecture:** Two new data sources (arXiv API, HF Daily Papers API) feed into a two-pass LLM pipeline: Pass 1 ranks ~20-30 candidates by relevance, Pass 2 generates structured HTML summaries for the top 5. Papers are stored in the existing `knowledge` table with `source="arxiv:{id}"` and rendered as a dedicated section above GitHub cards in the HTML report.

**Tech Stack:** Python 3.13, httpx/hishel (existing), xml.etree.ElementTree (stdlib), langchain (existing), SQLite (existing)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `dtech.py` | Modify | All new functions + modified report generator + modified `__main__` |
| `tests/test_papers.py` | Create | Tests for XML parsing, dedup logic, JSON rank parsing |
| `pyproject.toml` | Modify | Add pytest dev dependency |

---

### Task 1: Set up test infrastructure

**Files:**
- Modify: `pyproject.toml:1-14`
- Create: `tests/__init__.py`
- Create: `tests/test_papers.py`

- [ ] **Step 1: Add pytest dependency**

```bash
uv add --dev pytest
```

- [ ] **Step 2: Create test directory and init file**

```bash
mkdir -p tests
```

Write `tests/__init__.py` as an empty file.

- [ ] **Step 3: Create test file with placeholder**

Write `tests/test_papers.py`:

```python
"""Tests for the AI research papers pipeline."""
```

- [ ] **Step 4: Verify pytest runs**

Run: `uv run pytest tests/ -v`
Expected: 0 tests collected, no errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/
git commit -m "chore: add pytest and test infrastructure"
```

---

### Task 2: Add PaperCandidate dataclass and paper category metadata

**Files:**
- Modify: `dtech.py:1-8` (imports)
- Modify: `dtech.py:104-108` (after KnowledgeEntry)
- Modify: `dtech.py:218-230` (`_source_category`)
- Modify: `dtech.py:206-215` (`nice_source_label`)
- Modify: `dtech.py:242-293` (`_CATEGORY_META`)

- [ ] **Step 1: Add xml import**

In `dtech.py`, add after line 8 (`from urllib.parse import urlparse`):

```python
import xml.etree.ElementTree as ET
```

Also add `import json` after `import logging` (line 3) — needed later for parsing LLM ranking output.

- [ ] **Step 2: Add PaperCandidate dataclass**

In `dtech.py`, add after the `KnowledgeEntry` dataclass (after line 108):

```python

@dataclass(frozen=True)
class PaperCandidate:
    arxiv_id: str
    title: str
    abstract: str
    published: str
    pdf_url: str
    categories: str
    hf_trending: bool = False
```

- [ ] **Step 3: Update `_source_category` to recognize paper sources**

In `dtech.py`, modify `_source_category` (line 218-230). Add at the beginning of the function body, before the existing `if "/search/repositories"` check:

```python
def _source_category(url: str) -> str:
    """Classify a source URL into a category for visual grouping."""
    if url.startswith("arxiv:"):
        return "paper"
    if "/search/repositories" in url:
```

(The rest of the function stays the same.)

- [ ] **Step 4: Update `nice_source_label` to handle paper sources**

In `dtech.py`, modify `nice_source_label` (line 206-215). Add at the beginning of the function body:

```python
def nice_source_label(url: str) -> str:
    """Turn a GitHub API URL into a readable label, e.g. 'anthropics/claude-code · releases'."""
    if url.startswith("arxiv:"):
        return url.replace("arxiv:", "arXiv · ")
    path = urlparse(url).path.strip("/")
```

(The rest stays the same.)

- [ ] **Step 5: Add "paper" to `_CATEGORY_META`**

In `dtech.py`, add a new entry to `_CATEGORY_META` dict, after the `"other"` entry (before the closing `}`). Add it between `"release"` (line 275) and `"other"` (line 284):

```python
    "paper": (
        "Research Papers",
        "#0891b2",
        _svg(
            '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>'
            '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>'
        ),
    ),
```

- [ ] **Step 6: Write tests for category and label helpers**

In `tests/test_papers.py`:

```python
"""Tests for the AI research papers pipeline."""

import sys
from pathlib import Path

# Add project root to path so we can import dtech
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtech import _source_category, nice_source_label


def test_source_category_arxiv():
    assert _source_category("arxiv:2403.12345") == "paper"


def test_source_category_github_unchanged():
    assert _source_category(
        "https://api.github.com/repos/fastapi/fastapi/releases"
    ) == "release"


def test_nice_source_label_arxiv():
    assert nice_source_label("arxiv:2403.12345") == "arXiv · 2403.12345"


def test_nice_source_label_github_unchanged():
    url = "https://api.github.com/repos/fastapi/fastapi/releases"
    assert "fastapi/fastapi" in nice_source_label(url)
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_papers.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add dtech.py tests/test_papers.py
git commit -m "feat: add PaperCandidate dataclass and paper category metadata"
```

---

### Task 3: Implement `fetch_arxiv_papers()`

**Files:**
- Modify: `dtech.py` (add function after `_hint_for`, around line 144)
- Modify: `tests/test_papers.py` (add XML parsing test)

- [ ] **Step 1: Write test for arXiv XML parsing**

Add to `tests/test_papers.py`:

```python
from dtech import fetch_arxiv_papers

SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2403.12345v1</id>
    <title>  Test Paper: A Novel Approach
    to Something  </title>
    <summary>  This paper presents a novel approach
    to doing something interesting in ML.  </summary>
    <published>2026-03-27T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2403.12345v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2403.12345v1" title="pdf" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
    <category term="cs.AI"/>
  </entry>
</feed>"""


def test_fetch_arxiv_papers_parses_xml(monkeypatch):
    """Test that arXiv XML is correctly parsed into PaperCandidate objects."""
    import httpx

    class FakeResponse:
        status_code = 200
        text = SAMPLE_ARXIV_XML
        def raise_for_status(self):
            pass

    class FakeClient:
        def get(self, url, **kwargs):
            return FakeResponse()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr("dtech._build_paper_client", lambda: FakeClient())

    papers = fetch_arxiv_papers(days=3)
    assert len(papers) == 1
    p = papers[0]
    assert p.arxiv_id == "2403.12345"
    assert "Novel Approach" in p.title
    assert "novel approach" in p.abstract
    assert p.pdf_url == "http://arxiv.org/pdf/2403.12345v1"
    assert "cs.LG" in p.categories
    assert p.hf_trending is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_papers.py::test_fetch_arxiv_papers_parses_xml -v`
Expected: FAIL — `fetch_arxiv_papers` not defined.

- [ ] **Step 3: Implement `_build_paper_client` and `fetch_arxiv_papers`**

In `dtech.py`, add after the `_hint_for` function (after line 144):

```python

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

ARXIV_QUERY = (
    "http://export.arxiv.org/api/query"
    "?search_query=cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL"
    "&sortBy=submittedDate&sortOrder=descending&max_results=20"
)


def _build_paper_client() -> SyncCacheClient:
    """Create an httpx client for paper APIs (no GitHub auth headers)."""
    storage = SyncSqliteStorage(
        database_path=str(BASE_DIR / ".http_cache.db"),
        default_ttl=3600.0,
    )
    return SyncCacheClient(
        storage=storage,
        headers={"User-Agent": "daily-knowledge-bot"},
        timeout=30.0,
    )


def _parse_arxiv_id(raw_id: str) -> str:
    """Extract the numeric arXiv ID from a full URL like 'http://arxiv.org/abs/2403.12345v1'."""
    # Take the last path segment and strip version suffix
    segment = raw_id.rstrip("/").split("/")[-1]
    if "v" in segment:
        segment = segment[: segment.rfind("v")]
    return segment


def fetch_arxiv_papers(days: int = 3) -> list[PaperCandidate]:
    """Fetch recent AI papers from arXiv and return parsed candidates."""
    papers: list[PaperCandidate] = []
    with _build_paper_client() as client:
        try:
            resp = client.get(ARXIV_QUERY)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("Failed to fetch arXiv: %s", exc)
            return papers

    root = ET.fromstring(resp.text)
    cutoff = datetime.now() - timedelta(days=days)

    for entry in root.findall("atom:entry", ARXIV_NS):
        published_text = entry.findtext("atom:published", "", ARXIV_NS)
        if not published_text:
            continue
        published_dt = datetime.fromisoformat(published_text.replace("Z", "+00:00"))
        if published_dt.replace(tzinfo=None) < cutoff:
            continue

        raw_id = entry.findtext("atom:id", "", ARXIV_NS)
        arxiv_id = _parse_arxiv_id(raw_id)

        title = " ".join(entry.findtext("atom:title", "", ARXIV_NS).split())
        abstract = " ".join(entry.findtext("atom:summary", "", ARXIV_NS).split())

        pdf_url = ""
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        cats = [c.get("term", "") for c in entry.findall("atom:category", ARXIV_NS)]

        papers.append(
            PaperCandidate(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                published=published_text,
                pdf_url=pdf_url,
                categories=", ".join(cats),
            )
        )

    log.info("Fetched %d papers from arXiv", len(papers))
    return papers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_papers.py::test_fetch_arxiv_papers_parses_xml -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dtech.py tests/test_papers.py
git commit -m "feat: add fetch_arxiv_papers with XML parsing"
```

---

### Task 4: Implement `fetch_hf_daily_papers()`

**Files:**
- Modify: `dtech.py` (add function after `fetch_arxiv_papers`)
- Modify: `tests/test_papers.py` (add HF parsing test)

- [ ] **Step 1: Write test for HF daily papers parsing**

Add to `tests/test_papers.py`:

```python
import json as json_mod
from dtech import fetch_hf_daily_papers

SAMPLE_HF_JSON = json_mod.dumps([
    {
        "title": "HF Trending Paper About Agents",
        "paper": {
            "id": "2403.67890",
            "title": "HF Trending Paper About Agents",
            "summary": "This paper explores new agent architectures for LLMs.",
            "publishedAt": "2026-03-27T12:00:00.000Z",
        },
        "numUpvotes": 42,
    },
    {
        "title": "Another Paper",
        "paper": {
            "id": "2403.11111",
            "title": "Another Paper",
            "summary": "A second paper about fine-tuning.",
            "publishedAt": "2026-03-26T08:00:00.000Z",
        },
        "numUpvotes": 10,
    },
])


def test_fetch_hf_daily_papers_parses_json(monkeypatch):
    """Test that HF Daily Papers JSON is correctly parsed."""

    class FakeResponse:
        status_code = 200
        text = SAMPLE_HF_JSON
        def raise_for_status(self):
            pass
        def json(self):
            return json_mod.loads(self.text)

    class FakeClient:
        def get(self, url, **kwargs):
            return FakeResponse()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr("dtech._build_paper_client", lambda: FakeClient())

    papers = fetch_hf_daily_papers()
    assert len(papers) == 2
    assert papers[0].arxiv_id == "2403.67890"
    assert papers[0].hf_trending is True
    assert "agent architectures" in papers[0].abstract
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_papers.py::test_fetch_hf_daily_papers_parses_json -v`
Expected: FAIL — `fetch_hf_daily_papers` not defined.

- [ ] **Step 3: Implement `fetch_hf_daily_papers`**

In `dtech.py`, add after `fetch_arxiv_papers`:

```python

HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"


def fetch_hf_daily_papers() -> list[PaperCandidate]:
    """Fetch today's community-curated papers from HuggingFace."""
    papers: list[PaperCandidate] = []
    with _build_paper_client() as client:
        try:
            resp = client.get(HF_DAILY_PAPERS_URL)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("Failed to fetch HuggingFace daily papers: %s", exc)
            return papers

    for item in resp.json():
        paper_data = item.get("paper", {})
        arxiv_id = paper_data.get("id", "")
        if not arxiv_id:
            continue

        title = " ".join(paper_data.get("title", "").split())
        abstract = " ".join(paper_data.get("summary", "").split())
        published = paper_data.get("publishedAt", "")

        papers.append(
            PaperCandidate(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                published=published,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                categories="",
                hf_trending=True,
            )
        )

    log.info("Fetched %d papers from HuggingFace Daily Papers", len(papers))
    return papers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_papers.py::test_fetch_hf_daily_papers_parses_json -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dtech.py tests/test_papers.py
git commit -m "feat: add fetch_hf_daily_papers with JSON parsing"
```

---

### Task 5: Implement `deduplicate_papers()`

**Files:**
- Modify: `dtech.py` (add function after `fetch_hf_daily_papers`)
- Modify: `tests/test_papers.py` (add dedup test)

- [ ] **Step 1: Write test for dedup logic**

Add to `tests/test_papers.py`:

```python
from dtech import deduplicate_papers, PaperCandidate


def test_deduplicate_papers_merges_by_arxiv_id():
    """Papers appearing in both sources should be deduped, keeping HF trending flag."""
    arxiv = [
        PaperCandidate(
            arxiv_id="2403.12345",
            title="Shared Paper",
            abstract="Abstract from arXiv",
            published="2026-03-27T00:00:00Z",
            pdf_url="http://arxiv.org/pdf/2403.12345v1",
            categories="cs.LG, cs.AI",
            hf_trending=False,
        ),
        PaperCandidate(
            arxiv_id="2403.99999",
            title="arXiv Only",
            abstract="Only on arXiv",
            published="2026-03-27T00:00:00Z",
            pdf_url="http://arxiv.org/pdf/2403.99999v1",
            categories="cs.CL",
            hf_trending=False,
        ),
    ]
    hf = [
        PaperCandidate(
            arxiv_id="2403.12345",
            title="Shared Paper",
            abstract="Abstract from HF",
            published="2026-03-27T12:00:00.000Z",
            pdf_url="https://arxiv.org/pdf/2403.12345",
            categories="",
            hf_trending=True,
        ),
        PaperCandidate(
            arxiv_id="2403.77777",
            title="HF Only",
            abstract="Only on HF",
            published="2026-03-27T08:00:00.000Z",
            pdf_url="https://arxiv.org/pdf/2403.77777",
            categories="",
            hf_trending=True,
        ),
    ]

    result = deduplicate_papers(arxiv, hf)
    assert len(result) == 3

    ids = {p.arxiv_id for p in result}
    assert ids == {"2403.12345", "2403.99999", "2403.77777"}

    # The shared paper should have hf_trending=True and keep arXiv's richer data
    shared = next(p for p in result if p.arxiv_id == "2403.12345")
    assert shared.hf_trending is True
    assert shared.categories == "cs.LG, cs.AI"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_papers.py::test_deduplicate_papers_merges_by_arxiv_id -v`
Expected: FAIL — `deduplicate_papers` not defined.

- [ ] **Step 3: Implement `deduplicate_papers`**

In `dtech.py`, add after `fetch_hf_daily_papers`:

```python

def deduplicate_papers(
    arxiv: list[PaperCandidate],
    hf: list[PaperCandidate],
) -> list[PaperCandidate]:
    """Merge papers from both sources, deduplicating by arXiv ID.

    When a paper appears in both, keep arXiv's richer metadata but set hf_trending=True.
    """
    by_id: dict[str, PaperCandidate] = {}

    for p in arxiv:
        by_id[p.arxiv_id] = p

    for p in hf:
        if p.arxiv_id in by_id:
            # Upgrade existing arXiv entry with HF trending flag
            existing = by_id[p.arxiv_id]
            by_id[p.arxiv_id] = PaperCandidate(
                arxiv_id=existing.arxiv_id,
                title=existing.title,
                abstract=existing.abstract,
                published=existing.published,
                pdf_url=existing.pdf_url,
                categories=existing.categories,
                hf_trending=True,
            )
        else:
            by_id[p.arxiv_id] = p

    return list(by_id.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_papers.py::test_deduplicate_papers_merges_by_arxiv_id -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dtech.py tests/test_papers.py
git commit -m "feat: add deduplicate_papers for arXiv+HF merge"
```

---

### Task 6: Implement `rank_papers()` (LLM Pass 1)

**Files:**
- Modify: `dtech.py` (add function after `deduplicate_papers`)
- Modify: `tests/test_papers.py` (add JSON fallback test)

- [ ] **Step 1: Write test for JSON parsing fallback**

Add to `tests/test_papers.py`:

```python
from dtech import _parse_ranked_ids


def test_parse_ranked_ids_valid_json():
    raw = '[{"arxiv_id": "2403.111", "reason": "good"}, {"arxiv_id": "2403.222", "reason": "great"}]'
    assert _parse_ranked_ids(raw) == ["2403.111", "2403.222"]


def test_parse_ranked_ids_invalid_json():
    raw = "This is not valid JSON at all"
    assert _parse_ranked_ids(raw) == []


def test_parse_ranked_ids_strips_markdown_fences():
    raw = '```json\n[{"arxiv_id": "2403.333", "reason": "nice"}]\n```'
    assert _parse_ranked_ids(raw) == ["2403.333"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_papers.py::test_parse_ranked_ids_valid_json -v`
Expected: FAIL — `_parse_ranked_ids` not defined.

- [ ] **Step 3: Implement `_parse_ranked_ids` and `rank_papers`**

In `dtech.py`, add after `deduplicate_papers`:

```python

INTEREST_PROFILE = (
    "Python developer working with LLMs, agents (Google ADK, LangChain), "
    "ML pipelines, and applied AI. Interested in practical methodologies, "
    "training techniques, agent architectures, RAG, and optimization."
)


def _parse_ranked_ids(raw: str) -> list[str]:
    """Parse LLM ranking output into a list of arXiv IDs. Handles markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        items = json.loads(text)
        return [item["arxiv_id"] for item in items if "arxiv_id" in item]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def rank_papers(candidates: list[PaperCandidate], top_n: int = 5) -> list[PaperCandidate]:
    """LLM Pass 1: select the top-N most relevant papers for our interest profile."""
    if not candidates:
        return []

    paper_list = "\n\n".join(
        f"[{i+1}] ID: {p.arxiv_id} | Title: {p.title}"
        f"{' | HF-TRENDING' if p.hf_trending else ''}"
        f"\nAbstract: {p.abstract[:500]}"
        for i, p in enumerate(candidates)
    )

    prompt = f"""You are selecting the most relevant AI research papers for a developer.

DEVELOPER PROFILE: {INTEREST_PROFILE}

Below are {len(candidates)} recent papers. Select the top {top_n} most relevant to this
developer's work. Prefer papers with novel, practical methodologies. Papers marked
HF-TRENDING have community validation — give them a small relevance boost.

Return ONLY a JSON array (no markdown, no explanation):
[{{"arxiv_id": "...", "reason": "one-line justification"}}, ...]

PAPERS:
{paper_list}"""

    response = model.invoke(prompt)
    ranked_ids = _parse_ranked_ids(response.content)

    if not ranked_ids:
        log.warning("LLM ranking returned no valid IDs — falling back to first %d by recency", top_n)
        return candidates[:top_n]

    id_set = set(ranked_ids)
    by_id = {p.arxiv_id: p for p in candidates}
    selected = [by_id[aid] for aid in ranked_ids if aid in by_id]

    if not selected:
        log.warning("LLM ranked IDs not found in candidates — falling back to first %d", top_n)
        return candidates[:top_n]

    return selected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_papers.py -v -k "parse_ranked"`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add dtech.py tests/test_papers.py
git commit -m "feat: add rank_papers with LLM pass-1 selection"
```

---

### Task 7: Implement `summarize_paper()` (LLM Pass 2)

**Files:**
- Modify: `dtech.py` (add function after `rank_papers`)

- [ ] **Step 1: Implement `summarize_paper`**

In `dtech.py`, add after `rank_papers`:

```python

def summarize_paper(paper: PaperCandidate) -> str:
    """LLM Pass 2: generate a structured HTML summary for a single paper."""
    prompt = f"""You are writing a research paper briefing for a Python/AI developer.

PAPER TITLE: {paper.title}
ABSTRACT: {paper.abstract}

Generate a structured HTML fragment (no markdown, no backticks, no <html>/<body> tags).
Do NOT include authors. Structure:

<h3>{paper.title}</h3>
<p class="paper-tldr"><em>One-sentence TL;DR of what this paper does.</em></p>
<h4>Key Methodology</h4>
<ul>
  <li>2-3 bullets explaining their approach, technique, or architecture</li>
  <li>Focus on what's novel and how it works at a high level</li>
</ul>
<h4>Why It Matters For You</h4>
<ul>
  <li>1-2 bullets on practical takeaways for someone building with LLMs, agents, or ML pipelines in Python</li>
</ul>
<p><a href="https://arxiv.org/abs/{paper.arxiv_id}">arXiv</a> &middot; <a href="{paper.pdf_url}">PDF</a></p>

Keep it concise and useful. Focus on methodology, not hype."""

    response = model.invoke(prompt)
    return response.content
```

- [ ] **Step 2: Commit**

```bash
git add dtech.py
git commit -m "feat: add summarize_paper for LLM pass-2 structured summaries"
```

---

### Task 8: Implement `fetch_and_process_papers()` orchestrator

**Files:**
- Modify: `dtech.py` (add function after `summarize_paper`)

- [ ] **Step 1: Implement `_papers_fetched_today` and `fetch_and_process_papers`**

In `dtech.py`, add after `summarize_paper`:

```python

def _papers_fetched_today() -> bool:
    """Return True if any paper entries exist for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM knowledge WHERE date = ? AND source LIKE 'arxiv:%'",
            (today,),
        ).fetchone()
    return row[0] > 0


def fetch_and_process_papers() -> None:
    """Orchestrator: fetch → dedup → rank → summarize → store."""
    _init_db()

    if _papers_fetched_today():
        log.info("Papers already fetched today — skipping.")
        return

    log.info("Fetching AI research papers...")
    arxiv = fetch_arxiv_papers(days=3)
    hf = fetch_hf_daily_papers()

    if not arxiv and not hf:
        log.warning("No papers fetched from any source — skipping.")
        return

    candidates = deduplicate_papers(arxiv, hf)
    log.info("Deduplicated to %d unique candidates", len(candidates))

    ranked = rank_papers(candidates, top_n=5)
    log.info("LLM selected %d papers", len(ranked))

    for paper in tqdm(ranked, desc="Summarizing papers"):
        summary = summarize_paper(paper)
        store(
            KnowledgeEntry(
                timestamp=datetime.now().isoformat(),
                source=f"arxiv:{paper.arxiv_id}",
                summary=summary,
            )
        )

    log.info("Stored %d paper summaries", len(ranked))
```

- [ ] **Step 2: Commit**

```bash
git add dtech.py
git commit -m "feat: add fetch_and_process_papers orchestrator"
```

---

### Task 9: Update `generate_html_report()` with dedicated papers section

**Files:**
- Modify: `dtech.py:296-949` (`generate_html_report` function)

- [ ] **Step 1: Modify the DB query to separate papers from GitHub content**

In `generate_html_report()`, replace the single DB query block (lines 299-304):

```python
def generate_html_report() -> None:
    """Read knowledge DB and render a styled HTML dashboard."""
    _init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        paper_rows = conn.execute(
            "SELECT timestamp, source, summary FROM knowledge "
            "WHERE source LIKE 'arxiv:%' ORDER BY timestamp DESC LIMIT 10"
        ).fetchall()
        github_rows = conn.execute(
            "SELECT timestamp, source, summary FROM knowledge "
            "WHERE source NOT LIKE 'arxiv:%' ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()
    paper_items = [dict(r) for r in paper_rows]
    items = [dict(r) for r in github_rows]
```

- [ ] **Step 2: Update the stats counting to include papers**

Replace the stats counting block (original lines 309-313) with:

```python
    today_str = datetime.now().strftime("%A, %B %d, %Y")

    # Count entries per category for stats bar
    all_items = paper_items + items
    cat_counts: dict[str, int] = {}
    for item in all_items:
        cat = _source_category(item["source"])
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
```

- [ ] **Step 3: Add paper cards builder**

After the filter buttons block and before the GitHub cards block, add a paper section builder. Insert after the `filter_buttons` construction and before the `# Build cards` comment:

```python
    # Build paper section
    paper_section = ""
    if paper_items:
        paper_section += """
        <section class="paper-section">
            <div class="section-divider">
                <span class="section-divider-icon" style="--accent:#0891b2">
                    {icon} Research Papers
                </span>
            </div>""".format(icon=_CATEGORY_META["paper"][2])

        for idx, item in enumerate(paper_items):
            ts = datetime.fromisoformat(item["timestamp"])
            human_time = ts.strftime("%b %d, %Y &middot; %H:%M")
            source_label = nice_source_label(item["source"])
            color = "#0891b2"
            label = "Research Papers"
            icon = _CATEGORY_META["paper"][2]

            delay = idx * 60
            paper_section += f"""
            <article class="card" data-cat="paper"
                     style="--accent:{color};animation-delay:{delay}ms">
                <div class="card-accent"></div>
                <div class="card-inner">
                    <div class="card-header">
                        <div class="card-meta">
                            <span class="badge" style="--accent:{color}">{icon} {label}</span>
                            <span class="source-label">{source_label}</span>
                        </div>
                        <time class="time">{human_time}</time>
                    </div>
                    <div class="summary">{item["summary"]}</div>
                </div>
            </article>"""

        paper_section += "\n        </section>"
```

- [ ] **Step 4: Update the total count and insert paper section into HTML**

Update the `total` calculation:

```python
    total = len(paper_items) + len(items)
```

In the HTML template, replace the `{cards}` insertion (around line 890) with:

```
{paper_section}

{cards}
```

- [ ] **Step 5: Add CSS for paper section**

In the `<style>` block, add after the `.card.hidden` rule (after line 594 in original):

```css
        /* ── Paper section ── */
        .paper-section {{
            margin-bottom: 40px;
        }}
        .section-divider {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
            padding-bottom: 12px;
            border-bottom: 2px solid rgba(8, 145, 178, 0.15);
        }}
        .section-divider-icon {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 18px;
            font-weight: 700;
            color: #0891b2;
            letter-spacing: -0.01em;
        }}
        .section-divider-icon svg {{
            opacity: 0.85;
        }}
        .summary .paper-tldr {{
            font-style: italic;
            color: var(--text-secondary);
            font-size: 15px;
            margin: 0 0 16px;
        }}
        .summary h4 {{
            font-size: 12.5px;
            font-weight: 600;
            color: color-mix(
                in srgb, var(--accent) 80%, #1a1a2e
            );
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 18px 0 8px;
        }}

        /* ── GitHub section divider ── */
        .github-section-divider {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
            padding-bottom: 12px;
            border-bottom: 2px solid rgba(0, 0, 0, 0.06);
        }}
        .github-section-divider span {{
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }}
```

- [ ] **Step 6: Add a GitHub section divider before GitHub cards**

In the HTML template, between `{paper_section}` and `{cards}`, add:

```html
        <div class="github-section-divider">
            <span>GitHub Activity</span>
        </div>
```

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest tests/test_papers.py -v`
Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add dtech.py
git commit -m "feat: render papers in dedicated HTML section above GitHub cards"
```

---

### Task 10: Update `__main__` to call papers pipeline

**Files:**
- Modify: `dtech.py:994-998` (`__main__` block)

- [ ] **Step 1: Update `__main__`**

Replace the `__main__` block (lines 994-998) with:

```python
if __name__ == "__main__":
    fetch_and_process_papers()
    fetch_and_process(days=7)
    generate_html_report()
    print("🧠 Daily knowledge added.")
    os.startfile(REPORT_PATH)
```

- [ ] **Step 2: Lint and format**

Run: `uv run ruff check dtech.py --fix && uv run ruff format dtech.py`
Expected: No errors (or auto-fixed).

Run: `uv run ruff check tests/ --fix && uv run ruff format tests/`
Expected: No errors.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add dtech.py tests/
git commit -m "feat: wire papers pipeline into main and lint"
```

---

### Task 11: End-to-end manual test

- [ ] **Step 1: Run the full app**

Run: `uv run python dtech.py`

Expected behavior:
1. "Fetching AI research papers..." appears in logs
2. arXiv and HF papers are fetched (or gracefully skipped on failure)
3. LLM ranking selects top 5
4. Each paper gets a structured summary
5. GitHub sources are fetched as before
6. HTML report opens with a "Research Papers" section at the top
7. Filter bar includes a "Research" button
8. Paper cards show structured summaries with TL;DR, Key Methodology, Why It Matters

- [ ] **Step 2: Verify dedup works on re-run**

Run: `uv run python dtech.py` again.
Expected: "Papers already fetched today — skipping." in logs.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete AI research papers integration"
```
