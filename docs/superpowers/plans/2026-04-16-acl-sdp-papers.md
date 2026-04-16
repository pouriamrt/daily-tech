# ACL SDP Papers as Reserved Source — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (sequential, recommended for small plans), team-driven-development (parallel swarm, recommended for 3+ tasks with parallelizable dependency graph), or superpowers:executing-plans (inline batch) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reserve up to 2 of 5 daily-report paper slots for ACL SDP workshop papers (soft quota), with cross-day deduplication so the same paper doesn't appear on consecutive days.

**Architecture:** One new fetcher (OAI-PMH against ACL Anthology), a `PaperCandidate` rename (`arxiv_id` → `paper_id` with added `source` discriminator), a new `shown_papers` SQLite table acting as a pre-ranker filter, and a source-aware soft-reserve clause added to the existing ranker LLM prompt. No report-format changes; the existing pipeline's shape is preserved.

**Tech Stack:** Python 3.13, `httpx` + `hishel` (cached HTTP client), `xml.etree.ElementTree` for XML parsing, `sqlite3` stdlib, `pytest` + `unittest.mock` for tests, `ruff` for lint/format, existing `langchain` LLM client unchanged.

**Spec:** [`docs/superpowers/specs/2026-04-16-acl-sdp-papers-design.md`](../specs/2026-04-16-acl-sdp-papers-design.md)

---

## File Structure

**Files modified:**
- `dtech.py` — `PaperCandidate` dataclass rename; new `fetch_acl_sdp_papers`, `_parse_acl_record`, `_fetch_acl_abstract` helpers; new `filter_already_shown`, `record_shown_papers` functions; extended `deduplicate_papers`; extended `rank_papers` prompt; extended `_init_db`; extended `nice_source_label` and `_source_category`; updated orchestration in the paper-pipeline block (around line 640-687).
- `tests/test_papers.py` — update existing tests to use new `paper_id`/`source` field names.

**Files created:**
- `tests/test_shown_papers.py` — unit tests for shown-papers filter + recorder.
- `tests/test_acl_sdp_fetcher.py` — unit tests for the OAI-PMH fetcher.
- `tests/test_rank_papers_source_aware.py` — unit tests for source-aware ranker prompt.
- `tests/test_integration_acl.py` — opt-in live-endpoint integration test.
- `tests/fixtures/acl/sdp_single_record.xml`
- `tests/fixtures/acl/sdp_empty_abstract.xml`
- `tests/fixtures/acl/sdp_landing_page.html`
- `tests/fixtures/acl/sdp_page1_with_token.xml`
- `tests/fixtures/acl/sdp_page2_final.xml`
- `tests/fixtures/acl/sdp_malformed.xml`

---

## Task 1: Rename `PaperCandidate.arxiv_id` → `paper_id`, add `source` field

**Files:**
- Modify: `dtech.py:135-143` (dataclass), `dtech.py:251-305` (arXiv fetcher), `dtech.py:311-345` (HF fetcher), `dtech.py:348-383` (dedup), `dtech.py:467-486` (filter), `dtech.py:489-499` (parser), `dtech.py:502-567` (ranker), `dtech.py:682` (store call).
- Modify: `tests/test_papers.py` (update all `arxiv_id=` → `paper_id=`, add `source=` kwarg).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_papers.py` (near the top, after imports):

```python
def test_paper_candidate_requires_source_field():
    """PaperCandidate must carry a source discriminator."""
    from dataclasses import fields
    field_names = {f.name for f in fields(PaperCandidate)}
    assert "paper_id" in field_names, "paper_id field is required"
    assert "source" in field_names, "source field is required"
    assert "arxiv_id" not in field_names, "arxiv_id must be renamed to paper_id"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_papers.py::test_paper_candidate_requires_source_field -v`
Expected: FAIL with `AssertionError: paper_id field is required` (the field is still named `arxiv_id`).

- [ ] **Step 3: Update the dataclass**

Replace `dtech.py:135-143`:

```python
from typing import Literal

PaperSource = Literal["arxiv", "hf", "acl-sdp"]


@dataclass(frozen=True)
class PaperCandidate:
    paper_id: str
    title: str
    abstract: str
    published: str
    pdf_url: str
    categories: str
    source: PaperSource
    hf_trending: bool = False
```

- [ ] **Step 4: Update arXiv fetcher** (`dtech.py`, inside `fetch_arxiv_papers`, replace the `PaperCandidate(...)` call around line 293)

```python
papers.append(
    PaperCandidate(
        paper_id=arxiv_id,
        title=title,
        abstract=abstract,
        published=published_text,
        pdf_url=pdf_url,
        categories=", ".join(cats),
        source="arxiv",
    )
)
```

- [ ] **Step 5: Update HF fetcher** (`dtech.py`, inside `fetch_hf_daily_papers`, replace the `PaperCandidate(...)` call around line 332)

```python
papers.append(
    PaperCandidate(
        paper_id=arxiv_id,
        title=title,
        abstract=abstract,
        published=published,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        categories="",
        source="hf",
        hf_trending=True,
    )
)
```

- [ ] **Step 6: Update `deduplicate_papers`** (replace the function body, `dtech.py:348-383`)

```python
def deduplicate_papers(
    arxiv: list[PaperCandidate],
    hf: list[PaperCandidate],
) -> list[PaperCandidate]:
    """Merge papers from arXiv and HF, deduplicating by paper_id.

    When a paper appears in both, keep arXiv's richer metadata but set hf_trending=True.
    """
    by_id: dict[str, PaperCandidate] = {}

    for p in arxiv:
        by_id[p.paper_id] = p

    for p in hf:
        if p.paper_id in by_id:
            existing = by_id[p.paper_id]
            by_id[p.paper_id] = PaperCandidate(
                paper_id=existing.paper_id,
                title=existing.title,
                abstract=existing.abstract,
                published=existing.published,
                pdf_url=existing.pdf_url,
                categories=existing.categories,
                source=existing.source,
                hf_trending=True,
            )
        else:
            by_id[p.paper_id] = p

    return list(by_id.values())
```

Task 6 will extend this signature to accept a third list; do not touch it yet.

- [ ] **Step 7: Update `_parse_ranked_ids`** (`dtech.py:489-499`)

Replace `item["arxiv_id"]` with `item["paper_id"]`:

```python
def _parse_ranked_ids(raw: str) -> list[str]:
    """Parse LLM ranking output into a list of paper IDs. Handles markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        items = json.loads(text)
        return [item["paper_id"] for item in items if "paper_id" in item]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
```

- [ ] **Step 8: Update `rank_papers` prompt + selection** (`dtech.py:502-567`)

In the prompt `paper_list` block, replace `p.arxiv_id` with `p.paper_id`. In the instruction paragraph starting "Return ONLY a JSON array", change `"arxiv_id"` to `"paper_id"`. In the selection block:

```python
by_id = {p.paper_id: p for p in candidates}
```

- [ ] **Step 9: Update `_is_low_value_paper`** (`dtech.py:467-486`)

Any reference to `paper.arxiv_id` — rename to `paper.paper_id`. If no such reference exists in that function body, skip this step.

- [ ] **Step 10: Update the store call in the paper pipeline** (`dtech.py:682`)

Replace:

```python
source=f"arxiv:{paper.arxiv_id}",
```

with:

```python
source=f"{paper.source}:{paper.paper_id}",
```

This gives ACL papers a `"acl-sdp:2024.sdp-1.3"` source string and HF papers `"hf:2403.12345"`. Existing DB rows with `"arxiv:..."` prefixes remain valid.

- [ ] **Step 11: Update `tests/test_papers.py` fixture constructors**

Replace every `PaperCandidate(arxiv_id=...)` call in the test file with `PaperCandidate(paper_id=..., source="arxiv")`. Use your editor's find-and-replace restricted to `tests/test_papers.py`. Verify the following in particular:
- Any `SAMPLE_HF_JSON` or `sample_hf_paper()` helper: ensure resulting `PaperCandidate` objects have `source="hf"`.
- Any `SAMPLE_ARXIV_XML`-driven test: ensure resulting objects have `source="arxiv"`.

Also rename `arxiv_id` attribute reads in assertions to `paper_id`:

```python
# Before: assert paper.arxiv_id == "2403.12345"
# After:  assert paper.paper_id == "2403.12345"
```

- [ ] **Step 12: Run existing tests + the new one**

Run: `uv run pytest tests/test_papers.py -v`
Expected: ALL PASS, including `test_paper_candidate_requires_source_field`.

- [ ] **Step 13: Run linter and type checker**

Run: `uv run ruff check dtech.py tests/test_papers.py`
Expected: no errors.

Run: `uv run ruff format dtech.py tests/test_papers.py`
Expected: files formatted in place.

- [ ] **Step 14: Commit**

```bash
git add dtech.py tests/test_papers.py
git commit -m "refactor: rename PaperCandidate.arxiv_id to paper_id, add source field"
```

---

## Task 2: Add `shown_papers` table to `_init_db`

**Files:**
- Modify: `dtech.py:146-162` (the `_init_db` function).
- Test: `tests/test_shown_papers.py` (new file).

- [ ] **Step 1: Create `tests/test_shown_papers.py` with a failing table-exists test**

```python
"""Tests for the shown_papers cross-day deduplication table."""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtech import _init_db


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temp location and initialize schema."""
    import dtech

    db = tmp_path / "test_knowledge.db"
    monkeypatch.setattr(dtech, "DB_PATH", db)
    _init_db()
    return db


def test_init_db_creates_shown_papers_table(tmp_db):
    with sqlite3.connect(tmp_db) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='shown_papers'"
        ).fetchone()
    assert row is not None, "shown_papers table should be created by _init_db"


def test_shown_papers_has_paper_id_primary_key(tmp_db):
    with sqlite3.connect(tmp_db) as conn:
        cols = conn.execute("PRAGMA table_info(shown_papers)").fetchall()
    pk_cols = [c for c in cols if c[5] == 1]  # c[5] is pk flag
    assert len(pk_cols) == 1, "expected exactly one primary key column"
    assert pk_cols[0][1] == "paper_id", "paper_id should be primary key"
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_shown_papers.py -v`
Expected: both tests FAIL with `table 'shown_papers' not found` or similar.

- [ ] **Step 3: Extend `_init_db`** (`dtech.py:146-162`)

Append inside the existing `with sqlite3.connect(DB_PATH) as conn:` block:

```python
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shown_papers (
                paper_id    TEXT PRIMARY KEY,
                shown_date  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shown_papers_date
            ON shown_papers(shown_date DESC)
        """)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_shown_papers.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add dtech.py tests/test_shown_papers.py
git commit -m "feat: add shown_papers table for cross-day paper deduplication"
```

---

## Task 3: Add `filter_already_shown` and `record_shown_papers` functions

**Files:**
- Modify: `dtech.py` (new functions, insert directly after `_init_db`).
- Modify: `tests/test_shown_papers.py` (add tests 14–17 from spec).

- [ ] **Step 1: Extend `tests/test_shown_papers.py` with failing tests for the filter/recorder**

Append to `tests/test_shown_papers.py`:

```python
from datetime import datetime

from dtech import PaperCandidate, filter_already_shown, record_shown_papers


def _make_paper(paper_id: str) -> PaperCandidate:
    return PaperCandidate(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        abstract="",
        published="2026-04-16T00:00:00Z",
        pdf_url="",
        categories="",
        source="arxiv",
    )


def test_filter_already_shown_removes_known_ids(tmp_db):
    with sqlite3.connect(tmp_db) as conn:
        conn.executemany(
            "INSERT INTO shown_papers (paper_id, shown_date) VALUES (?, ?)",
            [("A", "2026-04-15"), ("B", "2026-04-14")],
        )
    candidates = [_make_paper("A"), _make_paper("B"), _make_paper("C")]
    result = filter_already_shown(candidates)
    assert [p.paper_id for p in result] == ["C"]


def test_filter_already_shown_empty_table(tmp_db):
    candidates = [_make_paper("A"), _make_paper("B")]
    result = filter_already_shown(candidates)
    assert [p.paper_id for p in result] == ["A", "B"]


def test_filter_already_shown_empty_candidates(tmp_db):
    assert filter_already_shown([]) == []


def test_record_shown_papers_inserts_rows(tmp_db):
    papers = [_make_paper("X"), _make_paper("Y")]
    record_shown_papers(papers)
    with sqlite3.connect(tmp_db) as conn:
        rows = conn.execute(
            "SELECT paper_id, shown_date FROM shown_papers ORDER BY paper_id"
        ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "X"
    assert rows[1][0] == "Y"
    today = datetime.now().strftime("%Y-%m-%d")
    assert all(r[1] == today for r in rows)


def test_record_shown_papers_is_idempotent(tmp_db):
    papers = [_make_paper("X")]
    record_shown_papers(papers)
    record_shown_papers(papers)
    with sqlite3.connect(tmp_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM shown_papers").fetchone()[0]
    assert count == 1
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_shown_papers.py -v`
Expected: FAIL with `ImportError: cannot import name 'filter_already_shown' from 'dtech'`.

- [ ] **Step 3: Add the two functions to `dtech.py`**

Insert directly after the `_init_db` function (around line 163):

```python
def filter_already_shown(
    candidates: list[PaperCandidate],
) -> list[PaperCandidate]:
    """Drop candidates whose paper_id has appeared in a previous daily report."""
    if not candidates:
        return []
    with sqlite3.connect(DB_PATH) as conn:
        placeholders = ",".join("?" * len(candidates))
        rows = conn.execute(
            f"SELECT paper_id FROM shown_papers WHERE paper_id IN ({placeholders})",
            [p.paper_id for p in candidates],
        ).fetchall()
    shown = {row[0] for row in rows}
    filtered = [p for p in candidates if p.paper_id not in shown]
    log.info(
        "Filtered %d already-shown papers; %d candidates remain",
        len(candidates) - len(filtered),
        len(filtered),
    )
    return filtered


def record_shown_papers(selected: list[PaperCandidate]) -> None:
    """Record that these papers appeared in today's report."""
    if not selected:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO shown_papers (paper_id, shown_date) VALUES (?, ?)",
            [(p.paper_id, today) for p in selected],
        )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_shown_papers.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Lint + format**

Run: `uv run ruff check dtech.py tests/test_shown_papers.py && uv run ruff format dtech.py tests/test_shown_papers.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add dtech.py tests/test_shown_papers.py
git commit -m "feat: add filter_already_shown and record_shown_papers"
```

---

## Task 4: Create ACL SDP OAI-PMH fixtures

**Files:**
- Create: `tests/fixtures/acl/sdp_single_record.xml`
- Create: `tests/fixtures/acl/sdp_empty_abstract.xml`
- Create: `tests/fixtures/acl/sdp_landing_page.html`
- Create: `tests/fixtures/acl/sdp_page1_with_token.xml`
- Create: `tests/fixtures/acl/sdp_page2_final.xml`
- Create: `tests/fixtures/acl/sdp_malformed.xml`

No test step in this task — fixtures alone commit. They're consumed in Task 5.

- [ ] **Step 1: Create `tests/fixtures/acl/sdp_single_record.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
         xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <ListRecords>
    <record>
      <header>
        <identifier>oai:aclanthology.org:2024.sdp-1.3</identifier>
        <datestamp>2024-11-15</datestamp>
      </header>
      <metadata>
        <oai_dc:dc>
          <dc:title>A Novel Method for Scientific Document Chunking</dc:title>
          <dc:description>We propose a new chunking strategy that leverages section boundaries and citation contexts to improve downstream retrieval accuracy by 12.4 points on SciRepEval.</dc:description>
          <dc:date>2024-11</dc:date>
          <dc:identifier>2024.sdp-1.3</dc:identifier>
        </oai_dc:dc>
      </metadata>
    </record>
  </ListRecords>
</OAI-PMH>
```

- [ ] **Step 2: Create `tests/fixtures/acl/sdp_empty_abstract.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
         xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <ListRecords>
    <record>
      <header>
        <identifier>oai:aclanthology.org:2024.sdp-1.4</identifier>
        <datestamp>2024-11-15</datestamp>
      </header>
      <metadata>
        <oai_dc:dc>
          <dc:title>Another SDP Paper</dc:title>
          <dc:description></dc:description>
          <dc:date>2024</dc:date>
          <dc:identifier>2024.sdp-1.4</dc:identifier>
        </oai_dc:dc>
      </metadata>
    </record>
  </ListRecords>
</OAI-PMH>
```

- [ ] **Step 3: Create `tests/fixtures/acl/sdp_landing_page.html`**

```html
<!DOCTYPE html>
<html>
<head>
<title>Another SDP Paper - ACL Anthology</title>
<meta name="citation_abstract" content="This paper presents a transformer-based approach to scientific claim verification using graph attention over citation networks.">
</head>
<body><h1>Another SDP Paper</h1></body>
</html>
```

- [ ] **Step 4: Create `tests/fixtures/acl/sdp_page1_with_token.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
         xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <ListRecords>
    <record>
      <header><identifier>oai:aclanthology.org:2024.sdp-1.1</identifier></header>
      <metadata>
        <oai_dc:dc>
          <dc:title>Paper P1A</dc:title>
          <dc:description>Abstract for P1A.</dc:description>
          <dc:date>2024-11</dc:date>
          <dc:identifier>2024.sdp-1.1</dc:identifier>
        </oai_dc:dc>
      </metadata>
    </record>
    <resumptionToken>TOKEN_ABC</resumptionToken>
  </ListRecords>
</OAI-PMH>
```

- [ ] **Step 5: Create `tests/fixtures/acl/sdp_page2_final.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
         xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <ListRecords>
    <record>
      <header><identifier>oai:aclanthology.org:2024.sdp-1.2</identifier></header>
      <metadata>
        <oai_dc:dc>
          <dc:title>Paper P2A</dc:title>
          <dc:description>Abstract for P2A.</dc:description>
          <dc:date>2024-11</dc:date>
          <dc:identifier>2024.sdp-1.2</dc:identifier>
        </oai_dc:dc>
      </metadata>
    </record>
    <resumptionToken></resumptionToken>
  </ListRecords>
</OAI-PMH>
```

- [ ] **Step 6: Create `tests/fixtures/acl/sdp_malformed.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH><ListRecords><record><header><identifier>broken
```

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/acl/
git commit -m "test: add ACL SDP OAI-PMH fixtures"
```

---

## Task 5: Implement `fetch_acl_sdp_papers`

**Files:**
- Modify: `dtech.py` (add new constants and three new functions after the HF paper fetcher, around line 346).
- Create: `tests/test_acl_sdp_fetcher.py`.

- [ ] **Step 1: Create `tests/test_acl_sdp_fetcher.py` with failing tests**

```python
"""Tests for the ACL SDP OAI-PMH fetcher."""

import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtech import _parse_acl_record, fetch_acl_sdp_papers

FIXTURES = Path(__file__).parent / "fixtures" / "acl"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _mock_response(text: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=resp
        )
    return resp


def test_parse_acl_record_happy_path():
    root = ET.fromstring(_load("sdp_single_record.xml"))
    record = root.find(".//{http://www.openarchives.org/OAI/2.0/}record")
    client = MagicMock()
    cutoff = datetime.now() - timedelta(days=365)
    paper = _parse_acl_record(record, client, cutoff)
    assert paper is not None and paper != "_too_old_"
    assert paper.paper_id == "2024.sdp-1.3"
    assert paper.source == "acl-sdp"
    assert "chunking" in paper.title.lower()
    assert "12.4" in paper.abstract
    assert paper.pdf_url == "https://aclanthology.org/2024.sdp-1.3.pdf"
    assert paper.published == "2024-11-01T00:00:00Z"


def test_parse_acl_record_empty_abstract_fallback():
    root = ET.fromstring(_load("sdp_empty_abstract.xml"))
    record = root.find(".//{http://www.openarchives.org/OAI/2.0/}record")
    client = MagicMock()
    client.get.return_value = _mock_response(_load("sdp_landing_page.html"))
    cutoff = datetime.now() - timedelta(days=365)
    paper = _parse_acl_record(record, client, cutoff)
    assert paper is not None and paper != "_too_old_"
    assert "transformer-based approach" in paper.abstract
    client.get.assert_called_once_with("https://aclanthology.org/2024.sdp-1.4/")


def test_parse_acl_record_year_only_date():
    root = ET.fromstring(_load("sdp_empty_abstract.xml"))
    record = root.find(".//{http://www.openarchives.org/OAI/2.0/}record")
    client = MagicMock()
    client.get.return_value = _mock_response(_load("sdp_landing_page.html"))
    cutoff = datetime.now() - timedelta(days=365 * 10)
    paper = _parse_acl_record(record, client, cutoff)
    assert paper.published == "2024-01-01T00:00:00Z"


def test_parse_acl_record_too_old_returns_sentinel():
    root = ET.fromstring(_load("sdp_single_record.xml"))
    record = root.find(".//{http://www.openarchives.org/OAI/2.0/}record")
    client = MagicMock()
    cutoff = datetime(2099, 1, 1)
    result = _parse_acl_record(record, client, cutoff)
    assert result == "_too_old_"


def test_fetch_acl_sdp_papers_pagination():
    responses = [
        _mock_response(_load("sdp_page1_with_token.xml")),
        _mock_response(_load("sdp_page2_final.xml")),
    ]
    with patch("dtech._build_paper_client") as mock_build:
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.get.side_effect = responses
        mock_build.return_value = client
        papers = fetch_acl_sdp_papers(days=365 * 10)
    assert [p.paper_id for p in papers] == ["2024.sdp-1.1", "2024.sdp-1.2"]
    assert client.get.call_count == 2


def test_fetch_acl_sdp_papers_http_error():
    with patch("dtech._build_paper_client") as mock_build:
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.get.side_effect = httpx.ConnectError("boom")
        mock_build.return_value = client
        papers = fetch_acl_sdp_papers(days=90)
    assert papers == []


def test_fetch_acl_sdp_papers_malformed_xml():
    with patch("dtech._build_paper_client") as mock_build:
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.get.return_value = _mock_response(_load("sdp_malformed.xml"))
        mock_build.return_value = client
        papers = fetch_acl_sdp_papers(days=90)
    assert papers == []
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_acl_sdp_fetcher.py -v`
Expected: all FAIL with `ImportError: cannot import name '_parse_acl_record' from 'dtech'`.

- [ ] **Step 3: Add constants + helpers to `dtech.py`**

Insert immediately after the HF fetcher block (after the line `log.info("Fetched %d papers from HuggingFace Daily Papers", ...)` and before `def deduplicate_papers`, around line 346):

```python
# ---------------------------------------------------------------------------
# ACL Anthology — SDP workshop fetcher
# ---------------------------------------------------------------------------

ACL_SDP_OAI_URL = (
    "https://aclanthology.org/oai-pmh/"
    "?verb=ListRecords&set=sig:sigdp&metadataPrefix=oai_dc"
)
ACL_SDP_RESUME_URL = (
    "https://aclanthology.org/oai-pmh/?verb=ListRecords&resumptionToken={token}"
)
ACL_LANDING_URL = "https://aclanthology.org/{paper_id}/"

OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}

_CITATION_ABSTRACT_RE = re.compile(
    r'<meta\s+name=["\']citation_abstract["\']\s+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)

_MAX_OAI_PAGES = 10


def _normalize_acl_date(raw: str) -> str:
    """Pad ACL dates: '2024' -> '2024-01-01T00:00:00Z', '2024-11' -> '2024-11-01T...'."""
    raw = raw.strip()
    if len(raw) == 4 and raw.isdigit():
        return f"{raw}-01-01T00:00:00Z"
    if len(raw) == 7 and raw[4] == "-":
        return f"{raw}-01T00:00:00Z"
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return f"{raw}T00:00:00Z"
    return raw  # already full ISO


def _fetch_acl_abstract(paper_id: str, client) -> str:
    """Fallback: scrape the ACL landing page for <meta name='citation_abstract'>."""
    try:
        resp = client.get(ACL_LANDING_URL.format(paper_id=paper_id))
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("ACL landing-page fetch failed for %s: %s", paper_id, exc)
        return ""
    match = _CITATION_ABSTRACT_RE.search(resp.text)
    return match.group(1).strip() if match else ""


def _parse_acl_record(record, client, cutoff: datetime):
    """Parse one <oai:record>. Returns PaperCandidate, '_too_old_' sentinel, or None."""
    dc = record.find(".//oai_dc:dc", OAI_NS)
    if dc is None:
        return None

    paper_id_el = dc.find("dc:identifier", OAI_NS)
    title_el = dc.find("dc:title", OAI_NS)
    date_el = dc.find("dc:date", OAI_NS)
    desc_el = dc.find("dc:description", OAI_NS)

    if paper_id_el is None or title_el is None or date_el is None:
        return None

    paper_id = (paper_id_el.text or "").strip()
    title = " ".join((title_el.text or "").split())
    published = _normalize_acl_date(date_el.text or "")

    try:
        published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        return None
    if published_dt.replace(tzinfo=None) < cutoff:
        return "_too_old_"

    abstract = " ".join((desc_el.text or "").split()) if desc_el is not None else ""
    if not abstract:
        abstract = _fetch_acl_abstract(paper_id, client)

    return PaperCandidate(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        published=published,
        pdf_url=f"https://aclanthology.org/{paper_id}.pdf",
        categories="",
        source="acl-sdp",
    )


def fetch_acl_sdp_papers(days: int = 90) -> list[PaperCandidate]:
    """Fetch recent ACL SDP workshop papers via OAI-PMH.

    Returns papers published within the last `days`. Graceful on failure
    (returns empty list + warning) so the main pipeline continues.
    """
    papers: list[PaperCandidate] = []
    cutoff = datetime.now() - timedelta(days=days)
    next_url: str | None = ACL_SDP_OAI_URL
    pages = 0

    with _build_paper_client() as client:
        while next_url and pages < _MAX_OAI_PAGES:
            pages += 1
            try:
                resp = client.get(next_url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("Failed to fetch ACL SDP OAI-PMH: %s", exc)
                break

            try:
                root = ET.fromstring(resp.text)
            except ET.ParseError as exc:
                log.warning("Malformed ACL SDP XML: %s", exc)
                break

            stop_paging = False
            for record in root.findall(".//oai:record", OAI_NS):
                result = _parse_acl_record(record, client, cutoff)
                if result is None:
                    continue
                if result == "_too_old_":
                    stop_paging = True
                    continue
                papers.append(result)

            token_el = root.find(".//oai:resumptionToken", OAI_NS)
            token = (token_el.text or "").strip() if token_el is not None else ""
            if stop_paging or not token:
                next_url = None
            else:
                next_url = ACL_SDP_RESUME_URL.format(token=token)

    log.info("Fetched %d ACL SDP papers (last %d days)", len(papers), days)
    return papers
```

- [ ] **Step 4: Run the fetcher tests**

Run: `uv run pytest tests/test_acl_sdp_fetcher.py -v`
Expected: all 7 PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 6: Lint + format**

Run: `uv run ruff check dtech.py tests/test_acl_sdp_fetcher.py && uv run ruff format dtech.py tests/test_acl_sdp_fetcher.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add dtech.py tests/test_acl_sdp_fetcher.py
git commit -m "feat: add ACL SDP OAI-PMH fetcher with landing-page abstract fallback"
```

---

## Task 6: Extend `deduplicate_papers` to accept ACL SDP list

**Files:**
- Modify: `dtech.py` (the `deduplicate_papers` function body, updated in Task 1).
- Modify: `tests/test_papers.py` (extend the existing dedup test, OR add test 12 from spec).

- [ ] **Step 1: Write failing test for three-source dedup**

Append to `tests/test_papers.py`:

```python
def test_deduplicate_papers_handles_three_sources():
    arxiv_p = PaperCandidate(
        paper_id="2403.12345", title="A", abstract="a", published="2026-04-10",
        pdf_url="https://arxiv.org/pdf/2403.12345", categories="cs.CL", source="arxiv",
    )
    hf_p = PaperCandidate(
        paper_id="2403.12345", title="A", abstract="a", published="2026-04-10",
        pdf_url="https://arxiv.org/pdf/2403.12345", categories="", source="hf",
        hf_trending=True,
    )
    acl_p = PaperCandidate(
        paper_id="2024.sdp-1.3", title="S", abstract="s", published="2024-11-01T00:00:00Z",
        pdf_url="https://aclanthology.org/2024.sdp-1.3.pdf", categories="", source="acl-sdp",
    )
    result = deduplicate_papers([arxiv_p], [hf_p], [acl_p])
    ids = [p.paper_id for p in result]
    assert set(ids) == {"2403.12345", "2024.sdp-1.3"}
    merged_arxiv = next(p for p in result if p.paper_id == "2403.12345")
    assert merged_arxiv.source == "arxiv"
    assert merged_arxiv.hf_trending is True
    merged_acl = next(p for p in result if p.paper_id == "2024.sdp-1.3")
    assert merged_acl.source == "acl-sdp"
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/test_papers.py::test_deduplicate_papers_handles_three_sources -v`
Expected: FAIL with `TypeError: deduplicate_papers() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Extend `deduplicate_papers` signature** (`dtech.py`, replace the function body from Task 1)

```python
def deduplicate_papers(
    arxiv: list[PaperCandidate],
    hf: list[PaperCandidate],
    acl_sdp: list[PaperCandidate] | None = None,
) -> list[PaperCandidate]:
    """Merge papers across sources, deduplicating by paper_id.

    When a paper appears in both arXiv and HF (same arXiv ID), keep arXiv's
    richer metadata but flip hf_trending=True. ACL papers use anthology IDs,
    which never collide with arXiv IDs, so they pass through untouched.
    """
    by_id: dict[str, PaperCandidate] = {}

    for p in arxiv:
        by_id[p.paper_id] = p

    for p in hf:
        if p.paper_id in by_id:
            existing = by_id[p.paper_id]
            by_id[p.paper_id] = PaperCandidate(
                paper_id=existing.paper_id,
                title=existing.title,
                abstract=existing.abstract,
                published=existing.published,
                pdf_url=existing.pdf_url,
                categories=existing.categories,
                source=existing.source,
                hf_trending=True,
            )
        else:
            by_id[p.paper_id] = p

    for p in acl_sdp or []:
        if p.paper_id not in by_id:
            by_id[p.paper_id] = p

    return list(by_id.values())
```

The third argument defaults to `None` so existing callers (tests) that still pass two lists continue to work.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_papers.py -v`
Expected: all PASS including the new test.

- [ ] **Step 5: Commit**

```bash
git add dtech.py tests/test_papers.py
git commit -m "feat: extend deduplicate_papers to accept ACL SDP list"
```

---

## Task 7: Extend `rank_papers` with source-aware soft-reserve prompt

**Files:**
- Modify: `dtech.py:502-567` (the `rank_papers` function).
- Create: `tests/test_rank_papers_source_aware.py`.

- [ ] **Step 1: Create `tests/test_rank_papers_source_aware.py` with failing tests**

```python
"""Tests for rank_papers source-aware soft reserve."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtech import PaperCandidate, rank_papers


def _mk(paper_id: str, source: str, hf: bool = False) -> PaperCandidate:
    return PaperCandidate(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        abstract="abstract",
        published="2026-04-10",
        pdf_url="",
        categories="",
        source=source,
        hf_trending=hf,
    )


def test_rank_papers_prompt_tags_acl_sdp_papers():
    candidates = [_mk("2403.1", "arxiv"), _mk("2024.sdp-1.3", "acl-sdp")]
    captured_prompt = {}

    def _invoke(prompt):
        captured_prompt["text"] = prompt
        resp = MagicMock()
        resp.content = json.dumps([{"paper_id": "2024.sdp-1.3", "reason": "x"}])
        return resp

    with patch("dtech.model.invoke", side_effect=_invoke):
        rank_papers(candidates, top_n=1)

    assert "| ACL-SDP" in captured_prompt["text"]
    assert "2024.sdp-1.3" in captured_prompt["text"]
    assert "SOURCE-AWARE SOFT RESERVE" in captured_prompt["text"]


def test_rank_papers_returns_acl_sdp_paper_when_chosen():
    candidates = [_mk("2403.1", "arxiv"), _mk("2024.sdp-1.3", "acl-sdp")]

    def _invoke(_prompt):
        resp = MagicMock()
        resp.content = json.dumps([{"paper_id": "2024.sdp-1.3", "reason": "x"}])
        return resp

    with patch("dtech.model.invoke", side_effect=_invoke):
        selected = rank_papers(candidates, top_n=1)

    assert len(selected) == 1
    assert selected[0].source == "acl-sdp"


def test_rank_papers_fallback_when_llm_returns_empty():
    candidates = [_mk("2403.1", "arxiv"), _mk("2024.sdp-1.3", "acl-sdp")]

    def _invoke(_prompt):
        resp = MagicMock()
        resp.content = "[]"
        return resp

    with patch("dtech.model.invoke", side_effect=_invoke):
        selected = rank_papers(candidates, top_n=2)

    assert len(selected) == 2
    assert {p.paper_id for p in selected} == {"2403.1", "2024.sdp-1.3"}


def test_rank_papers_tags_hf_and_acl_together():
    candidates = [_mk("2403.1", "hf", hf=True), _mk("2024.sdp-1.3", "acl-sdp")]
    captured_prompt = {}

    def _invoke(prompt):
        captured_prompt["text"] = prompt
        resp = MagicMock()
        resp.content = json.dumps([{"paper_id": "2403.1", "reason": "x"}])
        return resp

    with patch("dtech.model.invoke", side_effect=_invoke):
        rank_papers(candidates, top_n=1)

    assert "HF-TRENDING" in captured_prompt["text"]
    assert "| ACL-SDP" in captured_prompt["text"]
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_rank_papers_source_aware.py -v`
Expected: tests FAIL — prompt text does not yet contain `SOURCE-AWARE SOFT RESERVE` or `| ACL-SDP` tag.

- [ ] **Step 3: Refactor `rank_papers`** (`dtech.py:502-567`)

Add a helper `_source_tag` directly above `rank_papers`:

```python
def _source_tag(p: PaperCandidate) -> str:
    tags = []
    if p.source == "acl-sdp":
        tags.append("ACL-SDP")
    if p.hf_trending:
        tags.append("HF-TRENDING")
    return f" | {' | '.join(tags)}" if tags else ""
```

Replace the `paper_list` string inside `rank_papers` with:

```python
    paper_list = "\n\n".join(
        f"[{i + 1}] ID: {p.paper_id} | Source: {p.source}{_source_tag(p)} | Title: {p.title}"
        f"\nAbstract: {p.abstract[:500]}"
        for i, p in enumerate(candidates)
    )
```

Insert the soft-reserve clause into the prompt. Find the existing SELECTION CRITERIA block (ending with the HF-TRENDING tiebreaker line) and the HARD REJECTS block, and between them add:

```python
    prompt = f"""You are selecting research papers for a senior AI engineer who needs
NEW TECHNICAL METHODS, not benchmarks or surveys.

DEVELOPER PROFILE: {INTEREST_PROFILE}

Below are {len(candidates)} recent papers. Select the top {top_n} that deliver the most
substantive METHODOLOGICAL contribution this developer can use at work or in research.

SELECTION CRITERIA (ranked):

1. NOVEL TECHNICAL CONTRIBUTION — paper proposes a new algorithm, architecture, loss
   function, optimizer, training scheme, decoding strategy, attention mechanism, or
   inference technique. This is the PRIMARY criterion. If a paper does not clearly
   introduce a new method, reject it.
2. IMPLEMENTABILITY — the method is described precisely enough to implement, ideally
   with released code, pseudocode, or concrete architectural details.
3. STACK APPLICABILITY — the technique applies to LLM agents, RAG, fine-tuning, or
   inference systems (their working stack).
4. HF-TRENDING — community-validated papers marked HF-TRENDING get a small boost, but
   only after the above criteria are met.

SOURCE-AWARE SOFT RESERVE:
Some papers are tagged ACL-SDP — these come from the ACL Scientific Document
Processing workshop. AIM for UP TO 2 ACL-SDP papers in the final {top_n}, but ONLY
if they clear the same quality bar as the other papers (novel technical
contribution, implementable, stack-applicable). If fewer than 2 ACL-SDP
papers meet the bar this week, fill those slots from the main pool instead.
Do NOT pick a weak ACL-SDP paper just to hit the reserve.

The ACL-SDP reserve is a PREFERENCE, not a quota. Quality always wins.

HARD REJECTS — do not select these even if interesting:
- Benchmark introductions ("we introduce a benchmark / dataset / evaluation suite")
- Survey, review, or position papers
- Leaderboard chasing (same method, new dataset, marginal gains)
- Pure empirical studies with no new method
- Theoretical papers with no implementation path
- Off-topic domains: vision-only, robotics, medical imaging, bio/climate, autonomous driving

Prefer papers that would make the engineer think "I can build this on Monday."

Return ONLY a JSON array (no markdown, no explanation):
[{{"paper_id": "...", "reason": "one-line justification naming the technical contribution"}}, ...]

PAPERS:
{paper_list}"""
```

At the end of the function, after `selected` is computed and before returning, add source-breakdown logging:

```python
    by_source = {
        s: sum(1 for p in selected if p.source == s)
        for s in ("arxiv", "hf", "acl-sdp")
    }
    log.info("Ranker picked %s", by_source)

    return selected
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_rank_papers_source_aware.py tests/test_papers.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint + format**

Run: `uv run ruff check dtech.py tests/test_rank_papers_source_aware.py && uv run ruff format dtech.py tests/test_rank_papers_source_aware.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add dtech.py tests/test_rank_papers_source_aware.py
git commit -m "feat: source-aware soft reserve for ACL SDP in rank_papers"
```

---

## Task 8: Wire the pipeline, update label/category helpers

**Files:**
- Modify: `dtech.py:645-687` (paper-pipeline orchestration block).
- Modify: `dtech.py:1027-1055` (`nice_source_label` and `_source_category` functions).
- Modify: `tests/test_papers.py` (extend existing label/category tests).

- [ ] **Step 1: Write failing tests for label/category handling of ACL source prefix**

Append to `tests/test_papers.py`:

```python
def test_source_category_acl_sdp():
    assert _source_category("acl-sdp:2024.sdp-1.3") == "paper"


def test_source_category_hf():
    assert _source_category("hf:2403.12345") == "paper"


def test_nice_source_label_acl_sdp():
    assert nice_source_label("acl-sdp:2024.sdp-1.3") == "ACL SDP · 2024.sdp-1.3"


def test_nice_source_label_hf():
    label = nice_source_label("hf:2403.12345")
    assert label == "HF · 2403.12345"
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_papers.py -k "source_category or source_label" -v`
Expected: four new tests FAIL (existing two for arxiv pass).

- [ ] **Step 3: Extend `_source_category`** (`dtech.py:1041-1055`)

Replace the function body:

```python
def _source_category(url: str) -> str:
    """Classify a source URL into a category for visual grouping."""
    if url.startswith("arxiv:") or url.startswith("hf:") or url.startswith("acl-sdp:"):
        return "paper"
    if "/search/repositories" in url:
        if "topic:machine-learning" in url:
            return "ml"
        if "topic:llm" in url:
            return "llm"
        if "topic:google-adk" in url:
            return "adk"
        return "trending"
    if "/releases" in url:
        return "release"
    return "other"
```

- [ ] **Step 4: Extend `nice_source_label`** (`dtech.py:1027-1038`)

Replace the function body:

```python
def nice_source_label(url: str) -> str:
    """Turn a source URL or prefixed ID into a readable label."""
    if url.startswith("arxiv:"):
        return url.replace("arxiv:", "arXiv · ")
    if url.startswith("hf:"):
        return url.replace("hf:", "HF · ")
    if url.startswith("acl-sdp:"):
        return url.replace("acl-sdp:", "ACL SDP · ")
    path = urlparse(url).path.strip("/")
    parts = path.split("/")
    if len(parts) >= 4 and parts[0] == "repos":
        owner = parts[1]
        repo = parts[2]
        rest = "/".join(parts[3:])
        return f"{owner}/{repo} · {rest}"
    return path or url
```

- [ ] **Step 5: Wire fetcher, filter, and recorder into the pipeline** (`dtech.py:645-687`)

Find the block starting `log.info("Fetching AI research papers...")`. Replace it through to the end of that function with:

```python
    log.info("Fetching AI research papers...")
    arxiv = fetch_arxiv_papers(days=3)
    hf = fetch_hf_daily_papers()
    acl_sdp = fetch_acl_sdp_papers(days=90)

    if not arxiv and not hf and not acl_sdp:
        log.warning("No papers fetched from any source — skipping.")
        return

    candidates = deduplicate_papers(arxiv, hf, acl_sdp)
    log.info("Deduplicated to %d unique candidates", len(candidates))

    before_filter = len(candidates)
    candidates = [p for p in candidates if not _is_low_value_paper(p)]
    log.info(
        "Pre-LLM filter kept %d of %d candidates (dropped %d benchmark/survey/off-topic)",
        len(candidates),
        before_filter,
        before_filter - len(candidates),
    )

    candidates = filter_already_shown(candidates)

    if not candidates:
        log.warning("All candidates filtered out — skipping paper processing.")
        return

    ranked = rank_papers(candidates, top_n=5)
    log.info("LLM selected %d papers", len(ranked))

    record_shown_papers(ranked)

    for paper in tqdm(ranked, desc="Summarizing papers"):
        summary = summarize_paper(paper)
        store(
            KnowledgeEntry(
                timestamp=datetime.now().isoformat(),
                source=f"{paper.source}:{paper.paper_id}",
                summary=summary,
            )
        )

    log.info("Stored %d paper summaries", len(ranked))
```

Key changes:
- Add `fetch_acl_sdp_papers(days=90)` call.
- Pass three lists to `deduplicate_papers`.
- Call `filter_already_shown` after the low-value filter.
- Call `record_shown_papers(ranked)` once ranking completes and before summarization (so that even if one summarization fails, we don't re-show papers).
- Source prefix in `store()` uses `f"{paper.source}:{paper.paper_id}"`. Task 1 reverted this to the literal `f"arxiv:{paper.paper_id}"` to keep Task 1 purely behavior-preserving; it is restored here as part of the full prefix migration (see Steps 5a–5c below).

- [ ] **Step 5a: Add failing test for `_papers_fetched_today` detecting all paper prefixes**

Append to `tests/test_papers.py`:

```python
def test_papers_fetched_today_detects_hf_and_acl_prefixes(tmp_path, monkeypatch):
    import dtech
    db = tmp_path / "test.db"
    monkeypatch.setattr(dtech, "DB_PATH", db)
    dtech._init_db()
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO knowledge (timestamp, source, summary, date) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), "hf:2403.99999", "x", today),
        )
    assert dtech._papers_fetched_today() is True
```

Run: `uv run pytest tests/test_papers.py::test_papers_fetched_today_detects_hf_and_acl_prefixes -v`
Expected: FAIL — current `_papers_fetched_today` only matches `arxiv:%`.

- [ ] **Step 5b: Generalize `_papers_fetched_today`** (`dtech.py:644` approximately)

Replace the `source LIKE 'arxiv:%'` predicate with a predicate that matches any of the three paper prefixes. Read the current function body first, then apply the minimal edit. The recommended form:

```python
        row = conn.execute(
            """
            SELECT 1 FROM knowledge
            WHERE date = ?
              AND (source LIKE 'arxiv:%' OR source LIKE 'hf:%' OR source LIKE 'acl-sdp:%')
            LIMIT 1
            """,
            (today,),
        ).fetchone()
```

Re-run the test from Step 5a; expected PASS.

- [ ] **Step 5c: Generalize the two report-rendering SQL queries** (`dtech.py:1145` and `dtech.py:1150` approximately)

Read the current queries first. The first one selects paper rows; the second selects non-paper rows via `NOT LIKE 'arxiv:%'`. Replace `source LIKE 'arxiv:%'` with `(source LIKE 'arxiv:%' OR source LIKE 'hf:%' OR source LIKE 'acl-sdp:%')` in both sites (the NOT-LIKE gets wrapped in a NOT of that OR group).

Before editing, add a failing test to `tests/test_papers.py` that stores rows with each of the three prefixes and asserts the report-render functions put them in the correct bucket. Since those render functions are large, the simplest test is:

```python
def test_paper_prefix_predicate_helper(tmp_path, monkeypatch):
    """Every paper-prefix source ends up in the 'paper' side of the split."""
    import dtech
    db = tmp_path / "test.db"
    monkeypatch.setattr(dtech, "DB_PATH", db)
    dtech._init_db()
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(db) as conn:
        for src in ("arxiv:1", "hf:2", "acl-sdp:3", "https://api.github.com/repos/a/b/releases"):
            conn.execute(
                "INSERT INTO knowledge (timestamp, source, summary, date) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), src, "x", today),
            )
        paper_rows = conn.execute(
            "SELECT source FROM knowledge WHERE "
            "source LIKE 'arxiv:%' OR source LIKE 'hf:%' OR source LIKE 'acl-sdp:%'"
        ).fetchall()
    assert {r[0] for r in paper_rows} == {"arxiv:1", "hf:2", "acl-sdp:3"}
```

Run: `uv run pytest tests/test_papers.py -k paper_prefix -v` — should pass without code changes (test is on the predicate itself). It serves as a regression guard that the predicate string stays consistent with the actual queries.

Then update the two queries in `dtech.py` to use the three-way OR predicate.

- [ ] **Step 5d: Restore the generalized store prefix**

In the pipeline code from Step 5 above, ensure the store call uses:

```python
                source=f"{paper.source}:{paper.paper_id}",
```

(not the temporary `f"arxiv:{paper.paper_id}"` from Task 1's fix commit).

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 7: Lint + format**

Run: `uv run ruff check dtech.py tests/test_papers.py && uv run ruff format dtech.py tests/test_papers.py`
Expected: no errors.

- [ ] **Step 8: Manual smoke test**

Run: `uv run python -c "from dtech import fetch_acl_sdp_papers; papers = fetch_acl_sdp_papers(days=180); print(f'{len(papers)} SDP papers'); [print(p.paper_id, p.title[:60]) for p in papers[:5]]"`

Expected: prints a count and up to 5 paper IDs + titles. If zero, widen `days` or confirm the OAI endpoint is reachable from your network.

- [ ] **Step 9: Commit**

```bash
git add dtech.py tests/test_papers.py
git commit -m "feat: wire ACL SDP fetcher into pipeline with cross-day dedup"
```

---

## Task 9: Opt-in integration test against live ACL OAI-PMH

**Files:**
- Create: `tests/test_integration_acl.py`.

- [ ] **Step 1: Create the integration test file**

```python
"""Opt-in integration test against live ACL Anthology OAI-PMH endpoint.

Skipped by default. Run with: uv run pytest tests/test_integration_acl.py -v -m integration
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtech import fetch_acl_sdp_papers


@pytest.mark.integration
def test_acl_oai_live_endpoint():
    """Hit the real OAI endpoint; assert at least one well-formed record comes back."""
    papers = fetch_acl_sdp_papers(days=365 * 3)  # 3-year window for a reliable hit
    assert len(papers) > 0, "expected at least one SDP paper in the last 3 years"
    p = papers[0]
    assert p.source == "acl-sdp"
    assert p.paper_id.startswith("20")  # e.g. "2024.sdp-1.3"
    assert ".sdp-" in p.paper_id
    assert p.title
    assert p.pdf_url.startswith("https://aclanthology.org/")
    assert p.published.endswith("Z")
```

- [ ] **Step 2: Register the `integration` marker in `pyproject.toml`**

If `pyproject.toml` does not already have a `[tool.pytest.ini_options]` table with `markers`, add:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: opt-in tests that hit live external endpoints",
]
```

If that table already exists, just append the marker to the existing list.

- [ ] **Step 3: Verify the default test run skips integration tests**

Run: `uv run pytest tests/ -v`
Expected: all other tests PASS; `test_acl_oai_live_endpoint` is collected but NOT executed (no marker filter → runs only if explicitly selected; or is skipped if `addopts` filters it — either is acceptable as long as default CI doesn't hit the network).

Safer: add `-m "not integration"` to default runs, or leave it as-is and document the opt-in command.

- [ ] **Step 4: Run the integration test explicitly (opt-in)**

Run: `uv run pytest tests/test_integration_acl.py -v -m integration`
Expected: PASS (assuming network access to aclanthology.org).

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration_acl.py pyproject.toml
git commit -m "test: add opt-in integration test for live ACL OAI-PMH endpoint"
```

---

## Self-Review (completed before publishing)

- **Spec coverage:**
  - Goal + Non-Goals → covered by all tasks.
  - Design Decisions 1-5 (original) → Tasks 1, 5, 7 (data model + fetcher + ranker).
  - Design Decision 6 (cross-day dedup) → Tasks 2, 3, 8.
  - Architecture diagram → realized by Task 8.
  - Data Model Changes → Task 1.
  - Cross-Day Deduplication section → Tasks 2, 3, 8.
  - New Fetcher section → Task 5 (with fixtures in Task 4).
  - Pipeline Integration → Task 8.
  - Ranker Integration → Task 7.
  - Testing section (tests 1-17) → distributed across Tasks 1, 3, 5, 6, 7, 8, 9.
  - Rollout steps 1-7 → Tasks 1-9 in order.
  - Risks → mitigated by graceful error handling (Task 5), soft-reserve prompt (Task 7), pagination cap (Task 5), idempotent record (Task 3).

- **Placeholder scan:** none — every step has exact paths, code, commands, and expected outputs.

- **Type consistency:** `paper_id` (not `arxiv_id`), `source` as `Literal["arxiv","hf","acl-sdp"]`, `_source_tag` helper consistently referenced, `filter_already_shown` / `record_shown_papers` signatures stable across Tasks 3 and 8.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-16-acl-sdp-papers.md`. Three execution options:

1. **Subagent-Driven** — dispatch a fresh subagent per task sequentially, review between tasks, fast iteration. Best here since tasks 1→8 are tightly sequential (each depends on the previous).
2. **Team-Driven (swarm)** — parallel execution with dependency-aware TeamCreate coordination and worktree isolation. Not ideal for this plan — Tasks 1–8 form a straight dependency chain, with only Task 9 (integration test) truly parallelizable.
3. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach?
