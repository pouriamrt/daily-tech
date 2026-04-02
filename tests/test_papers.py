"""Tests for the AI research papers pipeline."""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json as json_mod
from unittest.mock import MagicMock, patch

from dtech import (
    KnowledgeEntry,
    PaperCandidate,
    _fix_bare_quoted_nodes,
    _init_db,
    _normalize_html_tags,
    _parse_ranked_ids,
    _postprocess_summary,
    _reduce_excessive_bold,
    _sanitize_mermaid,
    _source_category,
    deduplicate_papers,
    fetch_arxiv_papers,
    fetch_hf_daily_papers,
    generate_relationship_map,
    nice_source_label,
    store,
    summarize,
    summarize_paper,
)


def test_source_category_arxiv():
    assert _source_category("arxiv:2403.12345") == "paper"


def test_source_category_github_unchanged():
    assert _source_category("https://api.github.com/repos/fastapi/fastapi/releases") == "release"


def test_nice_source_label_arxiv():
    assert nice_source_label("arxiv:2403.12345") == "arXiv · 2403.12345"


def test_nice_source_label_github_unchanged():
    url = "https://api.github.com/repos/fastapi/fastapi/releases"
    assert "fastapi/fastapi" in nice_source_label(url)


_RECENT_DATE = datetime.now().strftime("%Y-%m-%dT00:00:00Z")

SAMPLE_ARXIV_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2403.12345v1</id>
    <title>  Test Paper: A Novel Approach
    to Something  </title>
    <summary>  This paper presents a novel approach
    to doing something interesting in ML.  </summary>
    <published>{_RECENT_DATE}</published>
    <link href="http://arxiv.org/abs/2403.12345v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2403.12345v1"
          title="pdf" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
    <category term="cs.AI"/>
  </entry>
</feed>"""


def test_fetch_arxiv_papers_parses_xml(monkeypatch):
    """Test that arXiv XML is correctly parsed into PaperCandidate objects."""

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


SAMPLE_HF_JSON = json_mod.dumps(
    [
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
    ]
)


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

    shared = next(p for p in result if p.arxiv_id == "2403.12345")
    assert shared.hf_trending is True
    assert shared.categories == "cs.LG, cs.AI"


def test_parse_ranked_ids_valid_json():
    raw = (
        '[{"arxiv_id": "2403.111", "reason": "good"}, {"arxiv_id": "2403.222", "reason": "great"}]'
    )
    assert _parse_ranked_ids(raw) == ["2403.111", "2403.222"]


def test_parse_ranked_ids_invalid_json():
    raw = "This is not valid JSON at all"
    assert _parse_ranked_ids(raw) == []


def test_parse_ranked_ids_strips_markdown_fences():
    raw = '```json\n[{"arxiv_id": "2403.333", "reason": "nice"}]\n```'
    assert _parse_ranked_ids(raw) == ["2403.333"]


def test_summarize_paper_prompt_includes_mermaid_instruction():
    """The paper summarization prompt should instruct LLM to optionally include Mermaid."""
    paper = PaperCandidate(
        arxiv_id="2403.00001",
        title="Test Paper",
        abstract="A test abstract about methodology.",
        published="2026-03-27T00:00:00Z",
        pdf_url="http://arxiv.org/pdf/2403.00001v1",
        categories="cs.LG",
    )

    mock_response = MagicMock()
    mock_response.content = "<h3>Test Paper</h3><p>Summary</p>"

    with patch("dtech.model") as mock_model:
        mock_model.invoke.return_value = mock_response
        summarize_paper(paper)

        prompt_sent = mock_model.invoke.call_args[0][0]
        assert "mermaid" in prompt_sent.lower()
        assert '<pre class="mermaid">' in prompt_sent


def test_summarize_prompt_includes_mermaid_instruction():
    """The GitHub summarization prompt should instruct LLM to optionally include Mermaid."""
    mock_response = MagicMock()
    mock_response.content = "<h2>Test Repo</h2><p>Summary</p>"

    with patch("dtech.model") as mock_model:
        mock_model.invoke.return_value = mock_response
        summarize('{"items": []}', "https://api.github.com/repos/test/repo/releases")

        prompt_sent = mock_model.invoke.call_args[0][0]
        assert "mermaid" in prompt_sent.lower()
        assert '<pre class="mermaid">' in prompt_sent


# ---------------------------------------------------------------------------
# generate_relationship_map tests
# ---------------------------------------------------------------------------


def _setup_test_db(tmp_path, monkeypatch):
    """Create a temp DB and patch DB_PATH to use it."""
    test_db = tmp_path / "test_knowledge.db"
    monkeypatch.setattr("dtech.DB_PATH", test_db)
    _init_db()
    return test_db


def test_generate_relationship_map_stores_entry(tmp_path, monkeypatch):
    """When items exist and LLM finds connections, a meta entry is stored."""
    test_db = _setup_test_db(tmp_path, monkeypatch)
    today = datetime.now().strftime("%Y-%m-%d")

    store(
        KnowledgeEntry(
            timestamp=datetime.now().isoformat(),
            source="arxiv:2403.00001",
            summary="<h3>Paper About Agents</h3>",
        )
    )
    store(
        KnowledgeEntry(
            timestamp=datetime.now().isoformat(),
            source="https://api.github.com/repos/langchain-ai/langchain/releases",
            summary="<h2>LangChain 1.3</h2>",
        )
    )

    mermaid_code = "graph TD\n  P1[Paper About Agents] -.->|uses| R1[LangChain 1.3]"
    mock_response = MagicMock()
    mock_response.content = mermaid_code

    with patch("dtech.model") as mock_model:
        mock_model.invoke.return_value = mock_response
        generate_relationship_map()

    with sqlite3.connect(test_db) as conn:
        row = conn.execute(
            "SELECT summary FROM knowledge WHERE source = 'meta:relationship-map' AND date = ?",
            (today,),
        ).fetchone()
    assert row is not None
    assert "graph TD" in row[0]


def test_generate_relationship_map_skips_when_none(tmp_path, monkeypatch):
    """When LLM returns NONE, no meta entry is stored."""
    test_db = _setup_test_db(tmp_path, monkeypatch)

    store(
        KnowledgeEntry(
            timestamp=datetime.now().isoformat(),
            source="arxiv:2403.00001",
            summary="<h3>Paper About Agents</h3>",
        )
    )
    store(
        KnowledgeEntry(
            timestamp=datetime.now().isoformat(),
            source="https://api.github.com/repos/test/repo/releases",
            summary="<h2>Some Release</h2>",
        )
    )

    mock_response = MagicMock()
    mock_response.content = "NONE"

    with patch("dtech.model") as mock_model:
        mock_model.invoke.return_value = mock_response
        generate_relationship_map()

    with sqlite3.connect(test_db) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM knowledge WHERE source = 'meta:relationship-map'"
        ).fetchone()
    assert row[0] == 0


def test_generate_relationship_map_skips_if_already_exists(tmp_path, monkeypatch):
    """Should not call LLM if relationship map already exists for today."""
    _setup_test_db(tmp_path, monkeypatch)

    store(
        KnowledgeEntry(
            timestamp=datetime.now().isoformat(),
            source="meta:relationship-map",
            summary="graph TD\n  A --> B",
        )
    )

    with patch("dtech.model") as mock_model:
        generate_relationship_map()
        mock_model.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# _fix_bare_quoted_nodes tests
# ---------------------------------------------------------------------------


def test_fix_bare_quoted_nodes_basic():
    """Bare quoted strings on arrow lines should get node IDs."""
    code = 'graph LR\n  "Embeddings" --> "Index"\n  "Query" --> "Index"'
    result = _fix_bare_quoted_nodes(code)
    assert "BN" in result
    assert '["Embeddings"]' in result
    assert '["Index"]' in result
    assert '["Query"]' in result


def test_fix_bare_quoted_nodes_same_label_reuses_id():
    """The same label appearing twice should get the same node ID."""
    code = 'graph LR\n  "A" --> "B"\n  "B" --> "C"'
    result = _fix_bare_quoted_nodes(code)
    # "B" appears twice and should map to the same ID
    assert result.count('["B"]') == 2


def test_fix_bare_quoted_nodes_ignores_bracketed():
    """Already-bracketed nodes like A["Label"] should not be modified."""
    code = 'graph LR\n  A["Label"] --> B["Other"]'
    result = _fix_bare_quoted_nodes(code)
    assert result == code


def test_fix_bare_quoted_nodes_ignores_non_arrow_lines():
    """Quoted strings on non-arrow lines (subgraph, etc.) should not be touched."""
    code = 'graph TD\n  subgraph "My Group"\n    A --> B\n  end'
    result = _fix_bare_quoted_nodes(code)
    assert result == code


# ---------------------------------------------------------------------------
# _sanitize_mermaid integration tests
# ---------------------------------------------------------------------------


def test_sanitize_mermaid_fixes_bare_nodes_and_quotes():
    """Full sanitizer should fix bare quoted nodes and quote unquoted labels."""
    code = 'graph LR\n  "Input" --> "Output"\n  A[Unquoted] --> B'
    result = _sanitize_mermaid(code)
    assert '["Input"]' in result
    assert '["Output"]' in result
    assert '["Unquoted"]' in result


# ---------------------------------------------------------------------------
# _normalize_html_tags tests
# ---------------------------------------------------------------------------


def test_normalize_html_tags_b_to_strong():
    html = "<b>bold</b> and <i>italic</i>"
    result = _normalize_html_tags(html)
    assert "<strong>bold</strong>" in result
    assert "<em>italic</em>" in result
    assert "<b>" not in result
    assert "<i>" not in result


def test_normalize_html_tags_preserves_strong():
    html = "<strong>already</strong> and <em>fine</em>"
    result = _normalize_html_tags(html)
    assert result == html


# ---------------------------------------------------------------------------
# _reduce_excessive_bold tests
# ---------------------------------------------------------------------------


def test_reduce_excessive_bold_keeps_first_strips_rest():
    html = (
        "<li><strong>Label</strong>: text with <strong>term1</strong> "
        "and <strong>term2</strong></li>"
    )
    result = _reduce_excessive_bold(html)
    assert "<strong>Label</strong>" in result
    assert "<strong>term1</strong>" not in result
    assert "term1" in result
    assert "<strong>term2</strong>" not in result
    assert "term2" in result


def test_reduce_excessive_bold_no_strong():
    html = "<li>Plain text bullet</li>"
    result = _reduce_excessive_bold(html)
    assert result == html


def test_reduce_excessive_bold_single_strong_unchanged():
    html = "<li><strong>Only label</strong>: description</li>"
    result = _reduce_excessive_bold(html)
    assert result == html


def test_reduce_excessive_bold_works_on_paragraphs():
    html = "<p>Text with <strong>bold1</strong> and <strong>bold2</strong></p>"
    result = _reduce_excessive_bold(html)
    assert "<strong>bold1</strong>" in result
    assert "<strong>bold2</strong>" not in result
    assert "bold2" in result


def test_reduce_excessive_bold_skips_nested_lists():
    """Multi-line <li> with nested <ul> should not be corrupted."""
    html = (
        "<li><strong>Label</strong>\n"
        "  <ul>\n"
        "    <li><strong>Inner</strong>: text</li>\n"
        "  </ul>\n"
        "</li>"
    )
    result = _reduce_excessive_bold(html)
    # Inner single-line <li> should have its bold kept (first strong)
    assert "<li><strong>Inner</strong>: text</li>" in result
    # Outer line only has one strong, so stays intact
    assert "<li><strong>Label</strong>" in result


def test_reduce_excessive_bold_multiline_li_with_inline_bolds():
    """A <li> line that has nested content AND inline bolds should be fixed."""
    html = (
        "<li><strong>Label</strong> intro in <strong>v1.2</strong>, plus:\n"
        "  <ul>\n"
        "    <li>Sub-item</li>\n"
        "  </ul>\n"
        "</li>"
    )
    result = _reduce_excessive_bold(html)
    # First line had 2 strongs - second should be stripped
    assert "<strong>Label</strong>" in result
    assert "<strong>v1.2</strong>" not in result
    assert "v1.2" in result


# ---------------------------------------------------------------------------
# _postprocess_summary integration test
# ---------------------------------------------------------------------------


def test_postprocess_summary_full_pipeline():
    """postprocess_summary should normalize tags, fix mermaid, and reduce bold."""
    html = (
        "<li><b>Label</b>: <b>inline bold</b></li>"
        '<pre class="mermaid">graph LR\n  "A" --> "B"</pre>'
    )
    result = _postprocess_summary(html)
    # b -> strong, first bold kept, second stripped
    assert "<strong>Label</strong>" in result
    assert "<strong>inline bold</strong>" not in result
    assert "inline bold" in result
    # Mermaid bare nodes fixed
    assert '["A"]' in result
    assert '["B"]' in result
