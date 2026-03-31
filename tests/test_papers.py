"""Tests for the AI research papers pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json as json_mod

from dtech import (
    PaperCandidate,
    _parse_ranked_ids,
    _source_category,
    deduplicate_papers,
    fetch_arxiv_papers,
    fetch_hf_daily_papers,
    nice_source_label,
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
    from unittest.mock import MagicMock, patch

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
        from dtech import summarize_paper
        summarize_paper(paper)

        prompt_sent = mock_model.invoke.call_args[0][0]
        assert "mermaid" in prompt_sent.lower()
        assert '<pre class="mermaid">' in prompt_sent
