"""Tests for the AI research papers pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtech import _source_category, fetch_arxiv_papers, nice_source_label


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
