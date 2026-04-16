"""Tests for the ACL SDP fetcher (via acl-anthology Python library)."""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtech import _acl_paper_date, _acl_paper_published_iso, fetch_acl_sdp_papers


def _mock_paper(full_id: str, title: str, abstract: str, year: int, month: str | None = None):
    paper = MagicMock()
    paper.full_id = full_id
    paper.title = title
    paper.abstract = abstract
    paper.year = year
    paper.month = month
    pdf = MagicMock()
    pdf.url = f"https://aclanthology.org/{full_id}.pdf"
    paper.pdf = pdf
    return paper


def _mock_anthology(papers_by_year: dict[int, list]):
    """Build a mock Anthology with get_collection returning mock collections."""
    anthology = MagicMock()

    def get_collection(cid: str):
        year = int(cid.split(".")[0])
        if year not in papers_by_year:
            return None
        coll = MagicMock()
        vol = MagicMock()
        vol.papers.return_value = papers_by_year[year]
        coll.volumes.return_value = [vol]
        return coll

    anthology.get_collection.side_effect = get_collection
    return anthology


def test_acl_paper_date_with_month():
    dt = _acl_paper_date(2024, "August")
    assert dt == datetime(2024, 8, 1)


def test_acl_paper_date_no_month():
    dt = _acl_paper_date(2024, None)
    assert dt == datetime(2024, 1, 1)


def test_acl_paper_published_iso():
    assert _acl_paper_published_iso(2024, "November") == "2024-11-01T00:00:00Z"
    assert _acl_paper_published_iso(2024, None) == "2024-01-01T00:00:00Z"


def test_fetch_returns_papers_within_window():
    current_year = datetime.now().year
    papers = [
        _mock_paper(f"{current_year}.sdp-1.1", "Paper A", "abstract a", current_year, "January"),
        _mock_paper(f"{current_year}.sdp-1.2", "Paper B", "abstract b", current_year, "March"),
    ]
    mock_anth = _mock_anthology({current_year: papers})

    with patch("dtech.Anthology", create=True) as mock_anth_cls:
        mock_anth_cls.from_repo.return_value = mock_anth
        with patch.dict("sys.modules", {"acl_anthology": MagicMock(Anthology=mock_anth_cls)}):
            result = fetch_acl_sdp_papers(days=365)

    assert len(result) == 2
    assert result[0].source == "acl-sdp"
    assert result[0].paper_id == f"{current_year}.sdp-1.1"
    assert result[1].paper_id == f"{current_year}.sdp-1.2"


def test_fetch_filters_old_papers():
    papers = [
        _mock_paper("2020.sdp-1.1", "Old Paper", "abstract", 2020, "June"),
    ]
    mock_anth = _mock_anthology({2020: papers})

    with patch("dtech.Anthology", create=True) as mock_anth_cls:
        mock_anth_cls.from_repo.return_value = mock_anth
        with patch.dict("sys.modules", {"acl_anthology": MagicMock(Anthology=mock_anth_cls)}):
            result = fetch_acl_sdp_papers(days=90)

    assert result == []


def test_fetch_skips_proceedings_frontmatter():
    current_year = datetime.now().year
    papers = [
        _mock_paper(
            f"{current_year}.sdp-1.0", "Proceedings of SDP", "front", current_year, "January"
        ),
        _mock_paper(f"{current_year}.sdp-1.1", "Real Paper", "abstract", current_year, "January"),
    ]
    mock_anth = _mock_anthology({current_year: papers})

    with patch("dtech.Anthology", create=True) as mock_anth_cls:
        mock_anth_cls.from_repo.return_value = mock_anth
        with patch.dict("sys.modules", {"acl_anthology": MagicMock(Anthology=mock_anth_cls)}):
            result = fetch_acl_sdp_papers(days=365)

    assert len(result) == 1
    assert result[0].paper_id == f"{current_year}.sdp-1.1"


def test_fetch_graceful_on_import_error():
    with patch.dict("sys.modules", {"acl_anthology": None}):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = fetch_acl_sdp_papers(days=90)
    assert result == []


def test_fetch_graceful_on_repo_error():
    mock_module = MagicMock()
    mock_module.Anthology.from_repo.side_effect = RuntimeError("git clone failed")

    with patch.dict("sys.modules", {"acl_anthology": mock_module}):
        result = fetch_acl_sdp_papers(days=90)

    assert result == []


def test_fetch_paper_has_correct_fields():
    current_year = datetime.now().year
    papers = [
        _mock_paper(f"{current_year}.sdp-1.5", "Title X", "Abstract X", current_year, "February"),
    ]
    mock_anth = _mock_anthology({current_year: papers})

    with patch("dtech.Anthology", create=True) as mock_anth_cls:
        mock_anth_cls.from_repo.return_value = mock_anth
        with patch.dict("sys.modules", {"acl_anthology": MagicMock(Anthology=mock_anth_cls)}):
            result = fetch_acl_sdp_papers(days=365)

    assert len(result) == 1
    p = result[0]
    assert p.paper_id == f"{current_year}.sdp-1.5"
    assert p.title == "Title X"
    assert p.abstract == "Abstract X"
    assert p.source == "acl-sdp"
    assert p.pdf_url == f"https://aclanthology.org/{current_year}.sdp-1.5.pdf"
    assert p.published == f"{current_year}-02-01T00:00:00Z"
    assert p.categories == ""
    assert p.hf_trending is False
