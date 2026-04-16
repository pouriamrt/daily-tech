"""Tests for the shown_papers cross-day deduplication table."""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtech import PaperCandidate, _init_db, filter_already_shown, record_shown_papers


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
