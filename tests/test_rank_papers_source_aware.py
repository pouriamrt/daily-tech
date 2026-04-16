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

    with patch("dtech.model") as mock_model:
        mock_model.invoke.side_effect = _invoke
        rank_papers(candidates, top_n=1)

    assert "| ACL-SDP" in captured_prompt["text"]
    assert "2024.sdp-1.3" in captured_prompt["text"]
    assert "ACL-SDP RESERVED SLOTS" in captured_prompt["text"]


def test_rank_papers_returns_acl_sdp_paper_when_chosen():
    candidates = [_mk("2403.1", "arxiv"), _mk("2024.sdp-1.3", "acl-sdp")]

    def _invoke(_prompt):
        resp = MagicMock()
        resp.content = json.dumps([{"paper_id": "2024.sdp-1.3", "reason": "x"}])
        return resp

    with patch("dtech.model") as mock_model:
        mock_model.invoke.side_effect = _invoke
        selected = rank_papers(candidates, top_n=1)

    assert len(selected) == 1
    assert selected[0].source == "acl-sdp"


def test_rank_papers_fallback_when_llm_returns_empty():
    candidates = [_mk("2403.1", "arxiv"), _mk("2024.sdp-1.3", "acl-sdp")]

    def _invoke(_prompt):
        resp = MagicMock()
        resp.content = "[]"
        return resp

    with patch("dtech.model") as mock_model:
        mock_model.invoke.side_effect = _invoke
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

    with patch("dtech.model") as mock_model:
        mock_model.invoke.side_effect = _invoke
        rank_papers(candidates, top_n=1)

    assert "HF-TRENDING" in captured_prompt["text"]
    assert "| ACL-SDP" in captured_prompt["text"]
