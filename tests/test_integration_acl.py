"""Opt-in integration test against ACL Anthology via acl-anthology Python library.

Skipped by default. Run with: uv run pytest tests/test_integration_acl.py -v -m integration
Requires: acl-anthology package installed, internet access (first run clones the repo).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtech import fetch_acl_sdp_papers


@pytest.mark.integration
def test_acl_anthology_library():
    """Load ACL Anthology data via Python library; assert SDP papers are well-formed."""
    papers = fetch_acl_sdp_papers(days=365 * 3)
    assert len(papers) > 0, "expected at least one SDP paper in the last 3 years"
    p = papers[0]
    assert p.source == "acl-sdp"
    assert p.paper_id.startswith("20")
    assert ".sdp-" in p.paper_id or "sdp" in p.paper_id.lower()
    assert p.title
    assert p.pdf_url.startswith("https://aclanthology.org/")
    assert p.published.endswith("Z")
