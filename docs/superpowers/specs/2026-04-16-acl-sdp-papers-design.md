# Design: ACL SDP Papers as a Reserved Source

**Date:** 2026-04-16
**Status:** Approved
**Scope:** `dtech.py` paper pipeline + `tests/`

## Goal

In the daily report's 5-paper selection, reserve up to 2 slots for papers from
the ACL [Scientific Document Processing (SDP) workshop](https://aclanthology.org/venues/sdp/).
The reserve is soft: if fewer than 2 SDP papers clear the existing quality bar
(novel technical contribution, implementable, stack-applicable), the remaining
slots are filled from the main arXiv + HuggingFace Daily Papers pool.

## Non-Goals

- Hard quotas, exact-count guarantees, or forced inclusion of weak papers.
- Supporting ACL venues other than SDP in this iteration.
- Title-based cross-source deduplication (an SDP paper and its arXiv preprint
  may both appear as candidates; the ranker resolves it by quality).
- Evaluating LLM selection quality — that is an eval problem, out of scope.

## Design Decisions (from brainstorming)

| # | Question | Decision |
|---|----------|----------|
| 1 | Quota semantics | **Soft reserve**: aim for ~2 SDP papers if quality clears, otherwise fall back to main pool. |
| 2 | Freshness window | **Rolling 90-day window** for ACL SDP (vs 3 days for arXiv). |
| 3 | Data model | **Generalize** `PaperCandidate.arxiv_id` → `paper_id`; add `source: Literal["arxiv", "hf", "acl-sdp"]`. |
| 4 | Fetch method | **OAI-PMH** at `https://aclanthology.org/oai-pmh/?verb=ListRecords&set=sig:sigdp&metadataPrefix=oai_dc`, with landing-page fallback when `<dc:description>` is empty. |
| 5 | Ranker integration | **One-pass, source-aware prompt**: tag each candidate with its source, extend the prompt with a soft-reserve clause. |
| 6 | Cross-day dedup | **New `shown_papers` table** in SQLite — tracks paper IDs already selected in a previous daily report. Filter applied before ranking, uniformly across all sources. |

## Architecture

```
arXiv fetchers (7 queries)  ──┐
HF Daily Papers fetcher ──────┼──► deduplicate_papers ──► rank_papers (source-aware) ──► top 5 ──► summarize ──► HTML
ACL SDP fetcher (NEW)    ─────┘                              ↑
                                                             │
                                    per-candidate source tag; prompt aims for ≤2 SDP
                                    slots only if they clear the quality bar
```

No new pipeline stages, no DB schema migration, no report-format change. The
`PaperCandidate` dataclass is the only seam that crosses module boundaries, and
we extend it rather than branch around it.

## Data Model Changes

**Before (`dtech.py:135-143`):**
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

**After:**
```python
from typing import Literal

PaperSource = Literal["arxiv", "hf", "acl-sdp"]

@dataclass(frozen=True)
class PaperCandidate:
    paper_id: str            # arXiv ID (e.g., "2403.12345") or anthology ID (e.g., "2024.sdp-1.3")
    title: str
    abstract: str
    published: str           # ISO8601; ACL "YYYY" dates are padded to "YYYY-01-01T00:00:00Z"
    pdf_url: str
    categories: str          # arXiv cats; empty for HF and ACL
    source: PaperSource
    hf_trending: bool = False
```

**Call-site rename (mechanical):**
- `fetch_arxiv_papers()`: set `source="arxiv"`, `paper_id=<arxiv_id>`.
- `fetch_hf_daily_papers()`: set `source="hf"`, `paper_id=<arxiv_id>`, `hf_trending=True`.
- `deduplicate_papers()`: key on `paper_id`; grows to accept a third list.
- `rank_papers()`: read `p.paper_id` in the prompt's ID column.
- `_parse_ranked_ids()`: JSON key renamed `"arxiv_id"` → `"paper_id"`.
- `summarize_paper()` and HTML rendering: unchanged; still reads `pdf_url`.

**DB impact:** one new table, no schema changes to existing tables. See
"Cross-Day Deduplication" section below.

## Cross-Day Deduplication

**Problem:** The arXiv fetcher uses a 3-day window, so overlap between
consecutive daily reports is small but non-zero. The new ACL SDP fetcher uses
a 90-day window, which means the ranker will keep picking the "best" SDP
papers day after day — same content re-shown for months. A cross-day dedup
mechanism is needed.

**Solution:** a new SQLite table recording every paper ID that has appeared in
a daily report, queried as a pre-rank filter.

**Schema (applied at startup, alongside `_init_db()`):**
```sql
CREATE TABLE IF NOT EXISTS shown_papers (
    paper_id    TEXT PRIMARY KEY,
    shown_date  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_shown_papers_date
    ON shown_papers(shown_date DESC);
```

`PRIMARY KEY` on `paper_id` makes recording idempotent (safe to re-run a day's
report). Works uniformly for arXiv IDs, HF-derived arXiv IDs, and ACL anthology
IDs — no collisions by construction.

**Filter function (new):**
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
    log.info("Filtered %d already-shown papers; %d candidates remain",
             len(candidates) - len(filtered), len(filtered))
    return filtered
```

**Record function (new, called after final top-5 selection):**
```python
def record_shown_papers(selected: list[PaperCandidate]) -> None:
    """Record that these papers appeared in today's report."""
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO shown_papers (paper_id, shown_date) VALUES (?, ?)",
            [(p.paper_id, today) for p in selected],
        )
```

`INSERT OR IGNORE` keeps the table idempotent if a day's report re-runs.

**Pipeline position:** filter is applied AFTER `deduplicate_papers()` and
AFTER `_should_hard_reject()` but BEFORE `rank_papers()` — so the LLM never
sees already-shown candidates. Record happens AFTER ranking succeeds,
ensuring we never mark a paper "shown" when the report generation failed
partway through.

**Retention:** no pruning in v1. Storage is negligible (~5 rows/day → ~2000
rows/year). A future `--prune-older-than-days` CLI flag can be added if the
table grows unwieldy.

**Applies to all sources:** this filter is not SDP-specific. It also
eliminates the small overlap in the 3-day arXiv window, so the daily report
becomes strictly novel-content-only across days. This is a minor behavior
change for arXiv papers — documented here so it's not a surprise.

## New Fetcher: `fetch_acl_sdp_papers`

**Constants:**
```python
ACL_SDP_OAI_URL = (
    "https://aclanthology.org/oai-pmh/"
    "?verb=ListRecords&set=sig:sigdp&metadataPrefix=oai_dc"
)
ACL_LANDING_URL = "https://aclanthology.org/{paper_id}/"

OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}
```

**Top-level fetcher signature:**
```python
def fetch_acl_sdp_papers(days: int = 90) -> list[PaperCandidate]:
    """Fetch recent ACL SDP workshop papers via OAI-PMH.

    Returns papers published within the last `days`. Graceful on failure
    (returns empty list + warning) so the main pipeline continues.
    """
```

**Control flow:**
1. Initialize `next_url = ACL_SDP_OAI_URL` and `cutoff = now - timedelta(days=days)`.
2. Loop while `next_url` is set, with a safety cap of 10 resumption-token pages (1000 records).
3. GET `next_url` via `_build_paper_client()` (shared hishel-cached httpx client).
4. Parse XML; for each `<oai:record>`, call `_parse_acl_record(record, client, cutoff)`.
5. If the helper returns the sentinel `"_too_old_"`, set `stop_paging = True` and break out of pagination after the current page (OAI-PMH is newest-first).
6. Read `<oai:resumptionToken>`; if empty or missing or `stop_paging`, stop.

**Helper `_parse_acl_record(record, client, cutoff)`:**
- Extract `<dc:identifier>` → `paper_id` (e.g., `"2024.sdp-1.3"`).
- Extract `<dc:title>` → `title` (whitespace-collapsed).
- Extract `<dc:description>` → `abstract`. If blank, call `_fetch_acl_abstract(paper_id, client)` which GETs the landing page and parses `<meta name="citation_abstract">`.
- Extract `<dc:date>` → normalize: `"2024"` → `"2024-01-01T00:00:00Z"`, `"2024-09"` → `"2024-09-01T00:00:00Z"`, full ISO passes through.
- Compare normalized date to `cutoff`; if older, return `"_too_old_"` sentinel.
- `pdf_url = f"https://aclanthology.org/{paper_id}.pdf"`.
- Return `PaperCandidate(paper_id, title, abstract, published, pdf_url, categories="", source="acl-sdp", hf_trending=False)`.

**Helper `_fetch_acl_abstract(paper_id, client)`:**
- GET `ACL_LANDING_URL.format(paper_id=paper_id)`.
- Use a minimal regex or `html.parser` to find `<meta name="citation_abstract" content="...">`.
- Return the content, or empty string on any failure (the downstream ranker tolerates blank abstracts).

**Error handling:**
- `httpx.HTTPError` on the OAI-PMH endpoint → `log.warning(...)`, return whatever papers were collected so far.
- `ET.ParseError` on malformed XML → `log.warning(...)`, return partial list.
- Blank abstract after landing-page fallback → keep the candidate with empty abstract; `_should_hard_reject()` and `rank_papers()` handle it naturally.
- Pagination safety cap of 10 resumption pages — prevents runaway if the date
  short-circuit logic ever fails.

## Pipeline Integration

In the main run path (wherever `fetch_arxiv_papers` + `fetch_hf_daily_papers` are orchestrated today):

```python
arxiv = fetch_arxiv_papers(days=3)
hf = fetch_hf_daily_papers()
acl_sdp = fetch_acl_sdp_papers(days=90)
candidates = deduplicate_papers(arxiv, hf, acl_sdp)
candidates = [p for p in candidates if not _should_hard_reject(p)]
candidates = filter_already_shown(candidates)
selected = rank_papers(candidates, top_n=5)
record_shown_papers(selected)
```

`deduplicate_papers` signature becomes:

```python
def deduplicate_papers(
    arxiv: list[PaperCandidate],
    hf: list[PaperCandidate],
    acl_sdp: list[PaperCandidate],
) -> list[PaperCandidate]:
```

The existing merge rule (arXiv + HF collision → keep arXiv fields, flip `hf_trending=True`) is preserved. ACL papers are inserted by `paper_id`; anthology IDs cannot collide with arXiv IDs (different formats).

## Ranker Integration

**Per-candidate source tag** (replacing `rank_papers()` candidate-formatting block, currently `dtech.py:507-512`):

```python
def _source_tag(p: PaperCandidate) -> str:
    tags = []
    if p.source == "acl-sdp":
        tags.append("ACL-SDP")
    if p.hf_trending:
        tags.append("HF-TRENDING")
    return f" | {' | '.join(tags)}" if tags else ""

paper_list = "\n\n".join(
    f"[{i + 1}] ID: {p.paper_id} | Source: {p.source}{_source_tag(p)} | Title: {p.title}"
    f"\nAbstract: {p.abstract[:500]}"
    for i, p in enumerate(candidates)
)
```

**Prompt extension** (inserted between the existing "SELECTION CRITERIA" block and "HARD REJECTS"):

```
SOURCE-AWARE SOFT RESERVE:
Some papers are tagged ACL-SDP — these come from the ACL Scientific Document
Processing workshop. AIM for UP TO 2 ACL-SDP papers in the final 5, but ONLY
if they clear the same quality bar as the other papers (novel technical
contribution, implementable, stack-applicable). If fewer than 2 ACL-SDP
papers meet the bar this week, fill those slots from the main pool instead.
Do NOT pick a weak ACL-SDP paper just to hit the reserve.

The ACL-SDP reserve is a PREFERENCE, not a quota. Quality always wins.
```

**Parser:** `_parse_ranked_ids()` already updated in the data-model rename step
(JSON key is `"paper_id"`).

**Source-breakdown logging** added after selection:

```python
by_source = {s: sum(1 for p in selected if p.source == s) for s in ("arxiv", "hf", "acl-sdp")}
log.info("Ranker picked %s", by_source)
```

**Fallback behavior** (unchanged): if the LLM returns no valid IDs,
`rank_papers()` falls back to `candidates[:top_n]` by recency. SDP papers may
or may not appear in the fallback slice — acceptable for v1, since the
fallback is a safety net, not a guarantee path.

**Hard-filter interaction** (`_should_hard_reject()`): unchanged. The filter
acts on title/abstract content, not source, so SDP papers pass through on
merit.

## Testing

**Unit tests:**

`tests/test_acl_sdp_fetcher.py`:
1. `test_parse_acl_record_happy_path` — all fields populated.
2. `test_parse_acl_record_empty_abstract_fallback` — blank `<dc:description>`, landing-page fallback fills abstract.
3. `test_parse_acl_record_year_only_date` — `"2024"` padded to `"2024-01-01T00:00:00Z"`.
4. `test_fetch_acl_sdp_papers_pagination` — two-page fixture, resumption token followed.
5. `test_fetch_acl_sdp_papers_early_stop_on_old_records` — short-circuit when records fall below cutoff.
6. `test_fetch_acl_sdp_papers_http_error` — `httpx.HTTPError` → empty list, warning.
7. `test_fetch_acl_sdp_papers_malformed_xml` — bad XML → empty list, no crash.

`tests/test_paper_candidate.py`:
8. `test_paper_candidate_requires_source` — constructing without `source` raises.

`tests/test_rank_papers.py` (new or extend existing):
9. `test_rank_papers_prompt_tags_acl_sdp_papers` — capture prompt, assert `| ACL-SDP` appears for SDP candidates only.
10. `test_rank_papers_respects_soft_reserve` — mixed candidates, mocked LLM returns 2 SDP + 3 arXiv IDs; assert source breakdown.
11. `test_rank_papers_fallback_when_llm_returns_no_ids` — mock LLM returns `"[]"`; assert fallback works with new `paper_id` field.
12. `test_deduplicate_papers_handles_three_sources` — arXiv + HF collision merges as before; ACL paper passes through with `source="acl-sdp"`.

`tests/test_shown_papers.py` (new):
14. `test_filter_already_shown_removes_known_ids` — seed the table with 2 paper_ids, assert they are filtered from candidates.
15. `test_filter_already_shown_empty_table` — empty table, all candidates pass through.
16. `test_record_shown_papers_is_idempotent` — record same list twice, assert no duplicate rows (uses `INSERT OR IGNORE` + `PRIMARY KEY`).
17. `test_record_shown_papers_stores_today_iso_date` — assert `shown_date` column contains today's `YYYY-MM-DD`.

**Integration test** (opt-in, `@pytest.mark.integration`):

13. `test_acl_oai_live_endpoint` in `tests/test_integration_acl.py` — hits ACL OAI-PMH live, asserts at least one well-formed record. Skipped in default CI; guards against upstream API changes.

Test numbering: integration test stays #13; new unit tests #14–17 are added before it for `tests/test_shown_papers.py`.

**Fixtures** under `tests/fixtures/acl/`:
- `sdp_single_record.xml`
- `sdp_empty_abstract.xml`
- `sdp_landing_page.html`
- `sdp_page1_with_token.xml`
- `sdp_page2_final.xml`
- `sdp_malformed.xml`

**Coverage targets:** 95%+ on the new fetcher module; 90%+ on updated `rank_papers` surface.

**Not tested:**
- LLM selection quality (eval problem).
- ACL Anthology content quality.
- End-to-end `daily_report.html` rendering (existing integration flow covers it).

## Rollout

1. Merge the rename (`arxiv_id` → `paper_id`, add `source`) behind the existing test suite.
2. Add `shown_papers` table + `filter_already_shown` + `record_shown_papers` + their tests.
3. Add the ACL SDP fetcher and its tests.
4. Add the ranker prompt extension + source-breakdown logging.
5. Add the pipeline integration (third fetcher call, extended dedupe, filter + record wiring).
6. Run a full daily report manually; confirm the "Ranker picked" log shows sensible source breakdown, and that re-running the report on the same day produces an empty candidate pool (idempotency check).
7. Watch the next 3–5 daily reports for SDP quality and for absence of cross-day paper repetition; adjust the soft-reserve prompt if the ranker over- or under-selects SDP.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| ACL OAI-PMH endpoint goes down or changes schema | Graceful empty-list return; `test_acl_oai_live_endpoint` integration test surfaces schema drift. |
| Most SDP papers have empty abstracts → ranker can't judge them | Landing-page fallback fills most abstracts; worst case, weak SDP papers fall below quality bar and soft reserve gracefully yields slots to the main pool. |
| Ranker over-picks SDP (picks 2 weak SDP papers over stronger arXiv) | Prompt is explicit: "Quality always wins." If observed in practice, tune by moving the reserve clause below the HARD REJECTS block or adding an explicit counter-example. |
| Anthology IDs collide with arXiv IDs in dedup | Won't happen — formats are disjoint (`2403.12345` vs `2024.sdp-1.3`). Tested in test #12. |
| Pagination runs away | Hard cap of 10 resumption tokens (1000 records). |
| `shown_papers` table grows indefinitely | Negligible size (~2000 rows/year); pruning can be added later via a CLI flag if needed. |
| Re-run of same day produces empty report (everything already shown) | Expected and correct — matches user's mental model of "today's report". Recorded as a rollout step #6 idempotency check. |
