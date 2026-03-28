# AI Research Papers Integration — Design Spec

**Date**: 2026-03-28
**Status**: Approved

## Goal

Add a curated daily AI research papers section to the Daily Tech Intelligence report. 3-5 high-relevance papers per day with structured methodology breakdowns, rendered as a dedicated section above the existing GitHub content.

## Data Sources

### arXiv API

- **Endpoint**: `http://export.arxiv.org/api/query`
- **Query**: `cat:cs.AI OR cat:cs.LG OR cat:cs.CL`, sorted by `submittedDate` descending
- **Window**: Last 3 days of submissions
- **Max results**: 20 candidates
- **Format**: Atom XML, parsed with `xml.etree.ElementTree` (stdlib)
- **Fields extracted**: `id` (arXiv URL), `title`, `abstract`, `published`, `categories`, `pdf_link`

### HuggingFace Daily Papers

- **Endpoint**: `https://huggingface.co/api/daily_papers`
- **Format**: JSON with community-curated papers (typically 5-15/day)
- **Fields extracted**: `title`, `abstract` (from linked arXiv paper), `paper.id`, `upvotes`

### Deduplication

Both sources reference arXiv IDs. Before ranking, deduplicate by arXiv ID. Papers appearing in both sources get an `hf_trending: True` flag for the ranking prompt.

### Caching

Uses the existing `hishel` HTTP cache client with 1-hour TTL. No new cache configuration.

## LLM Two-Pass Pipeline

### Pass 1 — Rank & Select

Single LLM call with all deduplicated candidates (~20-30 abstracts).

**Prompt includes**:
- Interest profile: "Python developer working with LLMs, agents (Google ADK, LangChain), ML pipelines, and applied AI"
- All candidate titles + abstracts
- HF-trending flag as extra signal
- Instruction to return JSON array of top 5 arXiv IDs with one-line justification each

**Output**: JSON array, e.g.:
```json
[
  {"arxiv_id": "2403.12345", "reason": "Novel agent memory architecture"},
  ...
]
```

**Fallback**: If LLM returns invalid JSON, fall back to first 5 candidates by recency.

### Pass 2 — Deep Summary

One LLM call per selected paper. Generates structured HTML:

```html
<h3>Paper Title</h3>
<p class="paper-tldr">One-line TL;DR</p>
<h4>Key Methodology</h4>
<ul>
  <li>2-3 bullets on approach/technique</li>
</ul>
<h4>Why It Matters For You</h4>
<ul>
  <li>Practical takeaway for AI/Python/agent work</li>
</ul>
<p><a href="...">arXiv</a> &middot; <a href="...">PDF</a></p>
```

Uses existing `model` instance (`gpt-5.4-nano`). No new LLM configuration.

## Storage

Reuses existing `knowledge` table — no schema migration.

- Papers stored with `source` = `"arxiv:{arxiv_id}"` (e.g., `"arxiv:2403.12345"`)
- Existing `UNIQUE(source, date)` constraint handles dedup across runs
- Helper `_papers_fetched_today()` checks if papers were already processed today; skips entire pipeline if so

## HTML Report

### Dedicated Papers Section

Rendered **above** existing GitHub cards, with visual separation.

**New category metadata**:
```python
"paper": ("Research Papers", "#0891b2", <book-open SVG icon>)
```

**Section structure**:
- Section divider/header: "Research Papers" with teal accent
- 3-5 paper cards using existing card component style
- Each card has `data-cat="paper"` and the "Research Papers" badge
- New `.paper-tldr` CSS class (italic, muted text)
- Section separator before GitHub content

### Filter Integration

- New "Research" button in the filter bar
- Stat pills include "Research Papers" count
- Existing JS filter logic works unchanged via `data-cat="paper"`

## Code Organization

All code stays in `dtech.py`. New functions:

| Function | Purpose |
|----------|---------|
| `fetch_arxiv_papers(days=3)` | Query arXiv API, parse XML, return list of candidate dicts |
| `fetch_hf_daily_papers()` | Query HF daily papers API, return list of candidate dicts |
| `deduplicate_papers(arxiv, hf)` | Merge by arXiv ID, flag HF-sourced ones |
| `rank_papers(candidates)` | Pass 1 — LLM selects top 5, returns arXiv IDs |
| `summarize_paper(paper)` | Pass 2 — structured HTML summary for one paper |
| `fetch_and_process_papers()` | Orchestrator: fetch -> dedup -> rank -> summarize -> store |

**Modified functions**:
- `generate_html_report()` — query paper entries separately, render dedicated section above GitHub cards
- `_source_category()` — recognize `arxiv:` prefix, return `"paper"` category
- `__main__` — call `fetch_and_process_papers()` before `fetch_and_process()`

## Error Handling

| Scenario | Behavior |
|----------|----------|
| arXiv API fails | Log warning, continue with HF papers only |
| HF API fails | Log warning, continue with arXiv only |
| Both APIs fail | Skip papers entirely, report generates with GitHub content only |
| LLM ranking returns invalid JSON | Fall back to first 5 candidates by recency |
| LLM ranking returns <3 papers | Show whatever it picked |
| LLM ranking returns 0 papers | Skip papers section header entirely |

## Dependencies

**No new dependencies.** XML parsing uses stdlib `xml.etree.ElementTree`. HTTP uses existing `httpx`/`hishel`. LLM uses existing `langchain` model instance.

## LLM Cost per Run

- 1 ranking call (~20-30 abstracts, ~4-6k input tokens)
- 3-5 summary calls (~500 input tokens each, ~300 output tokens each)
- Total: ~6-8k input tokens, ~2k output tokens per run
