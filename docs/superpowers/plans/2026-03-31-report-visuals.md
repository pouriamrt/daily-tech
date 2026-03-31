# Report Visuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LLM-generated Mermaid diagrams to the daily HTML report -- per-item diagrams where useful, and a relationship map at the bottom.

**Architecture:** Modify existing `summarize()` and `summarize_paper()` prompts to optionally emit `<pre class="mermaid">` blocks. Add one new function `generate_relationship_map()` for the bottom synthesis diagram. Update `generate_html_report()` to load Mermaid.js and render the connections section. Single file (`dtech.py`) modified throughout.

**Tech Stack:** Mermaid.js v11 (CDN, client-side), existing LLM via `langchain.init_chat_model`, SQLite, Python 3.13+

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `dtech.py:377-405` | Modify | Add Mermaid diagram instruction to `summarize_paper()` prompt |
| `dtech.py:454-496` | Modify | Add Mermaid diagram instruction to `summarize()` prompt |
| `dtech.py` (new, after line 497) | Create function | `generate_relationship_map()` -- LLM call for connections diagram |
| `dtech.py:615-1488` | Modify | `generate_html_report()` -- Mermaid.js script, CSS, connections section, filter update |
| `dtech.py:1533-1538` | Modify | `__main__` block -- call `generate_relationship_map()` |
| `tests/test_papers.py` | Modify | Add tests for `generate_relationship_map()` and prompt changes |

---

### Task 1: Add Mermaid diagram instruction to `summarize_paper()`

**Files:**
- Modify: `dtech.py:377-405`
- Test: `tests/test_papers.py`

- [ ] **Step 1: Write a test verifying the paper prompt includes Mermaid instruction**

Add to `tests/test_papers.py`:

```python
from unittest.mock import MagicMock, patch


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
        from dtech import summarize_paper
        summarize_paper(paper)

        prompt_sent = mock_model.invoke.call_args[0][0]
        assert "mermaid" in prompt_sent.lower()
        assert '<pre class="mermaid">' in prompt_sent
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_papers.py::test_summarize_paper_prompt_includes_mermaid_instruction -v`
Expected: FAIL -- the current prompt does not contain "mermaid"

- [ ] **Step 3: Add Mermaid instruction to `summarize_paper()` prompt**

In `dtech.py`, replace the `summarize_paper` function prompt. The full function becomes:

```python
def summarize_paper(paper: PaperCandidate) -> str:
    """LLM Pass 2: generate a structured HTML summary for a single paper."""
    prompt = f"""You are writing a research paper briefing for a Python/AI developer.

PAPER TITLE: {paper.title}
ABSTRACT: {paper.abstract}

Generate a structured HTML fragment (no markdown, no backticks, no <html>/<body> tags).
Do NOT include authors. Structure:

<h3>{paper.title}</h3>
<p class="paper-tldr"><em>One-sentence TL;DR of what this paper does.</em></p>
<h4>Key Methodology</h4>
<ul>
  <li>2-3 bullets explaining their approach, technique, or architecture</li>
  <li>Focus on what's novel and how it works at a high level</li>
</ul>
<h4>Why It Matters For You</h4>
<ul>
  <li>1-2 bullets on practical takeaways for someone building with LLMs,
  agents, or ML pipelines in Python</li>
</ul>

If a visual diagram would help explain this paper's methodology (architecture, training
pipeline, data flow), include a Mermaid diagram using:
<pre class="mermaid">
graph LR
  A[Step 1] --> B[Step 2]
</pre>
Place the diagram after the Key Methodology section. If the methodology is simple or a
diagram would not add clarity, do NOT include one.

<p><a href="https://arxiv.org/abs/{paper.arxiv_id}">arXiv</a> &middot;
<a href="{paper.pdf_url}">PDF</a></p>

Keep it concise and useful. Focus on methodology, not hype."""

    response = model.invoke(prompt)
    return response.content
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_papers.py::test_summarize_paper_prompt_includes_mermaid_instruction -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dtech.py tests/test_papers.py
git commit -m "feat: add optional Mermaid diagram instruction to paper summarization prompt"
```

---

### Task 2: Add Mermaid diagram instruction to `summarize()`

**Files:**
- Modify: `dtech.py:454-496`
- Test: `tests/test_papers.py`

- [ ] **Step 1: Write a test verifying the GitHub prompt includes Mermaid instruction**

Add to `tests/test_papers.py`:

```python
def test_summarize_prompt_includes_mermaid_instruction():
    """The GitHub summarization prompt should instruct LLM to optionally include Mermaid."""
    mock_response = MagicMock()
    mock_response.content = "<h2>Title</h2><p>Summary</p>"

    with patch("dtech.model") as mock_model:
        mock_model.invoke.return_value = mock_response
        from dtech import summarize
        summarize('{"items": []}', "https://api.github.com/repos/test/repo/releases")

        prompt_sent = mock_model.invoke.call_args[0][0]
        assert "mermaid" in prompt_sent.lower()
        assert '<pre class="mermaid">' in prompt_sent
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_papers.py::test_summarize_prompt_includes_mermaid_instruction -v`
Expected: FAIL

- [ ] **Step 3: Add Mermaid instruction to `summarize()` prompt**

In `dtech.py`, modify the `summarize` function. The full function becomes:

```python
def summarize(text: str, source: str) -> str:
    """Ask the model to return a compact HTML snippet summarising raw GitHub API JSON."""
    hint = _hint_for(source)
    prompt = f"""
You are preparing a compact daily tech briefing for an experienced developer.

SOURCE: {source}

You are given raw GitHub API JSON. Extract only the most important, recent information.
Focus on:
- version names
- dates
- major breaking changes
- new features
- dropped Python versions
- security / compatibility notes
- one or two key upgrade recommendations

Return a SHORT HTML fragment only (no markdown, no backticks, no <html> or <body> tags).
Structure:

<h2>Title…</h2>
<p>One-sentence overview.</p>
<h3>Key releases</h3>
<ul>
  <li><strong>Version …</strong> – short description.</li>
  ...
</ul>
<h3>Upgrade notes</h3>
<ul>
  <li>Short bullet</li>
</ul>
<h3>Links</h3>
<ul>
  <li><a href="...">Label</a></li>
</ul>

If a visual diagram would help explain this content (architecture overview, component
relationships, migration path between versions), include a Mermaid diagram using:
<pre class="mermaid">
graph LR
  A[Component] --> B[Component]
</pre>
If the content is a simple changelog or list that does not benefit from a diagram, do NOT
include one. Only add a diagram when it genuinely clarifies the content.

Keep it concise and skimmable.{hint}
Here is the raw data to summarize:
{text}
"""
    response = model.invoke(prompt)
    return response.content
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_papers.py::test_summarize_prompt_includes_mermaid_instruction -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dtech.py tests/test_papers.py
git commit -m "feat: add optional Mermaid diagram instruction to GitHub summarization prompt"
```

---

### Task 3: Implement `generate_relationship_map()`

**Files:**
- Modify: `dtech.py` (new function after `store()`, around line 511)
- Test: `tests/test_papers.py`

- [ ] **Step 1: Write tests for the relationship map function**

Add to `tests/test_papers.py`:

```python
import sqlite3
from datetime import datetime
from dtech import _init_db, DB_PATH, generate_relationship_map, store, KnowledgeEntry


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

    # Insert some test items
    store(KnowledgeEntry(
        timestamp=datetime.now().isoformat(),
        source="arxiv:2403.00001",
        summary="<h3>Paper About Agents</h3>",
    ))
    store(KnowledgeEntry(
        timestamp=datetime.now().isoformat(),
        source="https://api.github.com/repos/langchain-ai/langchain/releases",
        summary="<h2>LangChain 1.3</h2>",
    ))

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
    _setup_test_db(tmp_path, monkeypatch)

    store(KnowledgeEntry(
        timestamp=datetime.now().isoformat(),
        source="arxiv:2403.00001",
        summary="<h3>Paper About Agents</h3>",
    ))

    mock_response = MagicMock()
    mock_response.content = "NONE"

    with patch("dtech.model") as mock_model:
        mock_model.invoke.return_value = mock_response
        generate_relationship_map()

    with sqlite3.connect(tmp_path / "test_knowledge.db") as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM knowledge WHERE source = 'meta:relationship-map'"
        ).fetchone()
    assert row[0] == 0


def test_generate_relationship_map_skips_if_already_exists(tmp_path, monkeypatch):
    """Should not call LLM if relationship map already exists for today."""
    _setup_test_db(tmp_path, monkeypatch)

    store(KnowledgeEntry(
        timestamp=datetime.now().isoformat(),
        source="meta:relationship-map",
        summary="graph TD\n  A --> B",
    ))

    with patch("dtech.model") as mock_model:
        generate_relationship_map()
        mock_model.invoke.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_papers.py -k "relationship_map" -v`
Expected: FAIL -- `generate_relationship_map` does not exist yet

- [ ] **Step 3: Implement `generate_relationship_map()`**

Add after the `store()` function in `dtech.py` (around line 511):

```python
def _extract_title_from_summary(summary: str) -> str:
    """Pull the first heading text from an HTML summary for use in relationship mapping."""
    for tag in ("h2", "h3"):
        start = summary.find(f"<{tag}>")
        end = summary.find(f"</{tag}>")
        if start != -1 and end != -1:
            return summary[start + len(tag) + 2 : end].strip()
    return summary[:80]


def generate_relationship_map() -> None:
    """Generate a Mermaid relationship diagram connecting today's briefing items."""
    _init_db()
    today = datetime.now().strftime("%Y-%m-%d")

    with sqlite3.connect(DB_PATH) as conn:
        # Skip if already generated today
        existing = conn.execute(
            "SELECT COUNT(*) FROM knowledge WHERE source = 'meta:relationship-map' AND date = ?",
            (today,),
        ).fetchone()
        if existing[0] > 0:
            log.info("Relationship map already generated today -- skipping.")
            return

        rows = conn.execute(
            "SELECT source, summary FROM knowledge "
            "WHERE date = ? AND source != 'meta:relationship-map' ORDER BY timestamp",
            (today,),
        ).fetchall()

    if len(rows) < 2:
        log.info("Fewer than 2 items today -- skipping relationship map.")
        return

    item_list = "\n".join(
        f"[{i + 1}] {_extract_title_from_summary(row[1])} (category: {_source_category(row[0])})"
        for i, row in enumerate(rows)
    )

    prompt = f"""You are analyzing today's tech briefing items to find meaningful connections.

ITEMS:
{item_list}

Identify connections between items: shared topics, framework dependencies,
paper-to-tool relevance, competing approaches, etc.

If meaningful connections exist, return a Mermaid graph using this structure:
- Use graph TD
- Group items by category using subgraph blocks
- Use dotted arrows -.->|relationship label| for cross-category connections
- Only include items that have at least one connection
- Keep labels short (under 5 words)

If no meaningful connections exist today, return exactly: NONE

Return ONLY the Mermaid code or NONE, no markdown fences, no explanation."""

    response = model.invoke(prompt)
    content = response.content.strip()

    if content.upper() == "NONE":
        log.info("LLM found no meaningful connections today.")
        return

    # Strip markdown fences if the LLM wraps them anyway
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    store(KnowledgeEntry(
        timestamp=datetime.now().isoformat(),
        source="meta:relationship-map",
        summary=content,
    ))
    log.info("Relationship map generated and stored.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_papers.py -k "relationship_map" -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add dtech.py tests/test_papers.py
git commit -m "feat: add generate_relationship_map() for cross-item connections diagram"
```

---

### Task 4: Update `generate_html_report()` -- Mermaid.js and CSS

**Files:**
- Modify: `dtech.py:615-1488` (the `generate_html_report()` function)

- [ ] **Step 1: Add Mermaid CSS to the `<style>` block**

In `dtech.py`, inside `generate_html_report()`, find the closing `</style>` tag (around the line with `    </style>`). Insert the following CSS just before `</style>`:

```python
        /* ── Mermaid diagrams ── */
        .summary pre.mermaid {{
            background: var(--bg-deep);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 20px;
            margin: 16px 0;
            text-align: center;
            color: var(--text-body);
        }}
        .summary pre.mermaid::before {{
            content: "diagram";
        }}

        /* ── Connections section ── */
        .connections-section {{
            margin-top: 48px;
            padding: 28px 24px 24px;
            background: linear-gradient(
                135deg,
                rgba(99,102,241,0.03) 0%,
                rgba(168,85,247,0.06) 100%
            );
            border-radius: var(--radius-lg);
            border: 1px solid rgba(99,102,241,0.1);
        }}
        .connections-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 24px;
            padding-bottom: 14px;
            border-bottom: 2px solid rgba(99,102,241,0.12);
        }}
        .connections-header-title {{
            font-size: 20px;
            font-weight: 800;
            color: #6366f1;
            letter-spacing: -0.02em;
        }}
        .connections-header svg {{
            opacity: 0.8;
        }}
```

- [ ] **Step 2: Add Mermaid.js `<script>` tag to `<head>`**

In the HTML template string, find the closing `</style>` and add the Mermaid script right after it, before `</head>`:

```html
    <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
    <script>
      mermaid.initialize({{
        startOnLoad: true,
        theme: 'neutral',
        themeVariables: {{
          fontFamily: 'Inter, system-ui, sans-serif',
          fontSize: '14px',
        }}
      }});
    </script>
```

- [ ] **Step 3: Run lint to verify no syntax errors**

Run: `uv run ruff check dtech.py`
Expected: No errors (or only pre-existing ones)

- [ ] **Step 4: Commit**

```bash
git add dtech.py
git commit -m "feat: add Mermaid.js CDN, diagram CSS, and connections section styles"
```

---

### Task 5: Update `generate_html_report()` -- connections section and filter logic

**Files:**
- Modify: `dtech.py:615-1488`

- [ ] **Step 1: Query the relationship map from the DB**

In `generate_html_report()`, after the existing queries for `paper_rows` and `github_rows` (around line 631), add a query for the relationship map:

```python
        relationship_row = conn.execute(
            "SELECT summary FROM knowledge "
            "WHERE source = 'meta:relationship-map' AND date = ?",
            (today,),
        ).fetchone()
    relationship_map = relationship_row[0] if relationship_row else ""
```

Note: this goes inside the existing `with sqlite3.connect(DB_PATH) as conn:` block, and the `relationship_map` assignment goes just after the `with` block closes.

- [ ] **Step 2: Build the connections section HTML**

After the `cards` string is built (around line 720, after the cards loop), add:

```python
    # Build connections section
    connections_section = ""
    if relationship_map:
        connections_section = f"""
        <section class="connections-section" data-cat="connections">
            <div class="connections-header">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                     stroke="#6366f1" stroke-width="2">
                    <circle cx="12" cy="5" r="3"/><circle cx="19" cy="19" r="3"/>
                    <circle cx="5" cy="19" r="3"/>
                    <line x1="12" y1="8" x2="19" y2="16"/>
                    <line x1="12" y1="8" x2="5" y2="16"/>
                </svg>
                <span class="connections-header-title">Today's Connections</span>
            </div>
            <pre class="mermaid">
{relationship_map}
            </pre>
        </section>"""
```

- [ ] **Step 3: Insert the connections section into the HTML template**

Find the section in the HTML template string where the footer is rendered (the line with `<footer class="footer">`). Insert `{connections_section}` on its own line just before the footer:

```python
{connections_section}

        <footer class="footer">
```

- [ ] **Step 4: Update the filter JS to hide connections section**

In the `applyFilter` JavaScript function inside the HTML template, after the `ghSec` toggle logic, add:

```javascript
            const connSec = document.querySelector('.connections-section');
            if (connSec) {{
                connSec.classList.toggle('hidden', f !== 'all');
            }}
```

- [ ] **Step 5: Exclude `meta:` sources from the GitHub rows query**

Update the `github_rows` query to also exclude meta entries. Change:

```sql
WHERE source NOT LIKE 'arxiv:%' AND date = ?
```

to:

```sql
WHERE source NOT LIKE 'arxiv:%' AND source NOT LIKE 'meta:%' AND date = ?
```

- [ ] **Step 6: Run lint**

Run: `uv run ruff check dtech.py`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add dtech.py
git commit -m "feat: render connections section with relationship map in HTML report"
```

---

### Task 6: Wire up `generate_relationship_map()` in `__main__`

**Files:**
- Modify: `dtech.py:1533-1538`

- [ ] **Step 1: Add the call between data processing and report generation**

Change the `__main__` block from:

```python
if __name__ == "__main__":
    fetch_and_process_papers()
    fetch_and_process(days=7)
    generate_html_report()
    print("Daily knowledge added.")
    os.startfile(REPORT_PATH)
```

to:

```python
if __name__ == "__main__":
    fetch_and_process_papers()
    fetch_and_process(days=7)
    generate_relationship_map()
    generate_html_report()
    print("Daily knowledge added.")
    os.startfile(REPORT_PATH)
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Run lint and format**

Run: `uv run ruff check dtech.py --fix && uv run ruff format dtech.py`
Expected: Clean

- [ ] **Step 4: Commit**

```bash
git add dtech.py
git commit -m "feat: wire generate_relationship_map() into main pipeline"
```

---

### Task 7: End-to-end manual verification

- [ ] **Step 1: Delete today's entries to force a fresh run**

```bash
uv run python -c "
import sqlite3
from datetime import datetime
db = sqlite3.connect('knowledge.db')
today = datetime.now().strftime('%Y-%m-%d')
db.execute('DELETE FROM knowledge WHERE date = ?', (today,))
db.commit()
print(f'Cleared entries for {today}')
"
```

- [ ] **Step 2: Run the full pipeline**

Run: `uv run python dtech.py`

Expected output should show:
- Paper fetching and summarization (with possible Mermaid blocks in summaries)
- GitHub source fetching
- "Relationship map generated and stored." log line
- "HTML report generated" log line
- Browser opens with the report

- [ ] **Step 3: Verify the report in browser**

Check:
1. At least some paper cards contain rendered Mermaid diagrams (flowcharts)
2. The "Today's Connections" section appears at the bottom with a rendered graph
3. Clicking a category filter hides the connections section
4. Clicking "All" shows it again
5. Diagrams render without JS errors (check browser console)

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add dtech.py
git commit -m "fix: address issues found during manual verification"
```

Only create this commit if fixes were needed. Skip if everything worked.
