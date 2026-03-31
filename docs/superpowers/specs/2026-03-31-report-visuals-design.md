# Report Visuals Design

Add LLM-generated Mermaid diagrams to the daily HTML report: per-item diagrams where useful, and a relationship map at the bottom connecting the day's items.

## Approach

**Two-prompt strategy**: modify existing summarization prompts to optionally include Mermaid diagrams (zero extra LLM calls), plus one new LLM call for the relationship map.

## Per-Item Diagrams

### Prompt changes

Both `summarize()` and `summarize_paper()` get an additional instruction appended to their prompts:

```
If a visual diagram would help explain this content (architecture, methodology flow,
migration path, component relationships), include a Mermaid diagram using:
<pre class="mermaid">
graph LR
  A[Step 1] --> B[Step 2]
</pre>

If the content is a simple list, changelog, or does not benefit from a diagram, do NOT
include one. Only add a diagram when it genuinely clarifies the content.
```

### Diagram types by category

| Category | Likely diagram type | When to include |
|----------|-------------------|-----------------|
| Papers | `graph LR` methodology flow | Almost always -- methodology is visual |
| Trending repos | `graph TD` architecture/component diagram | When repo has interesting architecture |
| Releases | `graph LR` migration path | Only for complex breaking changes |

### Placement in HTML

Diagrams appear inline within the card's `.summary` div, wherever the LLM places the `<pre class="mermaid">` block (typically after methodology bullets, before links).

## Relationship Map

### New function: `generate_relationship_map()`

Called after all items are stored, before `generate_html_report()`.

1. Query all today's items from SQLite (titles + sources only, not full summaries).
2. Send to LLM with this prompt:

```
You are analyzing today's tech briefing items to find meaningful connections.

ITEMS:
{numbered list of title + category}

Identify connections between items: shared topics, framework dependencies,
paper-to-tool relevance, competing approaches, etc.

If meaningful connections exist, return a Mermaid graph using this structure:
- Use `graph TD`
- Group items by category using `subgraph` blocks
- Use dotted arrows `-.->|relationship label|` for cross-category connections
- Only include items that have at least one connection
- Keep labels short (under 5 words)

If no meaningful connections exist today, return exactly: NONE

Return ONLY the Mermaid code or NONE, no markdown fences, no explanation.
```

3. If result is not "NONE", store as `source = "meta:relationship-map"` with the Mermaid code as the summary.
4. Uses same `UNIQUE(source, date)` dedup -- won't regenerate if already exists today.

### HTML rendering

- New section before footer: "Today's Connections"
- Styled container matching the paper section aesthetic (subtle gradient background, section divider header)
- Contains a single `<pre class="mermaid">` block with the relationship graph
- Hidden when any category filter is active (only visible on "All")
- Omitted entirely if no relationship map was generated

## HTML Template Changes

### Mermaid.js integration

Add to `<head>`:
```html
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({
    startOnLoad: true,
    theme: 'neutral',
    themeVariables: {
      fontFamily: 'Inter, system-ui, sans-serif',
      fontSize: '14px',
    }
  });
</script>
```

### New CSS

```css
/* Mermaid diagrams inside cards */
.summary pre.mermaid {
    background: var(--bg-deep);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 20px;
    margin: 16px 0;
    text-align: center;
}

/* Override the code label for mermaid blocks */
.summary pre.mermaid::before {
    content: "diagram";
}

/* Connections section */
.connections-section {
    margin-top: 48px;
    padding: 28px 24px 24px;
    background: linear-gradient(
        135deg,
        rgba(99,102,241,0.03) 0%,
        rgba(168,85,247,0.06) 100%
    );
    border-radius: var(--radius-lg);
    border: 1px solid rgba(99,102,241,0.1);
}
.connections-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 24px;
    padding-bottom: 14px;
    border-bottom: 2px solid rgba(99,102,241,0.12);
}
.connections-header span {
    font-size: 20px;
    font-weight: 800;
    color: #6366f1;
    letter-spacing: -0.02em;
}
```

### Filter behavior

The connections section gets `data-cat="connections"` and the filter JS hides it for all filters except "All":

```js
const connSec = document.querySelector('.connections-section');
if (connSec) {
    connSec.classList.toggle('hidden', f !== 'all');
}
```

## Database Changes

No schema changes. The relationship map is stored as a regular knowledge entry with `source = "meta:relationship-map"`. The `generate_html_report()` function queries it separately and excludes it from card rendering.

## File Changes

Only `dtech.py` is modified:

| Location | Change |
|----------|--------|
| `summarize()` prompt | Add diagram instruction paragraph |
| `summarize_paper()` prompt | Add diagram instruction paragraph |
| New `generate_relationship_map()` | ~30 lines, LLM call + store |
| `generate_html_report()` | Add Mermaid.js script tag, connections section HTML, mermaid CSS, filter update |
| `if __name__ == "__main__"` | Call `generate_relationship_map()` before `generate_html_report()` |

## Cost Impact

- Per-item diagrams: zero extra LLM calls (prompt addition to existing calls)
- Relationship map: 1 extra LLM call per run (lightweight -- just titles, not full content)
- Token increase per summarization call: ~80 tokens of prompt addition

## Non-Goals

- No dark mode Mermaid theme (report is light-only)
- No interactive/clickable diagrams
- No diagram caching beyond the existing SQLite dedup
- No new Python dependencies
