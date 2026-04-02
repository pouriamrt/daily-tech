from __future__ import annotations

import json  # noqa: F401  (used by upcoming paper-fetch tasks)
import logging
import os
import re
import sqlite3
import xml.etree.ElementTree as ET  # noqa: F401  (used by upcoming paper-fetch tasks)
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from hishel import SyncSqliteStorage
from hishel.httpx import SyncCacheClient
from langchain.chat_models import init_chat_model
from tqdm import tqdm

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "knowledge.db"
REPORT_PATH = BASE_DIR / "daily_report.html"

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

model = init_chat_model("gpt-5.4-nano", temperature=0.1, timeout=60)

HEADERS: dict[str, str] = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "daily-knowledge-bot",
    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN', '')}",
}

SOURCES: tuple[str, ...] = (
    # Trending Python repos (recent)
    "https://api.github.com/search/repositories?q=language:python+created:>={date}&sort=stars&order=desc",
    # Trending AI / ML repos
    "https://api.github.com/search/repositories?q=topic:machine-learning+created:>={date}&sort=stars&order=desc",
    "https://api.github.com/search/repositories?q=topic:llm+created:>={date}&sort=stars&order=desc",
    # Trending Google ADK repos
    "https://api.github.com/search/repositories?q=topic:google-adk+created:>={date}&sort=stars&order=desc",
    # FastAPI releases
    "https://api.github.com/repos/fastapi/fastapi/releases",
    # HuggingFace Transformers releases
    "https://api.github.com/repos/huggingface/transformers/releases",
    # LangChain releases
    "https://api.github.com/repos/langchain-ai/langchain/releases",
    # Google ADK releases
    "https://api.github.com/repos/google/adk-python/releases",
    # Claude Code releases
    "https://api.github.com/repos/anthropics/claude-code/releases",
)

# Per-source extra instructions appended to the summarisation prompt.
# Keys are substring-matched against the resolved URL.
SUMMARY_HINTS: dict[str, str] = {
    "langchain-ai/langchain": (
        "Be EXTREMELY brief. Only list the 3 most important releases "
        "with one-line descriptions. Skip minor patch bumps and partner "
        "packages. No upgrade notes section — just releases and links."
    ),
    "google/adk-python": (
        "After the releases section, add a <h3>Quick Example</h3> section "
        "with a small, self-contained Python code snippet (inside <pre><code>) "
        "showing a practical, learnable usage of a NEW feature from the latest "
        "release. Keep the example under 15 lines."
    ),
    "topic:google-adk": (
        "After the repos section, add a <h3>Quick Example</h3> section "
        "with a small, self-contained Python code snippet (inside <pre><code>) "
        "showing a practical, learnable usage of Google ADK based on the most "
        "interesting repo in the list (e.g. creating a simple agent, using "
        "tools, or a multi-agent pattern). Keep the example under 15 lines."
    ),
    "anthropics/claude-code": (
        "After the releases section, add a <h3>Quick Example</h3> section "
        "with a small, self-contained code snippet (inside <pre><code>) "
        "showing a practical, learnable usage of a NEW feature from the latest "
        "release. Keep the example under 15 lines."
    ),
    "fastapi/fastapi": (
        "After the releases section, add a <h3>Quick Example</h3> section "
        "with a small, self-contained Python code snippet (inside <pre><code>) "
        "showing a practical, learnable usage of a NEW feature from the latest "
        "release (e.g. SSE, streaming, strict content-type). Keep it under 15 lines."
    ),
    "topic:machine-learning": (
        "Focus on repos that introduce novel techniques, not wrappers or tutorials. "
        "For each notable repo, mention: what problem it solves, the core technique, "
        "and star count. After the repos list, add a <h3>Quick Example</h3> section "
        "with a Python code snippet (inside <pre><code>) showing the most interesting "
        "repo's key API or pattern. Keep the example under 15 lines."
    ),
    "topic:llm": (
        "Focus on repos with novel architectures, inference optimizations, or agent "
        "frameworks. For each notable repo, mention the core technical contribution. "
        "After the repos list, add a <h3>Quick Example</h3> section with a Python "
        "code snippet (inside <pre><code>) demonstrating the most interesting repo's "
        "API (e.g. agent setup, model loading, tool registration). Under 15 lines."
    ),
    "huggingface/transformers": (
        "Focus on new model architectures supported, pipeline changes, and performance "
        "improvements. Include model names and parameter counts. After the releases "
        "section, add a <h3>Quick Example</h3> section with a Python code snippet "
        "(inside <pre><code>) showing how to use a newly supported model or feature. "
        "Keep it under 15 lines."
    ),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeEntry:
    timestamp: str
    source: str
    summary: str


@dataclass(frozen=True)
class PaperCandidate:
    arxiv_id: str
    title: str
    abstract: str
    published: str
    pdf_url: str
    categories: str
    hf_trending: bool = False


def _init_db() -> None:
    """Create the knowledge table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT    NOT NULL,
                source    TEXT    NOT NULL,
                summary   TEXT    NOT NULL,
                date      TEXT    NOT NULL,
                UNIQUE(source, date)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_date
            ON knowledge(date DESC)
        """)


def _build_client() -> SyncCacheClient:
    """Create an httpx client with transparent HTTP caching via hishel."""
    storage = SyncSqliteStorage(
        database_path=str(BASE_DIR / ".http_cache.db"),
        default_ttl=3600.0,
    )
    return SyncCacheClient(storage=storage, headers=HEADERS, timeout=30.0)


def _hint_for(url: str) -> str:
    """Return any extra summarisation instruction that matches *url*."""
    for pattern, hint in SUMMARY_HINTS.items():
        if pattern in url:
            return f"\n\nADDITIONAL INSTRUCTION: {hint}"
    return ""


ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Multiple targeted queries: broad categories + keyword-focused searches
ARXIV_QUERIES: tuple[str, ...] = (
    # Broad AI/ML/NLP categories
    "https://export.arxiv.org/api/query"
    "?search_query=cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL"
    "&sortBy=submittedDate&sortOrder=descending&max_results=25",
    # LLM agents, tool-use, and agentic architectures
    "https://export.arxiv.org/api/query"
    "?search_query=all:agent+AND+(all:LLM+OR+all:language+model+OR+all:tool+use)"
    "&sortBy=submittedDate&sortOrder=descending&max_results=15",
    # RAG, retrieval-augmented generation, knowledge grounding
    "https://export.arxiv.org/api/query"
    "?search_query=all:retrieval+augmented+generation+OR+all:RAG+OR+all:knowledge+grounding"
    "&sortBy=submittedDate&sortOrder=descending&max_results=10",
    # Fine-tuning, RLHF, alignment, preference optimization
    "https://export.arxiv.org/api/query"
    "?search_query=all:fine-tuning+AND+(all:LLM+OR+all:RLHF+OR+all:DPO+OR+all:preference)"
    "&sortBy=submittedDate&sortOrder=descending&max_results=10",
)


def _build_paper_client() -> SyncCacheClient:
    """Create an httpx client for paper APIs (no GitHub auth headers)."""
    storage = SyncSqliteStorage(
        database_path=str(BASE_DIR / ".http_cache.db"),
        default_ttl=3600.0,
    )
    return SyncCacheClient(
        storage=storage,
        headers={"User-Agent": "daily-knowledge-bot"},
        timeout=30.0,
    )


def _parse_arxiv_id(raw_id: str) -> str:
    """Extract the numeric arXiv ID from a full URL like 'http://arxiv.org/abs/2403.12345v1'."""
    segment = raw_id.rstrip("/").split("/")[-1]
    if "v" in segment:
        segment = segment[: segment.rfind("v")]
    return segment


def fetch_arxiv_papers(days: int = 3) -> list[PaperCandidate]:
    """Fetch recent AI papers from arXiv across multiple targeted queries."""
    seen_ids: set[str] = set()
    papers: list[PaperCandidate] = []
    cutoff = datetime.now() - timedelta(days=days)

    with _build_paper_client() as client:
        for query_url in ARXIV_QUERIES:
            try:
                resp = client.get(query_url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("Failed to fetch arXiv query: %s", exc)
                continue

            root = ET.fromstring(resp.text)

            for entry in root.findall("atom:entry", ARXIV_NS):
                published_text = entry.findtext("atom:published", "", ARXIV_NS)
                if not published_text:
                    continue
                published_dt = datetime.fromisoformat(published_text.replace("Z", "+00:00"))
                if published_dt.replace(tzinfo=None) < cutoff:
                    continue

                raw_id = entry.findtext("atom:id", "", ARXIV_NS)
                arxiv_id = _parse_arxiv_id(raw_id)
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                title = " ".join(entry.findtext("atom:title", "", ARXIV_NS).split())
                abstract = " ".join(entry.findtext("atom:summary", "", ARXIV_NS).split())

                pdf_url = ""
                for link in entry.findall("atom:link", ARXIV_NS):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                        break

                cats = [c.get("term", "") for c in entry.findall("atom:category", ARXIV_NS)]

                papers.append(
                    PaperCandidate(
                        arxiv_id=arxiv_id,
                        title=title,
                        abstract=abstract,
                        published=published_text,
                        pdf_url=pdf_url,
                        categories=", ".join(cats),
                    )
                )

    log.info("Fetched %d unique papers from arXiv (%d queries)", len(papers), len(ARXIV_QUERIES))
    return papers


HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"


def fetch_hf_daily_papers() -> list[PaperCandidate]:
    """Fetch today's community-curated papers from HuggingFace."""
    papers: list[PaperCandidate] = []
    with _build_paper_client() as client:
        try:
            resp = client.get(HF_DAILY_PAPERS_URL)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("Failed to fetch HuggingFace daily papers: %s", exc)
            return papers

    for item in resp.json():
        paper_data = item.get("paper", {})
        arxiv_id = paper_data.get("id", "")
        if not arxiv_id:
            continue

        title = " ".join(paper_data.get("title", "").split())
        abstract = " ".join(paper_data.get("summary", "").split())
        published = paper_data.get("publishedAt", "")

        papers.append(
            PaperCandidate(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                published=published,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                categories="",
                hf_trending=True,
            )
        )

    log.info("Fetched %d papers from HuggingFace Daily Papers", len(papers))
    return papers


def deduplicate_papers(
    arxiv: list[PaperCandidate],
    hf: list[PaperCandidate],
) -> list[PaperCandidate]:
    """Merge papers from both sources, deduplicating by arXiv ID.

    When a paper appears in both, keep arXiv's richer metadata but set hf_trending=True.
    """
    by_id: dict[str, PaperCandidate] = {}

    for p in arxiv:
        by_id[p.arxiv_id] = p

    for p in hf:
        if p.arxiv_id in by_id:
            existing = by_id[p.arxiv_id]
            by_id[p.arxiv_id] = PaperCandidate(
                arxiv_id=existing.arxiv_id,
                title=existing.title,
                abstract=existing.abstract,
                published=existing.published,
                pdf_url=existing.pdf_url,
                categories=existing.categories,
                hf_trending=True,
            )
        else:
            by_id[p.arxiv_id] = p

    return list(by_id.values())


INTEREST_PROFILE = (
    "AI/ML engineer building production systems in Python. Core stack: "
    "Google ADK (multi-agent orchestration), LangChain (chains, retrieval), "
    "FastAPI (serving), HuggingFace Transformers (fine-tuning, inference). "
    "Actively working on: LLM agent architectures, tool-use and function calling, "
    "RAG pipelines with hybrid search, fine-tuning (LoRA/QLoRA, DPO, GRPO), "
    "multi-agent coordination patterns, prompt engineering, and evaluation/benchmarking. "
    "Cares about: novel training techniques, inference optimization (quantization, "
    "speculative decoding, KV-cache), agentic tool-use, retrieval strategies, "
    "code generation, and anything with reproducible results or open-source code."
)


def _parse_ranked_ids(raw: str) -> list[str]:
    """Parse LLM ranking output into a list of arXiv IDs. Handles markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        items = json.loads(text)
        return [item["arxiv_id"] for item in items if "arxiv_id" in item]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def rank_papers(candidates: list[PaperCandidate], top_n: int = 5) -> list[PaperCandidate]:
    """LLM Pass 1: select the top-N most relevant papers for our interest profile."""
    if not candidates:
        return []

    paper_list = "\n\n".join(
        f"[{i + 1}] ID: {p.arxiv_id} | Title: {p.title}"
        f"{' | HF-TRENDING' if p.hf_trending else ''}"
        f"\nAbstract: {p.abstract[:500]}"
        for i, p in enumerate(candidates)
    )

    prompt = f"""You are selecting the most relevant AI research papers for a developer.

DEVELOPER PROFILE: {INTEREST_PROFILE}

Below are {len(candidates)} recent papers. Select the top {top_n} most relevant to this
developer's daily work. Ranking criteria (in order of importance):

1. DIRECT APPLICABILITY — can the developer use this technique/tool/method in their stack?
   (agents, RAG, fine-tuning, serving, evaluation — highest priority)
2. NOVEL METHODOLOGY — introduces a new approach, architecture, or training technique
   that advances the state of the art (not incremental benchmarks on existing methods)
3. OPEN-SOURCE / REPRODUCIBLE — paper has code available or describes reproducible steps
4. HF-TRENDING — papers marked HF-TRENDING have community validation, give a relevance boost

De-prioritize: pure theoretical papers with no practical path, benchmark-only papers,
survey papers, and papers about domains unrelated to NLP/agents/ML-engineering.

Return ONLY a JSON array (no markdown, no explanation):
[{{"arxiv_id": "...", "reason": "one-line justification"}}, ...]

PAPERS:
{paper_list}"""

    response = model.invoke(prompt)
    ranked_ids = _parse_ranked_ids(response.content)

    if not ranked_ids:
        log.warning(
            "LLM ranking returned no valid IDs — falling back to first %d by recency", top_n
        )
        return candidates[:top_n]

    by_id = {p.arxiv_id: p for p in candidates}
    selected = [by_id[aid] for aid in ranked_ids if aid in by_id]

    if not selected:
        log.warning("LLM ranked IDs not found in candidates — falling back to first %d", top_n)
        return candidates[:top_n]

    return selected


def summarize_paper(paper: PaperCandidate) -> str:
    """LLM Pass 2: generate a structured, technical HTML summary for a single paper."""
    prompt = f"""You are writing a TECHNICAL research paper briefing for an AI/ML engineer
who builds production systems with LLMs, agents, RAG, and fine-tuning in Python.

PAPER TITLE: {paper.title}
ABSTRACT: {paper.abstract}

Generate a structured HTML fragment (no markdown, no backticks, no <html>/<body> tags).
Do NOT include authors. Be TECHNICAL and SPECIFIC -- include actual technique names,
architecture details, and numbers. Structure:

<h3>{paper.title}</h3>
<p class="paper-tldr"><em>One-sentence TL;DR: what problem, what approach, what result.</em></p>
<h4>Technical Approach</h4>
<ul>
  <li>3-4 bullets explaining the core technique with specifics: architecture names,
  loss functions, training strategies, model sizes, data pipeline design</li>
  <li>Name specific components (e.g. "cross-attention over retrieved chunks" not
  "uses retrieval")</li>
  <li>If they propose a new method, explain how it differs from the baseline technically</li>
</ul>
<h4>Key Results</h4>
<ul>
  <li>2-3 bullets with actual benchmark numbers, datasets, and comparisons</li>
  <li>Include percentage improvements, scores, or metrics when available</li>
  <li>Note computational cost or efficiency gains if mentioned</li>
</ul>
<h4>Practical Takeaways</h4>
<ul>
  <li>2-3 bullets on how this applies to someone building with Google ADK, LangChain,
  HuggingFace Transformers, or FastAPI</li>
  <li>Be specific: "You could adapt their retrieval scoring to your RAG pipeline" not
  "This is relevant to RAG"</li>
  <li>If the technique is implementable, sketch how (e.g. "add a re-ranking step after
  your vector search using their scoring formula")</li>
</ul>
If a visual diagram would help explain the architecture or pipeline (most papers benefit
from this), include a Mermaid diagram using:
<pre class="mermaid">
graph LR
  A["Step 1"] --> B["Step 2"]
</pre>
IMPORTANT Mermaid rules: Always quote node labels with double quotes inside brackets,
e.g. A["Label here"]. Use only plain text in labels -- no HTML entities, no special
characters like &, <, >, #, or parentheses. Keep labels short (3-5 words max).
Place the diagram after the Technical Approach section.

<p><a href="https://arxiv.org/abs/{paper.arxiv_id}">arXiv</a> &middot;
<a href="{paper.pdf_url}">PDF</a></p>

BOLDING RULE: Use <strong> ONLY for the leading label at the start of each <li> bullet
(e.g. "<li><strong>Core method</strong>: description..."). Do NOT bold inline terms within
the description text. Let technical terms stand on their own without emphasis markup.

Be dense and technical. No hype, no filler. Write for someone who reads papers regularly."""

    response = model.invoke(prompt)
    return _postprocess_summary(response.content)


def _papers_fetched_today() -> bool:
    """Return True if any paper entries exist for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM knowledge WHERE date = ? AND source LIKE 'arxiv:%'",
            (today,),
        ).fetchone()
    return row[0] > 0


def fetch_and_process_papers() -> None:
    """Orchestrator: fetch -> dedup -> rank -> summarize -> store."""
    _init_db()

    if _papers_fetched_today():
        log.info("Papers already fetched today — skipping.")
        return

    log.info("Fetching AI research papers...")
    arxiv = fetch_arxiv_papers(days=3)
    hf = fetch_hf_daily_papers()

    if not arxiv and not hf:
        log.warning("No papers fetched from any source — skipping.")
        return

    candidates = deduplicate_papers(arxiv, hf)
    log.info("Deduplicated to %d unique candidates", len(candidates))

    ranked = rank_papers(candidates, top_n=5)
    log.info("LLM selected %d papers", len(ranked))

    for paper in tqdm(ranked, desc="Summarizing papers"):
        summary = summarize_paper(paper)
        store(
            KnowledgeEntry(
                timestamp=datetime.now().isoformat(),
                source=f"arxiv:{paper.arxiv_id}",
                summary=summary,
            )
        )

    log.info("Stored %d paper summaries", len(ranked))


# Cap raw API text sent to the LLM to avoid context-window overflow.
# ~80K chars ≈ ~20K tokens for gpt-5.4-nano, well within limits.
_MAX_SUMMARIZE_CHARS = 80_000


def summarize(text: str, source: str) -> str:
    """Ask the model to return a compact HTML snippet summarising raw GitHub API JSON."""
    if len(text) > _MAX_SUMMARIZE_CHARS:
        text = text[:_MAX_SUMMARIZE_CHARS] + "\n\n[... truncated ...]"
    hint = _hint_for(source)
    prompt = f"""
You are preparing a TECHNICAL daily briefing for an AI/ML engineer who builds production
systems with Python (Google ADK, LangChain, FastAPI, HuggingFace Transformers).

SOURCE: {source}

You are given raw GitHub API JSON. Extract the most important, recent, TECHNICAL information.
Focus on:
- version names and dates
- NEW APIs, classes, or functions with their signatures/parameters
- major breaking changes with migration paths
- performance improvements with numbers (e.g. "2x faster inference", "30% less memory")
- new model support, architecture changes, or backend updates
- dropped Python versions or dependency changes
- security fixes with CVE IDs if available

Return a SHORT HTML fragment only (no markdown, no backticks, no <html> or <body> tags).
Structure:

<h2>Title…</h2>
<p>One-sentence technical overview.</p>
<h3>Key releases</h3>
<ul>
  <li><strong>Version …</strong> – description with specific API/feature names.</li>
  ...
</ul>
<h3>Technical Details</h3>
<ul>
  <li>New APIs or changed interfaces (mention class/function names)</li>
  <li>Performance or architecture changes with specifics</li>
  <li>Migration steps for breaking changes</li>
</ul>
<h3>Links</h3>
<ul>
  <li><a href="...">Label</a></li>
</ul>

If a visual diagram would help explain this content (architecture overview, component
relationships, migration path between versions), include a Mermaid diagram using:
<pre class="mermaid">
graph LR
  A["Component"] --> B["Component"]
</pre>
IMPORTANT Mermaid rules: Always quote node labels with double quotes inside brackets,
e.g. A["Label here"]. Use only plain text in labels -- no HTML entities, no special
characters like &, <, >, #, or parentheses. Keep labels short (3-5 words max).
If the content is a simple changelog or list that does not benefit from a diagram, do NOT
include one. Only add a diagram when it genuinely clarifies the content.

BOLDING RULE: Use <strong> ONLY for the leading label at the start of each <li> bullet
(e.g. "<li><strong>v1.2.3</strong> – description..."). Do NOT bold inline terms within
the description text. Let technical terms stand on their own without emphasis markup.

Be technical and specific. No generic descriptions.{hint}
Here is the raw data to summarize:
{text}
"""
    response = model.invoke(prompt)
    return _postprocess_summary(response.content)


def store(entry: KnowledgeEntry) -> None:
    """Insert a knowledge entry, skipping if source+date already exists."""
    date = entry.timestamp[:10]
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO knowledge
                (timestamp, source, summary, date)
            VALUES (?, ?, ?, ?)
            """,
            (entry.timestamp, entry.source, entry.summary, date),
        )


def _extract_title_from_summary(summary: str) -> str:
    """Pull the first heading text from an HTML summary for use in relationship mapping."""
    for tag in ("h2", "h3"):
        start = summary.find(f"<{tag}>")
        end = summary.find(f"</{tag}>")
        if start != -1 and end != -1:
            text = summary[start + len(tag) + 2 : end].strip()
            if text:
                return text
    return summary[:80]


def _fix_bare_quoted_nodes(code: str) -> str:
    """Fix bare quoted strings used as Mermaid nodes (e.g. ``"Label" --> "Other"``).

    Valid Mermaid requires ``NodeID["Label"]``, but LLMs sometimes emit bare
    quoted strings without IDs.  This assigns stable generated IDs so the same
    label always maps to the same node.
    """
    arrow_re = re.compile(r"(-->|-.->|-\.->|==>|~~>|-->>)")
    # Split lines by arrows and edge labels to isolate node tokens
    split_re = re.compile(r"(-->|-.->|-\.->|==>|~~>|-->>|\|[^|]*\|)")

    node_map: dict[str, str] = {}
    counter = 0

    for line in code.split("\n"):
        if not arrow_re.search(line):
            continue
        # Split into tokens around arrows/edge-labels, check each for bare "..."
        for token in split_re.split(line):
            t = token.strip()
            if t.startswith('"') and t.endswith('"') and len(t) > 2:
                label = t[1:-1]
                if label not in node_map:
                    counter += 1
                    node_map[label] = f"BN{counter}"

    if not node_map:
        return code

    for label, node_id in node_map.items():
        # Replace standalone "label" that is NOT inside brackets
        code = re.sub(
            r'(?<!\[)"' + re.escape(label) + r'"(?!\])',
            f'{node_id}["{label}"]',
            code,
        )
    return code


def _sanitize_mermaid(code: str) -> str:
    """Clean up LLM-generated Mermaid syntax to avoid common parse errors.

    Fixes: bare quoted nodes, unquoted labels, HTML entities, unicode symbols.
    """
    # Replace HTML entities
    code = code.replace("&amp;", "and").replace("&lt;", "").replace("&gt;", "")

    # Fix bare quoted strings used as nodes (before bracket quoting)
    code = _fix_bare_quoted_nodes(code)

    # Quote unquoted node labels: A[some label] -> A["some label"]
    # Matches [...] that isn't already ["..."]
    def _quote_label(m: re.Match) -> str:
        bracket_type = m.group(1)  # [ or (
        label = m.group(2)
        close = m.group(3)  # ] or )
        if label.startswith('"') and label.endswith('"'):
            return f"{bracket_type}{label}{close}"
        # Strip chars that break Mermaid even inside quotes
        clean = label.replace('"', "'")
        return f'{bracket_type}"{clean}"{close}'

    code = re.sub(r"(\[)([^\]]+?)(\])", _quote_label, code)
    code = re.sub(r"(\()([^)]+?)(\))", _quote_label, code)

    return code


def _sanitize_mermaid_in_html(html: str) -> str:
    """Find all <pre class="mermaid"> blocks in HTML and sanitize their Mermaid content."""

    def _replace_block(m: re.Match) -> str:
        raw_code = m.group(1)
        return f'<pre class="mermaid">{_sanitize_mermaid(raw_code)}</pre>'

    return re.sub(
        r'<pre class="mermaid">(.*?)</pre>',
        _replace_block,
        html,
        flags=re.DOTALL,
    )


def _normalize_html_tags(html: str) -> str:
    """Normalize ``<b>`` -> ``<strong>`` and ``<i>`` -> ``<em>`` for consistent CSS styling."""
    html = re.sub(r"<b\b([^>]*)>", r"<strong\1>", html)
    html = html.replace("</b>", "</strong>")
    html = re.sub(r"<i\b([^>]*)>", r"<em\1>", html)
    html = html.replace("</i>", "</em>")
    return html


def _reduce_excessive_bold(html: str) -> str:
    """Keep the first ``<strong>`` per line and strip subsequent ones.

    LLMs tend to bold every technical term, making everything look equally heavy.
    Processing line-by-line avoids corrupting nested HTML structures while
    still catching multi-line ``<li>`` elements with inline bolds.
    """

    def _fix_line(line: str) -> str:
        if line.count("<strong>") <= 1:
            return line
        first_end = line.find("</strong>")
        if first_end == -1:
            return line
        split_at = first_end + len("</strong>")
        before = line[:split_at]
        after = line[split_at:]
        after = re.sub(r"<strong>(.*?)</strong>", r"\1", after)
        return before + after

    return "\n".join(_fix_line(line) for line in html.split("\n"))


def _postprocess_summary(html: str) -> str:
    """Apply all HTML post-processing: mermaid sanitization, tag normalization, bold reduction."""
    html = _sanitize_mermaid_in_html(html)
    html = _normalize_html_tags(html)
    html = _reduce_excessive_bold(html)
    return html


def generate_relationship_map() -> None:
    """Generate a Mermaid relationship diagram connecting today's briefing items."""
    _init_db()
    today = datetime.now().strftime("%Y-%m-%d")

    with sqlite3.connect(DB_PATH) as conn:
        existing = conn.execute(
            "SELECT COUNT(*) FROM knowledge "
            "WHERE source IN ('meta:relationship-map', 'meta:novel-idea') AND date = ?",
            (today,),
        ).fetchone()
        if existing[0] >= 2:
            log.info("Relationship map and novel idea already generated today -- skipping.")
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
- IMPORTANT: Always quote node labels with double quotes, e.g. I1["Label here"]
- Use only plain text in labels -- no HTML entities, no special characters like &, <, >, #
- Keep labels short and simple (3-5 words)

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

    store(
        KnowledgeEntry(
            timestamp=datetime.now().isoformat(),
            source="meta:relationship-map",
            summary=_sanitize_mermaid(content),
        )
    )
    log.info("Relationship map generated and stored.")

    # Generate a novel idea from the connections
    summaries_text = "\n\n".join(
        f"[{_source_category(row[0]).upper()}] "
        f"{_extract_title_from_summary(row[1])}\n{row[1][:500]}"
        for row in rows
    )

    idea_prompt = f"""You are a creative AI researcher and engineer. Below are today's tech
briefing items -- research papers, trending repos, and release notes.

{summaries_text}

Based on the CONNECTIONS between these items, propose ONE novel, actionable project idea
that combines insights from at least 2-3 of today's items in a way that none of them
individually address. This should be something a Python/AI developer could realistically
start building.

Return an HTML fragment (no markdown, no backticks). Structure:

<h4>Novel Idea</h4>
<p class="idea-title"><strong>Title of the idea (one line)</strong></p>
<p>2-3 sentence description of what this project would do and why it's interesting.
Reference which of today's items inspired it.</p>
<h4>Quick Start Sketch</h4>
<ul>
<li>2-3 concrete first steps to prototype this</li>
</ul>

Be specific and practical, not vague. The idea should feel like a genuine insight that
emerges from combining today's items, not a generic suggestion."""

    idea_response = model.invoke(idea_prompt)
    idea_content = idea_response.content.strip()

    store(
        KnowledgeEntry(
            timestamp=datetime.now().isoformat(),
            source="meta:novel-idea",
            summary=idea_content,
        )
    )
    log.info("Novel idea generated and stored.")


def nice_source_label(url: str) -> str:
    """Turn a GitHub API URL into a readable label, e.g. 'anthropics/claude-code · releases'."""
    if url.startswith("arxiv:"):
        return url.replace("arxiv:", "arXiv · ")
    path = urlparse(url).path.strip("/")
    parts = path.split("/")
    if len(parts) >= 4 and parts[0] == "repos":
        owner = parts[1]
        repo = parts[2]
        rest = "/".join(parts[3:])
        return f"{owner}/{repo} · {rest}"
    return path or url


def _source_category(url: str) -> str:
    """Classify a source URL into a category for visual grouping."""
    if url.startswith("arxiv:"):
        return "paper"
    if "/search/repositories" in url:
        if "topic:machine-learning" in url:
            return "ml"
        if "topic:llm" in url:
            return "llm"
        if "topic:google-adk" in url:
            return "adk"
        return "trending"
    if "/releases" in url:
        return "release"
    return "other"


def _svg(body: str) -> str:
    """Wrap SVG body in a 14x14 icon element."""
    return (
        '<svg width="14" height="14" viewBox="0 0 24 24"'
        f' fill="none" stroke="currentColor" stroke-width="2">{body}</svg>'
    )


# Category display config: (label, accent color, icon SVG)
_CATEGORY_META: dict[str, tuple[str, str, str]] = {
    "trending": (
        "Trending",
        "#7c3aed",
        _svg(
            '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>'
        ),
    ),
    "ml": (
        "Machine Learning",
        "#db2777",
        _svg(
            '<circle cx="12" cy="12" r="3"/>'
            '<path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83'
            "M16.95 16.95l2.83 2.83M1 12h4M19 12h4"
            'M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>'
        ),
    ),
    "llm": (
        "LLM &amp; Agents",
        "#2563eb",
        _svg('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'),
    ),
    "adk": (
        "Google ADK",
        "#059669",
        _svg(
            '<polygon points="12 2 22 8.5 22 15.5'
            ' 12 22 2 15.5 2 8.5 12 2"/>'
            '<line x1="12" y1="22" x2="12" y2="15.5"/>'
            '<polyline points="22 8.5 12 15.5 2 8.5"/>'
        ),
    ),
    "release": (
        "Release",
        "#d97706",
        _svg(
            '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1'
            '-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>'
            '<line x1="7" y1="7" x2="7.01" y2="7"/>'
        ),
    ),
    "paper": (
        "Research Papers",
        "#0891b2",
        _svg(
            '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>'
            '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>'
        ),
    ),
    "other": (
        "Other",
        "#64748b",
        _svg(
            '<circle cx="12" cy="12" r="10"/>'
            '<line x1="12" y1="16" x2="12" y2="12"/>'
            '<line x1="12" y1="8" x2="12.01" y2="8"/>'
        ),
    ),
}


def generate_html_report() -> None:
    """Read knowledge DB and render a styled HTML dashboard."""
    _init_db()
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        paper_rows = conn.execute(
            "SELECT timestamp, source, summary FROM knowledge "
            "WHERE source LIKE 'arxiv:%' AND date = ? ORDER BY timestamp DESC",
            (today,),
        ).fetchall()
        github_rows = conn.execute(
            "SELECT timestamp, source, summary FROM knowledge "
            "WHERE source NOT LIKE 'arxiv:%' AND source NOT LIKE 'meta:%'"
            " AND date = ? ORDER BY timestamp DESC",
            (today,),
        ).fetchall()
        relationship_row = conn.execute(
            "SELECT summary FROM knowledge WHERE source = 'meta:relationship-map' AND date = ?",
            (today,),
        ).fetchone()
        novel_idea_row = conn.execute(
            "SELECT summary FROM knowledge WHERE source = 'meta:novel-idea' AND date = ?",
            (today,),
        ).fetchone()
    # Post-process summaries at render time (fixes old data stored before sanitization updates)
    paper_items = [
        {**dict(r), "summary": _postprocess_summary(dict(r)["summary"])} for r in paper_rows
    ]
    items = [{**dict(r), "summary": _postprocess_summary(dict(r)["summary"])} for r in github_rows]
    relationship_map = _sanitize_mermaid(relationship_row[0]) if relationship_row else ""
    novel_idea = _postprocess_summary(novel_idea_row[0]) if novel_idea_row else ""

    today_str = datetime.now().strftime("%A, %B %d, %Y")

    # Count entries per category for stats bar
    all_items = paper_items + items
    cat_counts: dict[str, int] = {}
    for item in all_items:
        cat = _source_category(item["source"])
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Build stat pills
    stat_pills = ""
    for cat, count in cat_counts.items():
        label, color, icon = _CATEGORY_META.get(cat, _CATEGORY_META["other"])
        stat_pills += (
            f'<span class="stat-pill" data-cat="{cat}" style="--accent:{color}">'
            f"{icon} {label} <strong>{count}</strong></span>"
        )

    # Build filter buttons
    filter_buttons = '<button class="filter-btn active" data-filter="all">All</button>'
    for cat in cat_counts:
        label, color, _icon = _CATEGORY_META.get(cat, _CATEGORY_META["other"])
        filter_buttons += (
            f'<button class="filter-btn" data-filter="{cat}" '
            f'style="--accent:{color}">{label}</button>'
        )

    # Build paper section
    paper_section = ""
    if paper_items:
        paper_count = len(paper_items)
        paper_icon = _CATEGORY_META["paper"][2]
        paper_section += (
            '\n        <section class="paper-section">'
            '\n            <div class="section-divider">'
            '\n                <span class="section-divider-icon">'
            f"\n                    {paper_icon} Research Papers"
            f'\n                    <span class="section-divider-count">'
            f"{paper_count} papers</span>"
            "\n                </span>"
            "\n            </div>"
        )

        for idx, item in enumerate(paper_items):
            source_label = nice_source_label(item["source"])
            color = "#0891b2"

            delay = idx * 60
            paper_section += f"""
            <article class="card" data-cat="paper"
                     style="--accent:{color};animation-delay:{delay}ms">
                <div class="card-accent"></div>
                <div class="card-inner">
                    <div class="card-header">
                        <span class="source-label">{source_label}</span>
                    </div>
                    <div class="summary">{item["summary"]}</div>
                </div>
            </article>"""

        paper_section += "\n        </section>"

    # Build cards
    cards = ""
    for idx, item in enumerate(items):
        ts = datetime.fromisoformat(item["timestamp"])
        human_time = ts.strftime("%b %d, %Y &middot; %H:%M")
        source_label = nice_source_label(item["source"])
        cat = _source_category(item["source"])
        label, color, icon = _CATEGORY_META.get(cat, _CATEGORY_META["other"])

        delay = idx * 60
        cards += f"""
        <article class="card" data-cat="{cat}"
                 style="--accent:{color};animation-delay:{delay}ms">
            <div class="card-accent"></div>
            <div class="card-inner">
                <div class="card-header">
                    <div class="card-meta">
                        <span class="badge" style="--accent:{color}">{icon} {label}</span>
                        <span class="source-label">{source_label}</span>
                    </div>
                    <time class="time">{human_time}</time>
                </div>
                <div class="summary">{item["summary"]}</div>
            </div>
        </article>"""

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
            <div style="position:relative">
                <div class="zoom-controls">
                    <button class="zoom-btn" id="zoomIn" title="Zoom in">+</button>
                    <button class="zoom-btn" id="zoomOut" title="Zoom out">&minus;</button>
                    <button class="zoom-btn" id="zoomReset" title="Fit to view">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                             stroke="currentColor" stroke-width="2.5">
                            <polyline points="15 3 21 3 21 9"/>
                            <polyline points="9 21 3 21 3 15"/>
                            <line x1="21" y1="3" x2="14" y2="10"/>
                            <line x1="3" y1="21" x2="10" y2="14"/>
                        </svg>
                    </button>
                </div>
                <div class="zoom-hint" id="zoomHint">Scroll to zoom &middot; Drag to pan</div>
                <div class="zoom-viewport" id="zoomViewport">
                    <div class="zoom-inner" id="zoomInner">
                        <pre class="mermaid" id="connMermaid">
{relationship_map}
                        </pre>
                    </div>
                </div>
            </div>"""

        if novel_idea:
            connections_section += f"""
            <div class="novel-idea">
                <div class="novel-idea-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                         stroke="#d97706" stroke-width="2">
                        <path d="M9 18h6M10 22h4M12 2a7 7 0 0 1 4 12.73V17a1 1 0
                        0 1-1 1H9a1 1 0 0 1-1-1v-2.27A7 7 0 0 1 12 2z"/>
                    </svg>
                    <span class="novel-idea-label">Sparked by Today's Connections</span>
                </div>
                <div class="novel-idea-content">{novel_idea}</div>
            </div>"""

        connections_section += """
        </section>"""

    total = len(paper_items) + len(items)
    github_count = len(items)
    gfonts = (
        "https://fonts.googleapis.com/css2"
        "?family=Inter:wght@400;500;600;700;800"
        "&family=JetBrains+Mono:wght@400;500&display=swap"
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Daily Tech Intelligence</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="{gfonts}" rel="stylesheet">
    <style>
        :root {{
            --bg-deep: #f8f7f4;
            --bg-surface: #ffffff;
            --bg-card: #ffffff;
            --bg-card-hover: #ffffff;
            --border-subtle: rgba(0, 0, 0, 0.06);
            --border-card: rgba(0, 0, 0, 0.08);
            --border-hover: rgba(0, 0, 0, 0.14);
            --text-primary: #1a1a2e;
            --text-body: #374151;
            --text-secondary: #6b7280;
            --text-muted: #9ca3af;
            --radius-lg: 16px;
            --radius-md: 12px;
            --radius-sm: 8px;
            --shadow-sm:
                0 1px 2px rgba(0,0,0,0.04),
                0 1px 3px rgba(0,0,0,0.06);
            --shadow-md:
                0 4px 12px rgba(0,0,0,0.05),
                0 1px 3px rgba(0,0,0,0.06);
            --shadow-lg:
                0 8px 24px rgba(0,0,0,0.07),
                0 2px 6px rgba(0,0,0,0.04);
            --font-sans: "Inter", system-ui,
                -apple-system, "Segoe UI", sans-serif;
            --font-mono: "JetBrains Mono", "Fira Code",
                "Cascadia Code", Consolas, monospace;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        html {{ scroll-behavior: smooth; }}

        body {{
            font-family: var(--font-sans);
            background: var(--bg-deep);
            color: var(--text-body);
            line-height: 1.7;
            min-height: 100vh;
            font-size: 16px;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}

        /* ── Ambient background ── */
        body::before {{
            content: "";
            position: fixed;
            inset: 0;
            background:
                radial-gradient(
                    ellipse 80% 50% at 10% 0%,
                    rgba(167,139,250,0.06) 0%,
                    transparent 50%
                ),
                radial-gradient(
                    ellipse 60% 40% at 90% 10%,
                    rgba(244,114,182,0.05) 0%,
                    transparent 45%
                ),
                radial-gradient(
                    ellipse 50% 40% at 50% 100%,
                    rgba(52,211,153,0.04) 0%,
                    transparent 45%
                );
            pointer-events: none;
            z-index: 0;
        }}

        .wrapper {{
            position: relative;
            z-index: 1;
            max-width: 1120px;
            margin: 0 auto;
            padding: 56px 24px 100px;
        }}

        /* ── Header ── */
        .header {{
            text-align: center;
            margin-bottom: 44px;
            padding-bottom: 36px;
            border-bottom: 1px solid var(--border-subtle);
        }}
        .header-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 64px;
            height: 64px;
            border-radius: 18px;
            background: linear-gradient(
                135deg, #6366f1, #a855f7
            );
            margin-bottom: 20px;
            box-shadow:
                0 4px 16px rgba(99,102,241,0.2),
                0 1px 3px rgba(0,0,0,0.08);
        }}
        .header-icon svg {{ color: #fff; }}
        .header-title {{
            font-size: 42px;
            font-weight: 800;
            letter-spacing: -0.035em;
            line-height: 1.15;
            background: linear-gradient(
                135deg,
                #1a1a2e 0%,
                #4f46e5 50%,
                #a855f7 100%
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }}
        .header-date {{
            font-size: 17px;
            color: var(--text-secondary);
            font-weight: 400;
            margin-bottom: 6px;
        }}
        .header-count {{
            font-size: 14px;
            color: var(--text-muted);
        }}
        .header-subtitle {{
            font-size: 15px;
            color: var(--text-secondary);
            font-weight: 400;
            margin-top: 4px;
            opacity: 0.8;
        }}

        /* ── Stats bar ── */
        .stats {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-bottom: 16px;
        }}
        .stat-pill {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 8px 16px;
            border-radius: 999px;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            font-size: 13px;
            color: color-mix(
                in srgb, var(--accent) 80%, #1a1a2e
            );
            font-weight: 500;
            transition: all 0.25s;
            box-shadow: var(--shadow-sm);
        }}
        .stat-pill:hover {{
            border-color: var(--border-hover);
            box-shadow: var(--shadow-md);
        }}
        .stat-pill strong {{
            font-weight: 700;
            font-size: 14px;
        }}
        .stat-pill svg {{ opacity: 0.85; }}

        /* ── Filter bar ── */
        .filters {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 8px;
            margin-bottom: 36px;
        }}
        .filter-btn {{
            padding: 9px 20px;
            border-radius: 999px;
            border: 1px solid var(--border-subtle);
            background: transparent;
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.25s ease;
            font-family: var(--font-sans);
        }}
        .filter-btn:hover {{
            border-color: var(--border-hover);
            color: var(--text-primary);
            background: rgba(0,0,0,0.02);
        }}
        .filter-btn.active {{
            background: var(--text-primary);
            border-color: var(--text-primary);
            color: #fff;
            font-weight: 600;
            box-shadow: var(--shadow-sm);
        }}

        /* ── Cards ── */
        .card {{
            position: relative;
            display: flex;
            margin-bottom: 20px;
            border-radius: var(--radius-lg);
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            overflow: hidden;
            transition:
                transform 0.3s cubic-bezier(.25,.8,.25,1),
                box-shadow 0.3s cubic-bezier(.25,.8,.25,1),
                border-color 0.3s ease;
            animation: fadeSlideIn 0.6s ease both;
            box-shadow: var(--shadow-sm);
        }}
        .card:hover {{
            border-color: var(--border-hover);
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }}
        .hidden {{ display: none; }}

        /* ── Paper section ── */
        .paper-section {{
            margin-bottom: 48px;
            padding: 28px 24px 24px;
            background: linear-gradient(
                135deg,
                rgba(8,145,178,0.03) 0%,
                rgba(6,182,212,0.06) 100%
            );
            border-radius: var(--radius-lg);
            border: 1px solid rgba(8,145,178,0.1);
        }}
        .section-divider {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
            padding-bottom: 14px;
            border-bottom: 2px solid rgba(8, 145, 178, 0.12);
        }}
        .section-divider-icon {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            font-size: 20px;
            font-weight: 800;
            color: #0891b2;
            letter-spacing: -0.02em;
        }}
        .section-divider-icon svg {{
            opacity: 0.9;
            width: 20px;
            height: 20px;
        }}
        .section-divider-count {{
            font-size: 13px;
            font-weight: 500;
            color: #0e7490;
            opacity: 0.7;
            margin-left: 4px;
        }}

        /* Paper cards: override h3 to be a proper title */
        .paper-section .summary h3 {{
            font-size: 19px;
            font-weight: 700;
            color: var(--text-primary);
            text-transform: none;
            letter-spacing: -0.01em;
            margin: 0 0 6px;
            padding-bottom: 0;
            border-bottom: none;
            display: block;
            line-height: 1.35;
        }}
        .summary .paper-tldr {{
            font-style: italic;
            color: var(--text-secondary);
            font-size: 14.5px;
            margin: 0 0 18px;
            line-height: 1.6;
        }}
        .summary h4 {{
            font-size: 11.5px;
            font-weight: 700;
            color: color-mix(
                in srgb, var(--accent) 85%, #1a1a2e
            );
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 20px 0 8px;
            padding-bottom: 5px;
            border-bottom: 1px solid
                color-mix(in srgb, var(--accent) 15%, transparent);
        }}

        /* Paper link pills */
        .paper-section .summary > p:last-child {{
            margin-top: 18px;
            margin-bottom: 0;
            display: flex;
            gap: 10px;
        }}
        .paper-section .summary > p:last-child a {{
            display: inline-flex;
            align-items: center;
            padding: 5px 14px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
            background: rgba(8,145,178,0.08);
            color: #0891b2;
            border: 1px solid rgba(8,145,178,0.18);
            border-bottom: 1px solid rgba(8,145,178,0.18);
            transition: all 0.2s;
        }}
        .paper-section .summary > p:last-child a:hover {{
            background: rgba(8,145,178,0.14);
            color: #0e7490;
            border-color: rgba(8,145,178,0.3);
        }}

        /* ── Section dividers ── */
        .github-section-divider {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 28px;
            padding-bottom: 14px;
            border-bottom: 2px solid rgba(0, 0, 0, 0.06);
        }}
        .github-section-divider svg {{
            opacity: 0.6;
        }}
        .github-section-divider span {{
            font-size: 20px;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }}
        .github-section-divider .section-count {{
            font-size: 13px;
            font-weight: 500;
            color: var(--text-muted);
            margin-left: 4px;
        }}

        .card-accent {{
            width: 4px;
            flex-shrink: 0;
            background: linear-gradient(
                180deg,
                var(--accent) 0%,
                color-mix(
                    in srgb,
                    var(--accent) 30%,
                    transparent
                ) 100%
            );
            opacity: 0.8;
            transition: opacity 0.3s;
        }}
        .card:hover .card-accent {{
            opacity: 1;
        }}

        .card-inner {{
            flex: 1;
            padding: 24px 28px 22px;
            min-width: 0;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 14px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }}
        .card-meta {{
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .source-label {{
            font-size: 14px;
            color: var(--text-secondary);
            font-weight: 500;
        }}
        .time {{
            font-size: 13px;
            color: var(--text-muted);
            white-space: nowrap;
            font-variant-numeric: tabular-nums;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 999px;
            background: color-mix(
                in srgb, var(--accent) 12%, #fff
            );
            color: color-mix(
                in srgb, var(--accent) 85%, #1a1a2e
            );
            font-size: 12.5px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border: 1px solid color-mix(
                in srgb, var(--accent) 20%, transparent
            );
        }}
        .badge svg {{ flex-shrink: 0; }}

        /* ── Summary content ── */
        .summary {{
            font-size: 15.5px;
            line-height: 1.75;
            color: var(--text-body);
        }}
        .summary h2 {{
            font-size: 21px;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0 0 10px;
            letter-spacing: -0.015em;
            line-height: 1.3;
        }}
        .summary h3 {{
            font-size: 13px;
            font-weight: 600;
            color: color-mix(
                in srgb, var(--accent) 80%, #1a1a2e
            );
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 22px 0 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .summary p {{
            margin: 0 0 14px;
            color: var(--text-secondary);
            font-size: 15px;
        }}
        .summary ul {{
            margin: 8px 0 14px 0;
            padding-left: 22px;
            list-style: none;
        }}
        .summary ul li {{
            position: relative;
            margin-bottom: 8px;
            padding-left: 6px;
            color: var(--text-body);
            font-size: 15px;
            line-height: 1.7;
        }}
        .summary ul li::before {{
            content: "";
            position: absolute;
            left: -15px;
            top: 11px;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: color-mix(
                in srgb,
                var(--accent, #6b7280) 70%,
                #1a1a2e
            );
            opacity: 0.7;
        }}
        .summary ul li strong {{
            color: var(--text-body);
            font-weight: 550;
        }}
        .summary ul li > strong:first-child {{
            color: var(--text-primary);
            font-weight: 600;
        }}
        .summary a {{
            color: #4338ca;
            text-decoration: none;
            border-bottom:
                1px solid rgba(67,56,202,0.2);
            transition: all 0.2s;
            font-weight: 500;
        }}
        .summary a:hover {{
            color: #3730a3;
            border-color: rgba(55,48,163,0.5);
        }}

        /* ── Code blocks ── */
        .summary pre {{
            background: #f4f4f5;
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: var(--radius-md);
            padding: 18px 22px;
            overflow-x: auto;
            font-size: 14px;
            line-height: 1.65;
            margin: 16px 0;
            position: relative;
            color: #1e1e2e;
        }}
        .summary pre::before {{
            content: "code";
            position: absolute;
            top: 8px;
            right: 12px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
            opacity: 0.7;
            font-family: var(--font-sans);
        }}
        .summary code {{
            font-family: var(--font-mono);
            font-size: 0.92em;
        }}
        .summary :not(pre) > code {{
            background: rgba(99,102,241,0.08);
            padding: 3px 7px;
            border-radius: 5px;
            font-size: 0.88em;
            color: #4338ca;
            border: 1px solid rgba(99,102,241,0.1);
        }}

        /* ── Spacing between cards ── */
        .card + .card {{
            margin-top: 20px;
        }}

        /* ── Footer ── */
        .footer {{
            margin-top: 56px;
            padding: 28px 0 0;
            border-top: 1px solid var(--border-subtle);
            text-align: center;
            font-size: 13px;
            color: var(--text-muted);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
        }}

        /* ── Scroll to top ── */
        .scroll-top {{
            position: fixed;
            bottom: 32px;
            right: 32px;
            width: 44px;
            height: 44px;
            border-radius: var(--radius-md);
            background: var(--bg-surface);
            border: 1px solid var(--border-card);
            color: var(--text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transform: translateY(12px);
            transition: all 0.35s;
            z-index: 100;
            box-shadow: var(--shadow-md);
        }}
        .scroll-top.visible {{
            opacity: 1;
            transform: translateY(0);
        }}
        .scroll-top:hover {{
            background: var(--text-primary);
            color: #fff;
            border-color: var(--text-primary);
            box-shadow: var(--shadow-lg);
        }}

        /* ── Animations ── */
        @keyframes fadeSlideIn {{
            from {{
                opacity: 0;
                transform: translateY(24px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        /* ── Responsive ── */
        @media (max-width: 640px) {{
            .wrapper {{
                padding: 28px 14px 64px;
            }}
            .header-title {{ font-size: 30px; }}
            .header-icon {{
                width: 52px;
                height: 52px;
            }}
            .card-inner {{ padding: 18px 16px; }}
            .card-header {{
                flex-direction: column;
                gap: 8px;
            }}
            .summary {{ font-size: 15px; }}
            .summary h2 {{ font-size: 19px; }}
            .paper-section .summary h3 {{ font-size: 17px; }}
            .filter-btn {{
                padding: 8px 14px;
            }}
            .paper-section {{
                padding: 20px 16px 18px;
                margin-bottom: 36px;
            }}
            .section-divider-icon {{
                font-size: 17px;
            }}
        }}

        /* ── Mermaid diagrams ── */
        .summary pre.mermaid {{
            background: linear-gradient(
                135deg,
                rgba(99,102,241,0.03) 0%,
                rgba(168,85,247,0.05) 50%,
                rgba(244,114,182,0.03) 100%
            );
            border: 1px solid rgba(167,139,250,0.15);
            border-radius: var(--radius-lg);
            padding: 32px 24px;
            margin: 20px 0;
            text-align: center;
            color: var(--text-body);
            overflow-x: auto;
        }}
        .summary pre.mermaid::before {{
            content: "diagram";
            color: #a78bfa;
            font-weight: 600;
        }}
        .summary pre.mermaid svg {{
            max-width: 100%;
            height: auto;
        }}
        /* Node styling */
        pre.mermaid .node rect,
        pre.mermaid .node circle,
        pre.mermaid .node ellipse,
        pre.mermaid .node polygon {{
            rx: 10px !important;
            ry: 10px !important;
            filter: drop-shadow(0 2px 4px rgba(99,102,241,0.12));
            stroke-width: 1.5px !important;
        }}
        pre.mermaid .node .label {{
            font-weight: 500;
            font-size: 13px !important;
        }}
        /* Edge styling */
        pre.mermaid .edgePath .path {{
            stroke-width: 1.8px !important;
            opacity: 0.75;
        }}
        pre.mermaid .edgeLabel {{
            font-size: 11px !important;
            font-weight: 500;
            background-color: rgba(255,255,255,0.9) !important;
            padding: 2px 6px;
            border-radius: 4px;
        }}
        pre.mermaid marker {{
            fill: #a78bfa;
        }}
        /* Subgraph styling */
        pre.mermaid .cluster rect {{
            rx: 12px !important;
            ry: 12px !important;
            stroke-dasharray: none !important;
            filter: drop-shadow(0 1px 3px rgba(0,0,0,0.06));
        }}
        pre.mermaid .cluster .nodeLabel {{
            font-weight: 700;
            font-size: 12px !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
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
            box-shadow: var(--shadow-sm);
        }}
        .connections-section pre.mermaid {{
            background: rgba(255,255,255,0.6);
            border: 1px solid rgba(167,139,250,0.12);
            border-radius: var(--radius-lg);
            padding: 0;
            margin: 0;
            text-align: center;
        }}
        .connections-section pre.mermaid::before {{
            content: none;
        }}
        .connections-section .zoom-viewport {{
            width: 100%;
            height: 500px;
            overflow: hidden;
            position: relative;
            cursor: grab;
        }}
        .connections-section .zoom-viewport:active {{
            cursor: grabbing;
        }}
        .connections-section .zoom-viewport .zoom-inner {{
            transform-origin: 0 0;
            display: inline-block;
        }}
        .connections-section .zoom-controls {{
            position: absolute;
            top: 12px;
            right: 12px;
            display: flex;
            gap: 4px;
            z-index: 10;
        }}
        .connections-section .zoom-btn {{
            width: 32px;
            height: 32px;
            border-radius: 8px;
            border: 1px solid rgba(167,139,250,0.2);
            background: rgba(255,255,255,0.9);
            color: #6366f1;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s;
            font-family: var(--font-sans);
            backdrop-filter: blur(8px);
        }}
        .connections-section .zoom-btn:hover {{
            background: #ede9fe;
            border-color: #a78bfa;
        }}
        .connections-section .zoom-hint {{
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 11px;
            color: #9ca3af;
            background: rgba(255,255,255,0.85);
            padding: 3px 10px;
            border-radius: 6px;
            pointer-events: none;
            z-index: 10;
            backdrop-filter: blur(4px);
            opacity: 1;
            transition: opacity 0.5s;
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

        /* ── Novel idea ── */
        .novel-idea {{
            margin-top: 24px;
            padding: 24px 28px;
            background: linear-gradient(
                135deg,
                rgba(251,191,36,0.06) 0%,
                rgba(245,158,11,0.08) 100%
            );
            border: 1px solid rgba(217,119,6,0.15);
            border-radius: var(--radius-md);
        }}
        .novel-idea-icon {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(217,119,6,0.12);
        }}
        .novel-idea-label {{
            font-size: 14px;
            font-weight: 700;
            color: #b45309;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .novel-idea-content {{
            font-size: 15px;
            line-height: 1.75;
            color: var(--text-body);
        }}
        .novel-idea-content h4 {{
            font-size: 13px;
            font-weight: 700;
            color: #92400e;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 18px 0 8px;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(217,119,6,0.1);
        }}
        .novel-idea-content h4:first-child {{
            margin-top: 0;
        }}
        .novel-idea-content .idea-title {{
            font-size: 18px;
            margin: 0 0 10px;
        }}
        .novel-idea-content .idea-title strong {{
            color: var(--text-primary);
            font-weight: 700;
        }}
        .novel-idea-content p {{
            margin: 0 0 12px;
            color: var(--text-secondary);
        }}
        .novel-idea-content ul {{
            margin: 8px 0 0 0;
            padding-left: 20px;
            list-style: none;
        }}
        .novel-idea-content ul li {{
            position: relative;
            padding-left: 6px;
            margin-bottom: 6px;
            color: var(--text-body);
            font-size: 14.5px;
            line-height: 1.65;
        }}
        .novel-idea-content ul li::before {{
            content: "";
            position: absolute;
            left: -14px;
            top: 10px;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #d97706;
            opacity: 0.6;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
    <script>
      mermaid.initialize({{
        startOnLoad: true,
        theme: 'base',
        flowchart: {{
          curve: 'basis',
          padding: 16,
          nodeSpacing: 40,
          rankSpacing: 50,
          htmlLabels: true,
          useMaxWidth: true,
        }},
        themeVariables: {{
          fontFamily: 'Inter, system-ui, sans-serif',
          fontSize: '13px',
          primaryColor: '#ede9fe',
          primaryTextColor: '#312e81',
          primaryBorderColor: '#8b5cf6',
          secondaryColor: '#fce7f3',
          secondaryTextColor: '#831843',
          secondaryBorderColor: '#ec4899',
          tertiaryColor: '#d1fae5',
          tertiaryTextColor: '#064e3b',
          tertiaryBorderColor: '#34d399',
          lineColor: '#8b5cf6',
          textColor: '#374151',
          nodeTextColor: '#1e1b4b',
          mainBkg: '#ede9fe',
          nodeBorder: '#8b5cf6',
          clusterBkg: 'rgba(245,243,255,0.7)',
          clusterBorder: '#a78bfa',
          edgeLabelBackground: 'rgba(255,255,255,0.95)',
          arrowheadColor: '#8b5cf6',
        }}
      }});
    </script>
</head>
<body>
    <div class="wrapper">
        <header class="header">
            <div class="header-icon">
                <svg width="30" height="30"
                     viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" stroke-width="1.5">
                    <path d="M13 2L3 14h9l-1 8
                             10-12h-9l1-8z"/>
                </svg>
            </div>
            <h1 class="header-title">Daily Tech Intelligence</h1>
            <p class="header-date">{today_str}</p>
            <p class="header-count">{total} briefings today</p>
        </header>

        <div class="stats">{stat_pills}</div>
        <nav class="filters">{filter_buttons}</nav>

{paper_section}

        <section class="github-section">
        <div class="github-section-divider">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37
                3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54
                6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07
                0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38
                0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07
                0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0
                5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9
                18.13V22"/>
            </svg>
            <span>GitHub Activity</span>
            <span class="section-count">{github_count} sources</span>
        </div>

{cards}
        </section>

{connections_section}

        <footer class="footer">
            <span>Generated by daily_tech</span>
        </footer>
    </div>

    <button class="scroll-top" id="scrollTop"
            title="Scroll to top">
        <svg width="20" height="20"
             viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5">
            <polyline points="18 15 12 9 6 15"/>
        </svg>
    </button>

    <script>
        /* ── Filter logic ── */
        const btns = document.querySelectorAll('.filter-btn');
        const cards = document.querySelectorAll('.card');
        const paperSec = document.querySelector('.paper-section');
        const ghSec = document.querySelector('.github-section');

        function applyFilter(f) {{
            cards.forEach(c => {{
                const hide = f !== 'all' && c.dataset.cat !== f;
                c.classList.toggle('hidden', hide);
            }});
            if (paperSec) {{
                paperSec.classList.toggle('hidden',
                    f !== 'all' && f !== 'paper');
            }}
            if (ghSec) {{
                ghSec.classList.toggle('hidden',
                    f === 'paper');
            }}
            const connSec = document.querySelector('.connections-section');
            if (connSec) {{
                connSec.classList.toggle('hidden', f !== 'all');
            }}
        }}

        btns.forEach(btn => {{
            btn.addEventListener('click', () => {{
                btns.forEach(b =>
                    b.classList.remove('active'));
                btn.classList.add('active');
                applyFilter(btn.dataset.filter);
            }});
        }});

        /* ── Scroll-to-top button ── */
        const sBtn = document.getElementById('scrollTop');
        window.addEventListener('scroll', () => {{
            sBtn.classList.toggle('visible',
                window.scrollY > 400);
        }});
        sBtn.addEventListener('click', () => {{
            window.scrollTo({{
                top: 0, behavior: 'smooth'
            }});
        }});

        /* ── Open all summary links in new tab ── */
        document.querySelectorAll('.summary a')
            .forEach(a => {{
                a.setAttribute('target', '_blank');
                a.setAttribute(
                    'rel', 'noopener noreferrer'
                );
            }});

        /* ── Pan & Zoom for connections diagram ── */
        (function() {{
            const vp = document.getElementById('zoomViewport');
            const inner = document.getElementById('zoomInner');
            if (!vp || !inner) return;

            let scale = 1, panX = 0, panY = 0, dragging = false;
            let startX, startY, startPanX, startPanY;
            const MIN_SCALE = 0.2, MAX_SCALE = 20;

            function apply() {{
                inner.style.transform =
                    `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
            }}

            function fitToView() {{
                /* Reset transform to measure natural size */
                inner.style.transform = 'none';
                const rect = inner.getBoundingClientRect();
                const vpW = vp.clientWidth;
                const vpH = vp.clientHeight;
                const natW = rect.width;
                const natH = rect.height;
                if (natW === 0 || natH === 0) return;
                scale = Math.min(vpW / natW, vpH / natH) * 0.92;
                panX = (vpW - natW * scale) / 2;
                panY = (vpH - natH * scale) / 2;
                apply();
            }}

            /* Wait for Mermaid to render, then fit */
            const observer = new MutationObserver(() => {{
                if (inner.querySelector('svg')) {{
                    observer.disconnect();
                    setTimeout(fitToView, 100);
                }}
            }});
            observer.observe(inner, {{ childList: true, subtree: true }});

            /* Wheel zoom */
            vp.addEventListener('wheel', (e) => {{
                e.preventDefault();
                const rect = vp.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                const ns = Math.min(MAX_SCALE,
                    Math.max(MIN_SCALE, scale * delta));
                const r = ns / scale;
                panX = mx - r * (mx - panX);
                panY = my - r * (my - panY);
                scale = ns;
                apply();
                const hint = document.getElementById('zoomHint');
                if (hint) hint.style.opacity = '0';
            }}, {{ passive: false }});

            /* Drag to pan */
            vp.addEventListener('mousedown', (e) => {{
                dragging = true;
                startX = e.clientX; startY = e.clientY;
                startPanX = panX; startPanY = panY;
                e.preventDefault();
            }});
            window.addEventListener('mousemove', (e) => {{
                if (!dragging) return;
                panX = startPanX + (e.clientX - startX);
                panY = startPanY + (e.clientY - startY);
                apply();
            }});
            window.addEventListener('mouseup', () => {{ dragging = false; }});

            /* Button controls */
            function zoomAt(factor) {{
                const cx = vp.clientWidth / 2, cy = vp.clientHeight / 2;
                const ns = Math.min(MAX_SCALE,
                    Math.max(MIN_SCALE, scale * factor));
                const r = ns / scale;
                panX = cx - r * (cx - panX);
                panY = cy - r * (cy - panY);
                scale = ns;
                apply();
            }}
            const zi = document.getElementById('zoomIn');
            const zo = document.getElementById('zoomOut');
            const zr = document.getElementById('zoomReset');
            if (zi) zi.addEventListener('click', () => zoomAt(1.3));
            if (zo) zo.addEventListener('click', () => zoomAt(0.75));
            if (zr) zr.addEventListener('click', fitToView);
        }})();
    </script>
</body>
</html>"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    log.info("HTML report generated: %s", REPORT_PATH)


def _sources_fetched_today() -> set[str]:
    """Return source URLs that already have entries for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT source FROM knowledge WHERE date = ?",
            (today,),
        ).fetchall()
    return {r[0] for r in rows}


def fetch_and_process(days: int = 7) -> None:
    """Fetch data from all sources and generate LLM summaries, skipping today's dupes."""
    _init_db()
    date_cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    already_done = _sources_fetched_today()

    with _build_client() as client:
        for url in tqdm(SOURCES, desc="Fetching sources"):
            resolved_url = url.format(date=date_cutoff) if "{date}" in url else url

            if resolved_url in already_done:
                log.info("Skipping (already fetched today): %s", resolved_url)
                continue

            try:
                resp = client.get(resolved_url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("Failed to fetch %s: %s", resolved_url, exc)
                continue

            summary = summarize(resp.text, resolved_url)
            store(
                KnowledgeEntry(
                    timestamp=datetime.now().isoformat(),
                    source=resolved_url,
                    summary=summary,
                )
            )


if __name__ == "__main__":
    fetch_and_process_papers()
    fetch_and_process(days=7)
    generate_relationship_map()
    generate_html_report()
    print("Daily knowledge added.")
    os.startfile(REPORT_PATH)
