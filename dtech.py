from __future__ import annotations

import logging
import os
import sqlite3
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
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeEntry:
    timestamp: str
    source: str
    summary: str


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

Keep it concise and skimmable.{hint}
Here is the raw data to summarize:
{text}
"""
    response = model.invoke(prompt)
    return response.content


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


def nice_source_label(url: str) -> str:
    """Turn a GitHub API URL into a readable label, e.g. 'anthropics/claude-code · releases'."""
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
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT timestamp, source, summary FROM knowledge ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()
    data = [dict(r) for r in rows]

    today_str = datetime.now().strftime("%A, %B %d, %Y")
    items = data[:20]

    # Count entries per category for stats bar
    cat_counts: dict[str, int] = {}
    for item in items:
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

    total = len(items)
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
            max-width: 920px;
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
        .card.hidden {{ display: none; }}

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
            .filter-btn {{
                padding: 8px 14px;
            }}
        }}
    </style>
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
            <p class="header-count">{total} briefings</p>
        </header>

        <div class="stats">{stat_pills}</div>
        <nav class="filters">{filter_buttons}</nav>

{cards}

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
        btns.forEach(btn => {{
            btn.addEventListener('click', () => {{
                btns.forEach(b =>
                    b.classList.remove('active'));
                btn.classList.add('active');
                const f = btn.dataset.filter;
                cards.forEach(c => {{
                    const hide = f !== 'all'
                        && c.dataset.cat !== f;
                    c.classList.toggle('hidden', hide);
                }});
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
    fetch_and_process(days=7)
    generate_html_report()
    print("🧠 Daily knowledge added.")
    os.startfile(REPORT_PATH)
