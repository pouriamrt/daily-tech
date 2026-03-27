from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import hishel
import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from tqdm import tqdm

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "knowledge.json"
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


def _build_client() -> hishel.CacheClient:
    """Create an httpx client with transparent HTTP caching via hishel."""
    storage = hishel.FileStorage(base_path=str(BASE_DIR / ".http_cache"))
    return hishel.CacheClient(storage=storage, headers=HEADERS, timeout=30.0)


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
    """Append a knowledge entry to the JSON database."""
    try:
        data: list[dict[str, str]] = json.loads(DB_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(asdict(entry))
    DB_PATH.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


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


def generate_html_report() -> None:
    """Read knowledge.json and render a styled HTML dashboard."""
    data: list[dict[str, str]] = json.loads(DB_PATH.read_text(encoding="utf-8"))
    data = sorted(data, key=lambda x: x["timestamp"], reverse=True)

    today_str = datetime.now().strftime("%A, %B %d, %Y")

    cards = ""
    for item in data[:20]:
        ts = datetime.fromisoformat(item["timestamp"])
        human_time = ts.strftime("%Y-%m-%d %H:%M")
        source_label = nice_source_label(item["source"])

        cards += f"""
        <div class="card">
            <div class="card-header">
                <div>
                    <div class="source-label">{source_label}</div>
                    <div class="badge"><span class="badge-dot"></span>GitHub</div>
                </div>
                <div class="time">{human_time}</div>
            </div>
            <div class="source-url">{item["source"]}</div>
            <div class="summary">{item["summary"]}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Daily Tech Intelligence</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: radial-gradient(circle at top left, #1e293b 0, #020617 55%);
            color: #e5e7eb;
        }}
        .wrapper {{ max-width: 980px; margin: 0 auto; padding: 32px 16px 56px; }}
        .header-title {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
        .header-subtitle {{ font-size: 14px; color: #9ca3af; margin-bottom: 24px; }}
        .card {{
            background: rgba(15,23,42,0.96);
            padding: 20px 24px;
            margin-bottom: 20px;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.3);
            box-shadow: 0 18px 45px rgba(0,0,0,0.55);
        }}
        .card-header {{
            display: flex; justify-content: space-between;
            align-items: baseline; gap: 12px; margin-bottom: 10px;
        }}
        .source-label {{
            font-size: 13px; color: #a5b4fc;
            text-transform: uppercase; letter-spacing: 0.16em;
        }}
        .time {{ font-size: 12px; color: #6ee7b7; white-space: nowrap; }}
        .source-url {{
            font-size: 12px; color: #38bdf8;
            word-break: break-all; margin-bottom: 8px;
        }}
        .summary {{ margin-top: 4px; line-height: 1.6; font-size: 14px; }}
        .summary h2, .summary h3 {{ color: #e5e7eb; margin: 10px 0 4px; }}
        .summary ul {{ margin: 4px 0 8px 20px; padding: 0; }}
        .summary li {{ margin-bottom: 4px; }}
        .summary a {{ color: #38bdf8; text-decoration: none; }}
        .summary a:hover {{ text-decoration: underline; }}
        .summary pre {{
            background: rgba(0,0,0,0.4); border-radius: 8px;
            padding: 12px 16px; overflow-x: auto; font-size: 13px;
        }}
        .summary code {{ font-family: "Fira Code", Consolas, monospace; }}
        .badge {{
            display: inline-flex; align-items: center;
            padding: 2px 8px; border-radius: 999px;
            background: rgba(56,189,248,0.1); color: #7dd3fc;
            font-size: 11px; font-weight: 500; gap: 4px;
        }}
        .badge-dot {{ width: 6px; height: 6px; border-radius: 999px; background: #38bdf8; }}
        .footer {{ margin-top: 18px; font-size: 11px; color: #6b7280; text-align: center; }}
    </style>
</head>
<body>
    <div class="wrapper">
        <div class="header-title">Daily Tech Intelligence</div>
        <div class="header-subtitle">{today_str}</div>
{cards}
        <div class="footer">Generated automatically by daily_tech.</div>
    </div>
</body>
</html>"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    log.info("HTML report generated: %s", REPORT_PATH)


def fetch_and_process(days: int = 7) -> None:
    """Fetch data from all sources and generate LLM summaries."""
    date_cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    with _build_client() as client:
        for url in tqdm(SOURCES, desc="Fetching sources"):
            resolved_url = url.format(date=date_cutoff) if "{date}" in url else url
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
    fetch_and_process(days=2)
    generate_html_report()
    print("🧠 Daily knowledge added.")
    os.startfile("C:/Users/pouri/Python/AI/Test/daily_tech")
