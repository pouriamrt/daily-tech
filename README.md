# Daily Tech Intelligence

Automated daily tech briefing that fetches trending GitHub repos and release notes, summarizes them with an LLM, and generates a polished HTML dashboard.

![Architecture](architecture.png)

## What It Does

Pulls data from **9 GitHub API sources** every day, asks an LLM to distill the raw JSON into concise developer briefings, stores everything in a local SQLite database, and renders a filterable HTML report that opens in your browser.

### Sources

| Category | What's Tracked |
|----------|---------------|
| Trending Python | Newly created Python repos by stars |
| Machine Learning | Repos tagged `machine-learning` |
| LLM & Agents | Repos tagged `llm` |
| Google ADK | Repos tagged `google-adk` |
| Releases | FastAPI, Transformers, LangChain, Google ADK, Claude Code |

### Features

- **LLM-powered summaries** with key releases, upgrade notes, and quick code examples
- **Per-source hints** that tailor the summarization (e.g. FastAPI gets SSE examples, ADK gets agent snippets)
- **SQLite storage** with `UNIQUE(source, date)` dedup -- same source won't be re-processed on the same day
- **HTTP caching** via hishel to avoid redundant GitHub API calls
- **Light-themed HTML dashboard** with category filters, color-coded cards, staggered animations, and responsive design

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- GitHub token (for API rate limits)
- OpenAI API key (for LLM summarization)

### Setup

```bash
git clone https://github.com/pouriamrt/daily-tech.git
cd daily-tech
uv sync
```

Create a `.env` file:

```
GITHUB_TOKEN=ghp_your_token_here
OPENAI_API_KEY=sk-your_key_here
```

### Run

```bash
uv run python dtech.py
```

This will:
1. Fetch all 9 sources (skipping any already fetched today)
2. Summarize each with the LLM
3. Store results in `knowledge.db`
4. Generate `daily_report.html`
5. Open the report in your browser

## Project Structure

```
daily-tech/
  dtech.py            # Single-file application (all logic)
  pyproject.toml       # Dependencies and tool config
  .env                 # API keys (gitignored)
  knowledge.db         # SQLite knowledge store (gitignored)
  daily_report.html    # Generated report (gitignored)
```

## Configuration

All configuration lives at the top of `dtech.py`:

- **`SOURCES`** -- tuple of GitHub API URLs to fetch
- **`SUMMARY_HINTS`** -- per-source extra instructions for the LLM (e.g. "add a Quick Example section")
- **`model`** -- LLM model used for summarization (default: `gpt-5.4-nano`)

## License

MIT
