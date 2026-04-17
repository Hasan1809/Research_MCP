# Research MCP — Architecture

> Last updated: 2026-04-17
> Update this file whenever a tool is added/removed, a service is changed, a new external dependency is introduced, or the data flow changes.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Directory Structure](#directory-structure)
4. [System Diagram](#system-diagram)
5. [MCP Tools](#mcp-tools)
6. [Services](#services)
7. [Data Flow Diagrams](#data-flow-diagrams)
8. [Data Storage](#data-storage)
9. [Logging System](#logging-system)
10. [Environment Variables](#environment-variables)
11. [Tool Dependency Map](#tool-dependency-map)
12. [Changelog](#changelog)

---

## Overview

A **Model Context Protocol (MCP)** server that gives an AI assistant (Claude Desktop) a set of research tools: searching academic APIs, ingesting PDFs, indexing them into a vector database, extracting structured insights with an LLM, detecting cross-paper research gaps, suggesting concrete experiments, and organizing papers into named projects.

The server runs as a local process. Claude Desktop connects to it over stdio. Every tool is a Python function registered with **FastMCP**.

---

## Tech Stack

| Layer | Technology | Version / Notes |
|---|---|---|
| MCP framework | `mcp` / FastMCP | `mcp.server.fastmcp.FastMCP` |
| LLM inference | IONOS AI Model Hub | OpenAI-compatible `/chat/completions` endpoint |
| LLM model | **Llama 3.3 70B Instruct** | Configured via `IONOS_MODEL` env var |
| Vector database | **ChromaDB** (persistent) | HNSW index, cosine distance |
| Embeddings | ChromaDB `DefaultEmbeddingFunction` | Local, no external call |
| PDF parsing | **pypdf** | Page-by-page text extraction |
| HTTP client | **httpx** | Sync, used for all external HTTP |
| Paper sources | **arXiv API** (Atom/XML) | `export.arxiv.org/api/query` |
| Paper sources | **Semantic Scholar API** | `api.semanticscholar.org/graph/v1` |
| Config | **python-dotenv** | `.env` loaded before all imports in `main.py` |
| Language | Python 3.11+ | Type hints throughout |

---

## Directory Structure

```
research_mcp/
├── server/
│   ├── main.py                        # Entry point — registers all 13 tools
│   ├── tools/                         # MCP tool functions (thin wrappers)
│   │   ├── search_papers.py
│   │   ├── analyze_papers.py
│   │   ├── ingest_paper.py
│   │   ├── extract_paper_insights.py
│   │   ├── index_paper.py
│   │   ├── retrieve_chunks.py
│   │   ├── synthesize_papers.py
│   │   ├── build_paper_profile.py
│   │   ├── detect_gaps.py
│   │   ├── manage_project.py          # create/add/list project tools
│   │   └── suggest_experiments.py
│   ├── services/
│   │   ├── retrieval/
│   │   │   ├── aggregator.py          # Merges arXiv + S2 results
│   │   │   ├── arxiv_service.py       # arXiv API client (with retry)
│   │   │   ├── semantic_scholar_service.py  # S2 API client (with retry)
│   │   │   └── vector_store.py        # ChromaDB read/write
│   │   ├── documents/
│   │   │   ├── pdf_service.py         # PDF download, section detection, cache
│   │   │   └── chunking.py            # Flat + section-aware chunking
│   │   ├── extraction/
│   │   │   └── llm_extractor.py       # All LLM prompts and calls
│   │   └── analysis/
│   │       ├── synthesis.py           # LLM analysis + statistical synthesis
│   │       ├── gap_detector.py        # LLM-powered gap detection
│   │       └── experiment_suggester.py  # LLM-powered experiment proposals
│   ├── services/
│   │   └── project_manager.py         # Project manifest CRUD
│   └── utils/
│       └── logger.py                  # Session logging + tool invocation logs
├── data/
│   ├── papers/                        # Cached ingested papers (JSON)
│   │   └── arxiv/<paper_id>.json
│   ├── insights/                      # Extracted insights (JSON)
│   │   └── arxiv/<paper_id>.json
│   ├── profiles/                      # Rich paper profiles (JSON)
│   │   └── arxiv/<paper_id>.json
│   ├── projects/                      # Project manifests (JSON)
│   │   └── <project-name>.json
│   ├── analysis/                      # Analysis outputs (JSON)
│   │   ├── gap_analysis_<timestamp>_<ids>.json
│   │   └── experiments_<timestamp>_<ids>.json
│   └── chroma/                        # ChromaDB persistent storage
│       ├── arxiv_papers/              # Content chunks (cosine)
│       ├── arxiv_papers_refs/         # Reference/bibliography chunks (L2)
│       └── arxiv_papers_sections/     # Section-aware chunks (cosine)
├── logs/
│   └── session_YYYYMMDD_HHMMSS/
│       ├── main.log                   # Full DEBUG-level session log
│       └── tools/
│           └── NNN_toolname.json      # Per-invocation record
└── tests/
    └── test_section_detection.py
```

---

## System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        Claude Desktop                             │
│                  (MCP client, calls tools by name)               │
└─────────────────────────────┬────────────────────────────────────┘
                              │ stdio (MCP protocol)
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    research-agent MCP Server                      │
│                        (server/main.py)                           │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                        Tools Layer (13 tools)               │  │
│  │  search  analyze  ingest  extract  index  retrieve          │  │
│  │  synthesize  build_profile  detect_gaps                     │  │
│  │  create_project  add_to_project  list_projects              │  │
│  │  suggest_experiments                                         │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
│                             │                                     │
│  ┌──────────────────────────▼─────────────────────────────────┐  │
│  │                      Services Layer                          │  │
│  │                                                              │  │
│  │  retrieval/         documents/         extraction/           │  │
│  │  ├ aggregator       ├ pdf_service      └ llm_extractor       │  │
│  │  ├ arxiv_service    └ chunking                               │  │
│  │  ├ semantic_scholar                   analysis/              │  │
│  │  └ vector_store                       ├ synthesis            │  │
│  │                                       ├ gap_detector         │  │
│  │  project_manager.py                   └ experiment_suggester │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
        │                │                    │
        ▼                ▼                    ▼
  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐
  │ External │   │  Local Disk  │   │    IONOS AI Hub     │
  │   APIs   │   │   (data/)    │   │  Llama 3.3 70B      │
  │ arXiv    │   │ ChromaDB     │   │  /chat/completions  │
  │ S2       │   │ JSON cache   │   └────────────────────┘
  └──────────┘   └──────────────┘
  (retry/backoff)
```

---

## MCP Tools

Each tool is a Python function registered via `mcp.tool()(fn)` in `main.py`. Tools are thin wrappers — they validate inputs, call services, handle logging, and return results to the MCP client.

### `search_papers_tool`
**File:** `tools/search_papers.py`
**Input:** `query: str, limit: int`
**Output:** `list[dict]` — each item has `paper_id, title, abstract, year, authors, source`
**What it does:** Queries arXiv and Semantic Scholar (concatenated), returns combined results. `paper_id` is the arXiv ID (e.g. `2401.04088`) — used to chain into `ingest_paper_tool`. Both APIs use retry with exponential backoff on 429/errors.

### `analyze_papers_tool`
**File:** `tools/analyze_papers.py`
**Input:** `papers: list[dict]` — raw search results
**Output:** `dict` with LLM-identified `themes[], trends[], methodology_landscape, paper_count, year_range`; falls back to word-frequency `themes[], common_topics[], limitations[], possible_gaps[]` if LLM unavailable
**What it does:** LLM-powered analysis of titles and abstracts. Themes reference specific paper titles. Falls back to `_analyze_papers_fallback()` (word counting) if IONOS is unreachable or returns bad JSON.

### `ingest_paper_tool`
**File:** `tools/ingest_paper.py`
**Input:** `paper_id: str, source: str` (only `"arxiv"` supported)
**Output:** `dict` with `paper_id, source, pdf_url, text_length, chunk_count, sections, section_chunks, metadata`
**What it does:**
1. Checks disk cache (`data/papers/arxiv/<id>.json`) — returns immediately if found
2. Downloads PDF from `https://arxiv.org/pdf/<paper_id>`
3. Extracts text page-by-page with pypdf
4. Detects section boundaries using heuristics (`_is_heading`)
5. Produces flat chunks (`chunk_text`) and section-aware chunks (`chunk_sections`)
6. Saves full result to disk cache

### `index_paper_tool`
**File:** `tools/index_paper.py`
**Input:** `paper_id: str, source: str`
**Output:** `dict` with chunk counts
**What it does:** Loads from disk cache, then:
- Upserts **flat chunks** into `arxiv_papers` ChromaDB collection (content) and `arxiv_papers_refs` (references, separated by regex)
- Upserts **section-aware chunks** into `arxiv_papers_sections` collection with rich metadata (`section`, `is_abstract`, `is_conclusion`, `position_ratio`)
- Must run after `ingest_paper_tool`

### `retrieve_chunks_tool`
**File:** `tools/retrieve_chunks.py`
**Input:** `query: str, paper_id: str, source: str, k: int`
**Output:** `list[dict]` — each item has `chunk_index, text, distance`
**What it does:** Semantic similarity search over the flat content collection for a specific paper. Distance is cosine distance (0 = identical, 2 = opposite). Must run after `index_paper_tool`.

### `extract_paper_insights_tool`
**File:** `tools/extract_paper_insights.py`
**Input:** `paper_id: str, source: str`
**Output:** `dict` with `methods, results, datasets, limitations, future_work, debug_notes`
**What it does:** Two-tier LLM extraction:
- **Primary (≤80k chars):** sends full section-reconstructed text to LLM in a single pass
- **Fallback (>80k chars):** per-field chunk retrieval + separate LLM call per field
- Saves to `data/insights/arxiv/<id>.json`
- Must run after `ingest_paper_tool` (fallback also requires `index_paper_tool`)

### `build_paper_profile_tool`
**File:** `tools/build_paper_profile.py`
**Input:** `paper_id: str, source: str`
**Output:** `dict` — 13-field rich profile
**What it does:** Three-tier context selection, then a single LLM call:
1. **Full text (≤80k chars):** send everything
2. **Priority sections (>80k):** extract sections matching `abstract, introduction, conclusion, discussion, method, approach, related work` and join them — if that fits under 80k, use it
3. **Chunk retrieval (last resort):** 3 retrieval queries → top 15 deduplicated chunks
- Saves to `data/profiles/arxiv/<id>.json`

**Profile schema:**
```
paper_type, research_problem, main_contribution, methods_or_approach,
key_findings, paper_intent, core_insight, paper_stance,
distinctive_elements[], datasets[], limitations[], future_work[],
plain_english_summary
```

### `synthesize_papers_tool`
**File:** `tools/synthesize_papers.py`
**Input:** `papers: list[dict]` OR `project: str` (project takes precedence if both given)
**Output:** `dict` with `common_methods, common_datasets, recurring_limitations, common_findings, notable_differences`
**What it does:** Statistical cross-paper synthesis. Prefers profiles over insights when both exist. Finds items appearing in ≥ half the papers (min 2). No LLM.

### `detect_gaps_tool`
**File:** `tools/detect_gaps.py`
**Input:** `papers: list[dict]` OR `project: str` (project takes precedence if both given); minimum 2 papers
**Output:** `dict` with `research_gaps, methodological_gaps, contradictions, connections, field_summary`
**What it does:** Loads profiles (or insights as fallback) for all papers, formats them, sends to LLM for gap analysis. Each gap/contradiction/connection references specific `paper_id`s. Saves to `data/analysis/gap_analysis_*.json`.
- LLM is instructed **not** to restate authors' own `future_work`/`limitations` — only gaps visible from cross-paper comparison.
- Must run `build_paper_profile_tool` (or `extract_paper_insights_tool`) on each paper first.

### `suggest_experiments_tool`
**File:** `tools/suggest_experiments.py`
**Input:** `papers: list[dict]` OR `project: str` (project takes precedence if both given); minimum 2 papers
**Output:** `dict` with `gaps` (full gap analysis) and `experiments[]`
**What it does:** Runs gap detection internally, then sends the gap analysis + all profiles to the LLM to propose concrete experiments. Each experiment has: `title, addresses_gap, hypothesis, method, baselines[], datasets[], expected_outcome, feasibility (high/medium/low), builds_on[]`. Saves to `data/analysis/experiments_*.json`.
- Must run `build_paper_profile_tool` on each paper first.

### `create_project_tool`
**File:** `tools/manage_project.py`
**Input:** `name: str`
**Output:** Project manifest `dict` with `name, created, papers[]`
**What it does:** Creates a new named project manifest at `data/projects/<name>.json`. Name is slugified (lowercase, hyphens). No-op if project already exists (returns existing manifest).

### `add_to_project_tool`
**File:** `tools/manage_project.py`
**Input:** `name: str, paper_id: str, source: str`
**Output:** Updated project manifest
**What it does:** Adds a paper reference to an existing project manifest. No-op if the paper is already in the project.

### `list_projects_tool`
**File:** `tools/manage_project.py`
**Input:** (none)
**Output:** `list[dict]` — each item has `name, created, paper_count, papers[]`
**What it does:** Lists all project manifests in `data/projects/`.

---

## Services

### `services/retrieval/aggregator.py`
Calls both `arxiv_service.fetch_papers()` and `semantic_scholar_service.fetch_papers()` and concatenates the results. No deduplication.

### `services/retrieval/arxiv_service.py`
- Calls `https://export.arxiv.org/api/query` with httpx
- Parses Atom XML with `xml.etree.ElementTree`
- Extracts `paper_id` from atom:id: `http://arxiv.org/abs/2401.04088v1` → `2401.04088`
- **Retry logic:** up to 4 attempts, exponential backoff (5s, 10s, 20s) on 429 or any `HTTPError`
- Returns: `paper_id, title, abstract, year, authors, source="arxiv"`

### `services/retrieval/semantic_scholar_service.py`
- Calls `https://api.semanticscholar.org/graph/v1/paper/search`
- Fields requested: `title,abstract,year,authors,externalIds`
- `paper_id` prefers `externalIds.ArXiv` (for ingest compatibility), falls back to S2 `paperId`
- Optional `S2_API_KEY` via `x-api-key` header
- **Retry logic:** same 4-attempt exponential backoff as arXiv service
- Returns: `paper_id, title, abstract, year, authors, source="semantic_scholar"`

### `services/retrieval/vector_store.py`
ChromaDB operations. Three collections per source:

| Collection | Distance | Contents |
|---|---|---|
| `{source}_papers` | **cosine** | Content chunks (all non-reference text) |
| `{source}_papers_refs` | L2 | Bibliography / reference list chunks |
| `{source}_papers_sections` | **cosine** | Section-aware chunks with rich metadata |

**Key functions:**
- `index_chunks(paper_id, source, chunks)` — classifies content vs references by regex, upserts both
- `index_structured_chunks(paper_id, source, chunks)` — indexes section chunks with `section, is_abstract, is_conclusion, position_ratio` metadata
- `query_chunks(query, paper_id, source, k)` — cosine similarity search filtered by `paper_id`
- `get_section_chunks(paper_id, source, section_name)` — retrieves all chunks from a named section
- Collection metadata conflict on distance metric change → auto-delete and recreate

### `services/documents/pdf_service.py`
- `download_and_extract_text(url)` — flat text extraction (backward compat)
- `extract_structured_text(url)` — downloads + calls detect
- `detect_sections_from_text(text, pages_text?)` — heuristic section detection
  - Heading detection: numbered patterns (`1.2 Methods`), keyword match (`abstract`, `introduction`, …), ALL CAPS
  - Fallback: wraps entire text as single `Full Text` section
  - Returns: `{full_text, sections[], metadata{title, abstract, page_count, char_count}}`
- `load_cached(source, paper_id)` / `save_cached(source, paper_id, data)` — JSON file cache at `data/papers/`

### `services/documents/chunking.py`
Two chunking strategies:

**`chunk_text(text) → list[str]`** (flat)
- Paragraph-aware (split on `\n\n`)
- `MAX_CHUNK_CHARS=1200`, `MIN_CHUNK_CHARS=100`, `OVERLAP_CHARS=80`
- Overlapping tail from previous chunk on split

**`chunk_sections(sections) → list[dict]`** (section-aware)
- Never splits across section boundaries
- Each chunk carries: `text, section, section_index, chunk_in_section, is_abstract, is_conclusion, position_ratio`
- `position_ratio`: float 0–1 for where the chunk sits in the full document

### `services/extraction/llm_extractor.py`
All LLM interaction. Three exported functions:

**`extract_insights(text) → (dict, raw_str)`**
- Single-pass extraction of: `methods, results, datasets, limitations, future_work, debug_notes`
- Strict no-fabrication rules in system prompt

**`extract_field(field, text) → (list, raw_str)`**
- Per-field extraction with field-specific system prompts
- `methods` and `results` have custom prompts; others use generic template
- Returns empty list on JSON parse failure (fault-tolerant)

**`build_profile(text) → (dict, raw_str)`**
- 13-field rich profile; `response_format: {"type": "json_object"}` enforced
- Timeout: 90s

**`_strip_code_fences(text)`** — strips ` ```json ` / ` ``` ` markdown wrapping before `json.loads()`. Imported and reused by `synthesis.py`, `gap_detector.py`, and `experiment_suggester.py`.

All functions read `IONOS_API_TOKEN`, `IONOS_BASE_URL`, `IONOS_MODEL` from environment.

### `services/analysis/synthesis.py`
**`analyze_papers(papers) → dict`** — LLM-powered analysis; sends titles + abstracts to the LLM and returns structured `themes[], trends[], methodology_landscape, paper_count, year_range`. Falls back to `_analyze_papers_fallback()` (word frequency) on any failure.

**`_analyze_papers_fallback(papers) → dict`** — original word-counting implementation; returns `themes[], common_topics[], limitations[], possible_gaps[]`.

**`synthesize_insights(insights) → dict`** — statistical cross-paper synthesis; finds items appearing in ≥ half the papers. Unchanged.

### `services/analysis/gap_detector.py`
**`detect_gaps(paper_profiles) → (dict, raw_str)`**
- Formats each profile with `_format_profile()` into a compact text block
- Single LLM call with 90s timeout, `response_format: json_object`
- System prompt rules include: only cross-paper gaps (not restatements of authors' own `future_work`/`limitations`), every item must reference specific `paper_id`s
- Returns: `research_gaps[], methodological_gaps[], contradictions[], connections[], field_summary`

### `services/analysis/experiment_suggester.py`
**`suggest_experiments(gap_analysis, paper_profiles) → (dict, raw_str)`**
- Takes the output of `detect_gaps()` plus the profiles it was run on
- Single LLM call with 90s timeout, `response_format: json_object`
- Each experiment proposal has: `title, addresses_gap, hypothesis, method, baselines[], datasets[], expected_outcome, feasibility, builds_on[]`
- Feasibility rated `high` (weeks) / `medium` (months) / `low` (significant resources)

### `services/project_manager.py`
Manifest CRUD for named research projects.
- `create_project(name)` — slugifies name, creates `data/projects/<name>.json`; returns existing manifest if already present
- `add_paper_to_project(name, paper_id, source)` — appends to `papers[]` with timestamp; no-op if duplicate
- `remove_paper_from_project(name, paper_id, source)` — removes matching entry
- `get_project(name)` — loads and returns manifest; raises `FileNotFoundError` if missing
- `list_projects()` — scans `data/projects/`, returns list with name, created, paper_count, papers
- `get_project_papers(name)` — convenience wrapper returning just the papers list

---

## Data Flow Diagrams

### Project-based pipeline (recommended)

```
create_project_tool("moe-efficiency")
       │ creates data/projects/moe-efficiency.json
       ▼
search_papers_tool(query, limit)
       │ arXiv (retry/backoff) + S2 (retry/backoff)
       │ returns [{paper_id, title, abstract, year, authors, source}, ...]
       ▼
for each relevant paper:
  ingest_paper_tool(paper_id, source="arxiv")
       │ cache check → download PDF → pypdf → section detection → chunking
       │ saves data/papers/arxiv/<id>.json
       ▼
  index_paper_tool(paper_id, source)
       │ flat chunks → arxiv_papers (cosine) + arxiv_papers_refs
       │ section chunks → arxiv_papers_sections (cosine)
       ▼
  build_paper_profile_tool(paper_id, source)
       │ three-tier context: full text → priority sections → chunk retrieval
       │ single LLM call → 13-field profile
       │ saves data/profiles/arxiv/<id>.json
       ▼
  add_to_project_tool("moe-efficiency", paper_id, source)
       │ appends to manifest

detect_gaps_tool(project="moe-efficiency")
       │ loads all profiles from manifest
       │ single LLM call → gaps JSON
       │ saves data/analysis/gap_analysis_*.json
       ▼
suggest_experiments_tool(project="moe-efficiency")
       │ re-runs gap detection + sends gaps + profiles to LLM
       │ saves data/analysis/experiments_*.json
       ▼
       Returns experiment proposals to Claude Desktop
```

### build_paper_profile context selection

```
cached paper data
       │
       ├── full_text ≤ 80,000 chars
       │       └──→ send full text to LLM  ✓
       │
       ├── full_text > 80,000 chars
       │       └── filter sections by keyword:
       │           {abstract, introduction, conclusion, discussion,
       │            method, methods, approach, related work}
       │           join in document order
       │           └── joined text ≤ 80,000 chars
       │                   └──→ send priority sections to LLM  ✓
       │           └── joined text > 80,000 chars (or no matches)
       │                   └──→ 3 retrieval queries × k=6 chunks
       │                       deduplicate, sort by index (max 15)
       │                       └──→ send top chunks to LLM  ✓
       ▼
build_profile(context_text)  →  13-field profile dict
```

### analyze_papers LLM + fallback

```
analyze_papers_tool(papers)
       │
       ▼
synthesis.analyze_papers(papers)
       │
       ├── IONOS env vars present?
       │   ├── No  →  _analyze_papers_fallback()  (word frequency)
       │   └── Yes →  LLM call (60s timeout)
       │               ├── Success + valid JSON
       │               │       └──→ return {themes, trends,
       │               │                    methodology_landscape,
       │               │                    paper_count, year_range}
       │               └── Any failure (HTTP error, bad JSON, timeout)
       │                       └──→ _analyze_papers_fallback()
       │                            return {themes, common_topics,
       │                                    limitations, possible_gaps}
```

### API retry behaviour (arXiv and S2)

```
fetch_papers(query, limit)
       │
       ├── attempt 1 ──→ 429 or HTTPError ──→ wait 5s  ──→ attempt 2
       ├── attempt 2 ──→ 429 or HTTPError ──→ wait 10s ──→ attempt 3
       ├── attempt 3 ──→ 429 or HTTPError ──→ wait 20s ──→ attempt 4
       ├── attempt 4 ──→ 429 or HTTPError ──→ raise
       └── any attempt ──→ 2xx ──→ parse + return immediately
```

---

## Data Storage

All data lives under `data/` relative to the repo root.

| Path | Format | Written by | Read by |
|---|---|---|---|
| `data/papers/{source}/{id}.json` | JSON | `ingest_paper_tool` | `index_paper_tool`, `extract_paper_insights_tool`, `build_paper_profile_tool` |
| `data/insights/{source}/{id}.json` | JSON | `extract_paper_insights_tool` | `synthesize_papers_tool`, `detect_gaps_tool` |
| `data/profiles/{source}/{id}.json` | JSON | `build_paper_profile_tool` | `synthesize_papers_tool`, `detect_gaps_tool`, `suggest_experiments_tool` |
| `data/projects/{name}.json` | JSON | `create_project_tool`, `add_to_project_tool` | `synthesize_papers_tool`, `detect_gaps_tool`, `suggest_experiments_tool` |
| `data/analysis/gap_analysis_*.json` | JSON | `detect_gaps_tool` | (audit / external) |
| `data/analysis/experiments_*.json` | JSON | `suggest_experiments_tool` | (audit / external) |
| `data/chroma/` | ChromaDB | `index_paper_tool` | `retrieve_chunks_tool`, `extract_paper_insights_tool`, `build_paper_profile_tool` |

**Paper cache schema** (`data/papers/arxiv/<id>.json`):
```json
{
  "paper_id": "2401.04088",
  "source": "arxiv",
  "pdf_url": "https://arxiv.org/pdf/2401.04088",
  "text_length": 95000,
  "full_text": "...",
  "chunk_count": 82,
  "chunks": ["...", "..."],
  "sections": [{"heading": "Introduction", "level": 1, "text": "...", "start_page": 1, "end_page": 2}],
  "section_chunks": [{"text": "...", "section": "Introduction", "section_index": 1, "chunk_in_section": 0, "is_abstract": false, "is_conclusion": false, "position_ratio": 0.0833}],
  "metadata": {"title": "...", "abstract": "...", "page_count": 12, "char_count": 95000}
}
```

**Project manifest schema** (`data/projects/<name>.json`):
```json
{
  "name": "moe-efficiency",
  "created": "2026-04-17T12:00:00",
  "papers": [
    {"paper_id": "2401.04088", "source": "arxiv", "added": "2026-04-17T12:01:00"}
  ]
}
```

---

## Logging System

**File:** `server/utils/logger.py`

Each server startup creates a new session directory:
```
logs/session_YYYYMMDD_HHMMSS/
├── main.log          # DEBUG-level stream for all modules
└── tools/
    ├── 001_search_papers_tool.json
    ├── 002_ingest_paper_tool.json
    └── ...
```

- `init_logging()` — called once at startup; sets up file + console handlers
- `get_logger(name)` — standard Python logger by module name
- `log_invocation(tool_name, arguments, output, error)` — writes numbered JSON file with timestamp, arguments, output or error

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `IONOS_API_TOKEN` | Yes | Bearer token for IONOS AI Model Hub |
| `IONOS_BASE_URL` | Yes | Base URL for IONOS OpenAI-compatible API |
| `IONOS_MODEL` | Yes | Model ID, e.g. `meta-llama/Llama-3.3-70B-Instruct` |
| `S2_API_KEY` | No | Semantic Scholar API key (higher rate limits) |
| `FULL_TEXT_CHAR_LIMIT` | No | Max chars for single-pass LLM (default: `80000`) |

Loaded from `.env` via `python-dotenv` before all other imports in `main.py`.

---

## Tool Dependency Map

```
search_papers_tool          (no prerequisites)
analyze_papers_tool         (no prerequisites — operates on search results)
create_project_tool         (no prerequisites)
list_projects_tool          (no prerequisites)
ingest_paper_tool           (no prerequisites)
add_to_project_tool         requires: create_project_tool
index_paper_tool            requires: ingest_paper_tool
retrieve_chunks_tool        requires: index_paper_tool
extract_paper_insights_tool requires: ingest_paper_tool
                            fallback also requires: index_paper_tool
build_paper_profile_tool    requires: ingest_paper_tool
                            fallback also requires: index_paper_tool
synthesize_papers_tool      requires: build_paper_profile_tool OR extract_paper_insights_tool
detect_gaps_tool            requires: build_paper_profile_tool OR extract_paper_insights_tool
                            (minimum 2 papers)
suggest_experiments_tool    requires: build_paper_profile_tool
                            (minimum 2 papers; runs gap detection internally)
```

**Recommended full pipeline:**
```
1. create_project_tool       → name the research session
2. search_papers_tool        → find paper IDs
3. ingest_paper_tool         → download + parse PDF           ┐ repeat
4. index_paper_tool          → embed into ChromaDB            │ for each
5. build_paper_profile_tool  → rich LLM profile               │ paper
6. add_to_project_tool       → register in project manifest   ┘
7. detect_gaps_tool          → cross-paper gap analysis  (project="...")
8. suggest_experiments_tool  → concrete experiment proposals (project="...")
```

---

## Changelog

| Date | Change |
|---|---|
| 2026-04-14 | Initial architecture documented |
| 2026-04-14 | Added `paper_id` to `search_papers_tool` output (arXiv atom:id + S2 externalIds) |
| 2026-04-14 | `build_paper_profile_tool`: added priority-sections fallback (three-tier context selection) |
| 2026-04-14 | `gap_detector.py`: added prompt rule prohibiting restatement of authors' own future_work/limitations |
| 2026-04-14 | Added project management system (`project_manager.py`, `manage_project.py`, 3 new tools) |
| 2026-04-14 | `detect_gaps_tool` and `synthesize_papers_tool`: added optional `project` parameter |
| 2026-04-14 | Added retry/backoff to `arxiv_service.py` and `semantic_scholar_service.py` (4 attempts, 5/10/20s) |
| 2026-04-17 | Added `suggest_experiments_tool` + `experiment_suggester.py` service |
| 2026-04-17 | `analyze_papers`: replaced word-counting with LLM-powered analysis + fallback |
