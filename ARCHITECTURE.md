# Research MCP Architecture

> Last updated: 2026-05-21
>
> Update this file when the MCP tool surface, LangChain baseline, artifact layout,
> workflow contract, or orchestration rules change.

## Overview

This repository implements a research-agent system with two comparable
orchestration surfaces:

- **MCP server**: exposed to Claude Desktop or any MCP-compatible host.
- **LangChain baseline**: a tool-calling LangChain agent that uses the same
  backend services.

The system is intentionally **modular**. It does not hide the whole research
workflow behind a single "run everything" tool. That is important for the
research goal of this project: studying how an LLM orchestrator uses MCP tools
to conduct research work. The backend provides reliable tools, state checks,
job execution, validation, artifact storage, and deterministic reporting; the
orchestrator remains responsible for planning, paper selection, and sequencing.

The expected user experience is still simple:

```text
Find research gaps and suggest experiments for LLM agent tool-use security.
```

The orchestrator should then use the workflow guide, search papers, create a
clean project, ingest/profile papers, detect gaps, validate them, suggest
experiments, generate a bibliography, generate the final report, and paste the
report Markdown in chat.

## Design Principles

1. **Modular tools, not a hidden pipeline**
   - Each stage is inspectable and logged.
   - The orchestrator decides how to plan the workflow.
   - The backend enforces guardrails where correctness matters.

2. **MCP and baseline parity**
   - The LangChain baseline exposes the same workflow shape as the MCP server.
   - Both surfaces reuse the same backend services wherever possible.
   - Differences should be orchestration-layer differences, not feature drift.

3. **Orchestrator-owned query planning**
   - The search tool executes one literal query at a time.
   - The orchestrator should generate 2-4 academic search queries for niche
     topics and call search once per query.
   - The backend does not hardcode topic-specific query expansion.

4. **Deterministic final output**
   - Final user-facing output should be the generated project report Markdown.
   - The agent should not create visual summaries, SVGs, extra indexes, or
     custom long summaries unless explicitly asked.

5. **Defensible research quality**
   - Gap validation is stricter than simple keyword matching.
   - Already-addressed gaps are excluded from experiments.
   - Experiments must be grounded in surviving validated gaps and project
     papers where possible.

6. **Reproducibility**
   - Every meaningful tool call is logged.
   - Intermediate artifacts are saved under `data/`.
   - Job results are persisted and resumable.

## High-Level Components

```text
research_mcp/
+-- server/
|   +-- main.py                       # MCP entrypoint and tool registration
|   +-- config.py                     # Shared limits and paths
|   +-- tools/                        # Thin MCP wrappers
|   +-- services/                     # Shared backend implementation
|   |   +-- analysis/                 # Gaps, validation, experiments
|   |   +-- artifacts/                # Bibliography generation
|   |   +-- documents/                # PDF download, parsing, chunking
|   |   +-- extraction/               # LLM profile extraction
|   |   +-- jobs/                     # Background job manager
|   |   +-- reports/                  # Deterministic project report
|   |   +-- retrieval/                # arXiv/S2 search and metadata
|   |   +-- project_manager.py        # Project manifests
|   |   +-- workflow_status.py        # Workflow readiness/resume checks
|   +-- utils/                        # Logging and usage tracking
+-- langchain_baseline/
|   +-- main.py                       # LangChain agent entrypoint
|   +-- tools/                        # LangChain tool wrappers
|   +-- services/__init__.py          # Reuses server-side implementations
|   +-- utils/                        # Baseline logging
+-- data/                             # Persistent research artifacts
+-- logs/                             # Per-session logs and tool calls
+-- tests/                            # Regression tests
```

## Runtime Surfaces

### MCP Server

`server/main.py` registers the current public MCP tool surface. The tools are
ordered to bias the orchestrator toward the correct workflow.

| Order | MCP tool | Purpose |
|---:|---|---|
| 1 | `get_research_workflow_guide_tool` | Returns the workflow contract and output rules. |
| 2 | `search_papers_tool` | Runs one literal academic search query. |
| 3 | `create_project_tool` | Creates or overwrites a clean project manifest. |
| 4 | `batch_ingest_papers_tool` | Downloads/parses selected papers in batch. |
| 5 | `start_batch_build_profiles_job` | Starts background profile extraction. |
| 6 | `get_job_status_tool` | Polls job completion. |
| 7 | `get_job_result_tool` | Fetches persisted job output. |
| 8 | `batch_add_to_project_tool` | Adds only successfully profiled papers to the project. |
| 9 | `get_workflow_status_tool` | Checks project readiness and latest artifacts. |
| 10 | `detect_gaps_tool` | Detects cross-paper candidate gaps. |
| 11 | `start_batch_validate_gaps_job` | Validates detected gaps against external literature. |
| 12 | `suggest_experiments_tool` | Suggests experiments from included validated gaps. |
| 13 | `generate_bibliography_tool` | Generates BibTeX from project metadata. |
| 14 | `generate_project_report_tool` | Generates and returns deterministic Markdown report. |
| 15 | `usage_summary_tool` | Returns token/cost/tool usage summary. |
| 16 | `cancel_job_tool` | Cancels background jobs when needed. |
| 17 | `clear_project_tool` | Clears project membership for recovery/testing. |
| 18 | `validate_gap_tool` | Validates a single gap for debugging or manual review. |

### LangChain Baseline

The LangChain baseline exposes the same workflow with LangChain tool names:

| MCP tool | LangChain baseline tool |
|---|---|
| `get_research_workflow_guide_tool` | `get_research_workflow_guide` |
| `search_papers_tool` | `search_papers` |
| `create_project_tool` | `create_project` |
| `batch_ingest_papers_tool` | `batch_ingest_papers` |
| `start_batch_build_profiles_job` | `start_batch_build_profiles_job` |
| `get_job_status_tool` | `get_job_status` |
| `get_job_result_tool` | `get_job_result` |
| `batch_add_to_project_tool` | `batch_add_to_project` |
| `get_workflow_status_tool` | `get_workflow_status` |
| `detect_gaps_tool` | `detect_research_gaps` |
| `start_batch_validate_gaps_job` | `start_batch_validate_gaps_job` |
| `suggest_experiments_tool` | `suggest_research_experiments` |
| `generate_bibliography_tool` | `generate_bibliography` |
| `generate_project_report_tool` | `generate_project_report` |
| `usage_summary_tool` | `usage_summary` |
| `cancel_job_tool` | `cancel_job` |
| `clear_project_tool` | `clear_project` |
| `validate_gap_tool` | `validate_research_gap` |

The baseline is for comparison against MCP orchestration. It should not add
capabilities that the MCP surface lacks, except for baseline-specific logging
and agent execution support.

## Canonical Workflow

For a simple user request like:

```text
Find research gaps and suggest experiments for <topic>.
```

the orchestrator should follow this sequence:

1. **Load the workflow guide**
   - Call `get_research_workflow_guide_tool`.
   - Follow its workflow and final-output contract.

2. **Plan search queries**
   - Infer the research topic from the user prompt.
   - Generate 2-4 academic search queries.
   - For niche terms, include adjacent academic language.
   - Do not ask the user to know internal tool details.

3. **Search**
   - Call `search_papers_tool(query, limit)` once per planned query.
   - Merge and deduplicate results mentally or in the orchestrator context.
   - Select a focused paper set, normally around 6-8 papers.

4. **Create a clean project**
   - Call `create_project_tool(name, overwrite=true)` for reproducibility.
   - The project name should be a concise slug/topic name.

5. **Ingest papers**
   - Call `batch_ingest_papers_tool`.
   - Use the default paper cap unless the user explicitly asks for more.

6. **Build profiles**
   - Start `start_batch_build_profiles_job`.
   - Poll with `get_job_status_tool` using meaningful waits.
   - Fetch final output with `get_job_result_tool` when needed.

7. **Add profiled papers to the project**
   - Call `batch_add_to_project_tool`.
   - The visible add tool requires successful profiles.
   - Failed/unprofiled papers should be skipped, not forced into the project.

8. **Check status**
   - Call `get_workflow_status_tool`.
   - Use it to confirm the project is ready for gap detection or to resume
     after failures.

9. **Detect gaps**
   - Call `detect_gaps_tool(project=...)`.
   - This saves a cached gap analysis artifact.

10. **Validate gaps**
   - Start `start_batch_validate_gaps_job(project=...)`.
   - This requires a cached gap analysis from step 9.
   - Poll until complete.

11. **Suggest experiments**
   - Call `suggest_experiments_tool(project=...)`.
   - It should use validation results when present.
   - Already-addressed gaps must not generate experiments.

12. **Generate bibliography**
   - Call `generate_bibliography_tool(project=...)`.
   - The report tool can also auto-generate it if missing.

13. **Generate final report**
   - Call `generate_project_report_tool(project=...)`.
   - The tool saves the report and returns `report_markdown`.

14. **Final response**
   - Paste the report Markdown in chat.
   - Do not create visual summaries, separate custom documents, or long
     alternative summaries unless the user asks.

## Data Flow

```text
User prompt
  |
Workflow guide
  |
Orchestrator plans 2-4 search queries
  |
search_papers_tool / search_papers
  |
Selected paper metadata
  |
create_project(overwrite=true)
  |
batch_ingest_papers
  |
data/papers/
  |
start_batch_build_profiles_job
  |
data/profiles/
  |
batch_add_to_project
  |
data/projects/<project>.json
  |
detect_gaps
  |
data/analysis/gap_analysis_*.json
  |
start_batch_validate_gaps_job
  |
data/analysis/gap_validations/
  |
suggest_experiments
  |
data/analysis/experiments_*.json
  |
generate_bibliography
  |
data/artifacts/bibliographies/
  |
generate_project_report
  |
data/artifacts/reports/
  |
report_markdown pasted in chat
```

## Artifact Layout

All persistent research state is stored under `data/`.

| Path | Purpose |
|---|---|
| `data/metadata/` | Search and citation metadata caches when available. |
| `data/papers/{source}/{id}.json` | Ingested paper text, sections, chunks, and metadata. |
| `data/profiles/{source}/{id}.json` | LLM-generated structured paper profiles. |
| `data/projects/{project}.json` | Project manifest and paper membership. |
| `data/analysis/gap_analysis_*.json` | MCP gap detection artifacts. |
| `data/analysis/lc_gap_analysis_*.json` | LangChain baseline gap detection artifacts. |
| `data/analysis/gap_validations/` | Single-gap validation artifacts. |
| `data/analysis/gap_validations/batches/` | Batch validation artifacts. |
| `data/analysis/experiments_*.json` | Experiment suggestion artifacts. |
| `data/artifacts/bibliographies/` | BibTeX outputs. |
| `data/artifacts/reports/` | Deterministic Markdown project reports. |
| `data/jobs/` | Background job state and result files. |
| `data/chroma/` | Persistent Chroma vector indexes used by legacy/deeper retrieval paths. |

## Logging Layout

Each MCP server run writes logs to:

```text
logs/session_YYYYMMDD_HHMMSS/
+-- main.log
+-- usage.log
+-- tools/
    +-- 001_<tool>.json
    +-- 002_<tool>.json
    +-- ...
```

Each LangChain baseline run writes logs to:

```text
logs/langchain_session_YYYYMMDD_HHMMSS/
+-- main.log
+-- usage.log
+-- final_output.md
+-- tools/
    +-- 001_<tool>.json
    +-- 002_<tool>.json
    +-- ...
```

Tool invocation logs should never be written to the repository root. The logger
initializes a session directory before any invocation log is written.

The report tool logs only metadata and character counts for the returned
Markdown, not the entire report body.

## Search Architecture

Search is deliberately simple:

```text
search_papers_tool(query: str, limit: int)
```

The tool executes the query against the configured academic sources and returns
paper metadata. It does not expand niche terminology, generate synonyms, or run
multiple hidden queries.

That work belongs to the orchestrator because:

- query planning is part of the research-agent behavior being studied;
- hardcoded backend expansions would bias specific domains;
- hidden expansion makes failures harder to explain;
- different orchestrators should be comparable on planning quality.

For a niche topic such as "vibecoding security", the orchestrator should search
both the literal term and adjacent academic terms such as AI-assisted coding,
LLM code generation security, secure code generation, and human-AI programming
workflows. The exact queries should be planned by the orchestrator at runtime,
not embedded in backend code.

## Project State and Paper Membership

Projects are reproducibility boundaries. A clean workflow should create the
project with overwrite enabled, then add only successfully profiled papers.

Important behavior:

- project names are slugified;
- `create_project(..., overwrite=true)` clears previous membership;
- `batch_add_to_project_tool` only adds papers with profiles;
- unprofiled or failed papers are skipped rather than silently included;
- `get_workflow_status_tool` reports readiness, latest artifacts, counts, and
  recommended next actions.

This prevents old papers from contaminating a new workflow run.

## Background Jobs

Long-running work is performed through background jobs:

- batch profile building;
- batch gap validation.

Job state and results are persisted under `data/jobs/`. The job manager uses
file locking and Windows-safe retry behavior to avoid transient file access
failures during concurrent writes.

The orchestrator should poll with `get_job_status_tool` and avoid tight polling
loops. When a job completes, the result can be read with `get_job_result_tool`.

## Gap Detection and Validation

Gap detection creates candidate gaps from the selected project papers. Gap
validation then checks whether each candidate is already addressed in external
literature.

Validation is intentionally conservative:

- direct evidence must match the domain, security/threat dimension where
  relevant, and the missing contribution;
- generic adjacent work should not close a gap;
- `already_addressed` requires strong direct evidence;
- incomplete evidence should become `partially_addressed`;
- vague but potentially useful gaps should become `needs_refinement`;
- gaps without strong external evidence remain `confirmed_candidate_gap`.

Validation artifacts include evidence classifications, scores, evidence titles,
and reasons so decisions are traceable.

## Experiment Suggestion Rules

Experiment generation should use validated gaps when validation exists.

Included statuses:

- `confirmed_candidate_gap`
- `partially_addressed`
- `needs_refinement`

Excluded statuses:

- `already_addressed`

If all gaps are already addressed:

- return zero experiments;
- do not ask the LLM to invent experiments anyway;
- save the zero-experiment artifact;
- explain that all detected gaps were externally addressed.

When experiments are generated:

- prefer fewer distinct experiments over near-duplicates;
- deduplicate deterministic repeats;
- ground each experiment in at least one project paper when possible;
- include title, hypothesis, method, feasibility, builds_on, and gap addressed.

## Report Generation

`generate_project_report_tool` is the canonical final-output tool.

It:

- reads the latest project artifacts;
- includes project summary, papers, gaps, validation, experiments, and
  bibliography;
- auto-generates a bibliography if requested and missing;
- saves the report under `data/artifacts/reports/`;
- returns the same Markdown as `report_markdown`.

The orchestrator should paste `report_markdown` into chat as the final answer.
It should not replace it with a custom executive summary unless explicitly
asked.

## MCP vs LangChain Baseline

The baseline exists to compare orchestration approaches, not to become a
separate product.

| Concern | MCP server | LangChain baseline |
|---|---|---|
| Orchestrator | Claude Desktop or MCP host | LangChain tool-calling agent |
| Backend services | `server/services/` | Reused from `server/services/` |
| Tool wrappers | `server/tools/` | `langchain_baseline/tools/` |
| Logs | `logs/session_*` | `logs/langchain_session_*` |
| Gap artifacts | `gap_analysis_*.json` | `lc_gap_analysis_*.json` |
| Final output | `report_markdown` | `report_markdown` |

The baseline prompt mirrors the MCP workflow guide:

- plan 2-4 search queries itself;
- use a clean project;
- ingest/profile/add papers in the correct order;
- detect gaps before validation;
- validate before experiments;
- generate bibliography and report;
- paste the deterministic report.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `IONOS_API_TOKEN` | Yes | Bearer token for IONOS AI Model Hub. |
| `IONOS_BASE_URL` | Yes | OpenAI-compatible IONOS endpoint. |
| `IONOS_MODEL` | Yes | Model used by backend LLM calls. |
| `S2_API_KEY` | No | Semantic Scholar API key for better rate limits. |
| `FULL_TEXT_CHAR_LIMIT` | No | Max chars for single-pass profile extraction. |
| `WORKFLOW_MAX_PAPERS` | No | Default max papers for normal workflow batches. |
| `LC_ORCHESTRATOR_PROVIDER` | Baseline only | `ionos` or `anthropic`. |
| `LC_ORCHESTRATOR_MODEL` | Baseline only | Overrides baseline orchestrator model. |
| `ANTHROPIC_API_KEY` | Baseline only | Required if using Anthropic baseline orchestration. |

## Running the System

### MCP Server

Claude Desktop launches the MCP server through its MCP configuration. The user
can then issue a natural prompt:

```text
Find research gaps and suggest experiments for MCP best practices and safety.
```

The orchestrator should use the workflow guide and tools to complete the
workflow.

### LangChain Baseline

```powershell
.\.venv\Scripts\python.exe .\langchain_baseline\main.py "Find research gaps and suggest experiments for LLM agent tool-use security."
```

The baseline writes logs to `logs/langchain_session_*` and saves the final chat
output to `final_output.md`.

### Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

## Non-Goals

The current architecture is not trying to:

- hide the complete workflow behind one opaque tool;
- hardcode topic-specific search expansions in the backend;
- generate presentation visuals by default;
- replace human literature review judgment;
- guarantee that every external paper can be ingested if no accessible PDF or
  metadata is available.

## Known Limitations

- Closed-access or malformed PDFs may fail ingestion.
- Search quality still depends on the orchestrator planning good queries.
- LLM extraction can produce imperfect profiles, especially from poor PDFs.
- Validation is conservative but still metadata-dependent.
- Human review is still required before treating gaps as thesis-level claims.
- The LangChain baseline may behave differently depending on the selected
  orchestrator model.

## Maintenance Checklist

When adding or removing tools:

1. Update `server/main.py`.
2. Update `langchain_baseline/tools/__init__.py` if baseline parity is needed.
3. Update `get_research_workflow_guide_tool`.
4. Update `langchain_baseline/main.py` prompt guidance if orchestration changes.
5. Update this file.
6. Add or update tests.
7. Run the full test suite.

When changing artifact paths or schemas:

1. Update the producing service.
2. Update `workflow_status.py`.
3. Update `project_report.py`.
4. Update baseline wrappers if needed.
5. Update this file.

## Recent Architecture Changes

| Date | Change |
|---|---|
| 2026-05-21 | Replaced old manual 13-tool architecture with the current guided workflow architecture. |
| 2026-05-21 | Added MCP and LangChain workflow guide parity. |
| 2026-05-21 | Made query expansion orchestrator-owned rather than backend-hardcoded. |
| 2026-05-21 | Made project reports the canonical final chat output via `report_markdown`. |
| 2026-05-21 | Added workflow status checks for ordering, resume, and artifact discovery. |
| 2026-05-21 | Hardened project state by using clean projects and profile-required membership. |
| 2026-05-21 | Added deterministic report/bibliography behavior and stricter experiment filtering. |
| 2026-05-21 | Fixed session logging so tool logs stay under per-session log directories. |
