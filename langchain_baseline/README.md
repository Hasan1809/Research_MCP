# LangChain Baseline

This directory provides a LangChain-based replica of the research workflow used by the MCP server:

`workflow_guide -> search -> create_project -> ingest -> profile_job -> add_profiled_papers -> workflow_status -> detect_gaps -> validate_gaps_job -> suggest_experiments -> bibliography -> report`

The goal is comparison, not feature divergence. The LangChain tools reuse the existing Python pipeline logic from `server/`, so the main difference is the orchestration layer:

- MCP version: Claude Desktop or another MCP host decides tool order
- LangChain version: a LangChain tool-calling agent decides tool order

Both surfaces expose the same workflow guardrails: a first-step workflow guide,
profile-required project membership, workflow status checks, validated-gap
experiment generation, bibliography generation, and deterministic report output.

## Files

- `main.py`: LangChain agent entrypoint
- `tools/`: LangChain `@tool` wrappers
- `services/__init__.py`: imports the existing server-side implementations
- `requirements.txt`: LangChain-only dependencies

## Usage

Install the extra dependencies into the project environment, then run:

```powershell
.\.venv\Scripts\python.exe .\langchain_baseline\main.py "Find research gaps on MCP security. Use 3 papers."
```

The script loads `.env` from the repo root and uses the same IONOS model configuration as the MCP server.

## Orchestrator Model

By default, the LangChain agent uses the IONOS OpenAI-compatible model for orchestration:

- `LC_ORCHESTRATOR_PROVIDER=ionos`
- `LC_ORCHESTRATOR_MODEL` defaults to `IONOS_MODEL`

If you want to match Claude Desktop more closely and use Haiku as the orchestration model while keeping the tool-internal LLM calls on IONOS, set:

```powershell
$env:LC_ORCHESTRATOR_PROVIDER="anthropic"
$env:LC_ORCHESTRATOR_MODEL="claude-3-5-haiku-latest"
$env:ANTHROPIC_API_KEY="..."
```

This keeps the tool pipeline unchanged and only swaps the LangChain orchestration model.

## Logging

- The LangChain run uses the same session logging system as the MCP server.
- `usage.log` is written into the same per-session directory as `main.log`.
- LangChain-originated usage records are prefixed with `lc_`, for example `lc_agent`, `lc_build_profile`, `lc_detect_gaps`.

## Comparison Notes

This baseline is intended for side-by-side evaluation of:

- token usage
- end-to-end latency
- orchestration overhead
- tool-selection behavior
- implementation complexity
