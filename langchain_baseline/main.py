import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT_DIR / "server"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")
os.environ["USAGE_TOOL_PREFIX"] = "lc_"

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_baseline.tools import TOOLS
from langchain_baseline.utils.logger import get_logger, get_session_dir, init_logging
from langchain_baseline.utils.usage_tracker import log_usage

init_logging()
logger = get_logger(__name__)

DEFAULT_NUM_PAPERS = 8

RESEARCH_OUTPUT_TEMPLATE = """---
## Research Analysis: {topic}

### Papers Analyzed
| # | Title | ID | Type | Core Contribution |
|---|-------|-----|------|-------------------|
| 1 | ... | ... | ... | 1 sentence |

### Research Gaps
| Gap | Type | Evidence | Papers |
|-----|------|----------|--------|
| ... | research/methodological | ... | IDs |

### Contradictions
| Finding A | Finding B | Papers | Nature |
|-----------|-----------|--------|--------|

### Suggested Experiments
| # | Title | Addresses | Feasibility | Method (1 sentence) |
|---|-------|-----------|-------------|---------------------|

### Connections
- [1 sentence per connection]

### Field Summary
[2-3 sentences from gap detection field_summary]
---"""


def get_resolved_orchestrator_model() -> str:
    provider = os.environ.get("LC_ORCHESTRATOR_PROVIDER", "ionos").strip().lower()
    if provider in {"ionos", "openai", "openai_compat"}:
        return os.environ.get("IONOS_MODEL", "")
    if provider == "anthropic":
        return os.environ.get("LC_ORCHESTRATOR_MODEL", "claude-3-5-haiku-latest")
    return os.environ.get("LC_ORCHESTRATOR_MODEL", os.environ.get("IONOS_MODEL", ""))


def normalize_research_topic(user_request: str) -> str:
    """Extract the research topic from common natural-language workflow requests."""
    text = (user_request or "").strip()
    if not text:
        return "Model Context Protocol security"
    lowered = text.lower()
    prefixes = [
        "find research gaps and suggest experiments for",
        "find gaps and suggest experiments for",
        "find research gaps for",
        "find gaps for",
        "suggest experiments for",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip(" .:")
    match = re.search(r"\bfor\s+(.+?)(?:\s+using\s+research\s+agent)?\.?$", text, flags=re.IGNORECASE)
    if match and any(verb in lowered for verb in ("find", "gap", "experiment", "research agent")):
        return match.group(1).strip(" .:")
    return text


def _extract_usage(response: Any) -> dict[str, int]:
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if usage:
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
            "completion_tokens": int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    generations = getattr(response, "generations", None) or []
    if generations and generations[0]:
        message = getattr(generations[0][0], "message", None)
        if message is not None:
            metadata = getattr(message, "response_metadata", None) or {}
            usage = metadata.get("token_usage") or metadata.get("usage") or {}
            if usage:
                return {
                    "prompt_tokens": int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                }

            usage_metadata = getattr(message, "usage_metadata", None) or {}
            if usage_metadata:
                return {
                    "prompt_tokens": int(usage_metadata.get("input_tokens", 0) or 0),
                    "completion_tokens": int(usage_metadata.get("output_tokens", 0) or 0),
                    "total_tokens": int(usage_metadata.get("total_tokens", 0) or 0),
                }

    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class LangChainUsageHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self._active_calls: deque[tuple[float, int]] = deque()

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        input_chars = sum(len(prompt) for prompt in prompts)
        self._active_calls.append((time.time(), input_chars))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        started_at, input_chars = self._active_calls.popleft() if self._active_calls else (time.time(), 0)
        usage = _extract_usage(response)
        if not usage["total_tokens"]:
            return

        log_usage(
            tool_name="agent",
            model=get_resolved_orchestrator_model(),
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            latency_seconds=max(time.time() - started_at, 0.0),
            input_chars=input_chars,
        )


def create_orchestrator_llm(usage_handler: BaseCallbackHandler):
    provider = os.environ.get("LC_ORCHESTRATOR_PROVIDER", "ionos").strip().lower()
    temperature = float(os.environ.get("LC_ORCHESTRATOR_TEMPERATURE", "0"))

    if provider in {"ionos", "openai", "openai_compat"}:
        from langchain_openai import ChatOpenAI

        model = os.environ["IONOS_MODEL"]
        base_url = os.environ.get("LC_ORCHESTRATOR_BASE_URL") or os.environ["IONOS_BASE_URL"]
        api_key = os.environ.get("LC_ORCHESTRATOR_API_KEY") or os.environ["IONOS_API_TOKEN"]
        os.environ["LC_ORCHESTRATOR_MODEL"] = model
        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            callbacks=[usage_handler],
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = os.environ.get("LC_ORCHESTRATOR_MODEL", "claude-3-5-haiku-latest")
        api_key = os.environ.get("LC_ORCHESTRATOR_API_KEY") or os.environ["ANTHROPIC_API_KEY"]
        os.environ["LC_ORCHESTRATOR_MODEL"] = model
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature,
            callbacks=[usage_handler],
        )

    raise ValueError(
        "Unsupported LC_ORCHESTRATOR_PROVIDER. Use one of: ionos, openai_compat, anthropic."
    )


def _build_full_workflow_prompt(topic: str, num_papers: int = DEFAULT_NUM_PAPERS) -> str:
    topic = normalize_research_topic(topic)
    search_limit = max(num_papers + 6, 12)
    return f"""Execute the complete research-agent workflow for: "{topic}"

INTENT:
Use this workflow whenever the user asks to find gaps, identify research gaps,
suggest experiments, build a research project, or run the whole workflow for a topic.

STEPS - execute every step in order, do not skip any:
1. get_research_workflow_guide(topic="{topic}", project="{topic}", num_papers={num_papers})
2. Plan 2-4 academic search queries yourself. For niche terms, include the user
   term plus broader scholarly terms that capture the same research area.
3. Call search_papers once for each planned query, using limit={search_limit}.
   You must make at least 2 distinct search_papers calls before creating the
   project. If the literal topic query returns off-topic results, search broader
   scholarly terms for the same area before selecting papers.
4. Pick up to {num_papers} relevant arxiv or semantic_scholar papers across all results. Only select
   papers whose title or abstract directly mentions the research topic. Reject
   papers that are only loosely related by domain, generic surveys outside the
   topic, or papers whose source cannot be ingested.
5. Call create_project(name="{topic}", overwrite=True) so this run starts from a clean project manifest.
6. Call batch_ingest_papers once with all selected papers as a list. Use the
   paper_id and source exactly as returned by search_papers; never pass a URL
   as paper_id. If some papers fail ingestion, continue with successful papers.
7. Call start_batch_build_profiles_job once with only successfully ingested papers
   as a list and max_workers=2. Poll get_job_status(job_id, wait_seconds=180)
   until status is completed, then call get_job_result(job_id). Continue only
   with successfully profiled papers.
8. Call batch_add_to_project once using project name "{topic}" and only successfully
   profiled papers as a list.
9. Call get_workflow_status(project="{topic}") and verify profiled_count equals
   paper_count before detecting gaps.
10. Call detect_research_gaps(project="{topic}").
11. Call start_batch_validate_gaps_job(project="{topic}", max_workers=2).
   Poll get_job_status(job_id, wait_seconds=180) until completed, then call get_job_result(job_id).
12. Call suggest_research_experiments(project="{topic}", compact=True). This step
   is mandatory. It must use only included/refined validated gaps. If all gaps
   are already addressed, accept zero experiments and do not invent any.
13. Call generate_bibliography(project_name="{topic}", format="bibtex", save=True).
14. Call generate_project_report(project="{topic}").
15. Call get_workflow_status(project="{topic}") once more and use its result only to verify completion.

OUTPUT - after all required steps are complete:
Paste the report_markdown returned by generate_project_report into chat.
Do not replace it with a summary. Do not create visual outputs, diagrams, SVGs,
charts, indexes, or extra documents.

RULES:
- Use project "{topic}" consistently for every project-scoped tool call.
- Select no more than {num_papers} papers unless the user explicitly asks for a larger corpus.
- Do not stop after search, ingest, profiling, gap detection, or validation. Continue until report generation finishes.
- Never say you will call a tool next; call the tool instead.
- Search results are not a valid final answer.
- You may only produce the final answer after generate_project_report and the final get_workflow_status call have completed.
- If fewer than 2 papers are successfully profiled, stop and explain that the workflow cannot detect cross-paper gaps.
- If asked to validate a gap after this workflow, call validate_research_gap(gap=<gap text>, project="{topic}").
- Do not add budget estimates, timelines, FTE counts, or resource requirements.
- Do not add executive summaries, next steps, or collaboration sections.
- Do not add any sections beyond the report_markdown.
- Do not add commentary before or after the output.
- The deterministic report is the deliverable. Present it and stop."""


def build_research_topic_prompt(topic: str, num_papers: int = DEFAULT_NUM_PAPERS) -> str:
    return _build_full_workflow_prompt(topic, num_papers)
    return f"""Execute this research workflow for: "{topic}"

STEPS — execute every step in order, do not skip any:
1. search_papers(query="{topic}", limit={num_papers + 2})
2. Pick the {num_papers} most relevant arxiv or semantic_scholar papers. Only select papers
   whose title or abstract directly mentions the research topic.
   Reject papers that are only loosely related by domain.
3. Call batch_ingest_papers once with all selected papers as a list.
   Use the paper_id and source exactly as returned by search_papers;
   never pass a URL as paper_id.
4. Call start_batch_build_profiles_job once with all papers as a list and max_workers=2.
   Poll get_job_status(job_id, wait_seconds=180) until status is completed, then call
   get_job_result(job_id). Do not start other profile-building work while
   the job is running.
5. Call batch_add_to_project once using project name "{topic}" and all
   selected papers as a list.
6. Call detect_research_gaps(project="{topic}").
7. Optionally call start_batch_validate_gaps_job(project="{topic}", max_workers=2)
   when validated gaps are needed before experiments. Poll get_job_status(job_id, wait_seconds=180)
   until completed, then call get_job_result(job_id).
8. Call suggest_research_experiments(project="{topic}", compact=True).
   This step is mandatory. It must use only included/refined validated gaps when validation
   is available. If all gaps are already addressed, do not invent experiments.
9. Call generate_bibliography(project_name="{topic}", format="bibtex", save=True).
10. Call generate_project_report(project="{topic}") and include the returned report_path in the final answer.

OUTPUT — after all required steps are complete, write your response
in this exact format and nothing else:

## Papers Analyzed
For each paper: paper_id, title, type, one sentence on core contribution.

## Research Gaps
For each gap from detect_research_gaps output:
- Gap name
- One sentence description
- Which paper IDs support it

## Methodological Gaps
Same format as Research Gaps.

## Contradictions
For each contradiction: what conflicts, which paper IDs, one sentence
on why it matters.

## Suggested Experiments
For each experiment from suggest_research_experiments output:
- Title
- One sentence hypothesis
- One sentence method
- Feasibility rating
- Which paper IDs it builds on

## Field Summary
The field_summary string from detect_research_gaps output verbatim.

## Report
The report_path returned by generate_project_report.

RULES:
- Do not create any files or documents except through explicit tools such as generate_project_report.
- If asked to validate a gap after this workflow, call validate_research_gap(gap=<gap text>, project="{topic}").
- Do not add budget estimates, timelines, FTE counts, or resource requirements.
- Do not add executive summaries, next steps, or collaboration sections.
- Do not add any sections not listed in the OUTPUT format above.
- Do not add commentary before or after the output.
- Keep every field to one sentence maximum unless stated otherwise.
- The tool outputs are the deliverable. Present them and stop."""


def save_final_output(text: str) -> Path:
    output_path = Path(get_session_dir()) / "final_output.md"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def create_agent(verbose: bool = True) -> AgentExecutor:
    usage_handler = LangChainUsageHandler()
    llm = create_orchestrator_llm(usage_handler)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a research assistant using tools to analyze academic papers. "
            "When the user asks to find gaps, suggest experiments, or gives a research topic, execute the full workflow silently: "
            "call get_research_workflow_guide, search, select the most relevant arxiv or semantic_scholar papers, "
            "create or overwrite a clean project, batch ingest, start the profile job, batch add successful profiled papers, "
            "check workflow status, detect gaps, validate gaps, suggest experiments, generate bibliography, generate the report, "
            "and check workflow status again. "
            "Use paper_id and source exactly "
            "as returned by search_papers, never a URL or placeholder. Use only successfully ingested/profiled papers. "
            "Make at least 2 distinct search_papers calls with different academic queries before creating the project. "
            "Do not create separate Markdown, SVG, index, or planning files outside generate_project_report. "
            "Do not create visual outputs, diagrams, charts, SVGs, or summary visualizations. "
            "Do not add budgets, team sizes, timelines, GPU estimates, or leadership-review language unless the user asks. "
            "Search results are not a valid final answer. Do not stop after search, ingestion, profiling, gap detection, or validation. "
            "Never say you will call a tool next; call the tool instead. "
            "You may only produce the final answer after generate_project_report and a final get_workflow_status call have completed. "
            "Do not stop after intermediate tool calls. After all workflow tools are complete, "
            "you must paste the report_markdown returned by generate_project_report into chat. "
            "Do not replace the report with a custom summary. Do not output raw JSON or narrate tool execution.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=verbose,
        max_iterations=24,
        early_stopping_method="generate",
    )


def main() -> None:
    user_request = sys.argv[1] if len(sys.argv) > 1 else "Model Context Protocol security"
    topic = normalize_research_topic(user_request)
    query = build_research_topic_prompt(topic, DEFAULT_NUM_PAPERS)
    logger.info("Starting LangChain baseline agent")
    logger.info("Session directory: %s", get_session_dir())
    logger.info(
        "Orchestrator provider=%s model=%s",
        os.environ.get("LC_ORCHESTRATOR_PROVIDER", "ionos"),
        get_resolved_orchestrator_model(),
    )
    logger.info("Research topic: %s", topic)

    started_at = time.time()
    agent = create_agent()
    result = agent.invoke({"input": query})
    final_output = str(result.get("output", "")).strip()
    output_path = save_final_output(final_output)
    logger.info("LangChain baseline completed in %.2fs", time.time() - started_at)
    logger.info("Final output saved to %s", output_path)
    print(final_output)


if __name__ == "__main__":
    main()
