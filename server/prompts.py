from mcp.server.fastmcp import FastMCP


def register_prompts(mcp: FastMCP):
    def _full_workflow_prompt(topic: str, num_papers: int, project: str | None = None) -> str:
        project_name = project or topic
        search_limit = max(num_papers + 6, 12)
        return f"""Execute the complete research-agent workflow for: "{topic}"

INTENT:
Use this workflow whenever the user asks to find gaps, identify research gaps,
suggest experiments, build a research project, or run the whole workflow for a topic.

STEPS - execute every step in order, do not skip any:
1. get_research_workflow_guide_tool(topic="{topic}", project="{project_name}", num_papers={num_papers})
2. Plan 2-4 academic search queries yourself. For niche terms, include the user
   term plus broader scholarly terms that capture the same research area.
3. Call search_papers_tool once for each planned query, using limit={search_limit}.
   You must make at least 2 distinct search_papers_tool calls before creating
   the project. If the literal topic query returns off-topic results, search
   broader scholarly terms for the same area before selecting papers.
4. Pick up to {num_papers} relevant arxiv or semantic_scholar papers across all results. Only select
   papers whose title or abstract directly mentions the research topic. Reject
   papers that are only loosely related by domain, generic surveys outside the
   topic, or papers whose source cannot be ingested.
5. Call create_project_tool(name="{project_name}", overwrite=True) so this run
   starts from a clean project manifest.
6. Call batch_ingest_papers_tool once with all selected papers as a list. Use
   the paper_id and source exactly as returned by search_papers_tool; never pass
   a URL as paper_id. If some papers fail ingestion, continue with successful papers.
7. Call start_batch_build_profiles_job once with only successfully ingested papers
   as a list and max_workers=2. Poll get_job_status_tool(job_id, wait_seconds=180)
   until status is completed, then call get_job_result_tool(job_id). Continue
   only with successfully profiled papers.
8. Call batch_add_to_project_tool once using project name "{project_name}" and
   only successfully profiled papers as a list.
9. Call get_workflow_status_tool(project="{project_name}") and verify profiled_count
   equals paper_count before detecting gaps.
10. Call detect_gaps_tool(project="{project_name}").
11. Call start_batch_validate_gaps_job(project="{project_name}", max_workers=2).
   Poll get_job_status_tool(job_id, wait_seconds=180) until completed, then call
   get_job_result_tool(job_id).
12. Call suggest_experiments_tool(project="{project_name}", compact=True). This
   step is mandatory. It must use only included/refined validated gaps. If all
   gaps are already addressed, accept zero experiments and do not invent any.
13. Call generate_bibliography_tool(project_name="{project_name}", format="bibtex").
14. Call generate_project_report_tool(project="{project_name}").
15. Call get_workflow_status_tool(project="{project_name}") once more and use its
    result only to verify completion.

OUTPUT - after all required steps are complete:
Paste the report_markdown returned by generate_project_report_tool into chat.
Do not replace it with a summary. Do not create visual outputs, diagrams, SVGs,
charts, indexes, or extra documents.

RULES:
- Use project "{project_name}" consistently for every project-scoped tool call.
- Select no more than {num_papers} papers unless the user explicitly asks for a larger corpus.
- Do not stop after search, ingest, profiling, gap detection, or validation. Continue until report generation finishes.
- Never say you will call a tool next; call the tool instead.
- Search results are not a valid final answer.
- If fewer than 2 papers are successfully profiled, stop and explain that the workflow cannot detect cross-paper gaps.
- If asked to validate a gap after this workflow, call validate_gap_tool(gap=<gap text>, project="{project_name}").
- Do not add budget estimates, timelines, FTE counts, or resource requirements.
- Do not add executive summaries, next steps, or collaboration sections.
- Do not add any sections beyond the report_markdown.
- Do not add commentary before or after the output.
- The deterministic report is the deliverable. Present it and stop."""

    @mcp.prompt()
    def research_topic(topic: str, num_papers: int = 8) -> str:
        """
        Complete research workflow for a topic: search, ingest, profile,
        detect gaps, validate gaps, suggest experiments, bibliography, report.
        """
        return _full_workflow_prompt(topic, num_papers)

    @mcp.prompt()
    def find_gaps_and_experiments(
        topic: str,
        num_papers: int = 8,
        project: str | None = None,
    ) -> str:
        """
        Run the standard workflow for requests like "find gaps and suggest
        experiments for <topic>".
        """
        return _full_workflow_prompt(topic, num_papers, project=project)

    @mcp.prompt()
    def analyze_paper(paper_id: str, source: str = "arxiv") -> str:
        """Deep analysis of a single paper."""
        return f"""Analyze paper {paper_id} from {source}:

STEPS:
1. batch_ingest_papers_tool(papers=[{{"paper_id": "{paper_id}", "source": "{source}"}}])
2. start_batch_build_profiles_job(papers=[{{"paper_id": "{paper_id}", "source": "{source}"}}], max_workers=1)
3. Poll get_job_status_tool(job_id, wait_seconds=180) until completed, then call get_job_result_tool(job_id).

OUTPUT FORMAT (use EXACTLY):

---
## {paper_id}

| Field | Value |
|-------|-------|
| Type | [paper_type] |
| Problem | [1-2 sentences] |
| Contribution | [1-2 sentences] |
| Methods | [1-2 sentences] |
| Key Findings | [1-2 sentences] |
| Core Insight | [1 sentence] |
| Datasets | [comma-separated list] |
| Limitations | [comma-separated list] |
| Future Work | [comma-separated list] |

**Summary**: [plain_english_summary]
---

Do NOT add commentary. Do NOT show JSON."""

    @mcp.prompt()
    def compare_papers(project: str) -> str:
        """Compare all papers in a project."""
        return f"""Compare papers in project "{project}":

STEPS - execute every step in order, do not skip any:
1. detect_gaps_tool(project="{project}")
2. Call start_batch_validate_gaps_job(project="{project}", max_workers=2).
   Poll get_job_status_tool(job_id, wait_seconds=180) until completed, then call get_job_result_tool(job_id).
3. suggest_experiments_tool(project="{project}", compact=True)
   This step is mandatory. Do not stop after detect_gaps_tool.

OUTPUT - after all steps are complete, write your response in this exact format and nothing else:

## Research Gaps
For each gap: one sentence description, which paper IDs support it.

## Methodological Gaps
Same format as Research Gaps.

## Contradictions
What conflicts, which paper IDs, one sentence on why it matters.

## Suggested Experiments
For each experiment: title, one sentence hypothesis, one sentence method, feasibility, which paper IDs it builds on.

## Field Summary
The field_summary string from detect_gaps_tool output verbatim.

RULES:
- Do not create any files or documents.
- If asked to validate a candidate gap, call validate_gap_tool(gap=<gap text>, project="{project}").
- Do not add budget estimates, timelines, FTE counts, or resource requirements.
- Do not add any sections not listed above.
- Keep every field to one sentence maximum unless stated otherwise."""
