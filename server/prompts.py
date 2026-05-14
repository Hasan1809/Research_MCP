from mcp.server.fastmcp import FastMCP


def register_prompts(mcp: FastMCP):
    @mcp.prompt()
    def research_topic(topic: str, num_papers: int = 3) -> str:
        """
        Complete research workflow: search, ingest, profile, detect gaps,
        suggest experiments.
        """
        return f"""Execute this research workflow for: "{topic}"

STEPS — execute every step in order, do not skip any:
1. search_papers_tool(query="{topic}", limit={num_papers + 2})
2. Pick the {num_papers} most relevant arxiv or semantic_scholar papers. Only select
   papers whose title or abstract directly mentions the research
   topic. Reject papers that are only loosely related by domain.
3. Call batch_ingest_papers_tool once with all selected papers
   as a list. Never call ingest_paper_tool individually when you
   have multiple papers. Use the paper_id and source exactly as
   returned by search_papers_tool; never pass a URL as paper_id.
4. Call batch_build_profiles_tool once with all papers as a list.
   Never call build_paper_profile_tool individually when you have
   multiple papers. Never use extract_paper_insights_tool.
5. Call batch_add_to_project_tool once using project name "{topic}"
   and all selected papers as a list. Never call add_to_project_tool
   individually when you have multiple papers.
6. Call detect_gaps_tool with all papers as a list.
7. Call suggest_experiments_tool with all papers as a list.
   This step is mandatory. Do not stop after detect_gaps_tool.

OUTPUT — after all seven steps are complete, write your response
in this exact format and nothing else:

## Papers Analyzed
For each paper: paper_id, title, type, one sentence on core contribution.

## Research Gaps
For each gap from detect_gaps_tool output:
- Gap name
- One sentence description
- Which paper IDs support it

## Methodological Gaps
Same format as Research Gaps.

## Contradictions
For each contradiction: what conflicts, which paper IDs, one sentence
on why it matters.

## Suggested Experiments
For each experiment from suggest_experiments_tool output:
- Title
- One sentence hypothesis
- One sentence method
- Feasibility rating
- Which paper IDs it builds on

## Field Summary
The field_summary string from detect_gaps_tool output verbatim.

RULES:
- Do not create any files or documents except the project manifest created by batch_add_to_project_tool.
- Do not add budget estimates, timelines, FTE counts, or resource requirements.
- Do not add executive summaries, next steps, or collaboration sections.
- Do not add any sections not listed in the OUTPUT format above.
- Do not add commentary before or after the output.
- Keep every field to one sentence maximum unless stated otherwise.
- The tool outputs are the deliverable. Present them and stop."""

    @mcp.prompt()
    def analyze_paper(paper_id: str, source: str = "arxiv") -> str:
        """Deep analysis of a single paper."""
        return f"""Analyze paper {paper_id} from {source}:

STEPS: ingest_paper_tool → build_paper_profile_tool

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

STEPS — execute every step in order, do not skip any:
1. detect_gaps_tool(project="{project}")
2. suggest_experiments_tool(project="{project}")
   This step is mandatory. Do not stop after detect_gaps_tool.

OUTPUT — after both steps are complete, write your response
in this exact format and nothing else:

## Research Gaps
For each gap: one sentence description, which paper IDs support it.

## Methodological Gaps
Same format as Research Gaps.

## Contradictions
What conflicts, which paper IDs, one sentence on why it matters.

## Suggested Experiments
For each experiment: title, one sentence hypothesis, one sentence method,
feasibility, which paper IDs it builds on.

## Field Summary
The field_summary string from detect_gaps_tool output verbatim.

RULES:
- Do not create any files or documents.
- Do not add budget estimates, timelines, FTE counts, or resource requirements.
- Do not add any sections not listed above.
- Keep every field to one sentence maximum unless stated otherwise."""
