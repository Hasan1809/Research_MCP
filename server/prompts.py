from mcp.server.fastmcp import FastMCP


def register_prompts(mcp: FastMCP):
    """Register MCP prompts for common research workflows."""

    @mcp.prompt()
    def research_topic(topic: str, num_papers: int = 3) -> str:
        """
        Complete research workflow: search papers on a topic, ingest and
        profile the most relevant ones, detect gaps, and suggest experiments.
        """
        return f"""You are a research assistant. Execute this workflow for the topic: "{topic}"

STEP 1: Create a project called "{topic.lower().replace(' ', '-')[:30]}"
- Use create_project_tool

STEP 2: Search for {num_papers + 2} papers on "{topic}"
- Use search_papers_tool with limit={num_papers + 2}
- From the results, pick the {num_papers} most relevant papers that are
  actually about {topic} (skip unrelated results)
- Only use papers with source="arxiv" (other sources can't be ingested)

STEP 3: For each selected paper, run this sequence:
- ingest_paper_tool(paper_id=..., source="arxiv")
- index_paper_tool(paper_id=..., source="arxiv")
- build_paper_profile_tool(paper_id=..., source="arxiv")
- add_to_project_tool(name="...", paper_id=..., source="arxiv")

STEP 4: Run gap detection
- detect_gaps_tool(project="...")

STEP 5: Run experiment suggestions
- suggest_experiments_tool(project="...")

FORMAT YOUR FINAL RESPONSE AS:

## Research Analysis: {topic}

### Papers Analyzed
For each paper, show:
- **Title**: [paper title]
- **ID**: [paper_id]
- **Core contribution**: [1 sentence from the profile]

### Research Gaps
For each gap, show:
- Gap description
- Which papers relate to this gap

### Suggested Experiments
For each experiment, show:
- **Title**: experiment title
- **Feasibility**: high/medium/low
- **What it tests**: the hypothesis
- **How**: 1-2 sentence method summary

### Field Summary
[The field_summary from gap detection]

Keep the response concise. Do not repeat raw JSON output."""

    @mcp.prompt()
    def analyze_paper(paper_id: str, source: str = "arxiv") -> str:
        """
        Deep analysis of a single paper: ingest, index, profile, and
        extract insights.
        """
        return f"""Analyze paper {paper_id} from {source}:

1. ingest_paper_tool(paper_id="{paper_id}", source="{source}")
2. index_paper_tool(paper_id="{paper_id}", source="{source}")
3. build_paper_profile_tool(paper_id="{paper_id}", source="{source}")
4. extract_paper_insights_tool(paper_id="{paper_id}", source="{source}")

FORMAT YOUR RESPONSE AS:

## Paper Analysis: [title from profile]

**Type**: [paper_type]
**Problem**: [research_problem - 2 sentences max]
**Contribution**: [main_contribution - 2 sentences max]
**Core Insight**: [core_insight]

### Methods
[methods_or_approach - summarized]

### Key Findings
[key_findings]

### Extracted Details
- **Datasets**: [list]
- **Limitations**: [list]
- **Future Work**: [list]

Keep it concise."""

    @mcp.prompt()
    def compare_papers(project: str) -> str:
        """
        Compare all papers in a project: synthesize findings, detect gaps,
        and suggest experiments.
        """
        return f"""Compare the papers in project "{project}":

1. synthesize_papers_tool(project="{project}")
2. detect_gaps_tool(project="{project}")
3. suggest_experiments_tool(project="{project}")

FORMAT YOUR RESPONSE AS:

## Comparative Analysis: {project}

### Common Methods
[from synthesis]

### Common Findings
[from synthesis]

### Key Differences
[notable_differences from synthesis]

### Research Gaps
[from gap detection - brief list]

### Suggested Experiments
[from experiment suggestions - title + feasibility + 1 sentence each]

### Field Summary
[field_summary]

Keep it concise and professional."""
