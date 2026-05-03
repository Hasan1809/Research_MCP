from mcp.server.fastmcp import FastMCP


def register_prompts(mcp: FastMCP):
    @mcp.prompt()
    def research_topic(topic: str, num_papers: int = 3) -> str:
        """
        Complete research workflow: search, ingest, profile, detect gaps,
        suggest experiments.
        """
        return f"""Execute this research workflow for: "{topic}"

STEPS (execute silently, do not narrate each step):
1. search_papers_tool(query="{topic}", limit={num_papers + 2})
2. Pick the {num_papers} most relevant arxiv papers
3. For each: ingest_paper_tool → build_paper_profile_tool
4. detect_gaps_tool(papers=[...])
5. suggest_experiments_tool(papers=[...])

OUTPUT FORMAT (use this EXACTLY):

---
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
---

RULES:
- Do NOT add commentary before or after the tables
- Do NOT explain what each tool does
- Do NOT show raw JSON
- Keep table cells to 1 sentence max
- Total response must be under 800 words"""

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

STEPS: detect_gaps_tool(project="{project}") → suggest_experiments_tool(project="{project}")

OUTPUT FORMAT (use EXACTLY):

---
## Comparison: {project}

### Gaps
| Gap | Type | Evidence |
|-----|------|----------|

### Experiments
| # | Title | Feasibility | Method |
|---|-------|-------------|--------|

### Field Summary
[2-3 sentences]
---

Do NOT add commentary."""
