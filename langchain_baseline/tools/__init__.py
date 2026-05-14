from langchain_baseline.tools.batch_ingest import batch_ingest_papers
from langchain_baseline.tools.batch_profiles import batch_build_profiles
from langchain_baseline.tools.bibliography import generate_bibliography
from langchain_baseline.tools.experiments import suggest_research_experiments
from langchain_baseline.tools.gaps import detect_research_gaps
from langchain_baseline.tools.profile import profile_paper
from langchain_baseline.tools.projects import (
    add_to_project,
    batch_add_to_project,
    create_project,
    list_projects,
)
from langchain_baseline.tools.search import search_papers

TOOLS = [
    search_papers,
    batch_ingest_papers,
    batch_build_profiles,
    create_project,
    batch_add_to_project,
    generate_bibliography,
    profile_paper,
    detect_research_gaps,
    suggest_research_experiments,
    add_to_project,
    list_projects,
]

__all__ = [
    "TOOLS",
    "search_papers",
    "batch_ingest_papers",
    "batch_build_profiles",
    "create_project",
    "batch_add_to_project",
    "generate_bibliography",
    "profile_paper",
    "detect_research_gaps",
    "suggest_research_experiments",
    "add_to_project",
    "list_projects",
]
