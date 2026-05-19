from langchain_baseline.tools.batch_ingest import batch_ingest_papers
from langchain_baseline.tools.batch_profiles import batch_build_profiles
from langchain_baseline.tools.batch_validate_gaps import batch_validate_research_gaps
from langchain_baseline.tools.bibliography import generate_bibliography
from langchain_baseline.tools.experiments import suggest_research_experiments
from langchain_baseline.tools.gaps import detect_research_gaps
from langchain_baseline.tools.jobs import (
    cancel_job,
    get_job_result,
    get_job_status,
    list_jobs,
    start_batch_build_profiles_job,
    start_batch_validate_gaps_job,
)
from langchain_baseline.tools.profile import profile_paper
from langchain_baseline.tools.projects import (
    add_to_project,
    batch_add_to_project,
    create_project,
    list_projects,
)
from langchain_baseline.tools.search import search_papers
from langchain_baseline.tools.validate_gap import validate_research_gap

TOOLS = [
    search_papers,
    batch_ingest_papers,
    start_batch_build_profiles_job,
    start_batch_validate_gaps_job,
    get_job_status,
    get_job_result,
    list_jobs,
    cancel_job,
    batch_build_profiles,
    batch_validate_research_gaps,
    create_project,
    batch_add_to_project,
    generate_bibliography,
    profile_paper,
    detect_research_gaps,
    suggest_research_experiments,
    validate_research_gap,
    add_to_project,
    list_projects,
]

__all__ = [
    "TOOLS",
    "search_papers",
    "batch_ingest_papers",
    "start_batch_build_profiles_job",
    "start_batch_validate_gaps_job",
    "get_job_status",
    "get_job_result",
    "list_jobs",
    "cancel_job",
    "batch_build_profiles",
    "batch_validate_research_gaps",
    "create_project",
    "batch_add_to_project",
    "generate_bibliography",
    "profile_paper",
    "detect_research_gaps",
    "suggest_research_experiments",
    "validate_research_gap",
    "add_to_project",
    "list_projects",
]
