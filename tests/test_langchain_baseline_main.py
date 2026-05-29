from langchain_baseline.main import normalize_research_topic


def test_normalize_research_topic_does_not_reparse_normalized_topic():
    topic = (
        "LLM agent tool-use security, prompt injection, tool poisoning, unsafe tool calls, "
        "and MCP security. Use 3 papers. Follow the full research workflow: search for "
        "papers, select relevant papers."
    )

    assert normalize_research_topic(topic) == topic


def test_normalize_research_topic_extracts_common_request_prefix():
    assert (
        normalize_research_topic("Find research gaps and suggest experiments for MCP security.")
        == "MCP security"
    )
