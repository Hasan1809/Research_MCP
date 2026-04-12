import re
from collections import Counter

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "with", "this", "that", "is", "are", "was", "were", "be", "been",
    "by", "from", "as", "we", "our", "their", "its", "it", "using", "based",
    "via", "into", "than", "which", "also", "has", "have", "not", "no",
    "can", "paper", "study", "show", "shows", "propose", "proposed",
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 3]


def analyze_papers(papers: list[dict]) -> dict:
    all_words: list[str] = []
    years: list[int] = []
    sources: list[str] = []

    for paper in papers:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        all_words.extend(_tokenize(text))
        if paper.get("year"):
            try:
                years.append(int(paper["year"]))
            except ValueError:
                pass
        if paper.get("source"):
            sources.append(paper["source"])

    word_counts = Counter(all_words)
    top_words = [word for word, _ in word_counts.most_common(30)]

    # Split top words into themes (top 5) and common topics (next 10)
    themes = top_words[:5]
    common_topics = top_words[5:15]

    limitations = []
    if len(papers) < 3:
        limitations.append("Too few papers to draw reliable conclusions")
    if years and (max(years) - min(years)) < 2:
        limitations.append("Papers span a narrow time range; trends may not be visible")
    if len(set(sources)) == 1:
        limitations.append(f"All papers from a single source ({sources[0]}); results may be biased")

    possible_gaps = []
    rare_words = [word for word, count in word_counts.items() if count == 1 and len(word) > 5]
    if rare_words:
        possible_gaps.append(f"Underexplored concepts: {', '.join(rare_words[:5])}")
    if years and max(years) < 2023:
        possible_gaps.append("No recent papers found; the field may have advanced since")

    return {
        "themes": themes,
        "common_topics": common_topics,
        "limitations": limitations,
        "possible_gaps": possible_gaps,
    }
