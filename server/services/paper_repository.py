import json
from pathlib import Path

from config import DATA_DIR


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_paper_cache(source: str, paper_id: str) -> dict | None:
    return _load_json(DATA_DIR / "papers" / source / f"{paper_id}.json")


def save_paper_cache(source: str, paper_id: str, data: dict) -> None:
    _save_json(DATA_DIR / "papers" / source / f"{paper_id}.json", data)


def load_paper_metadata(source: str, paper_id: str) -> dict | None:
    return _load_json(DATA_DIR / "metadata" / source / f"{paper_id}.json")


def save_paper_metadata(source: str, paper_id: str, metadata: dict) -> None:
    _save_json(DATA_DIR / "metadata" / source / f"{paper_id}.json", metadata)


def load_profile(source: str, paper_id: str) -> dict | None:
    return _load_json(DATA_DIR / "profiles" / source / f"{paper_id}.json")


def save_profile(source: str, paper_id: str, profile: dict) -> None:
    _save_json(DATA_DIR / "profiles" / source / f"{paper_id}.json", profile)


def load_insights(source: str, paper_id: str) -> dict | None:
    return _load_json(DATA_DIR / "insights" / source / f"{paper_id}.json")


def save_insights(source: str, paper_id: str, insights: dict) -> None:
    _save_json(DATA_DIR / "insights" / source / f"{paper_id}.json", insights)


def load_profile_or_insights(source: str, paper_id: str) -> dict:
    profile = load_profile(source, paper_id)
    if profile is not None:
        return profile

    insights = load_insights(source, paper_id)
    if insights is not None:
        return insights

    raise FileNotFoundError(
        f"No profile or insights found for paper_id={paper_id!r} source={source!r}."
    )
