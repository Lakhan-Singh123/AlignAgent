"""Persistent skill progress tracker — saves to progress.json."""

from __future__ import annotations

import json
from pathlib import Path

_FILE = Path("progress.json")

_NODE_LABELS = {
    "generate_queries": "Generating smart retrieval queries",
    "retrieve":         "Retrieving relevant document chunks",
    "grade_docs":       "Grading document relevance",
    "web_search":       "Running web search fallback",
    "generate_report":  "Generating skill gap report",
    "generate_resources": "Finding learning resources",
    "reflect":          "Self-reflection & hallucination check",
    "save_report":      "Saving final report",
}


def _load() -> dict:
    if _FILE.exists():
        try:
            return json.loads(_FILE.read_text())
        except Exception:
            pass
    return {"learned": [], "all_missing": []}


def _save(data: dict) -> None:
    _FILE.write_text(json.dumps(data, indent=2))


def get_learned_skills() -> list[str]:
    return _load().get("learned", [])


def get_all_missing() -> list[str]:
    return _load().get("all_missing", [])


def set_missing_skills(skills: list[str]) -> None:
    data = _load()
    # Add new missing skills; keep ones already learned
    existing = set(data.get("all_missing", []))
    for s in skills:
        existing.add(s)
    data["all_missing"] = sorted(existing)
    _save(data)


def toggle_skill(skill: str, learned: bool) -> None:
    data = _load()
    learned_set = set(data.get("learned", []))
    if learned:
        learned_set.add(skill)
    else:
        learned_set.discard(skill)
    data["learned"] = sorted(learned_set)
    _save(data)


def effective_score(base_score: int, matched: list, missing: list) -> int:
    """Recalculate readiness score after marking skills as learned."""
    learned = set(get_learned_skills())
    now_matched = len(matched) + len([s for s in missing if s in learned])
    total = len(matched) + len(missing)
    if total == 0:
        return base_score
    return round(now_matched / total * 100)


def node_label(node_name: str) -> str:
    return _NODE_LABELS.get(node_name, node_name.replace("_", " ").title())
