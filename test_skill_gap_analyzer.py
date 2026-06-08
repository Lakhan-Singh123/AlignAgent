"""Tests for Phase 2 skill-gap analysis."""

from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion import MultiTenantIngestionPipeline
from skill_gap_analyzer import SkillGapAnalyzer


class DeterministicFakeEmbeddings(Embeddings):
    """Small deterministic embedding model for local analyzer tests."""

    def __init__(self, dimension: int = 8) -> None:
        """Create a fake embeddings provider with a fixed vector dimension."""
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents into deterministic fixed-size vectors."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query into a deterministic fixed-size vector."""
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        """Create a deterministic vector based on character positions."""
        vector = [0.0] * self.dimension
        for index, character in enumerate(text):
            vector[index % self.dimension] += float(ord(character) % 31) / 31.0
        return vector


class FakeChatModel:
    """Deterministic chat model that returns a valid skill-gap JSON report."""

    def invoke(self, input: str) -> str:
        """Return a static report after verifying prompt evidence is present."""
        assert "Candidate evidence:" in input
        assert "Internship evidence:" in input
        return json.dumps(
            {
                "candidate_summary": "Candidate has Python and SQL experience.",
                "internship_summary": "Internship requires Python, SQL, Docker, and CI/CD.",
                "matched_skills": ["Python", "SQL"],
                "missing_skills": ["Docker", "CI/CD"],
                "learning_plan": [
                    "Build a Dockerized Python API.",
                    "Add automated tests and CI/CD.",
                ],
                "recommended_projects": [
                    "Internship matching dashboard with Docker deployment."
                ],
                "readiness_score": 72,
            }
        )


def test_skill_gap_analyzer_generates_structured_report(tmp_path, monkeypatch) -> None:
    """Analyze two populated workspaces and return a structured report."""
    embeddings = DeterministicFakeEmbeddings()
    base_dir = tmp_path / "workspaces"

    candidate_pipeline = MultiTenantIngestionPipeline(
        workspace_id="candidate",
        embeddings=embeddings,
        base_workspace_dir=base_dir,
        embedding_dimension=embeddings.dimension,
    )
    candidate_pipeline.ingest(
        [
            Document(
                page_content="Candidate knows Python, SQL, Pandas, Flask, and Git.",
                metadata={"source": "resume"},
            )
        ]
    )

    internship_pipeline = MultiTenantIngestionPipeline(
        workspace_id="internship",
        embeddings=embeddings,
        base_workspace_dir=base_dir,
        embedding_dimension=embeddings.dimension,
    )
    internship_pipeline.ingest(
        [
            Document(
                page_content="Internship requires Python, SQL, Docker, CI/CD, and APIs.",
                metadata={"source": "job-description"},
            )
        ]
    )

    original_init = MultiTenantIngestionPipeline.__init__

    def patched_init(self, *args, **kwargs) -> None:
        kwargs["embeddings"] = embeddings
        kwargs["embedding_dimension"] = embeddings.dimension
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(MultiTenantIngestionPipeline, "__init__", patched_init)
    analyzer = SkillGapAnalyzer(
        candidate_workspace_id="candidate",
        internship_workspace_id="internship",
        base_workspace_dir=base_dir,
        llm=FakeChatModel(),
    )

    report = analyzer.analyze()

    assert report.readiness_score == 72
    assert "Python" in report.matched_skills
    assert "Docker" in report.missing_skills
