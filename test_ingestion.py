"""Tests for the AlignAgent ingestion pipeline."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion import MultiTenantIngestionPipeline


class DeterministicFakeEmbeddings(Embeddings):
    """Small deterministic embedding model for local ingestion tests."""

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


def test_ingest_sample_document_persists_faiss_and_parent_store(tmp_path) -> None:
    """Ingest a sample Document and verify tenant files are persisted."""
    workspace_id = "test_workspace"
    embeddings = DeterministicFakeEmbeddings()
    pipeline = MultiTenantIngestionPipeline(
        workspace_id=workspace_id,
        embeddings=embeddings,
        base_workspace_dir=tmp_path / "workspaces",
        embedding_dimension=embeddings.dimension,
    )
    documents = [
        Document(
            page_content=(
                "AlignAgent analyzes internship descriptions, compares them "
                "against candidate skills, and identifies practical learning gaps."
            ),
            metadata={"source": "sample-text"},
        )
    ]

    pipeline.ingest(documents)

    assert (pipeline.faiss_index_dir / "index.faiss").exists()
    assert (pipeline.faiss_index_dir / "index.pkl").exists()
    assert any(pipeline.parent_store_dir.iterdir())


def test_get_retriever_loads_existing_faiss_index(tmp_path) -> None:
    """Verify an existing tenant FAISS index can be loaded without crashing."""
    workspace_id = "reload_workspace"
    embeddings = DeterministicFakeEmbeddings()
    kwargs = {
        "workspace_id": workspace_id,
        "embeddings": embeddings,
        "base_workspace_dir": tmp_path / "workspaces",
        "embedding_dimension": embeddings.dimension,
    }
    first_pipeline = MultiTenantIngestionPipeline(**kwargs)
    first_pipeline.ingest([Document(page_content="A reusable tenant document.")])

    second_pipeline = MultiTenantIngestionPipeline(**kwargs)
    retriever = second_pipeline.get_retriever()

    assert retriever.vectorstore.index.ntotal > 0
