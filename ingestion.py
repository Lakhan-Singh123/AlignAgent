"""Multi-tenant document ingestion for AlignAgent.

This module builds a tenant-isolated Parent-Child document ingestion pipeline
using LangChain, FAISS, Google GenAI embeddings, and a local parent document
store.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

try:
    import faiss
except ImportError:  # pragma: no cover - dependency error is raised at runtime.
    faiss = None

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import FastEmbedEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - older LangChain installations.
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_classic.storage import LocalFileStore
except ImportError:  # pragma: no cover - older LangChain installations.
    from langchain.storage import LocalFileStore

try:
    from langchain_classic.retrievers import ParentDocumentRetriever
except ImportError:  # pragma: no cover - older LangChain installations.
    from langchain.retrievers import ParentDocumentRetriever


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


class _PdfWithOcrLoader:
    """pdfplumber first; pytesseract OCR fallback for image-based / vector-path PDFs."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[Document]:
        text = ""

        # Step 1: pdfplumber — layout-aware, handles multi-column
        try:
            import pdfplumber
            with pdfplumber.open(self.file_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            logger.warning("pdfplumber failed for %s: %s", self.file_path, e)

        # Step 2: OCR fallback when text is too sparse
        if len(text.strip()) < 200:
            try:
                from pdf2image import convert_from_path
                import pytesseract
                images = convert_from_path(self.file_path)
                ocr_text = "\n".join(pytesseract.image_to_string(img) for img in images)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    logger.info("OCR fallback used for %s — %d chars", self.file_path, len(text.strip()))
            except Exception as e:
                logger.warning("OCR fallback failed (pdf2image/pytesseract not installed?): %s", e)

        return [Document(page_content=text, metadata={"source": self.file_path})]


class MultiTenantIngestionPipeline:
    """Ingest raw documents into isolated FAISS and parent-store workspaces."""

    PARENT_CHUNK_SIZE = 1200
    PARENT_CHUNK_OVERLAP = 200
    CHILD_CHUNK_SIZE = 300
    CHILD_CHUNK_OVERLAP = 50
    DEFAULT_EMBEDDING_DIMENSION = 384
    DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def __init__(
        self,
        workspace_id: str,
        embeddings: Embeddings | None = None,
        base_workspace_dir: str | Path = "workspaces",
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        embedding_model: str | None = None,
    ) -> None:
        """Initialize a tenant-specific ingestion pipeline.

        Args:
            workspace_id: Unique tenant/workspace identifier. It is used to
                isolate both FAISS child chunks and parent document files.
            embeddings: Optional embeddings implementation. When omitted,
                Google GenAI embeddings using ``models/text-embedding-004`` are
                created for production usage.
            base_workspace_dir: Root directory where tenant workspaces live.
            embedding_dimension: Vector dimension used when creating an empty
                FAISS index for the selected embedding model.
                ``nomic-embed-text`` returns 768 dimensions.
            embedding_model: Optional Ollama embedding model. When omitted,
                ``OLLAMA_EMBEDDING_MODEL`` is read from the environment
                before falling back to ``nomic-embed-text``.
        """
        if not workspace_id or not workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string.")

        self.workspace_id = workspace_id.strip()
        self.workspace_dir = Path(base_workspace_dir) / self.workspace_id
        self.faiss_index_dir = self.workspace_dir / "faiss_index"
        self.parent_store_dir = self.workspace_dir / "parent_store"
        self.embedding_dimension = embedding_dimension

        self.faiss_index_dir.mkdir(parents=True, exist_ok=True)
        self.parent_store_dir.mkdir(parents=True, exist_ok=True)
        logger.info("📁 Workspace ready: %s", self.workspace_dir)
        logger.info("📁 FAISS index path: %s", self.faiss_index_dir)
        logger.info("📁 Parent store path: %s", self.parent_store_dir)

        self.embedding_model = (
            embedding_model
            or os.getenv("EMBEDDING_MODEL")
            or self.DEFAULT_EMBEDDING_MODEL
        )
        self.embeddings = embeddings or FastEmbedEmbeddings(
            model_name=self.embedding_model
        )
        logger.info("🧠 Embeddings initialized: %s", self.embedding_model)

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.PARENT_CHUNK_SIZE,
            chunk_overlap=self.PARENT_CHUNK_OVERLAP,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHILD_CHUNK_SIZE,
            chunk_overlap=self.CHILD_CHUNK_OVERLAP,
        )
        logger.info(
            "✂️ Splitters initialized: parent=%s/%s child=%s/%s",
            self.PARENT_CHUNK_SIZE,
            self.PARENT_CHUNK_OVERLAP,
            self.CHILD_CHUNK_SIZE,
            self.CHILD_CHUNK_OVERLAP,
        )

        self.parent_store = LocalFileStore(str(self.parent_store_dir))
        logger.info("💾 Local parent file store initialized.")

    def load_documents(self, file_path: str) -> list[Document]:
        """Load LangChain documents from a file or directory.

        Args:
            file_path: Path to a supported file or a directory containing
                supported files. Supported extensions are PDF, DOCX, and TXT.

        Returns:
            A list of raw LangChain ``Document`` objects. Files that fail to
            parse are skipped and logged so one corrupted file does not halt the
            ingestion run.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {file_path}")

        candidate_files = list(self._iter_supported_files(path))
        logger.info("🔎 Found %s supported file(s) to load.", len(candidate_files))

        documents: list[Document] = []
        for candidate in candidate_files:
            try:
                logger.info("📄 Loading file: %s", candidate)
                loader = self._build_loader(candidate)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                logger.info("✅ Loaded %s document page(s): %s", len(loaded_docs), candidate)
            except Exception:
                logger.exception("⚠️ Failed to load file, skipping: %s", candidate)

        logger.info("📚 Total raw documents loaded: %s", len(documents))
        return documents

    def get_retriever(self) -> ParentDocumentRetriever:
        """Build a parent-document retriever for this tenant workspace.

        Returns:
            A configured ``ParentDocumentRetriever`` that uses FAISS for child
            chunk similarity search and ``LocalFileStore`` for parent document
            persistence.
        """
        vectorstore = self._load_or_create_vectorstore()
        logger.info("🔗 ParentDocumentRetriever linked to tenant storage.")
        return ParentDocumentRetriever(
            vectorstore=vectorstore,
            byte_store=self.parent_store,
            parent_splitter=self.parent_splitter,
            child_splitter=self.child_splitter,
        )

    def ingest(self, documents: list[Document]) -> None:
        """Ingest raw documents and persist child/parent chunks immediately.

        Args:
            documents: Raw LangChain ``Document`` objects to split, embed, and
                store in the tenant-specific FAISS and parent-store directories.
        """
        if not documents:
            logger.warning("⚠️ No documents provided for ingestion; skipping.")
            return

        logger.info("🚀 Starting ingestion for %s raw document(s).", len(documents))
        retriever = self.get_retriever()
        logger.info("✂️ Splitting parents and children, then embedding child chunks.")
        retriever.add_documents(documents)
        logger.info("💾 Saving FAISS child index to disk: %s", self.faiss_index_dir)
        retriever.vectorstore.save_local(str(self.faiss_index_dir))
        logger.info("✅ Ingestion complete for workspace: %s", self.workspace_id)

    def _iter_supported_files(self, path: Path) -> Iterable[Path]:
        """Yield supported files from a single path or directory tree."""
        if path.is_file():
            if path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                yield path
            else:
                logger.warning("⚠️ Unsupported file extension skipped: %s", path)
            return

        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and candidate.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                yield candidate

    def _build_loader(self, path: Path) -> _PdfWithOcrLoader | Docx2txtLoader | TextLoader:
        """Create the appropriate LangChain loader for a file path."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return _PdfWithOcrLoader(str(path))
        if suffix == ".docx":
            return Docx2txtLoader(str(path))
        if suffix == ".txt":
            return TextLoader(str(path), encoding="utf-8")
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    def _load_or_create_vectorstore(self) -> FAISS:
        """Load an existing FAISS index or create an empty one for this tenant."""
        index_file = self.faiss_index_dir / "index.faiss"
        pkl_file = self.faiss_index_dir / "index.pkl"

        if index_file.exists() and pkl_file.exists():
            logger.info("📦 Existing FAISS index found. Loading from disk.")
            return FAISS.load_local(
                str(self.faiss_index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        if faiss is None:
            raise ImportError(
                "faiss-cpu is required to create an empty FAISS vectorstore. "
                "Install it with `conda install -c conda-forge faiss-cpu` or "
                "`pip install faiss-cpu`."
            )

        logger.info("🆕 No existing FAISS index found. Creating an empty index.")
        index = faiss.IndexFlatL2(self.embedding_dimension)
        return FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
