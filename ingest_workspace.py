"""Command-line ingestion entry point for AlignAgent workspaces."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from ingestion import MultiTenantIngestionPipeline


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for workspace ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF, DOCX, and TXT documents into a tenant workspace."
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Tenant workspace ID used to isolate FAISS and parent-store files.",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Document file or directory to ingest.",
    )
    parser.add_argument(
        "--base-dir",
        default="workspaces",
        help="Base directory where tenant workspaces are stored.",
    )
    return parser.parse_args()


def main() -> None:
    """Load documents from disk and ingest them into a tenant workspace."""
    load_dotenv()
    args = parse_args()
    input_path = Path(args.path)

    logger.info("🚀 AlignAgent ingestion CLI started.")
    logger.info("🏷️ Workspace: %s", args.workspace)
    logger.info("📥 Input path: %s", input_path)

    pipeline = MultiTenantIngestionPipeline(
        workspace_id=args.workspace,
        base_workspace_dir=args.base_dir,
    )
    documents = pipeline.load_documents(str(input_path))
    pipeline.ingest(documents)

    logger.info("✅ CLI ingestion finished.")


if __name__ == "__main__":
    main()
