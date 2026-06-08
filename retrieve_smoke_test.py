"""Smoke-test retrieval from an AlignAgent tenant workspace."""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from ingestion import MultiTenantIngestionPipeline


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a retrieval smoke test."""
    parser = argparse.ArgumentParser(
        description="Query an existing AlignAgent workspace and print parent chunks."
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Tenant workspace ID to query.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Search query to run against the workspace.",
    )
    parser.add_argument(
        "--base-dir",
        default="workspaces",
        help="Base directory where tenant workspaces are stored.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of parent chunks to print.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a retrieval query and print the returned parent documents."""
    load_dotenv()
    args = parse_args()
    logger.info("🔎 AlignAgent retrieval smoke test started.")
    logger.info("🏷️ Workspace: %s", args.workspace)
    logger.info("❓ Query: %s", args.query)

    pipeline = MultiTenantIngestionPipeline(
        workspace_id=args.workspace,
        base_workspace_dir=args.base_dir,
    )
    retriever = pipeline.get_retriever()
    documents = retriever.invoke(args.query)

    print(f"\nRetrieved {len(documents)} parent document(s).\n")
    for index, document in enumerate(documents[: args.limit], start=1):
        source = document.metadata.get("source", "unknown")
        preview = document.page_content.strip().replace("\n", " ")
        print(f"--- Result {index} | source={source} ---")
        print(preview[:1200])
        print()

    logger.info("✅ Retrieval smoke test finished.")


if __name__ == "__main__":
    main()
