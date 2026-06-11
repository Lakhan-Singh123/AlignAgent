"""Bridge between the Streamlit UI and the full agentic LangGraph pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from ingestion import MultiTenantIngestionPipeline
from graph import app as agent_graph

logger = logging.getLogger(__name__)


def _cleanup(paths: list[str]) -> None:
    """Delete temp files and workspace directories silently."""
    for p in paths:
        try:
            path = Path(p)
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except Exception:
            pass


def run_agentic_analysis(
    file_bytes: bytes,
    filename: str,
    jd_text: str,
    base_workspace_dir: str = "workspaces",
):
    """
    Ingest an uploaded resume + job description into temporary workspaces,
    then run the full 8-node agentic graph.

    Yields LangGraph stream events so the caller can show real-time progress.
    Cleans up temp files and session workspaces after the graph completes.
    """
    session_id    = uuid.uuid4().hex[:8]
    candidate_id  = f"session_{session_id}_candidate"
    internship_id = f"session_{session_id}_internship"

    workspace_base = Path(base_workspace_dir)
    candidate_dir  = workspace_base / candidate_id
    internship_dir = workspace_base / internship_id

    temp_files: list[str] = []

    # ── Ingest resume ──────────────────────────────────────────────────────────
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(file_bytes)
        resume_path = f.name
    temp_files.append(resume_path)

    pipeline = MultiTenantIngestionPipeline(
        workspace_id=candidate_id,
        base_workspace_dir=base_workspace_dir,
    )
    pipeline.ingest(pipeline.load_documents(resume_path))

    # ── Ingest job description ─────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
        f.write(jd_text)
        jd_path = f.name
    temp_files.append(jd_path)

    jd_pipeline = MultiTenantIngestionPipeline(
        workspace_id=internship_id,
        base_workspace_dir=base_workspace_dir,
    )
    jd_pipeline.ingest(jd_pipeline.load_documents(jd_path))

    # ── Run graph (streaming) ──────────────────────────────────────────────────
    initial_state = {
        "candidate_id":         candidate_id,
        "internship_id":        internship_id,
        "analysis_report_path": f"reports/{session_id}_report.json",
        "candidate_queries":    [],
        "internship_queries":   [],
        "candidate_docs":       [],
        "internship_docs":      [],
        "candidate_context":    "",
        "internship_context":   "",
        "web_context":          "",
        "raw_report":           "",
        "resources":            {},
        "grounding_passed":     False,
        "retry_count":          0,
    }

    try:
        for event in agent_graph.stream(initial_state):
            yield event
    finally:
        # Always clean up — even if the graph errors mid-run
        _cleanup(temp_files + [str(candidate_dir), str(internship_dir)])
        logger.info("🧹 Cleaned up session workspaces for %s", session_id)
