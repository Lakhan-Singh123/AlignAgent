"""Bridge between the Streamlit UI and the full agentic LangGraph pipeline."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from ingestion import MultiTenantIngestionPipeline
from graph import app as agent_graph


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
    Returns the final state dict as the last yielded value.
    """
    session_id   = uuid.uuid4().hex[:8]
    candidate_id = f"session_{session_id}_candidate"
    internship_id = f"session_{session_id}_internship"

    # ── Ingest resume ──────────────────────────────────────────────────────────
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(file_bytes)
        resume_path = f.name

    pipeline = MultiTenantIngestionPipeline(
        workspace_id=candidate_id,
        base_workspace_dir=base_workspace_dir,
    )
    pipeline.ingest(pipeline.load_documents(resume_path))

    # ── Ingest job description ─────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
        f.write(jd_text)
        jd_path = f.name

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

    final_state = initial_state
    for event in agent_graph.stream(initial_state):
        final_state = event
        yield event   # caller uses this for progress display

    return final_state
