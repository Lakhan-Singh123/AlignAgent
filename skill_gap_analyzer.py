"""Skill-gap analysis for AlignAgent candidate and internship workspaces."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from ingestion import MultiTenantIngestionPipeline


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


class ChatModel(Protocol):
    """Minimal chat-model protocol used by the analyzer."""

    def invoke(self, input: str) -> Any:
        """Generate a response from a prompt."""


@dataclass(frozen=True)
class SkillGapReport:
    """Structured skill-gap analysis result."""

    candidate_summary: str
    internship_summary: str
    matched_skills: list[str]
    missing_skills: list[str]
    learning_plan: list[dict]        # [{week, skill, goal, tasks, project}]
    resources: dict                  # {skill: [{title, url, type}]}
    timeline_weeks: int
    recommended_projects: list[str]
    readiness_score: int
    raw_response: str


class SkillGapAnalyzer:
    """Compare candidate and internship workspaces and generate a gap report."""

    DEFAULT_ANALYSIS_MODEL = "llama-3.3-70b-versatile"
    RESUME_QUERY = (
        "Candidate technical skills, tools, projects, experience, education, "
        "strengths, and internship interests."
    )
    INTERNSHIP_QUERY = (
        "Internship responsibilities, required skills, preferred skills, tools, "
        "qualifications, projects, and evaluation criteria."
    )

    def __init__(
        self,
        candidate_workspace_id: str,
        internship_workspace_id: str,
        base_workspace_dir: str | Path = "workspaces",
        llm: ChatModel | None = None,
        analysis_model: str | None = None,
        fallback_only: bool = False,
    ) -> None:
        """Initialize a skill-gap analyzer for two tenant workspaces.

        Args:
            candidate_workspace_id: Workspace containing candidate/resume data.
            internship_workspace_id: Workspace containing internship/JD data.
            base_workspace_dir: Base directory containing tenant workspaces.
            llm: Optional chat model. Tests can inject a deterministic fake.
            analysis_model: Optional Ollama model for production analysis.
            fallback_only: When true, skip the LLM and use deterministic local
                skill extraction.
        """
        self.candidate_workspace_id = candidate_workspace_id
        self.internship_workspace_id = internship_workspace_id
        self.base_workspace_dir = base_workspace_dir
        self.fallback_only = fallback_only
        if llm is not None:
            self.llm = llm
        elif self.fallback_only:
            self.llm = None
        else:
            self.llm = ChatGroq(
                model=analysis_model
                or os.getenv("GROQ_MODEL")
                or self.DEFAULT_ANALYSIS_MODEL,
                temperature=0.2,
                groq_api_key=os.getenv("GROQ_API_KEY"),
            )

    def analyze(self) -> SkillGapReport:
        """Retrieve evidence, generate a report, and audit it for quality."""
        logger.info("🚀 Starting skill-gap analysis.")
        
        # 1. Retrieve Context
        candidate_docs = self._retrieve_context(self.candidate_workspace_id, self.RESUME_QUERY)
        internship_docs = self._retrieve_context(self.internship_workspace_id, self.INTERNSHIP_QUERY)
        candidate_context = self._format_documents(candidate_docs)
        internship_context = self._format_documents(internship_docs)
        
        # 2. Generate Initial Report
        if self.fallback_only:
            logger.info("🧮 Fallback-only mode enabled; skipping LLM analysis.")
            raw_response = self._build_fallback_report(candidate_context, internship_context)
        else:
            try:
                logger.info("🧠 Generating initial report with Ollama.")
                prompt = self._build_prompt(candidate_context, internship_context)
                response = self.llm.invoke(prompt)
                raw_response = self._extract_text(response)
                
                # 3. Trigger Auditor (The new quality gate)
                logger.info("🕵️ Running Auditor Node.")
                report_to_audit = self._parse_report(raw_response)
                audited_report = self.audit_report(report_to_audit, candidate_context, internship_context)
                raw_response = audited_report.raw_response
                
            except Exception:
                logger.exception("⚠️ LLM analysis failed; using local fallback analyzer.")
                raw_response = self._build_fallback_report(candidate_context, internship_context)
        
        report = self._parse_report(raw_response)
        logger.info("✅ Skill-gap analysis complete.")
        return report

    def save_report(self, report: SkillGapReport, output_path: str | Path) -> None:
        """Save a skill-gap report as pretty JSON.

        Args:
            report: Structured report returned by ``analyze``.
            output_path: Destination JSON file path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "candidate_summary": report.candidate_summary,
            "internship_summary": report.internship_summary,
            "matched_skills": report.matched_skills,
            "missing_skills": report.missing_skills,
            "learning_plan": report.learning_plan,
            "resources": report.resources,
            "timeline_weeks": report.timeline_weeks,
            "recommended_projects": report.recommended_projects,
            "readiness_score": report.readiness_score,
            "raw_response": report.raw_response,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("💾 Skill-gap report saved: %s", path)

    def _retrieve_context(self, workspace_id: str, query: str) -> list[Document]:
        """Retrieve parent documents for a query from one workspace."""
        logger.info("🔎 Retrieving evidence from workspace: %s", workspace_id)
        pipeline = MultiTenantIngestionPipeline(
            workspace_id=workspace_id,
            base_workspace_dir=self.base_workspace_dir,
        )
        retriever = pipeline.get_retriever()
        documents = retriever.invoke(query)
        logger.info("📚 Retrieved %s parent document(s).", len(documents))
        return documents

    def _format_documents(self, documents: list[Document]) -> str:
        """Format retrieved documents into compact prompt context."""
        if not documents:
            return "No relevant documents were retrieved."

        sections: list[str] = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "unknown")
            content = document.page_content.strip()
            sections.append(f"[Document {index} | source={source}]\n{content}")
        return "\n\n".join(sections)
    
    def audit_report(self, report: SkillGapReport, candidate_context: str, internship_context: str) -> SkillGapReport:
        """Audits the generated report for hallucinations or missing evidence."""
        
        audit_prompt = f"""
        You are a strict data auditor for AlignAgent. 
        Verify the following generated report against the provided source documents.
        
        Report: {report.raw_response}
        
        Source Documents:
        Candidate: {candidate_context}
        Internship: {internship_context}
        
        Task:
        1. Check if every skill in 'matched_skills' is supported by the Candidate evidence.
        2. If a skill is unsupported, mark it as a hallucination.
        3. If you find hallucinations, output the corrected JSON. 
        4. If the report is accurate, output the original report.
        """
        
        # Invoke the LLM again to audit its own work
        audit_response = self.llm.invoke(audit_prompt)
        
        # Parse the corrected JSON and return a new SkillGapReport
        return self._parse_report(self._extract_text(audit_response))
    
    def _build_prompt(self, candidate_context: str, internship_context: str) -> str:
        """Build the structured JSON-only prompt for Ollama."""
        return f"""
You are AlignAgent, an internship readiness and skill-gap analyzer.

Compare the candidate evidence against the internship evidence. Be specific,
honest, and practical. Use only the evidence provided. If something is missing
from the evidence, mark it as a gap instead of inventing it.

Return only valid JSON with exactly these keys:
- candidate_summary: string
- internship_summary: string
- matched_skills: array of strings
- missing_skills: array of strings
- learning_plan: array of strings
- recommended_projects: array of strings
- readiness_score: integer from 0 to 100

Candidate evidence:
{candidate_context}

Internship evidence:
{internship_context}
""".strip()

    def _extract_text(self, response: Any) -> str:
        """Extract plain text from a LangChain chat response."""
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()

    def _parse_report(self, raw_response: str) -> SkillGapReport:
        """Parse a JSON model response into a ``SkillGapReport``."""
        cleaned = self._strip_code_fence(raw_response)
        data = json.loads(cleaned)
        return SkillGapReport(
            candidate_summary=str(data.get("candidate_summary", "")),
            internship_summary=str(data.get("internship_summary", "")),
            matched_skills=list(data.get("matched_skills", [])),
            missing_skills=list(data.get("missing_skills", [])),
            learning_plan=list(data.get("learning_plan", [])),
            resources=dict(data.get("resources", {})),
            timeline_weeks=int(data.get("timeline_weeks", 0)),
            recommended_projects=list(data.get("recommended_projects", [])),
            readiness_score=int(data.get("readiness_score", 0)),
            raw_response=raw_response,
        )

    def _build_fallback_report(
        self,
        candidate_context: str,
        internship_context: str,
    ) -> str:
        """Build a deterministic report when the LLM is unavailable.

        The fallback is intentionally simple: it extracts known technical skill
        terms from both evidence blocks, computes overlap and gaps, and returns
        the same JSON schema as the LLM prompt.
        """
        candidate_skills = self._extract_candidate_skills(candidate_context)
        internship_skills = self._extract_skills(internship_context)
        matched_skills = sorted(candidate_skills & internship_skills)
        missing_skills = sorted(internship_skills - candidate_skills)
        readiness_score = self._calculate_readiness_score(
            matched_skills=matched_skills,
            missing_skills=missing_skills,
        )

        learning_plan = [
            {
                "week": i + 1,
                "skill": skill,
                "goal": f"Gain working knowledge of {skill}.",
                "tasks": [f"Read official docs for {skill}", f"Build a small demo using {skill}"],
                "project": f"Mini-project applying {skill} to a real use case.",
            }
            for i, skill in enumerate(missing_skills[:5])
        ]
        if not learning_plan:
            learning_plan = [
                {
                    "week": 1,
                    "skill": "Depth",
                    "goal": "Deepen matched skills with production-style work.",
                    "tasks": ["Pick one matched skill and build a production-quality project"],
                    "project": "Deploy and monitor a small service using matched skills.",
                }
            ]

        resources = {
            skill: [{"title": f"{skill} — official documentation", "url": "", "type": "docs"}]
            for skill in missing_skills
        }

        recommended_projects = [
            "Build a resume and internship matching dashboard using Python APIs.",
            "Create a RAG-based document parser with FAISS-backed retrieval.",
        ]
        if {"Docker", "CI/CD"} & set(missing_skills):
            recommended_projects.append(
                "Dockerize the matching service and deploy it with a CI/CD pipeline."
            )

        payload = {
            "candidate_summary": (
                "Candidate evidence shows skills in "
                f"{', '.join(sorted(candidate_skills)) or 'no extracted technical skills'}."
            ),
            "internship_summary": (
                "Internship evidence emphasizes "
                f"{', '.join(sorted(internship_skills)) or 'no extracted technical skills'}."
            ),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "learning_plan": learning_plan,
            "resources": resources,
            "timeline_weeks": len(learning_plan),
            "recommended_projects": recommended_projects,
            "readiness_score": readiness_score,
        }
        return json.dumps(payload)

    def _extract_candidate_skills(self, text: str) -> set[str]:
        """Extract candidate skills while removing explicitly stated gaps.

        Two-layer defence:
          1. Strip sentences signalling a gap before scanning for skills, so a
             keyword buried in "I am currently learning Docker" never enters
             the acquired set in the first place.
          2. Belt-and-suspenders: subtract gap skills found in the full text.
        """
        gap_sentence_re = re.compile(
            r"[^.!\n]*"
            r"(?:gaps?\s*(?:include|:)"
            r"|currently\s+learning"
            r"|needs?\s+to\s+(?:learn|improve)"
            r"|missing"
            r"|weak(?:ness)?\s+in"
            r"|hoping\s+to\s+learn"
            r"|plan(?:ning)?\s+to\s+learn"
            r"|(?:want|looking)\s+to\s+(?:learn|develop))"
            r"[^.!\n]*[.!\n]?",
            flags=re.IGNORECASE,
        )
        clean_text = gap_sentence_re.sub("", text)
        candidate_skills = self._extract_skills(clean_text)
        candidate_gap_skills = self._extract_candidate_gap_skills(text)
        return candidate_skills - candidate_gap_skills

    def _extract_candidate_gap_skills(self, text: str) -> set[str]:
        """Extract skills mentioned inside candidate gap statements.

        Covers: "Gaps: X", "gaps include X", "currently learning X",
        "missing X", "needs to learn X", "need to improve in X",
        "weak in X", "hoping/planning to learn X", "want to learn X".
        """
        gap_patterns = [
            r"(?:current\s+)?gaps?\s*(?:include|:)\s*([^.\n]+)",
            r"(?:missing|needs?\s+to\s+(?:learn|improve(?:\s+in)?))\s+([^.\n]+)",
            r"(?:currently\s+)?learning\s+([^.\n]+)",
            r"(?:hoping|plan(?:ning)?)\s+to\s+learn\s+([^.\n]+)",
            r"weak(?:ness)?\s+in\s+([^.\n]+)",
            r"need\s+to\s+improve\s+(?:in\s+)?([^.\n]+)",
            r"(?:want|looking)\s+to\s+(?:learn|develop)\s+([^.\n]+)",
        ]
        gap_text = " ".join(
            match.group(1)
            for pattern in gap_patterns
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        )
        return self._extract_skills(gap_text)

    def _extract_skills(self, text: str) -> set[str]:
        """Extract known technical skill terms from evidence text."""
        skill_patterns = {
            "Python": r"\bpython\b",
            "SQL": r"\bsql\b",
            "Machine Learning": r"\b(machine learning|ml)\b",
            "Data Analysis": r"\bdata analysis\b",
            "Flask": r"\bflask\b",
            "Git": r"\bgit\b",
            "Cloud Deployment": r"\bcloud deployment\b",
            "Pandas": r"\bpandas\b",
            "NumPy": r"\bnumpy\b",
            "REST APIs": r"\b(rest api|rest apis|apis?)\b",
            "HTML": r"\bhtml\b",
            "CSS": r"\bcss\b",
            "JavaScript": r"\bjavascript\b",
            "LangChain": r"\blangchain\b",
            "Embeddings": r"\bembeddings?\b",
            "Vector Databases": r"\bvector databases?\b",
            "RAG": r"\b(rag|retrieval-augmented generation)\b",
            "Docker": r"\bdocker\b",
            "CI/CD": r"\bci/cd\b",
            "System Design": r"\bsystem design\b",
            "Scalable Databases": r"\bscalable database design|scalable databases\b",
            "LLM Evaluation": r"\bevaluation of llm outputs|llm evaluation\b",
            "Production Monitoring": r"\bproduction monitoring\b",
            "Technical Communication": r"\btechnical communication\b",
            "Document Parsing": r"\bdocument parsing\b",
        }
        found: set[str] = set()
        for skill, pattern in skill_patterns.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                found.add(skill)
        return found

    def _calculate_readiness_score(
        self,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> int:
        """Calculate a simple readiness score from matched and missing skills."""
        total = len(matched_skills) + len(missing_skills)
        if total == 0:
            return 0
        return round((len(matched_skills) / total) * 100)

    def _strip_code_fence(self, text: str) -> str:
        """Remove Markdown code fences around a JSON response if present."""
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()


def parse_args() -> argparse.Namespace:
    """Parse command-line options for skill-gap analysis."""
    parser = argparse.ArgumentParser(
        description="Generate a skill-gap report from candidate and internship workspaces."
    )
    parser.add_argument("--candidate-workspace", required=True)
    parser.add_argument("--internship-workspace", required=True)
    parser.add_argument("--base-dir", default="workspaces")
    parser.add_argument(
        "--output",
        default="reports/skill_gap_report.json",
        help="Path where the JSON report should be saved.",
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip Ollama analysis and use the deterministic local analyzer.",
    )
    return parser.parse_args()


def main() -> None:
    """Run skill-gap analysis from the command line."""
    load_dotenv()
    args = parse_args()
    analyzer = SkillGapAnalyzer(
        candidate_workspace_id=args.candidate_workspace,
        internship_workspace_id=args.internship_workspace,
        base_workspace_dir=args.base_dir,
        fallback_only=args.fallback_only,
    )
    report = analyzer.analyze()
    analyzer.save_report(report, args.output)
    print(json.dumps(json.loads(report.raw_response), indent=2))


if __name__ == "__main__":
    main()