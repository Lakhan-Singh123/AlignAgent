import json
import logging
import os
import re

from dotenv import load_dotenv
try:
    from langchain_groq import ChatGroq
except ImportError as e:
    raise ImportError(
        "langchain-groq is not installed. Run: pip install langchain-groq"
    ) from e
from pypdf import PdfReader

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Hard cap so we never overflow llama3.2's context window
_MAX_RESUME_CHARS = 3000
_MAX_JD_CHARS     = 2000

# Skills the local fallback knows how to detect
_SKILL_PATTERNS = {
    "Python": r"\bpython\b",
    "SQL": r"\bsql\b",
    "Machine Learning": r"\b(machine.?learning|ml)\b",
    "Data Analysis": r"\bdata.?analysis\b",
    "Flask": r"\bflask\b",
    "FastAPI": r"\bfastapi\b",
    "Django": r"\bdjango\b",
    "Git": r"\bgit\b",
    "Docker": r"\bdocker\b",
    "CI/CD": r"\bci/?cd\b",
    "Pandas": r"\bpandas\b",
    "NumPy": r"\bnumpy\b",
    "REST APIs": r"\b(rest.?api|apis?)\b",
    "HTML": r"\bhtml\b",
    "CSS": r"\bcss\b",
    "JavaScript": r"\bjavascript\b",
    "TypeScript": r"\btypescript\b",
    "React": r"\breact\b",
    "LangChain": r"\blangchain\b",
    "RAG": r"\b(rag|retrieval.?augmented)\b",
    "Embeddings": r"\bembeddings?\b",
    "Vector Databases": r"\bvector.?database\b",
    "LLM Evaluation": r"\b(llm.?eval|evaluation.?of.?llm)\b",
    "Cloud Deployment": r"\bcloud.?deploy\b",
    "System Design": r"\bsystem.?design\b",
    "Production Monitoring": r"\bproduction.?monitor\b",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


def _extract_json(text: str) -> dict:
    """Strip fences and pull the first JSON object out of any LLM response.
    Attempts to recover truncated JSON by closing open brackets."""
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    raw = match.group() if match else cleaned

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to repair truncated JSON by closing open structures
        repaired = _repair_json(raw)
        return json.loads(repaired)


def _repair_json(text: str) -> str:
    """Truncate at the last fully-closed JSON object boundary, close remaining brackets."""
    closing = {"[": "]", "{": "}"}
    stack = []
    in_string = False
    escape_next = False
    # last_closed[depth] = last index+1 where depth went from d+1 to d
    last_good_pos = 0  # after the last char where depth was 1 (inside top object)
    depth = 0

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
            depth += 1
        elif ch in "}]":
            if stack:
                stack.pop()
            depth -= 1
            # Record any position where we just finished a child element (depth back to 1)
            if depth == 1:
                last_good_pos = i + 1

    if last_good_pos:
        truncated = text[:last_good_pos].rstrip().rstrip(",")
        # Re-compute open brackets for this truncated slice
        open_stack = []
        in_s = False
        esc = False
        for ch in truncated:
            if esc: esc = False; continue
            if ch == "\\" and in_s: esc = True; continue
            if ch == '"': in_s = not in_s; continue
            if in_s: continue
            if ch in "{[": open_stack.append(ch)
            elif ch in "}]" and open_stack: open_stack.pop()
        for ch in reversed(open_stack):
            truncated += closing.get(ch, "")
        return truncated

    # Nothing safely closed — just close what's open
    result = text.rstrip().rstrip(",")
    for ch in reversed(stack):
        result += closing.get(ch, "")
    return result


def _local_skills(text: str) -> set:
    return {skill for skill, pat in _SKILL_PATTERNS.items()
            if re.search(pat, text, re.IGNORECASE)}


# ── PDF Extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ── Step 1: Skills + Score ────────────────────────────────────────────────────

def _get_skills_and_score(model, resume: str, jd: str) -> dict:
    prompt = f"""Compare this resume against the job description.

RESUME:
{resume}

JOB DESCRIPTION:
{jd}

Return ONLY a JSON object — no prose, no markdown:
{{"matched_skills": ["skill1"], "missing_skills": ["skill2"], "readiness_score": 70}}"""

    raw = model.invoke(prompt).content
    logger.info("Step 1 raw: %s", raw[:200])
    return _extract_json(raw)


# ── Step 2: Learning Plan + Resources ────────────────────────────────────────

def _get_plan_and_resources(model, missing_skills: list, jd: str) -> dict:
    if not missing_skills:
        return {"learning_plan": [], "resources": {}, "timeline_weeks": 0,
                "recommended_projects": []}

    top_skills = missing_skills[:5]
    skills_str = ", ".join(top_skills)

    # ── 2a: learning plan only (one skill at a time to stay short) ────────────
    plan_prompt = f"""Write a week-by-week study plan. Skills to cover: {skills_str}

Return ONLY this JSON array (no markdown, keep values short — under 12 words each):
[
  {{"week": 1, "skill": "X", "goal": "short goal", "tasks": ["task 1", "task 2"], "project": "mini project"}},
  {{"week": 2, "skill": "Y", "goal": "short goal", "tasks": ["task 1", "task 2"], "project": "mini project"}}
]"""

    try:
        raw = model.invoke(plan_prompt).content
        logger.info("Step 2a raw: %s", raw[:150])
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        arr_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        plan = json.loads(arr_match.group()) if arr_match else []
    except Exception as e:
        logger.warning("Step 2a failed: %s", e)
        plan = [
            {"week": i + 1, "skill": s, "goal": f"Learn the fundamentals of {s}.",
             "tasks": [f"Read {s} docs", f"Build a small {s} demo"],
             "project": f"Apply {s} in a practical mini-project."}
            for i, s in enumerate(top_skills)
        ]

    # ── 2b: resources only ────────────────────────────────────────────────────
    res_prompt = f"""Suggest ONE free learning resource per skill for: {skills_str}

Return ONLY this JSON object (no markdown):
{{
  "skill name": [{{"title": "resource title", "url": "url or empty string", "type": "course"}}]
}}"""

    try:
        raw2 = model.invoke(res_prompt).content
        logger.info("Step 2b raw: %s", raw2[:150])
        resources = _extract_json(raw2)
    except Exception as e:
        logger.warning("Step 2b failed: %s", e)
        resources = {s: [{"title": f"{s} — official docs", "url": "", "type": "docs"}]
                     for s in top_skills}

    return {
        "learning_plan": plan,
        "resources": resources,
        "timeline_weeks": len(top_skills),
        "recommended_projects": [
            "Build a portfolio project combining your matched skills.",
            "Contribute to an open-source project in your target domain.",
        ],
    }


# ── Local Fallback ────────────────────────────────────────────────────────────

def _local_fallback(resume: str, jd: str) -> dict:
    logger.warning("⚠️  Using local fallback analyser.")
    candidate_skills = _local_skills(resume)
    required_skills  = _local_skills(jd)
    matched  = sorted(candidate_skills & required_skills)
    missing  = sorted(required_skills - candidate_skills)
    total    = len(matched) + len(missing)
    score    = round(len(matched) / total * 100) if total else 0

    plan = [
        {"week": i + 1, "skill": s,
         "goal": f"Build working knowledge of {s}.",
         "tasks": [f"Read the official {s} docs", f"Build a small demo with {s}"],
         "project": f"Apply {s} in a mini end-to-end project."}
        for i, s in enumerate(missing[:6])
    ]
    resources = {
        s: [{"title": f"{s} — official documentation", "url": "", "type": "docs"}]
        for s in missing
    }
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "readiness_score": score,
        "learning_plan": plan,
        "resources": resources,
        "timeline_weeks": len(plan),
        "recommended_projects": [
            "Build a portfolio project combining your matched skills.",
            "Contribute to an open-source project in your target domain.",
        ],
    }


# ── URL Scraper ───────────────────────────────────────────────────────────────

def scrape_jd_from_url(url: str) -> str:
    """Fetch a job description page and return its visible text."""
    import requests
    from bs4 import BeautifulSoup
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return "\n".join(lines)[:_MAX_JD_CHARS * 2]
    except Exception as e:
        raise ValueError(f"Could not scrape URL: {e}")


# ── Interview Questions ───────────────────────────────────────────────────────

def get_interview_questions(missing_skills: list, matched_skills: list, jd: str) -> list[str]:
    """Generate likely interview questions based on the skill gap."""
    if not missing_skills and not matched_skills:
        return []
    model = _llm()
    skills_str = ", ".join((missing_skills + matched_skills)[:8])
    prompt = f"""You are an interview coach. Generate 6 realistic interview questions for a candidate
applying for this role: {jd[:400]}

Focus on these skills: {skills_str}
Include 3 questions about skills they might lack, and 3 about skills they have.

Return ONLY a JSON array of 6 question strings (no markdown):
["Question 1?", "Question 2?", ...]"""
    try:
        raw = model.invoke(prompt).content
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        return json.loads(match.group()) if match else []
    except Exception as e:
        logger.warning("Interview questions failed: %s", e)
        return [f"Tell me about your experience with {s}?" for s in (missing_skills + matched_skills)[:6]]


# ── Resume Suggestions ────────────────────────────────────────────────────────

def get_resume_suggestions(resume_text: str, jd: str, missing_skills: list) -> list[str]:
    """Return actionable suggestions to improve the resume for this specific JD."""
    model = _llm()
    prompt = f"""You are a professional resume coach. Review this resume against the job description.

RESUME (excerpt):
{resume_text[:1500]}

JOB DESCRIPTION (excerpt):
{jd[:600]}

Missing skills: {', '.join(missing_skills[:5])}

Give 5 specific, actionable suggestions to improve the resume for this role.
Focus on: missing keywords, weak phrasing, missing sections, quantifying impact.

Return ONLY a JSON array of 5 suggestion strings (no markdown):
["Suggestion 1", "Suggestion 2", ...]"""
    try:
        raw = model.invoke(prompt).content
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        return json.loads(match.group()) if match else []
    except Exception as e:
        logger.warning("Resume suggestions failed: %s", e)
        return [
            "Add measurable impact to your bullet points (e.g., 'increased X by Y%').",
            f"Add a Skills section listing: {', '.join(missing_skills[:3])} if you have any exposure.",
            "Use keywords from the job description in your experience bullet points.",
            "Quantify projects with numbers: users, uptime %, lines of code, etc.",
            "Add a summary section tailored to this specific role.",
        ]


# ── ATS Score Checker ────────────────────────────────────────────────────────

_ATS_SECTIONS = {
    "Contact Info":  r"[\w.+-]+@[\w-]+\.[a-z]{2,}|\+?\d[\d\s\-()]{7,}",
    "LinkedIn":      r"linkedin\.com/in/",
    "Education":     r"\b(education|university|college|degree|bachelor|master|b\.?tech|m\.?tech|b\.?sc|m\.?sc|gpa)\b",
    "Experience":    r"\b(experience|work history|employment|internship|intern at|worked at|job)\b",
    "Skills":        r"\b(skills|technical skills|technologies|proficiencies|competencies)\b",
    "Projects":      r"\b(projects?|portfolio|built|developed|created|implemented)\b",
    "Achievements":  r"\d+\s*(%|percent|users?|customers?|ms\b|x\b|times?|hours?|days?|months?)",
}

_ATS_ACTION_VERBS = [
    "developed","built","designed","implemented","led","managed","created","improved",
    "increased","reduced","optimised","optimized","delivered","launched","automated",
    "architected","analysed","analyzed","collaborated","deployed","maintained",
]


def _ats_local(resume_text: str, jd_text: str) -> dict:
    """Fast regex-based ATS checks — no API call."""
    text_lower = resume_text.lower()

    # Section presence
    section_hits = {s: bool(re.search(p, resume_text, re.IGNORECASE))
                    for s, p in _ATS_SECTIONS.items()}
    section_score = round(sum(section_hits.values()) / len(section_hits) * 100)

    # Keyword match (skills from JD vs resume)
    jd_skills   = {s for s, p in _SKILL_PATTERNS.items() if re.search(p, jd_text, re.IGNORECASE)}
    resume_skills = {s for s, p in _SKILL_PATTERNS.items() if re.search(p, resume_text, re.IGNORECASE)}
    kw_found   = sorted(jd_skills & resume_skills)
    kw_missing = sorted(jd_skills - resume_skills)
    kw_score   = round(len(kw_found) / len(jd_skills) * 100) if jd_skills else 50

    # Action verbs
    verb_count = sum(1 for v in _ATS_ACTION_VERBS if v in text_lower)
    verb_score = min(100, verb_count * 12)

    # Word count
    word_count = len(resume_text.split())
    length_ok  = 200 <= word_count <= 900

    # Build issues list
    issues = []
    if not section_hits["Contact Info"]:
        issues.append("No email or phone number detected — ATS cannot contact you")
    if not section_hits["LinkedIn"]:
        issues.append("LinkedIn URL missing — many ATS and recruiters check this")
    if not section_hits["Education"]:
        issues.append("Education section not clearly labelled")
    if not section_hits["Experience"]:
        issues.append("Experience section not clearly labelled")
    if not section_hits["Skills"]:
        issues.append("No dedicated Skills section — ATS may miss your tech stack")
    if not section_hits["Achievements"]:
        issues.append("No quantified achievements found (add numbers, %, impact metrics)")
    if verb_count < 3:
        issues.append("Too few strong action verbs — use 'built', 'led', 'improved', etc.")
    if not length_ok:
        issues.append(
            "Resume too short (under 200 words)" if word_count < 200
            else "Resume may be too long for a 1-page ATS parse (over 900 words)"
        )

    # Overall ATS score (weighted)
    ats_score = round(kw_score * 0.40 + section_score * 0.35 + verb_score * 0.25)

    return {
        "ats_score":      ats_score,
        "keyword_score":  kw_score,
        "section_score":  section_score,
        "keywords_found": kw_found,
        "keywords_missing": kw_missing,
        "issues":         issues,
        "word_count":     word_count,
    }


def check_ats_score(resume_text: str, job_description: str) -> dict:
    """Return ATS compatibility report (local checks + LLM suggestions)."""
    local = _ats_local(resume_text, job_description)

    # LLM for specific, actionable suggestions
    suggestions = []
    try:
        model = _llm()
        prompt = f"""You are an ATS (Applicant Tracking System) expert.
Review this resume against the job description and provide 5 specific, actionable suggestions
to improve ATS compatibility. Be concise — one sentence per suggestion.

RESUME (excerpt):
{resume_text[:1800]}

JOB DESCRIPTION (excerpt):
{job_description[:800]}

KNOWN ISSUES: {', '.join(local['issues'][:4]) if local['issues'] else 'none detected'}

Return ONLY a JSON array of 5 suggestion strings:
["Suggestion 1", "Suggestion 2", "Suggestion 3", "Suggestion 4", "Suggestion 5"]"""

        raw = model.invoke(prompt).content
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        arr = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if arr:
            suggestions = json.loads(arr.group())
    except Exception as e:
        logger.warning("ATS LLM suggestions failed: %s", e)
        suggestions = [
            "Add measurable impact metrics (numbers, %, scale) to every bullet point.",
            f"Include these missing keywords from the JD: {', '.join(local['keywords_missing'][:4])}." if local['keywords_missing'] else "Mirror the exact job title and key phrases from the JD in your summary.",
            "Use standard section headings: Education, Experience, Skills, Projects.",
            "Add your LinkedIn URL and GitHub profile link.",
            "Keep resume to one page; prioritise most recent and relevant experience.",
        ]

    return {**local, "suggestions": suggestions}


# ── Cover Letter ─────────────────────────────────────────────────────────────

def generate_cover_letter(resume_text: str, job_description: str, matched_skills: list) -> str:
    """Generate a tailored cover letter based on the resume and job description."""
    model = _llm()
    skills_str = ", ".join(matched_skills[:8]) if matched_skills else "various relevant skills"
    prompt = f"""You are an expert career coach. Write a professional cover letter for this candidate.

RESUME (excerpt):
{resume_text[:2000]}

JOB DESCRIPTION (excerpt):
{job_description[:1000]}

CANDIDATE'S MATCHING SKILLS: {skills_str}

Write a compelling, personalized cover letter that:
- Opens with a strong hook (1 sentence referencing the specific role)
- Highlights 2-3 of the candidate's most relevant achievements with specifics from their resume
- Connects their matched skills directly to the job requirements
- Addresses enthusiasm for the company/role
- Closes with a confident call to action
- Is 3-4 paragraphs, professional but not robotic
- Does NOT use generic phrases like "I am writing to apply" or "I believe I would be a great fit"

Return ONLY the cover letter text, no subject line, no date, no address headers. Start directly with "Dear Hiring Manager," or the hiring manager's name if mentioned."""

    try:
        response = model.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.warning("Cover letter generation failed: %s", e)
        name_line = resume_text.strip().split("\n")[0] if resume_text.strip() else "Candidate"
        return f"""Dear Hiring Manager,

I am excited to apply for this position. With my background in {skills_str}, I am confident I can contribute meaningfully to your team.

Throughout my career I have developed strong expertise in the skills required for this role. I am eager to bring this experience to your organization and help drive results.

I would welcome the opportunity to discuss how my background aligns with your needs. Thank you for your consideration.

Sincerely,
{name_line}"""


# ── Public API ────────────────────────────────────────────────────────────────

def get_ai_analysis(resume_text: str, job_description: str) -> dict:
    resume = resume_text[:_MAX_RESUME_CHARS]
    jd     = job_description[:_MAX_JD_CHARS]
    model  = _llm()

    # Step 1 — skills + score
    try:
        step1 = _get_skills_and_score(model, resume, jd)
        matched = step1.get("matched_skills", [])
        missing = step1.get("missing_skills", [])
        score   = step1.get("readiness_score", 0)
        logger.info("✅ Step 1 OK — score %s, matched %d, missing %d", score, len(matched), len(missing))
    except Exception as e:
        logger.error("Step 1 failed: %s — falling back.", e)
        return _local_fallback(resume, jd)

    # Step 2 — plan + resources
    try:
        step2 = _get_plan_and_resources(model, missing, jd)
        logger.info("✅ Step 2 OK")
    except Exception as e:
        logger.warning("Step 2 failed: %s — using basic plan.", e)
        step2 = _local_fallback(resume, jd)
        step2["matched_skills"] = matched
        step2["missing_skills"] = missing
        step2["readiness_score"] = score

    # Step 3 — interview questions + resume suggestions (non-blocking)
    interview  = get_interview_questions(missing, matched, jd)
    resume_tips = get_resume_suggestions(resume, jd, missing)

    return {
        "matched_skills":       matched,
        "missing_skills":       missing,
        "readiness_score":      score,
        "learning_plan":        step2.get("learning_plan", []),
        "resources":            step2.get("resources", {}),
        "timeline_weeks":       step2.get("timeline_weeks", len(missing)),
        "recommended_projects": step2.get("recommended_projects", []),
        "interview_questions":  interview,
        "resume_suggestions":   resume_tips,
    }
