import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

# ── Streamlit Cloud: bridge st.secrets → os.environ so all modules can use
# os.getenv() regardless of whether we're running locally (.env) or in the cloud.
for _key in ["GROQ_API_KEY", "GROQ_MODEL", "EMBEDDING_MODEL"]:
    if _key in st.secrets and _key not in os.environ:
        os.environ[_key] = st.secrets[_key]

# Ensure required directories exist (not committed to git, must be created at runtime)
Path("reports").mkdir(exist_ok=True)
Path("workspaces").mkdir(exist_ok=True)

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

from analyzer import (
    _MAX_RESUME_CHARS,
    check_ats_score,
    extract_text_from_pdf,
    generate_cover_letter,
    generate_tailored_resume,
    get_ai_analysis,
    get_interview_questions,
    get_resume_suggestions,
    scrape_jd_from_url,
)
from export import generate_html_report, generate_json_bytes, generate_resume_pdf
from progress import (
    effective_score,
    get_all_missing,
    get_learned_skills,
    node_label,
    set_missing_skills,
    toggle_skill,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlignAgent",
    layout="wide",
    page_icon="◆",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

  /* ── Base ── */
  .stApp { background:#FFFFFF; color:#000000;
           font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
  .main .block-container { background:#FFFFFF; padding-top:1.5rem; max-width:1140px; }
  #MainMenu,footer,header { visibility:hidden; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background:#FFF9C4 !important;
    border-right:4px solid #000000 !important;
  }
  [data-testid="stSidebar"] * { color:#000000 !important; }

  /* ── Inputs ── */
  textarea, input[type="text"] {
    background:#FFFFFF !important; color:#000000 !important;
    border:3px solid #000000 !important; border-radius:0 !important;
    transition:border-color .15s !important;
  }
  textarea:focus, input[type="text"]:focus {
    border-color:#FFE000 !important; box-shadow:4px 4px 0px #000000 !important;
  }
  textarea::placeholder, input::placeholder { color:#888888 !important; }

  .stFileUploader section {
    background:#FFFDE7 !important;
    border-radius:0 !important; border:3px dashed #000000 !important;
    transition:border-color .15s !important;
  }
  .stFileUploader section:hover { border-color:#FFE000 !important; }
  .stFileUploader section small { color:#555555 !important; }

  /* ── Buttons ── */
  .stButton>button {
    background:#FFE000 !important;
    color:#000000 !important; border:3px solid #000000 !important; border-radius:0 !important;
    font-weight:800 !important; padding:0.6rem 1.8rem !important;
    box-shadow:5px 5px 0px #000000 !important;
    transition:all .1s !important; letter-spacing:.05em !important;
    text-transform:uppercase !important;
  }
  .stButton>button:hover {
    background:#FFD000 !important;
    box-shadow:3px 3px 0px #000000 !important;
    transform:translate(2px, 2px) !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background:#FFFFFF; border-radius:0; padding:0;
    gap:0; border:3px solid #000000;
  }
  .stTabs [data-baseweb="tab"] {
    background:#FFFFFF; color:#000000; border-radius:0;
    font-weight:700; font-size:.875rem; padding:.5rem 1rem; border:none;
    border-right:2px solid #000000;
    transition:all .1s;
  }
  .stTabs [aria-selected="true"] {
    background:#FFE000 !important;
    color:#000000 !important; box-shadow:none !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top:1.4rem; }

  /* ── Progress bar ── */
  .stProgress>div>div {
    background:#FFE000 !important;
    border-radius:0 !important;
  }
  .stProgress>div { background:#E8E8E8 !important; border-radius:0 !important; height:10px !important; border:2px solid #000000 !important; }

  /* ── Checkbox ── */
  .stCheckbox label { color:#000000 !important; }

  /* ── All native widget text → black on white ── */
  .stRadio label, .stRadio p,
  .stSelectbox label,
  .stTextInput label,
  .stTextArea label,
  .stFileUploader label,
  .stMarkdown p, .stMarkdown li,
  [data-testid="stText"], [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stCaption"] p,
  .stCaption, div[data-testid="stCaptionContainer"] p,
  .element-container p, .element-container li,
  label, p, li, span:not([class*="tag"]) { color:#000000; }

  /* ── Radio button options ── */
  [data-baseweb="radio"] label span,
  [data-baseweb="radio"] div { color:#000000 !important; }

  /* ── Captions / help text ── */
  small, .stCaption p,
  [data-testid="stCaptionContainer"] { color:#444444 !important; }

  /* ── Select/dropdown text ── */
  [data-baseweb="select"] div { color:#000000 !important; }

  /* ── Expander ── */
  [data-testid="stExpander"] { border:2px solid #000000 !important; border-radius:0 !important; }
  [data-testid="stExpander"] summary { color:#000000 !important; font-weight:700 !important; }
  [data-testid="stExpander"] summary p { color:#000000 !important; }

  /* ── Spinner text ── */
  [data-testid="stSpinner"] p { color:#000000 !important; }

  /* ── Alerts ── */
  [data-testid="stAlert"] { border-radius:0 !important; border:2px solid #000000 !important; }
  [data-testid="stAlert"] p { color:#000000 !important; }

  /* ── Hero ── */
  .hero {
    padding:1.6rem 0 2rem; border-bottom:5px solid #000000; margin-bottom:2rem;
    background:#FFE000;
  }
  .hero-badge {
    display:inline-block;
    background:#000000;
    color:#FFE000; font-size:.68rem; font-weight:800; letter-spacing:2px;
    text-transform:uppercase; padding:4px 14px; border-radius:0;
    border:2px solid #000000; margin-bottom:.7rem;
  }
  .hero h1 {
    font-size:2.5rem; font-weight:900; margin:0 0 .4rem; letter-spacing:-.5px;
    color:#000000;
    -webkit-text-fill-color:#000000;
    background:none;
  }
  .hero p { color:#333333; margin:0; font-size:.95rem; font-weight:500; }

  /* ── Input labels ── */
  .input-label {
    font-size:.72rem; font-weight:800; letter-spacing:.8px;
    text-transform:uppercase; color:#000000; margin-bottom:.4rem;
  }

  /* ── Score cards ── */
  .score-card {
    background:#FFFFFF;
    border:3px solid #000000; border-radius:0;
    padding:1.6rem 1rem; text-align:center; margin-bottom:.75rem;
    box-shadow:5px 5px 0px #000000;
    transition:transform .1s, box-shadow .1s;
  }
  .score-card:hover { transform:translate(-2px,-2px); box-shadow:7px 7px 0px #000000; }
  .score-label {
    font-size:.68rem; font-weight:800; letter-spacing:.8px;
    text-transform:uppercase; color:#555555; margin-bottom:.3rem;
  }
  .score-number { font-size:3.2rem; font-weight:900; line-height:1.1; }
  .score-sub { font-size:.8rem; font-weight:700; margin-top:.25rem; text-transform:uppercase; }

  .green  { color:#000000; }
  .yellow { color:#CC7700; }
  .red    { color:#E65000; }
  .purple { color:#FF6600; }

  /* ── Skill tags ── */
  .tag-wrap { display:flex; flex-wrap:wrap; gap:8px; margin-top:.6rem; }
  .tag { display:inline-block; padding:5px 14px; border-radius:0; font-size:.79rem; font-weight:700; border:2px solid #000000; }
  .tag-green {
    background:#FFE000;
    color:#000000;
  }
  .tag-red {
    background:#FF6600;
    color:#000000;
  }

  /* ── Section titles ── */
  .section-title {
    font-size:.68rem; font-weight:800; letter-spacing:.8px; text-transform:uppercase;
    color:#000000; margin:1.3rem 0 .7rem; padding-bottom:.4rem;
    border-bottom:3px solid #FFE000;
  }

  /* ── Timeline ── */
  .timeline-item {
    display:flex; gap:1rem; padding:1.1rem 0; border-bottom:2px solid #000000;
    transition:background .1s;
  }
  .week-dot {
    flex-shrink:0; width:38px; height:38px; border-radius:0;
    background:#FFE000;
    border:2px solid #000000; color:#000000; font-weight:800; font-size:.7rem;
    display:flex; align-items:center; justify-content:center; margin-top:2px;
  }
  .week-content h4 { margin:0 0 .15rem; font-size:.93rem; font-weight:700; color:#000000; }
  .week-content .goal { margin:0; font-size:.82rem; color:#555555; }
  .task-list { margin:.4rem 0 0; padding-left:1.1rem; }
  .task-list li { font-size:.82rem; color:#555555; margin-bottom:3px; }
  .project-chip {
    display:inline-block; margin-top:.5rem;
    background:#FFE000;
    color:#000000; border:2px solid #000000;
    border-radius:0; padding:3px 10px; font-size:.75rem; font-weight:700;
  }

  /* ── Resource cards ── */
  .resource-card {
    background:#FFFFFF;
    border:2px solid #000000; border-radius:0;
    padding:.8rem 1rem; margin-bottom:.5rem; display:flex;
    align-items:center; gap:.8rem; transition:all .1s;
    box-shadow:3px 3px 0px #000000;
  }
  .resource-card:hover { box-shadow:5px 5px 0px #FFE000; transform:translate(-2px,-2px); }
  .resource-type {
    flex-shrink:0; font-size:.6rem; font-weight:800; text-transform:uppercase;
    letter-spacing:.6px; padding:3px 8px; border-radius:0;
    background:#000000; color:#FFE000; border:none;
  }
  .resource-title { font-size:.87rem; font-weight:600; color:#000000; }
  .resource-title a { color:#CC6600; text-decoration:underline; transition:color .15s; }
  .resource-title a:hover { color:#FF6600; }

  /* ── Interview questions ── */
  .interview-q {
    background:#FFFDE7;
    border:2px solid #000000; border-left:5px solid #FFE000;
    border-radius:0; padding:.9rem 1.1rem; margin-bottom:.5rem;
    font-size:.88rem; color:#000000; transition:border-color .15s;
  }
  .interview-q:hover { border-left-color:#FF6600; }
  .interview-q strong { color:#CC6600; }

  /* ── Resume tips ── */
  .tip-card {
    background:#FFF9C4;
    border:2px solid #000000; border-left:5px solid #FF6600; border-radius:0;
    padding:.75rem 1rem; margin-bottom:.5rem; font-size:.87rem; color:#000000;
    transition:border-left-color .15s;
  }
  .tip-card:hover { border-left-color:#FFE000; }

  /* ── Compare cards ── */
  .compare-card {
    background:#FFFFFF;
    border:3px solid #000000; border-radius:0;
    padding:1.3rem; text-align:center;
    box-shadow:5px 5px 0px #000000;
    transition:transform .1s, box-shadow .1s;
  }
  .compare-card:hover { transform:translate(-2px,-2px); box-shadow:7px 7px 0px #FFE000; }
  .compare-score { font-size:2.6rem; font-weight:900; }
  .compare-title { font-size:.82rem; color:#555555; font-weight:700; margin-bottom:.5rem; text-transform:uppercase; }

  /* ── Utility ── */
  .divider { border:none; border-top:3px solid #000000; margin:1.5rem 0; }

  .empty-state { text-align:center; padding:4rem 0; color:#555555; border:3px dashed #000000; }
  .empty-state .icon { margin-bottom:.9rem; display:flex; justify-content:center; }
  .empty-state .icon svg { opacity:.55; }
  .empty-state h3 { color:#000000; font-size:.95rem; font-weight:800; margin:0 0 .3rem; text-transform:uppercase; }
  .empty-state p  { font-size:.83rem; margin:0; }

  /* ── Agent progress nodes ── */
  .node-step {
    background:#FFFFFF; border:2px solid #000000; border-radius:0;
    padding:.55rem 1rem; margin-bottom:.35rem; font-size:.84rem; color:#555555;
    transition:all .2s;
  }
  .node-step.done {
    background:#FFE000;
    border-color:#000000; color:#000000; font-weight:700;
  }

  /* ── ATS ── */
  .ats-issue {
    background:#FFF0E6;
    border:2px solid #000000; border-left:5px solid #FF6600; border-radius:0;
    padding:.65rem 1rem; margin-bottom:.4rem; font-size:.86rem; color:#000000;
  }
  .ats-suggestion {
    background:#FFFDE7;
    border:2px solid #000000; border-left:5px solid #FFE000; border-radius:0;
    padding:.65rem 1rem; margin-bottom:.4rem; font-size:.86rem; color:#000000;
  }

  /* ── Mobile sidebar toggle ── */
  [data-testid="collapsedControl"] {
    background:#FFE000 !important;
    border:3px solid #000000 !important; border-radius:0 !important;
    color:#000000 !important; width:28px !important;
    box-shadow:3px 0 0px #000000 !important;
  }

  /* ── Mobile responsive ── */
  @media (max-width: 768px) {
    .hero { padding:1rem 0 1.4rem; }
    .hero h1 { font-size:1.8rem !important; }
    .hero p  { font-size:.85rem; }
    .hero-badge { font-size:.6rem; }

    .score-number  { font-size:2.2rem !important; }
    .score-card    { padding:1rem .7rem; }
    .compare-score { font-size:1.8rem !important; }

    .main .block-container {
      padding-left:.6rem !important;
      padding-right:.6rem !important;
    }

    .tag { font-size:.72rem; padding:4px 10px; }

    .section-title { font-size:.62rem; }

    .timeline-item { gap:.6rem; }
    .week-dot { width:30px; height:30px; font-size:.62rem; }

    .resource-card { flex-wrap:wrap; }
    .interview-q   { font-size:.82rem; }
  }
</style>
""", unsafe_allow_html=True)


# ── Startup: clean up old workspaces and reports ─────────────────────────────
def _cleanup_old_data(max_age_days: int = 7) -> None:
    cutoff = datetime.now() - timedelta(days=max_age_days)
    for base_dir in ["workspaces", "reports"]:
        base = Path(base_dir)
        if not base.exists():
            continue
        for item in base.iterdir():
            if item.name.startswith("."):
                continue
            try:
                if datetime.fromtimestamp(item.stat().st_mtime) < cutoff:
                    shutil.rmtree(item) if item.is_dir() else item.unlink()
            except Exception:
                pass

if "_cleanup_done" not in st.session_state:
    _cleanup_old_data()
    st.session_state._cleanup_done = True


# ── Sidebar: Progress Tracker ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### AlignAgent")
    st.markdown("<hr style='border-color:#000000;margin:.5rem 0 1rem'>", unsafe_allow_html=True)
    st.markdown("#### Skill Progress Tracker")
    st.caption("Check off skills you've learned — your score updates automatically.")

    all_missing = get_all_missing()
    learned     = set(get_learned_skills())

    if all_missing:
        for skill in all_missing:
            is_learned = skill in learned
            checked = st.checkbox(skill, value=is_learned, key=f"prog_{skill}")
            if checked != is_learned:
                toggle_skill(skill, checked)
                st.rerun()
    else:
        st.caption("Run an analysis to populate your tracker.")

    st.markdown("<hr style='border-color:#000000;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("#### About")
    st.caption("Powered by **Groq** (llama-3.3-70b). Fast, free tier.")

    st.markdown("<hr style='border-color:#000000;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("#### Privacy")
    st.caption(
        "Your resume and job description text are sent to **Groq's API** (USA) for AI processing. "
        "No data is stored on Groq beyond the duration of the request. "
        "Do not upload documents containing passwords or financial information."
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">Skill Gap Analyser</div>
  <h1>AlignAgent</h1>
  <p>Upload your resume and a job description — get a full skill gap report, week-by-week plan, interview prep, and resume suggestions.</p>
</div>
""", unsafe_allow_html=True)


# ── Mode Tabs ─────────────────────────────────────────────────────────────────
mode_tab, compare_tab = st.tabs(["Analyse a Job", "Compare Multiple Jobs"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════
with mode_tab:

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="input-label">Resume (PDF)</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("resume", type=["pdf"], label_visibility="collapsed")
        if uploaded_file:
            _MAX_MB = 5
            if uploaded_file.size > _MAX_MB * 1024 * 1024:
                st.error(f"File too large. Maximum size is {_MAX_MB} MB.")
                uploaded_file = None
            else:
                st.success(f"✓  {uploaded_file.name}")

    with col_right:
        st.markdown('<div class="input-label">Job / Internship Description</div>', unsafe_allow_html=True)
        jd_source = st.radio("Source", ["Paste text", "Enter URL"],
                             horizontal=True, label_visibility="collapsed")
        if jd_source == "Paste text":
            job_description = st.text_area("jd", placeholder="Paste the full job description here…",
                                           height=140, label_visibility="collapsed")
        else:
            jd_url = st.text_input("URL", placeholder="https://linkedin.com/jobs/...",
                                   label_visibility="collapsed")
            job_description = ""
            if jd_url:
                from urllib.parse import urlparse
                _parsed = urlparse(jd_url)
                _host = (_parsed.hostname or "").lower()
                _blocked = ("localhost", "127.", "0.0.0.0", "10.", "192.168.", "169.254.", "::1")
                if _parsed.scheme not in ("http", "https"):
                    st.error("Only http/https URLs are allowed.")
                elif any(_host.startswith(b) for b in _blocked):
                    st.error("That URL is not allowed.")
                else:
                    with st.spinner("Scraping job description…"):
                        try:
                            job_description = scrape_jd_from_url(jd_url)
                            st.success(f"✓  Scraped {len(job_description)} characters")
                        except Exception as e:
                            st.error(str(e))

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    pipeline_mode = st.radio(
        "Analysis depth",
        ["⚡ Quick (2-step LLM)", "🧠 Deep Agentic (full 8-node graph)"],
        horizontal=True,
        help="Quick is faster (~40s). Deep uses the full agentic RAG pipeline (~2-3 min) with grading, reflection, and web search.",
    )

    run_btn = st.button("Analyse my fit →")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Run analysis ──────────────────────────────────────────────────────────
    if run_btn:
        if not uploaded_file or not job_description.strip():
            st.warning("Upload a resume **and** provide a job description to continue.")
            st.stop()

        resume_text = extract_text_from_pdf(uploaded_file)

        if len(resume_text.strip()) < 200:
            st.warning(
                f"Only {len(resume_text.strip())} characters were extracted from your PDF. "
                "This usually means the resume was exported as an image (Canva, Adobe Illustrator) "
                "or uses a complex multi-column layout. Results may be inaccurate. "
                "For best results, upload a text-based PDF or export from Google Docs / Microsoft Word."
            )
        elif len(resume_text) > _MAX_RESUME_CHARS:
            st.info(
                f"Your resume is {len(resume_text):,} characters — only the first "
                f"{_MAX_RESUME_CHARS:,} will be analysed. Consider trimming older or less relevant experience."
            )

        report = {}

        if "Quick" in pipeline_mode:
            try:
                with st.spinner("Running quick analysis…"):
                    report = get_ai_analysis(resume_text, job_description)
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate limit" in err:
                    st.error("Groq rate limit hit. Wait 60 seconds and try again.")
                else:
                    st.error(f"Analysis failed — Groq may be temporarily unavailable. ({e})")
                st.stop()
        else:
            # Deep agentic pipeline with real-time node progress
            from pipeline import run_agentic_analysis
            import json as _json

            completed_nodes = []
            st.markdown('<div class="section-title">Agent Progress</div>', unsafe_allow_html=True)
            progress_area = st.empty()

            def _render_nodes(nodes):
                html = ""
                for n in nodes:
                    html += f'<div class="node-step done">✓ {node_label(n)}</div>'
                progress_area.markdown(html, unsafe_allow_html=True)

            try:
                with st.spinner("Running deep agentic analysis…"):
                    final_state = {}
                    uploaded_file.seek(0)  # extract_text_from_pdf already consumed the stream
                    for event in run_agentic_analysis(
                        uploaded_file.read(), uploaded_file.name, job_description
                    ):
                        node_name = list(event.keys())[0]
                        completed_nodes.append(node_name)
                        _render_nodes(completed_nodes)
                        final_state = event[node_name]
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate limit" in err:
                    st.error(
                        "Groq rate limit reached even after retrying. "
                        "Wait 60 seconds and try again, or switch to Quick mode."
                    )
                else:
                    st.error(
                        f"Deep analysis failed mid-run — Groq may be temporarily down. "
                        f"Try Quick mode as a fallback. ({e})"
                    )
                st.stop()

            # Load report from saved file
            report_path = final_state.get("analysis_report_path", "")
            try:
                with open(report_path) as f:
                    report = _json.load(f)
            except Exception:
                report = {}

            # Enrich with interview questions + resume suggestions (not in graph)
            if report:
                report["interview_questions"] = get_interview_questions(
                    report.get("missing_skills", []),
                    report.get("matched_skills", []),
                    job_description,
                )
                report["resume_suggestions"] = get_resume_suggestions(
                    resume_text, job_description, report.get("missing_skills", [])
                )

        if not report:
            st.error("Analysis returned no results. Try the Quick mode or re-run.")
            st.stop()

        # Persist missing skills to progress tracker
        set_missing_skills(report.get("missing_skills", []))

        # Save everything needed for rendering — tab buttons trigger reruns
        # where run_btn is False, so we must read from session_state
        st.session_state._result = {
            "report":          report,
            "resume_text":     resume_text,
            "job_description": job_description,
            "filename":        uploaded_file.name,
        }
        st.session_state.ats_report   = None
        st.session_state.cover_letter = ""

    # ── Render results (persists across reruns triggered by tab buttons) ──────
    _r = st.session_state.get("_result")
    if _r:
        report          = _r["report"]
        resume_text     = _r["resume_text"]
        job_description = _r["job_description"]

        score    = report.get("readiness_score", 0)
        timeline = report.get("timeline_weeks")
        matched  = report.get("matched_skills", [])
        missing  = report.get("missing_skills", [])
        plan     = report.get("learning_plan", [])
        resources= report.get("resources", {})
        projects = report.get("recommended_projects", [])
        interview= report.get("interview_questions", [])
        tips     = report.get("resume_suggestions", [])
    
        # Adjusted score based on progress tracker
        adj_score = effective_score(score, matched, missing)
        learned   = set(get_learned_skills())
    
        # ── Score cards ───────────────────────────────────────────────────────
        sc, tc, ec = st.columns([1, 1, 1], gap="large")
    
        score_cls = "green" if adj_score >= 70 else ("yellow" if adj_score >= 40 else "red")
        score_sub = "Strong fit" if adj_score >= 70 else ("Getting there" if adj_score >= 40 else "Needs work")
    
        with sc:
            st.markdown(f"""
            <div class="score-card">
              <div class="score-label">Readiness Score</div>
              <div class="score-number {score_cls}">{adj_score}%</div>
              <div class="score-sub {score_cls}">{score_sub}</div>
            </div>""", unsafe_allow_html=True)
            st.progress(adj_score / 100)
            st.caption("AI estimate — treat as a guide, not a precise measurement.")
    
        with tc:
            if timeline:
                remaining_weeks = max(0, timeline - len(learned))
                st.markdown(f"""
                <div class="score-card">
                  <div class="score-label">Weeks Remaining</div>
                  <div class="score-number purple">{remaining_weeks}</div>
                  <div class="score-sub" style="color:#555555;">of {timeline} week plan</div>
                </div>""", unsafe_allow_html=True)
    
        with ec:
            skills_learned = len([s for s in missing if s in learned])
            pct = round(skills_learned / len(missing) * 100) if missing else 100
            st.markdown(f"""
            <div class="score-card">
              <div class="score-label">Skills Learned</div>
              <div class="score-number" style="color:#000000;">{skills_learned}/{len(missing)}</div>
              <div class="score-sub" style="color:#555555;">{pct}% of gaps closed</div>
            </div>""", unsafe_allow_html=True)
    
        # ── Charts row ───────────────────────────────────────────────────────
        if _PLOTLY_OK:
            ch1, ch2, ch3 = st.columns([1.1, 1, 1.4], gap="large")
    
            with ch1:
                gauge_color = "#FFE000" if adj_score >= 70 else ("#FF6600" if adj_score >= 40 else "#E65000")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=adj_score,
                    number={"suffix": "%", "font": {"size": 36, "color": "#000000"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#555555",
                                 "tickfont": {"color": "#555555", "size": 10}},
                        "bar": {"color": gauge_color, "thickness": 0.28},
                        "bgcolor": "#FFFFFF",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0,  40], "color": "#FFE8CC"},
                            {"range": [40, 70], "color": "#FFF3CC"},
                            {"range": [70, 100], "color": "#FFFDE7"},
                        ],
                        "threshold": {
                            "line": {"color": gauge_color, "width": 3},
                            "thickness": 0.82,
                            "value": adj_score,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#000000"}, height=210,
                    margin=dict(l=20, r=20, t=30, b=10),
                    title={"text": "Readiness Gauge", "font": {"size": 11, "color": "#555555"},
                           "x": 0.5, "xanchor": "center"},
                )
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
    
            with ch2:
                donut_labels = ["Skills Matched", "Skills Missing"]
                donut_values = [max(len(matched), 1), max(len(missing), 0)]
                fig_donut = go.Figure(go.Pie(
                    labels=donut_labels,
                    values=donut_values,
                    hole=0.68,
                    marker=dict(colors=["#FFE000", "#FF6600"],
                                line=dict(color="#FFFFFF", width=3)),
                    textfont={"color": "#000000", "size": 11},
                    hovertemplate="%{label}: %{value}<extra></extra>",
                ))
                fig_donut.add_annotation(
                    text=f"<b>{len(matched)}/{len(matched)+len(missing)}</b>",
                    x=0.5, y=0.5, font_size=20, font_color="#000000", showarrow=False,
                )
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#000000"}, height=210,
                    showlegend=True,
                    legend=dict(font=dict(color="#000000", size=10),
                                bgcolor="rgba(0,0,0,0)", orientation="h",
                                x=0.5, xanchor="center", y=-0.05),
                    margin=dict(l=10, r=10, t=30, b=10),
                    title={"text": "Skills Breakdown", "font": {"size": 11, "color": "#555555"},
                           "x": 0.5, "xanchor": "center"},
                )
                st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
    
            with ch3:
                all_skills = matched + [s for s in missing if s not in learned]
                bar_colors = (["#FFE000"] * len(matched) +
                              ["#FF6600"] * len([s for s in missing if s not in learned]))
                bar_labels = (["✓ " + s for s in matched] +
                              ["✗ " + s for s in missing if s not in learned])
                if all_skills:
                    fig_bar = go.Figure(go.Bar(
                        y=bar_labels,
                        x=[1] * len(bar_labels),
                        orientation="h",
                        marker=dict(color=bar_colors, line=dict(color="#000000", width=1)),
                        text=bar_labels,
                        textposition="inside",
                        textfont={"color": "#000000", "size": 10},
                        hoverinfo="skip",
                    ))
                    fig_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#000000"},
                        height=max(180, len(bar_labels) * 26 + 50),
                        xaxis=dict(visible=False, range=[0, 1.05]),
                        yaxis=dict(visible=False),
                        margin=dict(l=5, r=5, t=30, b=10),
                        bargap=0.25,
                        title={"text": "Skill Map", "font": {"size": 11, "color": "#555555"},
                               "x": 0.5, "xanchor": "center"},
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    
        # ── Skill tags ────────────────────────────────────────────────────────
        sk1, sk2 = st.columns(2, gap="large")
        with sk1:
            st.markdown('<div class="section-title">Skills you have</div>', unsafe_allow_html=True)
            tags = "".join(f'<span class="tag tag-green">{s}</span>'
                           for s in matched + [s for s in missing if s in learned])
            st.markdown(f'<div class="tag-wrap">{tags or "<em style=color:#555555>None detected</em>"}</div>',
                        unsafe_allow_html=True)
    
        with sk2:
            st.markdown('<div class="section-title">Skills still to develop</div>', unsafe_allow_html=True)
            remaining_missing = [s for s in missing if s not in learned]
            tags = "".join(f'<span class="tag tag-red">{s}</span>' for s in remaining_missing)
            if remaining_missing:
                st.markdown(f'<div class="tag-wrap">{tags}</div>', unsafe_allow_html=True)
            else:
                st.success("🎉 You've closed all skill gaps!")
    
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    
        # ── Result tabs ───────────────────────────────────────────────────────
        t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
            "Learning Plan", "Resources",
            "Interview Prep", "Resume Tips",
            "Projects", "ATS Check", "Cover Letter",
            "Resume Builder", "Export",
        ])
    
        with t1:
            if plan:
                # Gantt chart — week-by-week learning timeline
                gantt_skills, gantt_starts, gantt_ends, gantt_colors, gantt_text = [], [], [], [], []
                for step in plan:
                    if isinstance(step, dict):
                        w = step.get("week", 1)
                        s = step.get("skill", f"Skill {w}")
                        done_g = s in learned
                        gantt_skills.append(s)
                        gantt_starts.append(w - 1)
                        gantt_ends.append(w)
                        gantt_colors.append("#FFE000" if done_g else "#222222")
                        gantt_text.append("✓ Done" if done_g else f"Week {w}")
                if gantt_skills and _PLOTLY_OK:
                    fig_gantt = go.Figure()
                    for i, (skill_g, start, end, color, txt) in enumerate(
                        zip(gantt_skills, gantt_starts, gantt_ends, gantt_colors, gantt_text)
                    ):
                        txt_color = "#000000" if color == "#FFE000" else "#FFFFFF"
                        fig_gantt.add_trace(go.Bar(
                            x=[end - start], base=[start], y=[skill_g],
                            orientation="h",
                            marker=dict(color=color, line=dict(color="#000000", width=1)),
                            text=txt, textposition="inside",
                            textfont={"color": txt_color, "size": 10},
                            showlegend=False,
                            hovertemplate=f"<b>{skill_g}</b><br>Week {start+1}<extra></extra>",
                        ))
                    max_week = max(gantt_ends)
                    fig_gantt.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#000000"},
                        height=max(200, len(gantt_skills) * 36 + 80),
                        barmode="overlay", bargap=0.3,
                        xaxis=dict(
                            title="Week", range=[0, max_week + 0.2],
                            tickvals=list(range(1, max_week + 1)),
                            ticktext=[f"Wk {i}" for i in range(1, max_week + 1)],
                            gridcolor="#E0E0E0", color="#555555",
                        ),
                        yaxis=dict(gridcolor="#E0E0E0", color="#333333"),
                        margin=dict(l=10, r=10, t=10, b=30),
                    )
                    st.plotly_chart(fig_gantt, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    
                for step in plan:
                    if isinstance(step, dict):
                        week  = step.get("week", "?")
                        skill = step.get("skill", "")
                        goal  = step.get("goal", "")
                        tasks = step.get("tasks", [])
                        proj  = step.get("project", "")
                        done  = skill in learned
    
                        task_html = "".join(f"<li>{t}</li>" for t in tasks)
                        chip      = f'<div class="project-chip">🔨 {proj}</div>' if proj else ""
                        opacity   = "opacity:.45;" if done else ""
    
                        st.markdown(f"""
                        <div class="timeline-item" style="{opacity}">
                          <div class="week-dot">W{week}</div>
                          <div class="week-content">
                            <h4>{skill} {'✓' if done else ''}</h4>
                            <p class="goal">{goal}</p>
                            <ul class="task-list">{task_html}</ul>
                            {chip}
                          </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"- {step}")
            else:
                st.info("No learning plan generated.")
    
        with t2:
            if resources:
                st.caption("Links are AI-generated — verify each URL before clicking, as some may be inaccurate.")
                for skill, links in resources.items():
                    st.markdown(f'<div class="section-title">{skill}</div>', unsafe_allow_html=True)
                    for r in links:
                        if isinstance(r, dict):
                            title = r.get("title", "")
                            url   = r.get("url", "")
                            rtype = r.get("type", "link").upper()
                            t_html = f'<a href="{url}" target="_blank">{title}</a>' if url else title
                            st.markdown(f"""
                            <div class="resource-card">
                              <span class="resource-type">{rtype}</span>
                              <span class="resource-title">{t_html}</span>
                            </div>""", unsafe_allow_html=True)
            else:
                st.info("No resources found.")
    
        with t3:
            if interview:
                for i, q in enumerate(interview, 1):
                    st.markdown(f"""
                    <div class="interview-q">
                      <strong>Q{i}.</strong> {q}
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No interview questions generated.")
    
        with t4:
            if tips:
                for tip in tips:
                    st.markdown(f'<div class="tip-card">💡 {tip}</div>', unsafe_allow_html=True)
            else:
                st.info("No resume suggestions generated.")
    
        with t5:
            if projects:
                for i, proj in enumerate(projects, 1):
                    st.markdown(f"""
                    <div class="resource-card">
                      <span class="resource-type">#{i}</span>
                      <span class="resource-title">{proj}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No projects generated.")
    
        with t6:
            st.markdown('<div class="section-title">ATS Compatibility Check</div>', unsafe_allow_html=True)
            st.caption("See how well your resume will pass Applicant Tracking Systems before it reaches a human.")
    
            if "ats_report" not in st.session_state:
                st.session_state.ats_report = None
    
            ats_col, _ = st.columns([1, 3])
            with ats_col:
                ats_btn = st.button("Run ATS Check", key="run_ats")
    
            if ats_btn:
                with st.spinner("Analysing ATS compatibility…"):
                    st.session_state.ats_report = check_ats_score(resume_text, job_description)
    
            if st.session_state.ats_report:
                ar = st.session_state.ats_report
                ats_score    = ar.get("ats_score", 0)
                kw_score     = ar.get("keyword_score", 0)
                sec_score    = ar.get("section_score", 0)
                kw_found     = ar.get("keywords_found", [])
                kw_missing   = ar.get("keywords_missing", [])
                ats_issues   = ar.get("issues", [])
                ats_tips     = ar.get("suggestions", [])
                word_count   = ar.get("word_count", 0)
    
                ats_cls = "green" if ats_score >= 75 else ("yellow" if ats_score >= 50 else "red")
                ats_label = "ATS Ready" if ats_score >= 75 else ("Needs Work" if ats_score >= 50 else "High Risk")
    
                # Score cards row
                ac1, ac2, ac3 = st.columns(3, gap="large")
                with ac1:
                    st.markdown(f"""
                    <div class="score-card">
                      <div class="score-label">ATS Score</div>
                      <div class="score-number {ats_cls}">{ats_score}%</div>
                      <div class="score-sub {ats_cls}">{ats_label}</div>
                    </div>""", unsafe_allow_html=True)
                    st.progress(ats_score / 100)
                with ac2:
                    kw_cls = "green" if kw_score >= 70 else ("yellow" if kw_score >= 40 else "red")
                    st.markdown(f"""
                    <div class="score-card">
                      <div class="score-label">Keyword Match</div>
                      <div class="score-number {kw_cls}">{kw_score}%</div>
                      <div class="score-sub" style="color:#555555;">{len(kw_found)} of {len(kw_found)+len(kw_missing)} JD skills</div>
                    </div>""", unsafe_allow_html=True)
                    st.progress(kw_score / 100)
                with ac3:
                    sec_cls = "green" if sec_score >= 75 else ("yellow" if sec_score >= 50 else "red")
                    st.markdown(f"""
                    <div class="score-card">
                      <div class="score-label">Structure Score</div>
                      <div class="score-number {sec_cls}">{sec_score}%</div>
                      <div class="score-sub" style="color:#555555;">{word_count} words</div>
                    </div>""", unsafe_allow_html=True)
                    st.progress(sec_score / 100)
    
                # ATS gauge chart
                if _PLOTLY_OK:
                    ats_gauge_color = "#FFE000" if ats_score >= 75 else ("#FF6600" if ats_score >= 50 else "#E65000")
                    fig_ats = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=ats_score,
                        number={"suffix": "%", "font": {"size": 36, "color": "#000000"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#555555",
                                     "tickfont": {"color": "#555555", "size": 10}},
                            "bar": {"color": ats_gauge_color, "thickness": 0.28},
                            "bgcolor": "#FFFFFF", "borderwidth": 0,
                            "steps": [
                                {"range": [0,  50], "color": "#FFE8CC"},
                                {"range": [50, 75], "color": "#FFF3CC"},
                                {"range": [75, 100], "color": "#FFFDE7"},
                            ],
                        },
                    ))
                    fig_ats.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#000000"}, height=200,
                        margin=dict(l=20, r=20, t=20, b=10),
                    )
                    gc, _ = st.columns([1, 2])
                    with gc:
                        st.plotly_chart(fig_ats, use_container_width=True, config={"displayModeBar": False})
    
                # Keywords
                kw1, kw2 = st.columns(2, gap="large")
                with kw1:
                    st.markdown('<div class="section-title">Keywords Found</div>', unsafe_allow_html=True)
                    if kw_found:
                        tags = "".join(f'<span class="tag tag-green">{k}</span>' for k in kw_found)
                        st.markdown(f'<div class="tag-wrap">{tags}</div>', unsafe_allow_html=True)
                    else:
                        st.caption("No matching keywords detected.")
                with kw2:
                    st.markdown('<div class="section-title">Keywords Missing from JD</div>', unsafe_allow_html=True)
                    if kw_missing:
                        tags = "".join(f'<span class="tag tag-red">{k}</span>' for k in kw_missing)
                        st.markdown(f'<div class="tag-wrap">{tags}</div>', unsafe_allow_html=True)
                    else:
                        st.success("All JD keywords found in your resume!")
    
                # Issues
                if ats_issues:
                    st.markdown('<div class="section-title">Issues Detected</div>', unsafe_allow_html=True)
                    for issue in ats_issues:
                        st.markdown(f'<div class="ats-issue">⚠️ {issue}</div>', unsafe_allow_html=True)
    
                # Suggestions
                if ats_tips:
                    st.markdown('<div class="section-title">How to Improve</div>', unsafe_allow_html=True)
                    for tip in ats_tips:
                        st.markdown(f'<div class="ats-suggestion">💡 {tip}</div>', unsafe_allow_html=True)
    
            else:
                st.markdown("""
                <div class="empty-state">
                  <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#000000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                      <rect x="2" y="6" width="20" height="12" rx="2"/><path d="M12 12h.01"/><path d="M17 12h.01"/><path d="M7 12h.01"/>
                    </svg>
                  </div>
                  <h3>ATS check not run yet</h3>
                  <p>Click <strong>Run ATS Check</strong> above to analyse your resume.</p>
                </div>""", unsafe_allow_html=True)
    
        with t7:
            st.markdown('<div class="section-title">Tailored Cover Letter</div>', unsafe_allow_html=True)
            st.caption("AI-generated based on your resume and the job description. Review and personalise before sending.")
    
            if "cover_letter" not in st.session_state:
                st.session_state.cover_letter = ""
    
            gen_col, _ = st.columns([1, 3])
            with gen_col:
                gen_btn = st.button("Generate Cover Letter", key="gen_cl")
    
            if gen_btn:
                with st.spinner("Writing your cover letter…"):
                    st.session_state.cover_letter = generate_cover_letter(
                        resume_text, job_description, matched
                    )
    
            if st.session_state.cover_letter:
                cl_text = st.session_state.cover_letter
    
                edited = st.text_area(
                    "Edit before copying",
                    value=cl_text,
                    height=400,
                    label_visibility="collapsed",
                    key="cl_edit",
                )
    
                dl_col, _ = st.columns([1, 3])
                with dl_col:
                    st.download_button(
                        "Download .txt",
                        data=edited.encode("utf-8"),
                        file_name="cover_letter.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                st.caption("Tip: copy the text above and paste it into your application portal.")
            else:
                st.markdown("""
                <div class="empty-state">
                  <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#000000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                      <rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
                    </svg>
                  </div>
                  <h3>No cover letter yet</h3>
                  <p>Click <strong>Generate Cover Letter</strong> above to create one.</p>
                </div>""", unsafe_allow_html=True)
    
        with t8:
            st.markdown('<div class="section-title">AI Resume Builder</div>', unsafe_allow_html=True)
            st.caption("Rewrites your resume content to be tailored and optimised for this specific job. Your original data is never changed.")

            if "resume_data" not in st.session_state:
                st.session_state.resume_data = None

            rb_col, _ = st.columns([1, 3])
            with rb_col:
                rb_btn = st.button("Generate Tailored Resume", key="gen_resume")

            if rb_btn:
                with st.spinner("Rewriting your resume for this role… (~30s)"):
                    st.session_state.resume_data = generate_tailored_resume(
                        resume_text, job_description, matched, missing
                    )

            if st.session_state.resume_data:
                rd = st.session_state.resume_data

                # Preview — editable fields
                st.markdown('<div class="section-title">Preview & Edit</div>', unsafe_allow_html=True)
                st.caption("Edit any field below before downloading.")

                pc1, pc2 = st.columns(2, gap="large")
                with pc1:
                    rd["name"]     = st.text_input("Full Name",     value=rd.get("name", ""),     key="rb_name")
                    rd["email"]    = st.text_input("Email",          value=rd.get("email", ""),    key="rb_email")
                    rd["phone"]    = st.text_input("Phone",          value=rd.get("phone", ""),    key="rb_phone")
                with pc2:
                    rd["linkedin"] = st.text_input("LinkedIn URL",   value=rd.get("linkedin", ""), key="rb_linkedin")
                    rd["github"]   = st.text_input("GitHub URL",     value=rd.get("github", ""),   key="rb_github")

                rd["summary"] = st.text_area(
                    "Professional Summary",
                    value=rd.get("summary", ""),
                    height=100,
                    key="rb_summary",
                )

                # Skills — comma-editable
                skills_str = st.text_input(
                    "Skills (comma-separated)",
                    value=", ".join(rd.get("skills", [])),
                    key="rb_skills",
                )
                rd["skills"] = [s.strip() for s in skills_str.split(",") if s.strip()]

                # Experience bullets — one text area per job
                st.markdown('<div class="section-title">Experience</div>', unsafe_allow_html=True)
                for idx, job in enumerate(rd.get("experience", [])):
                    with st.expander(f"{job.get('title','')} — {job.get('company','')}  |  {job.get('duration','')}"):
                        bullets_text = st.text_area(
                            "Bullets (one per line)",
                            value="\n".join(job.get("bullets", [])),
                            height=120,
                            key=f"rb_bullets_{idx}",
                        )
                        job["bullets"] = [b.strip() for b in bullets_text.splitlines() if b.strip()]

                # Download
                st.markdown('<div class="section-title">Download</div>', unsafe_allow_html=True)
                dl1, dl2, _ = st.columns([1, 1, 2], gap="medium")
                try:
                    pdf_bytes = generate_resume_pdf(rd)
                    with dl1:
                        st.download_button(
                            "Download PDF",
                            data=pdf_bytes,
                            file_name="tailored_resume.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

                with dl2:
                    import json as _json2
                    st.download_button(
                        "Download JSON",
                        data=_json2.dumps(rd, indent=2).encode(),
                        file_name="tailored_resume.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                st.caption("PDF tip: open in browser → File → Print → Save as PDF for best quality.")

            else:
                st.markdown("""
                <div class="empty-state">
                  <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#000000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>
                    </svg>
                  </div>
                  <h3>No resume generated yet</h3>
                  <p>Click <strong>Generate Tailored Resume</strong> above to create a job-specific version of your resume.</p>
                </div>""", unsafe_allow_html=True)

        with t9:
            st.markdown('<div class="section-title">Download your report</div>', unsafe_allow_html=True)
    
            dl1, dl2 = st.columns(2)
            candidate_name = _r.get("filename", "resume").replace(".pdf", "")
    
            with dl1:
                html_report = generate_html_report(report, candidate_name)
                st.download_button(
                    "Download HTML Report",
                    data=html_report.encode("utf-8"),
                    file_name="alignagent_report.html",
                    mime="text/html",
                    use_container_width=True,
                )
                st.caption("Open in browser → Print → Save as PDF")
    
            with dl2:
                st.download_button(
                    "Download JSON Report",
                    data=generate_json_bytes(report),
                    file_name="alignagent_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.caption("Raw data for integration or backup")
    
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#000000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>
            </svg>
          </div>
          <h3>Your analysis will appear here</h3>
          <p>Upload a resume, paste or scrape a job description, then hit <strong>Analyse my fit</strong>.</p>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE JOBS
# ═══════════════════════════════════════════════════════════════════════════════
with compare_tab:
    st.markdown("**Upload your resume once — compare against up to 3 jobs.**")
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    cmp_file = st.file_uploader("Resume for comparison (PDF)", type=["pdf"],
                                key="cmp_resume", label_visibility="collapsed")
    if cmp_file:
        st.success(f"✓  {cmp_file.name}")

    c1, c2, c3 = st.columns(3, gap="medium")
    jd_inputs = []
    for col, label in zip([c1, c2, c3], ["Job 1", "Job 2", "Job 3"]):
        with col:
            st.markdown(f'<div class="input-label">{label}</div>', unsafe_allow_html=True)
            jd_inputs.append(st.text_area(label, placeholder=f"Paste {label} description…",
                                          height=180, label_visibility="collapsed", key=f"cmp_jd{label}"))

    compare_btn = st.button("Compare Jobs →", key="cmp_btn")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if compare_btn:
        filled_jds = [(f"Job {i+1}", jd) for i, jd in enumerate(jd_inputs) if jd.strip()]
        if not cmp_file or not filled_jds:
            st.warning("Upload a resume and fill in at least one job description.")
            st.stop()

        cmp_resume_text = extract_text_from_pdf(cmp_file)

        if len(cmp_resume_text.strip()) < 200:
            st.warning(
                f"Only {len(cmp_resume_text.strip())} characters were extracted from your PDF. "
                "The resume may be image-based or have a complex layout — comparison results may be inaccurate."
            )

        results = []

        with st.spinner(f"Analysing {len(filled_jds)} job(s)…"):
            for label, jd_text in filled_jds:
                r = get_ai_analysis(cmp_resume_text, jd_text)
                r["_label"] = label
                results.append(r)

        # Sort by score descending
        results.sort(key=lambda x: x.get("readiness_score", 0), reverse=True)
        best = results[0]["_label"]

        st.markdown(f"""
        <div style="background:#FFE000;border:3px solid #000000;border-radius:0;
                    padding:1rem 1.5rem;margin-bottom:1.5rem;color:#000000;font-weight:800;">
          Best fit: <strong>{best}</strong> — highest readiness score
        </div>""", unsafe_allow_html=True)

        cols = st.columns(len(results), gap="large")
        for col, r in zip(cols, results):
            score = r.get("readiness_score", 0)
            score_cls = "green" if score >= 70 else ("yellow" if score >= 40 else "red")
            matched_count = len(r.get("matched_skills", []))
            missing_count = len(r.get("missing_skills", []))
            label = r["_label"]
            crown = " 🏆" if label == best else ""

            with col:
                st.markdown(f"""
                <div class="compare-card">
                  <div class="compare-title">{label}{crown}</div>
                  <div class="compare-score {score_cls}">{score}%</div>
                  <div style="color:#555555;font-size:.8rem;margin:.3rem 0;">Readiness</div>
                  <div style="margin-top:.75rem;">
                    <span class="tag tag-green" style="margin:2px;">{matched_count} matched</span>
                    <span class="tag tag-red"   style="margin:2px;">{missing_count} gaps</span>
                  </div>
                </div>""", unsafe_allow_html=True)

                st.progress(score / 100)

                with st.expander("View skill details"):
                    st.markdown("**✅ You have:**")
                    for s in r.get("matched_skills", []):
                        st.markdown(f"- {s}")
                    st.markdown("**⚠️ You're missing:**")
                    for s in r.get("missing_skills", []):
                        st.markdown(f"- {s}")

    else:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#000000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>
            </svg>
          </div>
          <h3>Compare your fit across multiple jobs</h3>
          <p>Upload your resume and paste up to 3 job descriptions to see which is the best fit.</p>
        </div>""", unsafe_allow_html=True)
