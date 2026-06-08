import json
import streamlit as st

from analyzer import (
    extract_text_from_pdf,
    get_ai_analysis,
    get_interview_questions,
    get_resume_suggestions,
    scrape_jd_from_url,
)
from export import generate_html_report, generate_json_bytes
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
    page_icon="🎯",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background:#0D1117; color:#E6EDF3;
           font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; }
  .main .block-container { background:#0D1117; padding-top:1.5rem; max-width:1120px; }
  #MainMenu,footer,header { visibility:hidden; }

  /* Sidebar */
  [data-testid="stSidebar"] { background:#0D1117 !important; border-right:1px solid #21262D; }
  [data-testid="stSidebar"] * { color:#E6EDF3 !important; }

  /* Inputs */
  textarea,input[type="text"] {
    background:#161B22 !important; color:#E6EDF3 !important;
    border:1px solid #30363D !important; border-radius:8px !important;
  }
  textarea::placeholder,input::placeholder { color:#6E7681 !important; }
  .stFileUploader section { background:#161B22 !important; border-radius:10px !important;
                             border:1.5px dashed #30363D !important; }
  .stFileUploader section small { color:#8B949E !important; }

  /* Buttons */
  .stButton>button {
    background:#1F6FEB !important; color:#fff !important;
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; padding:0.55rem 1.6rem !important;
    transition:background .15s !important;
  }
  .stButton>button:hover { background:#388BFD !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background:#161B22; border-radius:10px; padding:4px;
    gap:4px; border:1px solid #30363D;
  }
  .stTabs [data-baseweb="tab"] {
    background:transparent; color:#8B949E; border-radius:8px;
    font-weight:500; font-size:.875rem; padding:.4rem .9rem; border:none;
  }
  .stTabs [aria-selected="true"] { background:#21262D !important; color:#E6EDF3 !important; }
  .stTabs [data-baseweb="tab-panel"] { padding-top:1.2rem; }

  /* Progress bar */
  .stProgress>div>div { background:#1F6FEB !important; border-radius:999px !important; }
  .stProgress>div     { background:#21262D !important; border-radius:999px !important; height:8px !important; }

  /* Checkbox */
  .stCheckbox label { color:#E6EDF3 !important; }

  /* Alerts */
  [data-testid="stAlert"] { border-radius:10px !important; }

  /* Custom components */
  .hero { padding:1.2rem 0 1.8rem; border-bottom:1px solid #21262D; margin-bottom:1.8rem; }
  .hero-badge {
    display:inline-block; background:#162032; color:#58A6FF;
    font-size:.7rem; font-weight:700; letter-spacing:1px; text-transform:uppercase;
    padding:3px 10px; border-radius:999px; border:1px solid #1F6FEB44; margin-bottom:.6rem;
  }
  .hero h1 { font-size:1.9rem; font-weight:700; color:#E6EDF3; margin:0 0 .3rem; letter-spacing:-.5px; }
  .hero p  { color:#8B949E; margin:0; font-size:.95rem; }

  .input-label { font-size:.78rem; font-weight:700; letter-spacing:.5px;
                 text-transform:uppercase; color:#8B949E; margin-bottom:.35rem; }

  .score-card {
    background:#161B22; border:1px solid #30363D; border-radius:14px;
    padding:1.5rem; text-align:center; margin-bottom:.75rem;
  }
  .score-label { font-size:.72rem; font-weight:700; letter-spacing:.6px;
                 text-transform:uppercase; color:#8B949E; }
  .score-number { font-size:3rem; font-weight:700; line-height:1.1; }
  .score-sub { font-size:.82rem; font-weight:500; margin-top:.2rem; }
  .green  { color:#3FB950; } .yellow { color:#D29922; }
  .red    { color:#F85149; } .purple { color:#BC8CFF; }

  .tag-wrap { display:flex; flex-wrap:wrap; gap:7px; margin-top:.5rem; }
  .tag { display:inline-block; padding:5px 13px; border-radius:999px; font-size:.8rem; font-weight:500; }
  .tag-green { background:#0F2A1A; color:#3FB950; border:1px solid #238636; }
  .tag-red   { background:#2A0F0F; color:#F85149; border:1px solid #6E2424; }

  .section-title { font-size:.72rem; font-weight:700; letter-spacing:.6px; text-transform:uppercase;
                   color:#8B949E; margin:1.2rem 0 .6rem; padding-bottom:.4rem;
                   border-bottom:1px solid #21262D; }

  .timeline-item { display:flex; gap:1rem; padding:1rem 0; border-bottom:1px solid #21262D; }
  .week-dot { flex-shrink:0; width:36px; height:36px; border-radius:50%; background:#162032;
              border:1.5px solid #1F6FEB44; color:#58A6FF; font-weight:700; font-size:.72rem;
              display:flex; align-items:center; justify-content:center; margin-top:2px; }
  .week-content h4 { margin:0 0 .15rem; font-size:.92rem; font-weight:600; color:#E6EDF3; }
  .week-content .goal { margin:0; font-size:.82rem; color:#8B949E; }
  .task-list { margin:.4rem 0 0; padding-left:1.1rem; }
  .task-list li { font-size:.82rem; color:#8B949E; margin-bottom:3px; }
  .project-chip { display:inline-block; margin-top:.5rem; background:#0F2A1A; color:#3FB950;
                  border:1px solid #238636; border-radius:6px; padding:3px 9px; font-size:.76rem; }

  .resource-card { background:#161B22; border:1px solid #30363D; border-radius:10px;
                   padding:.75rem 1rem; margin-bottom:.4rem; display:flex;
                   align-items:center; gap:.75rem; transition:border-color .15s; }
  .resource-card:hover { border-color:#58A6FF; }
  .resource-type { flex-shrink:0; font-size:.62rem; font-weight:700; text-transform:uppercase;
                   letter-spacing:.5px; padding:3px 7px; border-radius:5px;
                   background:#21262D; color:#8B949E; }
  .resource-title { font-size:.87rem; font-weight:500; color:#E6EDF3; }
  .resource-title a { color:#58A6FF; text-decoration:none; }
  .resource-title a:hover { text-decoration:underline; }

  .interview-q { background:#161B22; border:1px solid #30363D; border-radius:10px;
                 padding:.85rem 1rem; margin-bottom:.4rem; font-size:.88rem; color:#E6EDF3; }
  .interview-q strong { color:#58A6FF; }

  .tip-card { background:#1A1208; border-left:3px solid #D29922; border-radius:0 8px 8px 0;
              padding:.7rem 1rem; margin-bottom:.4rem; font-size:.87rem; color:#E6EDF3; }

  .compare-card { background:#161B22; border:1px solid #30363D; border-radius:12px;
                  padding:1.2rem; text-align:center; }
  .compare-score { font-size:2.4rem; font-weight:700; }
  .compare-title { font-size:.82rem; color:#8B949E; font-weight:500; margin-bottom:.5rem; }

  .divider { border:none; border-top:1px solid #21262D; margin:1.5rem 0; }
  .empty-state { text-align:center; padding:3.5rem 0; color:#6E7681; }
  .empty-state .icon { font-size:2.5rem; margin-bottom:.6rem; }
  .empty-state h3 { color:#8B949E; font-size:.95rem; font-weight:600; margin:0 0 .25rem; }
  .empty-state p  { font-size:.83rem; margin:0; }

  .node-step { background:#161B22; border:1px solid #30363D; border-radius:8px;
               padding:.5rem .9rem; margin-bottom:.3rem; font-size:.85rem; color:#8B949E; }
  .node-step.done { border-color:#238636; color:#3FB950; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: Progress Tracker ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 AlignAgent")
    st.markdown("<hr style='border-color:#21262D;margin:.5rem 0 1rem'>", unsafe_allow_html=True)
    st.markdown("#### 📊 Skill Progress Tracker")
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

    st.markdown("<hr style='border-color:#21262D;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("#### ℹ️ About")
    st.caption("Powered by **Ollama** (local, free). No data leaves your machine.")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">Local · Free · Agentic RAG</div>
  <h1>AlignAgent</h1>
  <p>Upload your resume and a job description — get a full skill gap report, week-by-week plan, interview prep, and resume suggestions.</p>
</div>
""", unsafe_allow_html=True)


# ── Mode Tabs ─────────────────────────────────────────────────────────────────
mode_tab, compare_tab = st.tabs(["🔍  Analyse a Job", "⚖️  Compare Multiple Jobs"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════
with mode_tab:

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="input-label">Resume (PDF)</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("resume", type=["pdf"], label_visibility="collapsed")
        if uploaded_file:
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
        report = {}

        if "Quick" in pipeline_mode:
            with st.spinner("Running quick analysis…"):
                report = get_ai_analysis(resume_text, job_description)
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

            with st.spinner("Running deep agentic analysis…"):
                final_state = {}
                for event in run_agentic_analysis(
                    uploaded_file.read(), uploaded_file.name, job_description
                ):
                    node_name = list(event.keys())[0]
                    completed_nodes.append(node_name)
                    _render_nodes(completed_nodes)
                    final_state = event[node_name]

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

        with tc:
            if timeline:
                remaining_weeks = max(0, timeline - len(learned))
                st.markdown(f"""
                <div class="score-card">
                  <div class="score-label">Weeks Remaining</div>
                  <div class="score-number purple">{remaining_weeks}</div>
                  <div class="score-sub" style="color:#8B949E;">of {timeline} week plan</div>
                </div>""", unsafe_allow_html=True)

        with ec:
            skills_learned = len([s for s in missing if s in learned])
            pct = round(skills_learned / len(missing) * 100) if missing else 100
            st.markdown(f"""
            <div class="score-card">
              <div class="score-label">Skills Learned</div>
              <div class="score-number" style="color:#58A6FF;">{skills_learned}/{len(missing)}</div>
              <div class="score-sub" style="color:#8B949E;">{pct}% of gaps closed</div>
            </div>""", unsafe_allow_html=True)

        # ── Skill tags ────────────────────────────────────────────────────────
        sk1, sk2 = st.columns(2, gap="large")
        with sk1:
            st.markdown('<div class="section-title">Skills you have</div>', unsafe_allow_html=True)
            tags = "".join(f'<span class="tag tag-green">{s}</span>'
                           for s in matched + [s for s in missing if s in learned])
            st.markdown(f'<div class="tag-wrap">{tags or "<em style=color:#6E7681>None detected</em>"}</div>',
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
        t1, t2, t3, t4, t5, t6 = st.tabs([
            "📅 Learning Plan", "📚 Resources",
            "🎤 Interview Prep", "📝 Resume Tips",
            "🛠 Projects", "📥 Export",
        ])

        with t1:
            if plan:
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
            st.markdown('<div class="section-title">Download your report</div>', unsafe_allow_html=True)

            dl1, dl2 = st.columns(2)
            candidate_name = uploaded_file.name.replace(".pdf", "") if uploaded_file else "Candidate"

            with dl1:
                html_report = generate_html_report(report, candidate_name)
                st.download_button(
                    "📄 Download HTML Report",
                    data=html_report.encode("utf-8"),
                    file_name="alignagent_report.html",
                    mime="text/html",
                    use_container_width=True,
                )
                st.caption("Open in browser → Print → Save as PDF")

            with dl2:
                st.download_button(
                    "📦 Download JSON Report",
                    data=generate_json_bytes(report),
                    file_name="alignagent_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.caption("Raw data for integration or backup")

    else:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">🎯</div>
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
        <div style="background:#0F2A1A;border:1px solid #238636;border-radius:12px;
                    padding:1rem 1.5rem;margin-bottom:1.5rem;color:#3FB950;font-weight:600;">
          🏆 Best fit: <strong>{best}</strong> — highest readiness score
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
                  <div style="color:#8B949E;font-size:.8rem;margin:.3rem 0;">Readiness</div>
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
          <div class="icon">⚖️</div>
          <h3>Compare your fit across multiple jobs</h3>
          <p>Upload your resume and paste up to 3 job descriptions to see which is the best fit.</p>
        </div>""", unsafe_allow_html=True)
