import json
import streamlit as st
import plotly.graph_objects as go

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
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  /* ── Base ── */
  .stApp { background:#080C14; color:#E2E8F0;
           font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
  .main .block-container { background:#080C14; padding-top:1.5rem; max-width:1140px; }
  #MainMenu,footer,header { visibility:hidden; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0D1220 0%,#080C14 100%) !important;
    border-right:1px solid #1E2A3A !important;
  }
  [data-testid="stSidebar"] * { color:#C9D1D9 !important; }

  /* ── Inputs ── */
  textarea, input[type="text"] {
    background:#0D1220 !important; color:#E2E8F0 !important;
    border:1px solid #1E2A3A !important; border-radius:10px !important;
    transition:border-color .2s !important;
  }
  textarea:focus, input[type="text"]:focus {
    border-color:#4F8EF7 !important; box-shadow:0 0 0 3px #4F8EF720 !important;
  }
  textarea::placeholder, input::placeholder { color:#6B7A8D !important; }

  .stFileUploader section {
    background:linear-gradient(135deg,#0D1220,#111827) !important;
    border-radius:12px !important; border:1.5px dashed #2D3748 !important;
    transition:border-color .2s !important;
  }
  .stFileUploader section:hover { border-color:#4F8EF7 !important; }
  .stFileUploader section small { color:#8899AA !important; }

  /* ── Buttons ── */
  .stButton>button {
    background:linear-gradient(135deg,#2563EB,#4F8EF7) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-weight:600 !important; padding:0.6rem 1.8rem !important;
    box-shadow:0 4px 15px #2563EB33 !important;
    transition:all .2s !important; letter-spacing:.01em !important;
  }
  .stButton>button:hover {
    background:linear-gradient(135deg,#3B82F6,#60A5FA) !important;
    box-shadow:0 6px 20px #3B82F640 !important;
    transform:translateY(-1px) !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background:#0D1220; border-radius:12px; padding:5px;
    gap:4px; border:1px solid #1E2A3A;
  }
  .stTabs [data-baseweb="tab"] {
    background:transparent; color:#8899AA; border-radius:9px;
    font-weight:500; font-size:.875rem; padding:.45rem 1rem; border:none;
    transition:all .15s;
  }
  .stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#1E293B,#162032) !important;
    color:#93C5FD !important; box-shadow:0 2px 8px #00000040 !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top:1.4rem; }

  /* ── Progress bar ── */
  .stProgress>div>div {
    background:linear-gradient(90deg,#2563EB,#7C3AED) !important;
    border-radius:999px !important;
  }
  .stProgress>div { background:#1E2A3A !important; border-radius:999px !important; height:8px !important; }

  /* ── Checkbox ── */
  .stCheckbox label { color:#C9D1D9 !important; }

  /* ── Alerts ── */
  [data-testid="stAlert"] { border-radius:12px !important; }

  /* ── Hero ── */
  .hero {
    padding:1.4rem 0 2rem; border-bottom:1px solid #1E2A3A; margin-bottom:2rem;
    background:linear-gradient(135deg,#080C14 60%,#0D1A2E 100%);
  }
  .hero-badge {
    display:inline-block;
    background:linear-gradient(135deg,#0F1E3A,#162032);
    color:#60A5FA; font-size:.68rem; font-weight:700; letter-spacing:1.5px;
    text-transform:uppercase; padding:4px 12px; border-radius:999px;
    border:1px solid #2563EB55; margin-bottom:.7rem;
    box-shadow:0 0 12px #2563EB22;
  }
  .hero h1 {
    font-size:2.1rem; font-weight:700; margin:0 0 .4rem; letter-spacing:-.6px;
    background:linear-gradient(135deg,#E2E8F0 30%,#93C5FD 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  }
  .hero p { color:#94A3B8; margin:0; font-size:.95rem; }

  /* ── Input labels ── */
  .input-label {
    font-size:.72rem; font-weight:700; letter-spacing:.8px;
    text-transform:uppercase; color:#4F8EF7; margin-bottom:.4rem;
  }

  /* ── Score cards ── */
  .score-card {
    background:linear-gradient(145deg,#0D1220,#111827);
    border:1px solid #1E2A3A; border-radius:16px;
    padding:1.6rem 1rem; text-align:center; margin-bottom:.75rem;
    box-shadow:0 4px 24px #00000040;
    transition:transform .2s, box-shadow .2s;
  }
  .score-card:hover { transform:translateY(-2px); box-shadow:0 8px 32px #00000055; }
  .score-label {
    font-size:.68rem; font-weight:700; letter-spacing:.8px;
    text-transform:uppercase; color:#8899AA; margin-bottom:.3rem;
  }
  .score-number { font-size:3.2rem; font-weight:700; line-height:1.1; }
  .score-sub { font-size:.8rem; font-weight:500; margin-top:.25rem; }

  .green  { color:#34D399; }
  .yellow { color:#FBBF24; }
  .red    { color:#F87171; }
  .purple { color:#A78BFA; }

  /* ── Skill tags ── */
  .tag-wrap { display:flex; flex-wrap:wrap; gap:8px; margin-top:.6rem; }
  .tag { display:inline-block; padding:5px 14px; border-radius:999px; font-size:.79rem; font-weight:600; }
  .tag-green {
    background:linear-gradient(135deg,#052E16,#064E3B);
    color:#6EE7B7; border:1px solid #059669;
    box-shadow:0 0 8px #05966922;
  }
  .tag-red {
    background:linear-gradient(135deg,#2D0A0A,#450A0A);
    color:#FCA5A5; border:1px solid #DC2626;
    box-shadow:0 0 8px #DC262622;
  }

  /* ── Section titles ── */
  .section-title {
    font-size:.68rem; font-weight:700; letter-spacing:.8px; text-transform:uppercase;
    color:#4F8EF7; margin:1.3rem 0 .7rem; padding-bottom:.4rem;
    border-bottom:1px solid #1E2A3A;
  }

  /* ── Timeline ── */
  .timeline-item {
    display:flex; gap:1rem; padding:1.1rem 0; border-bottom:1px solid #1E2A3A;
    transition:background .15s;
  }
  .week-dot {
    flex-shrink:0; width:38px; height:38px; border-radius:50%;
    background:linear-gradient(135deg,#1E3A5F,#162032);
    border:1.5px solid #3B82F660; color:#60A5FA; font-weight:700; font-size:.7rem;
    display:flex; align-items:center; justify-content:center; margin-top:2px;
    box-shadow:0 0 10px #3B82F625;
  }
  .week-content h4 { margin:0 0 .15rem; font-size:.93rem; font-weight:600; color:#E2E8F0; }
  .week-content .goal { margin:0; font-size:.82rem; color:#94A3B8; }
  .task-list { margin:.4rem 0 0; padding-left:1.1rem; }
  .task-list li { font-size:.82rem; color:#94A3B8; margin-bottom:3px; }
  .project-chip {
    display:inline-block; margin-top:.5rem;
    background:linear-gradient(135deg,#052E16,#064E3B);
    color:#6EE7B7; border:1px solid #059669;
    border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600;
  }

  /* ── Resource cards ── */
  .resource-card {
    background:linear-gradient(135deg,#0D1220,#111827);
    border:1px solid #1E2A3A; border-radius:12px;
    padding:.8rem 1rem; margin-bottom:.5rem; display:flex;
    align-items:center; gap:.8rem; transition:all .2s;
  }
  .resource-card:hover { border-color:#3B82F6; box-shadow:0 4px 16px #3B82F620; transform:translateX(3px); }
  .resource-type {
    flex-shrink:0; font-size:.6rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.6px; padding:3px 8px; border-radius:6px;
    background:#1E2A3A; color:#60A5FA; border:1px solid #2D3F55;
  }
  .resource-title { font-size:.87rem; font-weight:500; color:#C9D1D9; }
  .resource-title a { color:#60A5FA; text-decoration:none; transition:color .15s; }
  .resource-title a:hover { color:#93C5FD; text-decoration:underline; }

  /* ── Interview questions ── */
  .interview-q {
    background:linear-gradient(135deg,#0D1220,#0F1A2E);
    border:1px solid #1E2A3A; border-left:3px solid #3B82F6;
    border-radius:0 12px 12px 0; padding:.9rem 1.1rem; margin-bottom:.5rem;
    font-size:.88rem; color:#C9D1D9; transition:border-color .15s;
  }
  .interview-q:hover { border-left-color:#60A5FA; }
  .interview-q strong { color:#60A5FA; }

  /* ── Resume tips ── */
  .tip-card {
    background:linear-gradient(135deg,#1C1205,#1A1208);
    border-left:3px solid #FBBF24; border-radius:0 10px 10px 0;
    padding:.75rem 1rem; margin-bottom:.5rem; font-size:.87rem; color:#E2E8F0;
    transition:border-color .15s;
  }
  .tip-card:hover { border-left-color:#F59E0B; }

  /* ── Compare cards ── */
  .compare-card {
    background:linear-gradient(145deg,#0D1220,#111827);
    border:1px solid #1E2A3A; border-radius:14px;
    padding:1.3rem; text-align:center;
    box-shadow:0 4px 20px #00000040;
    transition:transform .2s, box-shadow .2s;
  }
  .compare-card:hover { transform:translateY(-2px); box-shadow:0 8px 30px #00000055; }
  .compare-score { font-size:2.6rem; font-weight:700; }
  .compare-title { font-size:.82rem; color:#8899AA; font-weight:500; margin-bottom:.5rem; }

  /* ── Utility ── */
  .divider { border:none; border-top:1px solid #1E2A3A; margin:1.5rem 0; }

  .empty-state { text-align:center; padding:4rem 0; color:#8899AA; }
  .empty-state .icon { font-size:2.8rem; margin-bottom:.7rem; }
  .empty-state h3 { color:#94A3B8; font-size:.95rem; font-weight:600; margin:0 0 .3rem; }
  .empty-state p  { font-size:.83rem; margin:0; }

  /* ── Agent progress nodes ── */
  .node-step {
    background:#0D1220; border:1px solid #1E2A3A; border-radius:10px;
    padding:.55rem 1rem; margin-bottom:.35rem; font-size:.84rem; color:#8899AA;
    transition:all .3s;
  }
  .node-step.done {
    background:linear-gradient(135deg,#052E16,#064E3B);
    border-color:#059669; color:#6EE7B7;
    box-shadow:0 0 10px #05966922;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: Progress Tracker ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 AlignAgent")
    st.markdown("<hr style='border-color:#1E2A3A;margin:.5rem 0 1rem'>", unsafe_allow_html=True)
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

    st.markdown("<hr style='border-color:#1E2A3A;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("#### ℹ️ About")
    st.caption("Powered by **Ollama** (local, free). No data leaves your machine.")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">⚡ Local · Free · Agentic RAG</div>
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
                  <div class="score-sub" style="color:#94A3B8;">of {timeline} week plan</div>
                </div>""", unsafe_allow_html=True)

        with ec:
            skills_learned = len([s for s in missing if s in learned])
            pct = round(skills_learned / len(missing) * 100) if missing else 100
            st.markdown(f"""
            <div class="score-card">
              <div class="score-label">Skills Learned</div>
              <div class="score-number" style="color:#58A6FF;">{skills_learned}/{len(missing)}</div>
              <div class="score-sub" style="color:#94A3B8;">{pct}% of gaps closed</div>
            </div>""", unsafe_allow_html=True)

        # ── Charts row ───────────────────────────────────────────────────────
        ch1, ch2, ch3 = st.columns([1.1, 1, 1.4], gap="large")

        with ch1:
            # Gauge — readiness score
            gauge_color = "#34D399" if adj_score >= 70 else ("#FBBF24" if adj_score >= 40 else "#F87171")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=adj_score,
                number={"suffix": "%", "font": {"size": 36, "color": "#E2E8F0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8899AA",
                             "tickfont": {"color": "#8899AA", "size": 10}},
                    "bar": {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": "#0D1220",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  40], "color": "#2D0A0A"},
                        {"range": [40, 70], "color": "#1C1205"},
                        {"range": [70, 100], "color": "#052E16"},
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
                font={"color": "#E2E8F0"}, height=210,
                margin=dict(l=20, r=20, t=30, b=10),
                title={"text": "Readiness Gauge", "font": {"size": 11, "color": "#8899AA"},
                       "x": 0.5, "xanchor": "center"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        with ch2:
            # Donut — matched vs missing
            donut_labels = ["Skills Matched", "Skills Missing"]
            donut_values = [max(len(matched), 1), max(len(missing), 0)]
            fig_donut = go.Figure(go.Pie(
                labels=donut_labels,
                values=donut_values,
                hole=0.68,
                marker=dict(colors=["#059669", "#DC2626"],
                            line=dict(color="#080C14", width=3)),
                textfont={"color": "#E2E8F0", "size": 11},
                hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            fig_donut.add_annotation(
                text=f"<b>{len(matched)}/{len(matched)+len(missing)}</b>",
                x=0.5, y=0.5, font_size=20, font_color="#E2E8F0", showarrow=False,
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#C9D1D9"}, height=210,
                showlegend=True,
                legend=dict(font=dict(color="#C9D1D9", size=10),
                            bgcolor="rgba(0,0,0,0)", orientation="h",
                            x=0.5, xanchor="center", y=-0.05),
                margin=dict(l=10, r=10, t=30, b=10),
                title={"text": "Skills Breakdown", "font": {"size": 11, "color": "#8899AA"},
                       "x": 0.5, "xanchor": "center"},
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        with ch3:
            # Horizontal bar — all skills (green = have, red = missing)
            all_skills  = matched + [s for s in missing if s not in learned]
            bar_colors  = (["#059669"] * len(matched) +
                           ["#DC2626"] * len([s for s in missing if s not in learned]))
            bar_labels  = (["✓ " + s for s in matched] +
                           ["✗ " + s for s in missing if s not in learned])
            if all_skills:
                fig_bar = go.Figure(go.Bar(
                    y=bar_labels,
                    x=[1] * len(bar_labels),
                    orientation="h",
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    text=bar_labels,
                    textposition="inside",
                    textfont={"color": "#E2E8F0", "size": 10},
                    hoverinfo="skip",
                ))
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#C9D1D9"},
                    height=max(180, len(bar_labels) * 26 + 50),
                    xaxis=dict(visible=False, range=[0, 1.05]),
                    yaxis=dict(visible=False),
                    margin=dict(l=5, r=5, t=30, b=10),
                    bargap=0.25,
                    title={"text": "Skill Map", "font": {"size": 11, "color": "#8899AA"},
                           "x": 0.5, "xanchor": "center"},
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # ── Skill tags ────────────────────────────────────────────────────────
        sk1, sk2 = st.columns(2, gap="large")
        with sk1:
            st.markdown('<div class="section-title">Skills you have</div>', unsafe_allow_html=True)
            tags = "".join(f'<span class="tag tag-green">{s}</span>'
                           for s in matched + [s for s in missing if s in learned])
            st.markdown(f'<div class="tag-wrap">{tags or "<em style=color:#94A3B8>None detected</em>"}</div>',
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
                        gantt_colors.append("#059669" if done_g else "#2563EB")
                        gantt_text.append("✓ Done" if done_g else f"Week {w}")
                if gantt_skills:
                    fig_gantt = go.Figure()
                    for i, (skill_g, start, end, color, txt) in enumerate(
                        zip(gantt_skills, gantt_starts, gantt_ends, gantt_colors, gantt_text)
                    ):
                        fig_gantt.add_trace(go.Bar(
                            x=[end - start], base=[start], y=[skill_g],
                            orientation="h",
                            marker=dict(color=color, line=dict(width=0)),
                            text=txt, textposition="inside",
                            textfont={"color": "#E2E8F0", "size": 10},
                            showlegend=False,
                            hovertemplate=f"<b>{skill_g}</b><br>Week {start+1}<extra></extra>",
                        ))
                    max_week = max(gantt_ends)
                    fig_gantt.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#C9D1D9"},
                        height=max(200, len(gantt_skills) * 36 + 80),
                        barmode="overlay", bargap=0.3,
                        xaxis=dict(
                            title="Week", range=[0, max_week + 0.2],
                            tickvals=list(range(1, max_week + 1)),
                            ticktext=[f"Wk {i}" for i in range(1, max_week + 1)],
                            gridcolor="#1E2A3A", color="#8899AA",
                        ),
                        yaxis=dict(gridcolor="#1E2A3A", color="#C9D1D9"),
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
                  <div style="color:#94A3B8;font-size:.8rem;margin:.3rem 0;">Readiness</div>
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
