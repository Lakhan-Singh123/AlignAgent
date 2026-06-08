"""Report export utilities — HTML (print-to-PDF) and JSON download."""

from __future__ import annotations

import json
from datetime import date


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:3px 10px;'
        f'border-radius:999px;font-size:0.78rem;font-weight:600;'
        f'margin:3px;display:inline-block;">{text}</span>'
    )


def generate_html_report(report: dict, candidate_name: str = "Candidate") -> str:
    """Return a self-contained HTML string suitable for browser print-to-PDF."""
    score      = report.get("readiness_score", 0)
    timeline   = report.get("timeline_weeks", "?")
    matched    = report.get("matched_skills", [])
    missing    = report.get("missing_skills", [])
    plan       = report.get("learning_plan", [])
    resources  = report.get("resources", {})
    projects   = report.get("recommended_projects", [])
    interview  = report.get("interview_questions", [])
    resume_tips= report.get("resume_suggestions", [])
    today      = date.today().strftime("%B %d, %Y")

    score_color = "#16A34A" if score >= 70 else ("#D97706" if score >= 40 else "#DC2626")

    matched_tags = " ".join(_badge(s, "#16A34A") for s in matched)
    missing_tags = " ".join(_badge(s, "#DC2626") for s in missing)

    plan_rows = ""
    for step in plan:
        if isinstance(step, dict):
            tasks_html = "".join(f"<li>{t}</li>" for t in step.get("tasks", []))
            plan_rows += f"""
            <tr>
              <td style="padding:10px;border-bottom:1px solid #e5e7eb;font-weight:600;
                         color:#1f2937;width:60px;text-align:center;">W{step.get('week','?')}</td>
              <td style="padding:10px;border-bottom:1px solid #e5e7eb;">
                <strong>{step.get('skill','')}</strong><br>
                <span style="color:#6b7280;font-size:0.85rem;">{step.get('goal','')}</span><br>
                <ul style="margin:6px 0;padding-left:18px;color:#374151;font-size:0.85rem;">{tasks_html}</ul>
                <span style="background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:6px;
                             font-size:0.78rem;">🔨 {step.get('project','')}</span>
              </td>
            </tr>"""

    res_html = ""
    for skill, links in resources.items():
        res_html += f'<p style="font-weight:600;margin:12px 0 4px;">{skill}</p>'
        for r in links:
            if isinstance(r, dict):
                url   = r.get("url", "")
                title = r.get("title", "")
                rtype = r.get("type", "").upper()
                link  = f'<a href="{url}" style="color:#2563eb;">{title}</a>' if url else title
                res_html += f'<div style="padding:5px 0;font-size:0.88rem;">[{rtype}] {link}</div>'

    interview_html = "".join(
        f'<div style="padding:8px 0;border-bottom:1px solid #f3f4f6;">'
        f'<strong>Q{i+1}.</strong> {q}</div>'
        for i, q in enumerate(interview)
    ) if interview else "<p style='color:#6b7280;'>No interview questions generated.</p>"

    tips_html = "".join(
        f'<div style="padding:6px 10px;margin:4px 0;background:#fef3c7;border-left:3px solid #f59e0b;'
        f'border-radius:4px;font-size:0.88rem;">{t}</div>'
        for t in resume_tips
    ) if resume_tips else "<p style='color:#6b7280;'>No suggestions generated.</p>"

    projects_html = "".join(
        f'<div style="padding:6px 10px;margin:4px 0;background:#eff6ff;border-left:3px solid #2563eb;'
        f'border-radius:4px;font-size:0.88rem;">{p}</div>'
        for p in projects
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AlignAgent Report — {candidate_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          max-width: 860px; margin: 0 auto; padding: 40px 32px; color: #111827; }}
  h1   {{ font-size: 1.8rem; margin: 0; }}
  h2   {{ font-size: 1.1rem; color: #374151; border-bottom: 1px solid #e5e7eb;
          padding-bottom: 6px; margin-top: 28px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  @media print {{ body {{ padding: 20px; }} }}
</style>
</head>
<body>
  <div style="display:flex;justify-content:space-between;align-items:flex-start;
              border-bottom:2px solid #111827;padding-bottom:16px;margin-bottom:24px;">
    <div>
      <div style="font-size:0.75rem;font-weight:700;letter-spacing:1px;
                  text-transform:uppercase;color:#6b7280;">AlignAgent</div>
      <h1>Skill Gap Report</h1>
      <div style="color:#6b7280;font-size:0.88rem;margin-top:4px;">{today}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:3rem;font-weight:700;color:{score_color};line-height:1;">{score}%</div>
      <div style="color:#6b7280;font-size:0.85rem;">Readiness Score</div>
      <div style="color:#7c3aed;font-weight:600;">{timeline} weeks</div>
    </div>
  </div>

  <h2>✅ Skills You Have</h2>
  <div style="margin:8px 0;">{matched_tags or '<em style="color:#6b7280;">None detected</em>'}</div>

  <h2>⚠️ Skills to Develop</h2>
  <div style="margin:8px 0;">{missing_tags or '<em style="color:#6b7280;">None — great fit!</em>'}</div>

  <h2>📅 Week-by-Week Learning Plan</h2>
  <table><tbody>{plan_rows}</tbody></table>

  <h2>📚 Learning Resources</h2>
  {res_html or '<p style="color:#6b7280;">No resources generated.</p>'}

  <h2>🎯 Interview Prep</h2>
  {interview_html}

  <h2>📝 Resume Suggestions</h2>
  {tips_html}

  <h2>🛠 Recommended Projects</h2>
  {projects_html or '<p style="color:#6b7280;">No projects generated.</p>'}

  <div style="margin-top:40px;padding-top:12px;border-top:1px solid #e5e7eb;
              color:#9ca3af;font-size:0.78rem;text-align:center;">
    Generated by AlignAgent · Powered by Ollama (local, free)
  </div>
</body>
</html>"""


def generate_json_bytes(report: dict) -> bytes:
    return json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
