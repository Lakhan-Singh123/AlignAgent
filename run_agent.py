from graph import app
import json

def main():
    initial_state = {
        "candidate_id": "candidate_demo",
        "internship_id": "internship_demo",
        "analysis_report_path": "reports/final_gap_report.json",
        "candidate_queries": [],
        "internship_queries": [],
        "candidate_docs": [],
        "internship_docs": [],
        "candidate_context": "",
        "internship_context": "",
        "web_context": "",
        "raw_report": "",
        "resources": {},
        "grounding_passed": False,
        "retry_count": 0,
    }

    print("🚀 AlignAgent: Starting agentic RAG analysis...")
    result = app.invoke(initial_state)
    print("✅ Analysis complete!")
    print(f"📄 Report saved to: {result['analysis_report_path']}\n")

    with open(result["analysis_report_path"]) as f:
        report = json.load(f)

    print(f"🎯 Readiness Score : {report['readiness_score']}%")
    print(f"⏱️  Total Timeline  : {report.get('timeline_weeks', '?')} weeks\n")

    print("✅ Matched Skills:")
    for s in report["matched_skills"]:
        print(f"   • {s}")

    print("\n⚠️  Missing Skills:")
    for s in report["missing_skills"]:
        print(f"   • {s}")

    print("\n📅 Week-by-Week Learning Plan:")
    for entry in report.get("learning_plan", []):
        if isinstance(entry, dict):
            print(f"\n  Week {entry.get('week', '?')} — {entry.get('skill', '')}")
            print(f"  Goal   : {entry.get('goal', '')}")
            for task in entry.get("tasks", []):
                print(f"    - {task}")
            print(f"  Project: {entry.get('project', '')}")
        else:
            print(f"  • {entry}")

    print("\n📚 Learning Resources:")
    for skill, links in report.get("resources", {}).items():
        print(f"\n  {skill}:")
        for r in links:
            if isinstance(r, dict):
                url = r.get("url", "")
                print(f"    [{r.get('type','').upper()}] {r.get('title','')} {('— ' + url) if url else ''}")

if __name__ == "__main__":
    main()
