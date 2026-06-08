from skill_gap_analyzer import SkillGapAnalyzer

# 1. Initialize the analyzer in fallback mode (no LLM needed)
analyzer = SkillGapAnalyzer(
    candidate_workspace_id="test", 
    internship_workspace_id="test", 
    fallback_only=True
)

# 2. Read your test bug text
text = "I have experience with Python and Git. However, my current gaps include Docker and CI/CD."

# 3. Print the results
acquired = analyzer._extract_candidate_skills(text)
gaps = analyzer._extract_candidate_gap_skills(text)

print(f"Skills Found: {acquired}")
print(f"Gaps Found: {gaps}")

# 4. Logical Check
if "Docker" in acquired:
    print("❌ BUG STILL PRESENT: Docker is wrongly flagged as a skill!")
else:
    print("✅ SUCCESS: Docker correctly excluded from skills.")