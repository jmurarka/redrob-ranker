def build_candidate_text(candidate: dict) -> str:
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    role = profile.get("current_title", "")

    advanced_skills = sorted(
        [s for s in skills if s.get("proficiency") in ("advanced", "expert")],
        key=lambda s: s.get("duration_months", 0), reverse=True
    )[:15]
    skills_str = ", ".join(s.get("name", "") for s in advanced_skills)

    titles = [r.get("title", "") for r in career[:5] if r.get("title")]
    career_str = ", ".join(titles)

    summary = (profile.get("summary") or "")[:150].strip()

    return (
        f"Role: {role}\n"
        f"Skills: {skills_str}\n"
        f"Career: {career_str}\n"
        f"Summary: {summary}"
    ).strip()
