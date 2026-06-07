from talentmind.config import (
    MAX_YOE_DELTA, MAX_EXPERT_ZERO_EVIDENCE_SKILLS,
    MAX_SINGLE_ROLE_MONTHS, MAX_SALARY_LPA
)

def is_invalid(candidate: dict) -> tuple[bool, str]:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals") or {}

    claimed_yoe = float(profile.get("years_of_experience", 0))
    career_months = sum(r.get("duration_months", 0) for r in career)
    career_years = career_months / 12.0

    if career_years > 0.5 and abs(claimed_yoe - career_years) > MAX_YOE_DELTA:
        return True, f"yoe_delta:{abs(claimed_yoe - career_years):.1f}"

    unverified_experts = sum(
        1 for s in skills
        if s.get("proficiency") == "expert"
        and s.get("duration_months", 0) == 0
        and s.get("endorsements", 0) == 0
    )
    if unverified_experts >= MAX_EXPERT_ZERO_EVIDENCE_SKILLS:
        return True, f"unverified_experts:{unverified_experts}"

    if any(r.get("duration_months", 0) > MAX_SINGLE_ROLE_MONTHS for r in career):
        return True, "tenure_overflow"

    salary = signals.get("expected_salary_range_inr_lpa", {}) or {}
    if salary.get("min", 0) > MAX_SALARY_LPA:
        return True, f"salary_impossible:{salary.get('min')}"

    for s in skills:
        if s.get("duration_months", 0) > career_months + 6:
            return True, f"skill_duration_overflow:{s.get('name','?')}"

    return False, ""
