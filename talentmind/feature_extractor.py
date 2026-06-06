from datetime import datetime
from talentmind.config import (
    SUBMISSION_DATE, CONSULTING_COMPANIES, TIER_A_SKILLS, TIER_B_SKILLS, TIER_C_SKILLS
)

def _days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
        return (SUBMISSION_DATE - d).days
    except Exception:
        return 365

SENIORITY_MAP = {
    "intern": 0, "trainee": 0, "junior": 1, "associate": 1,
    "mid": 2, "senior": 3, "lead": 4, "principal": 5, "staff": 5,
    "head": 6, "director": 7, "vp": 8, "chief": 9,
}
TECH_TITLE_SIGNALS = frozenset({
    "ml", "ai", "machine learning", "deep learning", "nlp", "data scien",
    "recommendation", "search", "ranking", "retrieval", "research scientist",
    "ai engineer", "ml engineer", "data engineer", "llm",
})
PROD_KEYWORDS = frozenset({"production", "deployed", "live", "users", "scale", "real-time"})
RESEARCH_KEYWORDS = frozenset({"paper", "arxiv", "published", "dataset", "benchmark"})


def _seniority_level(title: str) -> int:
    t = title.lower()
    best = 2  # default: mid
    for kw, lvl in SENIORITY_MAP.items():
        if kw in t:
            best = max(best, lvl)
    return best


def extract_career_features(candidate: dict) -> dict:
    career = candidate.get("career_history", [])
    if not career:
        return {
            "career_score": 0.10, "consulting_fraction": 0.0,
            "has_production_evidence": False, "growth_potential": "LOW",
            "top_career_note": "",
        }

    all_companies = [r.get("company", "").lower() for r in career]
    consulting_count = sum(
        1 for c in all_companies
        if any(cons in c for cons in CONSULTING_COMPANIES)
    )
    consulting_fraction = consulting_count / len(all_companies)

    if consulting_fraction >= 1.0:
        return {
            "career_score": 0.05, "consulting_fraction": 1.0,
            "has_production_evidence": False, "growth_potential": "LOW",
            "top_career_note": "entire career in IT services",
        }

    raw = -1.5 if consulting_fraction >= 0.6 else 0.0
    has_production_evidence = False
    top_career_note = ""

    for i, role in enumerate(career):
        title = role.get("title", "").lower()
        desc = role.get("description", "").lower()
        size = role.get("company_size", "1-10")
        duration = role.get("duration_months", 0)
        recency_mult = 2.0 if i == 0 else 1.0

        if any(sig in title for sig in TECH_TITLE_SIGNALS):
            raw += 1.5 * recency_mult
            if not top_career_note:
                top_career_note = f"{role.get('title','?')} at {role.get('company','?')}"

        if size not in ("1-10", "10001+"):
            raw += 0.4
        elif size == "10001+":
            co = role.get("company", "").lower()
            if not any(c in co for c in CONSULTING_COMPANIES):
                raw += 0.3

        prod_hits = sum(1 for kw in PROD_KEYWORDS if kw in desc)
        research_hits = sum(1 for kw in RESEARCH_KEYWORDS if kw in desc)
        if prod_hits > 0:
            has_production_evidence = True
        raw += prod_hits * 0.25 - research_hits * 0.12

        if 0 < duration < 12:
            raw -= 0.3
        elif duration > 24:
            raw += 0.15

    career_score = max(0.0, min(1.0, raw / 8.0))

    # Growth Potential (AUXILIARY — never enters final_score)
    titles_chron = [r.get("title", "") for r in reversed(career)]
    levels = [_seniority_level(t) for t in titles_chron]
    if len(levels) >= 2 and levels[-1] > levels[0]:
        delta = levels[-1] - levels[0]
        growth_potential = "HIGH" if delta >= 2 else "MEDIUM"
    elif has_production_evidence:
        growth_potential = "MEDIUM"
    else:
        growth_potential = "LOW"

    return {
        "career_score": career_score,
        "consulting_fraction": consulting_fraction,
        "has_production_evidence": has_production_evidence,
        "growth_potential": growth_potential,
        "top_career_note": top_career_note,
    }


def extract_behavior_features(signals: dict) -> dict:
    base = 0.50
    days_inactive = _days_since(signals.get("last_active_date", ""))

    if days_inactive <= 30:
        base += 0.20
    elif days_inactive <= 90:
        base += 0.10
    elif days_inactive > 180:
        base -= 0.25

    if signals.get("open_to_work_flag", False):
        base += 0.12

    rr = signals.get("recruiter_response_rate", 0.5)
    if rr > 0.60:
        base += 0.12
    elif rr >= 0.30:
        base += 0.06
    elif rr < 0.15:
        base -= 0.18

    icr = signals.get("interview_completion_rate", 0.7)
    if icr > 0.80:
        base += 0.08
    elif icr < 0.40:
        base -= 0.12

    gh = signals.get("github_activity_score", -1)
    if gh and gh > 50:
        base += 0.06

    if signals.get("profile_completeness_score", 0) > 80:
        base += 0.04

    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        nm = 1.0
    elif notice <= 60:
        nm = 0.92
    elif notice <= 90:
        nm = 0.80
    elif notice <= 120:
        nm = 0.65
    else:
        nm = 0.50

    return {
        "behavioral_score": max(0.10, min(1.0, base * nm)),
        "days_inactive": days_inactive,
        "notice_period_days": notice,
        "recruiter_response_rate": rr,
    }


def extract_trust_features(candidate: dict) -> dict:
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])

    trust = 0.60
    if signals.get("verified_email") and signals.get("verified_phone"):
        trust += 0.15
    if signals.get("linkedin_connected"):
        trust += 0.10
    endorsed = min(4, sum(1 for s in skills if s.get("endorsements", 0) > 5))
    trust += endorsed * 0.05
    if signals.get("profile_completeness_score", 0) > 80:
        trust += 0.10

    suspicious = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("endorsements", 0) == 0
        and s.get("duration_months", 0) < 6
    )
    trust -= min(0.25, suspicious * 0.05)

    claimed_yoe = float(profile.get("years_of_experience", 0))
    career_months = sum(r.get("duration_months", 0) for r in career)
    if career_months > 0 and abs(claimed_yoe - career_months / 12) > 1.5:
        trust -= 0.15

    oar = signals.get("offer_acceptance_rate", -1)
    icr = signals.get("interview_completion_rate", 0.7)
    if oar == -1 and icr < 0.40:
        trust -= 0.10

    return {"trust_score": max(0.10, min(1.0, trust))}


def extract_skill_features(candidate: dict) -> dict:
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    raw = 0.0
    top_skill_name = ""

    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")
        dur = s.get("duration_months", 0)
        end = s.get("endorsements", 0)

        pm = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.65, "beginner": 0.40}.get(prof, 0.40)
        tm = 0.50 if (dur == 0 and end == 0) else (1.10 if (dur > 24 or end > 5) else 1.0)

        tokens = frozenset(name.split())
        if TIER_A_SKILLS.intersection(tokens) or any(kw in name for kw in TIER_A_SKILLS):
            raw += 3.0 * pm * tm
            if not top_skill_name and dur > 6:
                top_skill_name = s.get("name", "")
        elif TIER_B_SKILLS.intersection(tokens) or any(kw in name for kw in TIER_B_SKILLS):
            raw += 2.0 * pm * tm
            if not top_skill_name and dur > 6:
                top_skill_name = s.get("name", "")
        elif TIER_C_SKILLS.intersection(tokens) or any(kw in name for kw in TIER_C_SKILLS):
            raw += 1.0 * pm * tm

    all_desc = " ".join(r.get("description", "").lower() for r in career)
    raw += sum(1.5 for kw in TIER_A_SKILLS if kw in all_desc)
    raw += sum(0.75 for kw in TIER_B_SKILLS if kw in all_desc)

    for skill_name, score in (signals.get("skill_assessment_scores") or {}).items():
        sn = skill_name.lower()
        if score > 60:
            if any(kw in sn for kw in TIER_A_SKILLS):
                raw += 0.5
            elif any(kw in sn for kw in TIER_B_SKILLS):
                raw += 0.25

    return {
        "skill_score": max(0.0, min(1.0, raw / 20.0)),
        "top_skill_name": top_skill_name,
    }


def extract_experience_features(candidate: dict) -> dict:
    yoe = float(candidate.get("profile", {}).get("years_of_experience", 0))
    if 5.0 <= yoe <= 9.0:
        exp = 1.0
    elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 11.0:
        exp = 0.85
    elif 3.0 <= yoe < 4.0 or 11.0 < yoe <= 14.0:
        exp = 0.70
    else:
        exp = 0.45
    return {"experience_score": exp, "years_of_experience": yoe}


def extract_all_features(candidate: dict) -> dict:
    f = {}
    f.update(extract_career_features(candidate))
    f.update(extract_behavior_features(candidate.get("redrob_signals", {})))
    f.update(extract_trust_features(candidate))
    f.update(extract_skill_features(candidate))
    f.update(extract_experience_features(candidate))
    f["candidate_id"] = candidate["candidate_id"]
    f["current_title"] = candidate.get("profile", {}).get("current_title", "Engineer")
    return f
