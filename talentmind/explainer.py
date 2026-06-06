def _concern(f: dict) -> str:
    if f.get("notice_period_days", 30) > 90:
        return f"notice period {f['notice_period_days']} days"
    if f.get("days_inactive", 0) > 180:
        return "low recent activity — availability uncertain"
    if f.get("recruiter_response_rate", 0.5) < 0.20:
        return "low recruiter response rate"
    if f.get("consulting_fraction", 0) > 0.60:
        return "primarily services-company background"
    if f.get("years_of_experience", 7) < 4:
        return "slightly below target experience range"
    return ""

def generate_reasoning(f: dict, rank: int, score: float, sem: float) -> str:
    title = f.get("current_title", "Engineer")
    yoe = f.get("years_of_experience", 0)
    note = f.get("top_career_note", "")
    prod = f.get("has_production_evidence", False)
    growth = f.get("growth_potential", "MEDIUM")
    skill = f.get("top_skill_name", "")
    concern = _concern(f)

    skill_phrase = f"with {skill} expertise" if skill else ""

    if rank <= 10:
        prod_phrase = " (production deployment evidence)" if prod else ""
        primary = (
            f"{yoe:.0f}-year {title} {skill_phrase}{prod_phrase}"
            + (f"; {note}" if note else "")
            + "; strong JD alignment."
        )
    elif rank <= 30:
        primary = f"{yoe:.0f}-year {title} {skill_phrase}; solid JD fit" + (f" — {note}." if note else ".")
    elif rank <= 60:
        primary = f"{yoe:.0f}-year {title}; moderate fit" + (f" — {note}" if note else "") + "."
    else:
        primary = f"{yoe:.0f}-year {title}; adjacent skills present, below primary threshold."

    secondary = (f"Note: {concern}." if concern
                 else f"Growth Potential: {growth.capitalize()}.")
    return (primary + " " + secondary).strip()[:500]
