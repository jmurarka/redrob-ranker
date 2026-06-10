from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ScoreBreakdown:
    """Dataclass holding candidate signal breakdown scores and evidence."""

    candidate_id: str
    semantic: float
    career: float
    skill: float
    experience: float
    behavioral: float
    trust: float
    growth: float
    logistics: float = 0.0

    semantic_pct: float = 0.0
    career_pct: float = 0.0
    skill_pct: float = 0.0
    experience_pct: float = 0.0
    behavioral_pct: float = 0.0
    trust_pct: float = 0.0
    growth_pct: float = 0.0
    logistics_pct: float = 0.0

    semantic_contrib: float = 0.0
    career_contrib: float = 0.0
    skill_contrib: float = 0.0
    experience_contrib: float = 0.0
    behavioral_contrib: float = 0.0
    trust_contrib: float = 0.0
    growth_contrib: float = 0.0
    logistics_contrib: float = 0.0

    matched_skills: List[str] = field(default_factory=list)
    promotion_evidence: List[str] = field(default_factory=list)
    leadership_evidence: List[str] = field(default_factory=list)


STANDARD_CASES = {
    "python": "Python",
    "pytorch": "PyTorch",
    "milvus": "Milvus",
    "faiss": "FAISS",
    "qdrant": "Qdrant",
    "pinecone": "Pinecone",
    "weaviate": "Weaviate",
    "opensearch": "OpenSearch",
    "elasticsearch": "Elasticsearch",
    "rag": "RAG",
    "llm": "LLM",
    "llms": "LLMs",
    "nlp": "NLP",
    "ai": "AI",
    "ml": "ML",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "a/b testing": "A/B Testing",
    "ab testing": "A/B Testing",
    "mlops": "MLOps",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "qlora": "QLoRA",
    "lora": "LoRA",
    "peft": "PEFT",
    "bge": "BGE",
    "bm25": "BM25",
    "pgvector": "pgvector",
}


def _clean_reasoning_text(text: str) -> str:
    return (
        str(text)
        .replace("â†’", "->")
        .replace("→", "->")
        .replace("â€”", "-")
        .replace("—", "-")
        .replace("â€“", "-")
        .replace("–", "-")
    )


def _display_skill(skill: str) -> str:
    key = str(skill).lower().strip()
    return STANDARD_CASES.get(key, str(skill).title())


def _title_case_role(text: str) -> str:
    words = []
    for raw in _clean_reasoning_text(text).split():
        stripped = raw.strip("()[]{}:,;")
        key = stripped.lower()
        words.append(STANDARD_CASES.get(key, raw.capitalize()))
    return " ".join(words)


def _concern(f: dict) -> str:
    notice   = f.get("notice_period_days", 30)
    inactive = f.get("days_inactive", 0)
    rr       = f.get("recruiter_response_rate", 0.5)
    # Availability is the hardest constraint for a founding hire — surface first
    if notice > 90 and inactive > 180:
        return f"notice period {notice}d and {inactive}d platform inactivity"
    if notice > 90:
        return f"notice period {notice} days"
    if inactive > 180:
        return f"{inactive} days platform inactivity"
    # Responsiveness
    if rr < 0.10:
        return f"recruiter response rate {int(rr*100)}% - high drop-off risk"
    if rr < 0.20:
        return f"recruiter response rate {int(rr*100)}%"
    # Logistics / background / seniority
    if f.get("logistics_score", 1.0) < 0.50:
        return f.get("logistics_note", "location/logistics fit is weak")
    if f.get("consulting_fraction", 0) > 0.60:
        return "primarily services-company background"
    if f.get("years_of_experience", 7) < 4:
        return "slightly below target experience range"
    return ""


def generate_reasoning(
    f: dict,
    rank: int,
    score: float,
    sem: float = None,
    breakdown: Optional[ScoreBreakdown] = None,
) -> str:
    """
    Generates evidence-backed reasoning strings for CSV review.
    Tone is calibrated to rank so top picks read as confident endorsements
    and lower picks honestly explain why they're included and what limits them.
    """
    title   = f.get("current_title", "Engineer")
    yoe     = f.get("years_of_experience", 0)
    note    = f.get("top_career_note", "")
    prod    = f.get("has_production_evidence", False)
    growth  = f.get("growth_potential", "MEDIUM")
    skill   = f.get("top_skill_name", "")
    concern = _concern(f)
    avail_ok = f.get("availability_multiplier", 1.0) >= 0.99

    skill_phrase = f"with {skill} expertise" if skill else ""
    prod_phrase  = " and production deployment evidence" if prod else ""

    if breakdown is None:
        weighted = f.get("weighted_signals")
        raw      = f.get("raw_signals")

        if weighted and raw and score > 0:
            contributions  = {k: (v / score) * 100 for k, v in weighted.items()}
            sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
            top1_name, top1_pct = sorted_contrib[0]
            top2_name, top2_pct = sorted_contrib[1]

            # Rank-aware primary sentence
            if rank <= 10:
                primary = (
                    f"Top-{rank} pick. {yoe:.0f}-year {title} {skill_phrase}{prod_phrase}. "
                    f"Score driven by {top1_name} ({top1_pct:.1f}%) and {top2_name} ({top2_pct:.1f}%)."
                    + (f" {note}." if note else "")
                )
            elif rank <= 30:
                primary = (
                    f"Ranked #{rank}. Strong fit — {yoe:.0f}-year {title} {skill_phrase}. "
                    f"Led by {top1_name} ({top1_pct:.1f}%) and {top2_name} ({top2_pct:.1f}%)."
                    + (f" {note}." if note else "")
                )
            elif rank <= 60:
                primary = (
                    f"Ranked #{rank}. Solid secondary candidate — {yoe:.0f}-year {title} {skill_phrase}. "
                    f"Primary signal: {top1_name} ({top1_pct:.1f}%)."
                    + (f" {note}." if note else "")
                )
            else:
                limiting = concern if concern else "profile below primary threshold on multiple signals"
                primary = (
                    f"Ranked #{rank}. Included for {top1_name} match ({top1_pct:.1f}%); "
                    f"limiting factor: {limiting}. "
                    f"{yoe:.0f}-year {title} {skill_phrase}."
                )

            # Rank-aware secondary (growth + concern framing)
            if rank <= 10:
                secondary = f"Growth: {growth.capitalize()}."
                if concern:
                    secondary += f" Caveat: {concern}."
            elif rank <= 30:
                secondary = f"Growth Potential: {growth.capitalize()}."
                if concern:
                    secondary += f" Note: {concern}."
            else:
                if concern:
                    secondary = f"Key limitation: {concern}. Growth: {growth.capitalize()}."
                else:
                    secondary = f"Growth Potential: {growth.capitalize()}."

            return _clean_reasoning_text(primary + " " + secondary).strip()[:500]

        # Fallback (no weighted_signals on feat dict)
        if rank <= 10:
            primary = (
                f"Top-{rank} pick. {yoe:.0f}-year {title} {skill_phrase}{prod_phrase}"
                + (f"; {note}" if note else "")
                + "; strong JD alignment."
            )
        elif rank <= 30:
            primary = f"{yoe:.0f}-year {title} {skill_phrase}; solid JD fit" + (f" - {note}." if note else ".")
        elif rank <= 60:
            primary = f"{yoe:.0f}-year {title}; moderate fit" + (f" - {note}" if note else "") + "."
        else:
            limiting = concern if concern else "profile below primary threshold on multiple signals"
            primary  = f"{yoe:.0f}-year {title}; included for skill match. Limiting factor: {limiting}."

        if rank <= 10:
            secondary = f"Growth: {growth.capitalize()}."
            if concern:
                secondary += f" Caveat: {concern}."
        elif rank <= 30:
            secondary = f"Growth Potential: {growth.capitalize()}."
            if concern:
                secondary += f" Note: {concern}."
        else:
            secondary = (f"Key limitation: {concern}. " if concern else "") + f"Growth: {growth.capitalize()}."

        return _clean_reasoning_text(primary + " " + secondary).strip()[:500]

    # ── Breakdown path (full ScoreBreakdown object) ──────────────────────────
    sem_top    = max(1, int(101.0 - breakdown.semantic_pct))
    career_top = max(1, int(101.0 - breakdown.career_pct))

    skills_clean = []
    for s in breakdown.matched_skills:
        formatted = _display_skill(s)
        if formatted not in skills_clean:
            skills_clean.append(formatted)
    skills_str = ", ".join(skills_clean[:3]) if skills_clean else "required AI/search stack"

    promo_str = (
        " -> ".join(_title_case_role(role) for role in breakdown.promotion_evidence[:4])
        if breakdown.promotion_evidence
        else "relevant ML/search roles"
    )

    if rank <= 10:
        primary = (
            f"Top-{rank} pick. Top {sem_top}% semantic match, top {career_top}% career trajectory. "
            f"Matched skills: {skills_str}."
        )
    elif rank <= 30:
        primary = (
            f"Ranked #{rank}. Top {sem_top}% semantic similarity and top {career_top}% career trajectory. "
            f"Matched skills: {skills_str}."
        )
    else:
        primary = (
            f"Ranked #{rank}. Semantic top {sem_top}%, career top {career_top}%. "
            f"Skills: {skills_str}."
        )

    if breakdown.promotion_evidence:
        primary += f" Promotion trail: {promo_str}."
    if breakdown.leadership_evidence:
        primary += f" Leadership: {_clean_reasoning_text(breakdown.leadership_evidence[0])}."

    growth_potential = f.get("growth_potential", "MEDIUM").capitalize()

    if rank <= 10:
        secondary = f"Growth: {growth_potential}."
        if concern:
            secondary += f" Caveat: {concern}."
    elif rank <= 30:
        secondary = f"Growth Potential: {growth_potential}."
        if concern:
            secondary += f" Note: {concern}."
    else:
        secondary = (f"Key limitation: {concern}. " if concern else "") + f"Growth: {growth_potential}."

    return _clean_reasoning_text(primary + " " + secondary).strip()[:500]


def generate_score_breakdown_table(breakdown: ScoreBreakdown) -> str:
    """
    Generates a markdown table of signal contributions for candidate evaluation.
    """
    weights = {
        "semantic": 0.18,
        "career": 0.24,
        "skill": 0.18,
        "experience": 0.10,
        "behavioral": 0.10,
        "trust": 0.10,
        "growth": 0.05,
        "logistics": 0.05,
    }

    raw_scores = {
        "semantic": breakdown.semantic,
        "career": breakdown.career,
        "skill": breakdown.skill,
        "experience": breakdown.experience,
        "behavioral": breakdown.behavioral,
        "trust": breakdown.trust,
        "growth": breakdown.growth,
        "logistics": breakdown.logistics,
    }

    contributions = {key: raw_scores[key] * weights[key] for key in weights}
    total_score = sum(contributions.values())

    lines = [
        "| Signal | Raw Score | Weight | Contribution | Percentage |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for key in weights:
        raw = raw_scores[key]
        weight = weights[key]
        contrib = contributions[key]
        pct = (contrib / total_score * 100.0) if total_score > 0 else 0.0
        lines.append(f"| {key.capitalize()} | {raw:.4f} | {weight:.2f} | {contrib:.4f} | {pct:.1f}% |")

    lines.append(f"| **Total** | | | **{total_score:.4f}** | **100.0%** |")
    return "\n".join(lines)