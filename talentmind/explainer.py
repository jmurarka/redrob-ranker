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
    if f.get("notice_period_days", 30) > 90:
        return f"notice period {f['notice_period_days']} days"
    if f.get("days_inactive", 0) > 180:
        return "low recent activity - availability uncertain"
    if f.get("recruiter_response_rate", 0.5) < 0.20:
        return "low recruiter response rate"
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
    """
    if breakdown is None:
        title = f.get("current_title", "Engineer")
        yoe = f.get("years_of_experience", 0)
        note = f.get("top_career_note", "")
        prod = f.get("has_production_evidence", False)
        growth = f.get("growth_potential", "MEDIUM")
        skill = f.get("top_skill_name", "")
        concern = _concern(f)

        weighted = f.get("weighted_signals")
        raw = f.get("raw_signals")

        if weighted and raw and score > 0:
            contributions = {k: (v / score) * 100 for k, v in weighted.items()}
            sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
            top1_name, top1_pct = sorted_contrib[0]
            top2_name, top2_pct = sorted_contrib[1]

            skill_phrase = f"with {skill} expertise" if skill else ""
            primary = (
                f"Ranked #{rank} (Score: {score:.3f}). Driven by "
                f"{top1_name} ({top1_pct:.1f}% contribution) and "
                f"{top2_name} ({top2_pct:.1f}% contribution). "
                f"{yoe:.0f}-year {title} {skill_phrase}."
            )
            secondary = f"Growth Potential: {growth.capitalize()}."
            if concern:
                secondary += f" Note: {concern}."
            return _clean_reasoning_text(primary + " " + secondary).strip()[:500]

        skill_phrase = f"with {skill} expertise" if skill else ""

        if rank <= 10:
            prod_phrase = " (production deployment evidence)" if prod else ""
            primary = (
                f"{yoe:.0f}-year {title} {skill_phrase}{prod_phrase}"
                + (f"; {note}" if note else "")
                + "; strong JD alignment."
            )
        elif rank <= 30:
            primary = f"{yoe:.0f}-year {title} {skill_phrase}; solid JD fit" + (f" - {note}." if note else ".")
        elif rank <= 60:
            primary = f"{yoe:.0f}-year {title}; moderate fit" + (f" - {note}" if note else "") + "."
        else:
            primary = f"{yoe:.0f}-year {title}; adjacent skills present, below primary threshold."

        secondary = f"Growth Potential: {growth.capitalize()}."
        if concern:
            secondary += f" Note: {concern}."
        return _clean_reasoning_text(primary + " " + secondary).strip()[:500]

    sem_top = max(1, int(101.0 - breakdown.semantic_pct))
    career_top = max(1, int(101.0 - breakdown.career_pct))

    skills_clean = []
    for skill in breakdown.matched_skills:
        formatted = _display_skill(skill)
        if formatted not in skills_clean:
            skills_clean.append(formatted)
    skills_str = ", ".join(skills_clean[:3]) if skills_clean else "required AI/search stack"

    promo_str = (
        " -> ".join(_title_case_role(role) for role in breakdown.promotion_evidence[:4])
        if breakdown.promotion_evidence
        else "relevant ML/search roles"
    )

    primary = f"Top {sem_top}% semantic similarity and top {career_top}% career trajectory."
    if breakdown.promotion_evidence:
        primary += f" Promotion history: {promo_str}."
    primary += f" Matched skills: {skills_str}."

    if breakdown.leadership_evidence:
        primary += f" Leadership: {_clean_reasoning_text(breakdown.leadership_evidence[0])}."

    growth_potential = f.get("growth_potential", "MEDIUM").capitalize()
    secondary = f"Growth Potential: {growth_potential}."
    concern = _concern(f)
    if concern:
        secondary += f" Note: {concern}."

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
