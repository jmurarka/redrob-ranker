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
    
    # Percentiles
    semantic_pct: float = 0.0
    career_pct: float = 0.0
    skill_pct: float = 0.0
    experience_pct: float = 0.0
    behavioral_pct: float = 0.0
    trust_pct: float = 0.0
    growth_pct: float = 0.0
    
    # Contributions
    semantic_contrib: float = 0.0
    career_contrib: float = 0.0
    skill_contrib: float = 0.0
    experience_contrib: float = 0.0
    behavioral_contrib: float = 0.0
    trust_contrib: float = 0.0
    growth_contrib: float = 0.0
    
    # Evidence fields
    matched_skills: List[str] = field(default_factory=list)
    promotion_evidence: List[str] = field(default_factory=list)
    leadership_evidence: List[str] = field(default_factory=list)


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


def generate_reasoning(
    f: dict, 
    rank: int, 
    score: float, 
    sem: float = None, 
    breakdown: Optional[ScoreBreakdown] = None
) -> str:
    """
    Generates reasoning strings. Supports new evidence-based formatting
    if breakdown is provided, otherwise falls back to the legacy parser.
    """
    # Legacy fallback path
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
            return (primary + " " + secondary).strip()[:500]
    
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
    
        secondary = f"Growth Potential: {growth.capitalize()}."
        if concern:
            secondary += f" Note: {concern}."
        return (primary + " " + secondary).strip()[:500]

    # New evidence-based format path
    sem_top = max(1, int(101.0 - breakdown.semantic_pct))
    career_top = max(1, int(101.0 - breakdown.career_pct))
    
    # Standard capitalization mapping
    standard_cases = {
        "python": "Python", "pytorch": "PyTorch", "milvus": "Milvus", "faiss": "FAISS",
        "qdrant": "Qdrant", "pinecone": "Pinecone", "weaviate": "Weaviate", "opensearch": "OpenSearch",
        "elasticsearch": "Elasticsearch", "rag": "RAG", "llm": "LLM", "nlp": "NLP",
        "langchain": "LangChain", "llamaindex": "LlamaIndex", "a/b testing": "A/B Testing",
        "ab testing": "A/B Testing", "mlops": "MLOps", "xgboost": "XGBoost", "lightgbm": "LightGBM"
    }
    
    skills_clean = []
    for s in breakdown.matched_skills:
        sc = standard_cases.get(s.lower(), s.title())
        if sc not in skills_clean:
            skills_clean.append(sc)
    skills_str = ", ".join(skills_clean[:3]) if skills_clean else "Required Tech Stack"
    
    promo_str = " → ".join(breakdown.promotion_evidence[:4]) if breakdown.promotion_evidence else "Relevant Roles"
    
    primary = f"Top {sem_top}% semantic similarity. Top {career_top}% career trajectory."
    if breakdown.promotion_evidence:
        primary += f" Promotion history: {promo_str}."
    primary += f" Matched skills: {skills_str}."
    
    if breakdown.leadership_evidence:
        primary += f" Leadership: {breakdown.leadership_evidence[0]}."
        
    concern = ""
    if f.get("notice_period_days", 30) > 90:
        concern = f"Note: notice period {f['notice_period_days']} days."
    elif f.get("days_inactive", 0) > 180:
        concern = "Note: low recent activity."
        
    growth_potential = f.get("growth_potential", "MEDIUM").capitalize()
    secondary = f"Growth Potential: {growth_potential}."
    if concern:
        secondary += f" {concern}"
        
    reasoning = (primary + " " + secondary).strip()
    return reasoning[:500]


def generate_score_breakdown_table(breakdown: ScoreBreakdown) -> str:
    """
    Generates a markdown table of signal contributions for candidate evaluation.
    """
    weights = {
        "semantic": 0.20,
        "career": 0.25,
        "skill": 0.20,
        "experience": 0.10,
        "behavioral": 0.10,
        "trust": 0.10,
        "growth": 0.05
    }
    
    raw_scores = {
        "semantic": breakdown.semantic,
        "career": breakdown.career,
        "skill": breakdown.skill,
        "experience": breakdown.experience,
        "behavioral": breakdown.behavioral,
        "trust": breakdown.trust,
        "growth": breakdown.growth
    }
    
    contributions = {k: raw_scores[k] * weights[k] for k in weights}
    total_score = sum(contributions.values())
    
    lines = [
        "| Signal | Raw Score | Weight | Contribution | Percentage |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]
    for k in weights:
        raw = raw_scores[k]
        w = weights[k]
        contrib = contributions[k]
        pct = (contrib / total_score * 100.0) if total_score > 0 else 0.0
        lines.append(f"| {k.capitalize()} | {raw:.4f} | {w:.2f} | {contrib:.4f} | {pct:.1f}% |")
    
    lines.append(f"| **Total** | | | **{total_score:.4f}** | **100.0%** |")
    return "\n".join(lines)
