import pytest
from talentmind.jd_intelligence import extract_jd_deep_profile, parse_jd, JDProfile
from talentmind.career_trajectory import CareerTrajectoryAnalyzer
from talentmind.feature_extractor import extract_career_features
from talentmind.config import WEIGHT_PROFILES
from talentmind.explainer import ScoreBreakdown, generate_reasoning, generate_score_breakdown_table
from rank import compute_candidate_relevance, compute_jd_satisfaction


def test_ontology_extraction():
    jd = """
    We are looking for a Senior AI Engineer.
    Our mission is to build a low-latency recommendation engine to scale search conversion.
    You will report to the VP of Engineering and collaborate with Product Managers.
    
    Responsibilities:
    - Design and build hybrid retrieval search layers.
    - Deploy ML models in production at scale.
    - Mentor junior developers.
    
    Requirements:
    - Must have deep experience with Python and PyTorch.
    - Core expertise in Milvus or Faiss.
    - 5-9 years of experience.
    
    Nice to have:
    - Experience with RAG and LLM fine-tuning is a plus.
    """
    
    profile = extract_jd_deep_profile(jd)
    
    assert profile.seniority_level == "Senior"
    assert profile.experience_range == {"min": 5, "max": 9}
    assert "recommendation engine" in profile.business_objective.lower()
    assert "VP of Engineering" in profile.team_context
    assert any("python" in s.lower() for s in profile.required_skills)
    assert any("milvus" in s.lower() for s in profile.required_skills)
    assert any("rag" in s.lower() for s in profile.preferred_skills)
    assert "Mentorship required" in profile.implicit_signals
    assert "mentorship" in profile.leadership_signals
    assert "design" in profile.responsibility_categories


def test_career_trajectory_metrics():
    career = [
        {"title": "Senior AI Engineer", "company": "Google", "duration_months": 24, "company_size": "10001+", "description": "led team, mentored junior devs, managed scaling"},
        {"title": "AI Engineer", "company": "Google", "duration_months": 24, "company_size": "10001+", "description": "built model in production"},
        {"title": "Junior Developer", "company": "Startup", "duration_months": 12, "company_size": "1-10", "description": "wrote code"}
    ]
    
    analyzer = CareerTrajectoryAnalyzer()
    
    # Assert individual metrics
    promotions = analyzer.detect_promotions(career)
    assert promotions == 1
    
    velocity = analyzer.compute_promotion_velocity(career)
    assert velocity > 0.0
    
    consistency = analyzer.compute_specialization_consistency(career)
    assert consistency == 1.0  # all titles contain "Engineer" or "Developer"
    
    growth = analyzer.compute_leadership_growth(career)
    assert growth > 0.0
    
    stability = analyzer.compute_tenure_stability(career)
    assert stability > 0.0
    
    domain_score = analyzer.compute_domain_focus_score(career)
    assert domain_score > 0.0
    
    # Test through feature extractor
    candidate = {
        "candidate_id": "CAND_TEST_CAREER",
        "career_history": career,
        "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 36, "endorsements": 10}]
    }
    feats = extract_career_features(candidate)
    
    assert feats["career_score"] > 0.0
    assert feats["promotion_velocity"] == velocity
    assert feats["specialization_consistency"] == consistency
    assert feats["leadership_growth"] == growth
    assert feats["tenure_stability"] == stability
    assert feats["domain_focus_score"] == domain_score
    assert feats["career_intelligence_score"] > 0.0


def test_symmetric_matching():
    # Setup mock candidate features and JDProfile
    feat = {
        "candidate_skills": ["python", "pytorch", "faiss"],
        "candidate_titles": ["junior developer", "ai engineer"],
        "years_of_experience": 6.0,
        "current_title": "AI Engineer",
        "specialization_consistency": 1.0,
        "domain_focus_score": 0.5
    }
    
    jd_profile = JDProfile(
        required_skills=["python", "pytorch", "faiss"],
        preferred_skills=["rag"],
        seniority_level="Senior",
        experience_range={"min": 5, "max": 9},
        business_objective="Build scalable vector search.",
        team_context="Reports to VP of Engineering.",
        industry="AI / SaaS",
        responsibility_list=["Design vector search solutions."],
        implicit_signals=["Mentorship required"]
    )
    
    relevance = compute_candidate_relevance(feat, jd_profile)
    satisfaction = compute_jd_satisfaction(feat, jd_profile)
    combined = 0.5 * relevance + 0.5 * satisfaction
    
    assert 0.0 <= relevance <= 1.0
    assert 0.0 <= satisfaction <= 1.0
    assert 0.0 <= combined <= 1.0


def test_weight_validation():
    # Test that all weight profiles sum to 1.0
    for profile_name, weights in WEIGHT_PROFILES.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-9, f"Profile {profile_name} weights sum to {sum(weights.values())}, not 1.0"


def test_explanation_generation():
    feat = {
        "candidate_id": "CAND_001",
        "current_title": "Senior AI Engineer",
        "years_of_experience": 6.0,
        "growth_potential": "HIGH"
    }
    
    breakdown = ScoreBreakdown(
        candidate_id="CAND_001",
        semantic=0.88,
        career=0.85,
        skill=0.82,
        experience=1.0,
        behavioral=0.9,
        trust=0.95,
        growth=0.8,
        semantic_pct=92.0,  # Top 9%
        career_pct=95.0,    # Top 6%
        skill_pct=88.0,
        experience_pct=85.0,
        behavioral_pct=80.0,
        trust_pct=85.0,
        growth_pct=75.0,
        matched_skills=["PyTorch", "Milvus", "Python"],
        promotion_evidence=["Engineer", "Senior Engineer", "Lead Engineer"],
        leadership_evidence=["Title: Lead Engineer"]
    )
    
    reasoning = generate_reasoning(feat, rank=3, score=0.85, breakdown=breakdown)
    
    assert "Top 9% semantic similarity" in reasoning
    assert "Top 6% career trajectory" in reasoning
    assert "Engineer → Senior Engineer → Lead Engineer" in reasoning
    assert "PyTorch, Milvus, Python" in reasoning
    assert len(reasoning) <= 500


def test_score_breakdown_table():
    breakdown = ScoreBreakdown(
        candidate_id="CAND_001",
        semantic=0.88,
        career=0.85,
        skill=0.82,
        experience=1.0,
        behavioral=0.9,
        trust=0.95,
        growth=0.8
    )
    
    table = generate_score_breakdown_table(breakdown)
    
    assert "| Signal | Raw Score | Weight | Contribution | Percentage |" in table
    assert "Semantic" in table
    assert "Career" in table
    assert "Skill" in table
    assert "**Total**" in table
