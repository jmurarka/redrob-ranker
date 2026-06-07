import pytest
import numpy as np
from datetime import date
from talentmind.qa_filter import is_invalid
from talentmind.feature_extractor import (
    extract_career_features,
    extract_behavior_features,
    extract_trust_features,
    extract_skill_features,
    extract_experience_features,
    extract_all_features
)
from talentmind.jd_intelligence import parse_jd
from talentmind.explainer import generate_reasoning
from talentmind.text_builder import build_candidate_text

def test_is_invalid_valid():
    # Valid candidate profile
    cand = {
        "candidate_id": "CAND_001",
        "profile": {"years_of_experience": 5.0},
        "career_history": [{"company": "Google", "duration_months": 60}],
        "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 36, "endorsements": 10}],
        "redrob_signals": {"expected_salary_range_inr_lpa": {"min": 50, "max": 80}}
    }
    invalid, reason = is_invalid(cand)
    assert invalid is False
    assert reason == ""

def test_is_invalid_yoe_delta():
    # Discrepancy > 3.0 between claimed YOE and sum of duration_months
    cand = {
        "candidate_id": "CAND_002",
        "profile": {"years_of_experience": 10.0},
        "career_history": [{"company": "Wipro", "duration_months": 24}],
        "skills": []
    }
    invalid, reason = is_invalid(cand)
    assert invalid is True
    assert "yoe_delta" in reason

def test_extract_career_features_medium():
    cand = {
        "career_history": [
            {
                "title": "Software Engineer",
                "company": "Startup Inc",
                "company_size": "51-200",
                "duration_months": 30,
                "description": "designed production services"
            }
        ]
    }
    f = extract_career_features(cand)
    assert f["career_score"] > 0.0
    assert f["growth_potential"] == "MEDIUM"

def test_extract_behavior_features():
    signals = {
        "last_active_date": "2026-05-15",
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.80,
        "notice_period_days": 30
    }
    f = extract_behavior_features(signals)
    assert 0.10 <= f["behavioral_score"] <= 1.0

def test_extract_trust_features():
    cand = {
        "redrob_signals": {
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        },
        "skills": [
            {"name": "FAISS", "proficiency": "expert", "duration_months": 12, "endorsements": 10}
        ]
    }
    f = extract_trust_features(cand)
    assert 0.10 <= f["trust_score"] <= 1.0

def test_extract_skill_features():
    cand = {
        "skills": [
            {"name": "Milvus", "proficiency": "expert", "duration_months": 25, "endorsements": 6}
        ],
        "career_history": []
    }
    f = extract_skill_features(cand)
    assert f["skill_score"] > 0.0
    assert f["top_skill_name"] == "Milvus"

def test_extract_experience_features():
    cand = {"profile": {"years_of_experience": 6.0}}
    f = extract_experience_features(cand)
    assert f["experience_score"] == 1.0

    cand2 = {"profile": {"years_of_experience": 15.0}}
    f2 = extract_experience_features(cand2)
    assert f2["experience_score"] == 0.45

def test_parse_jd():
    jd = "Seeking a Senior AI Engineer to build embeddings and vector search solutions using Pinecone. Need 5-9 years of experience."
    parsed = parse_jd(jd)
    assert parsed["role_category"] == "ML_ENGINEER"
    assert parsed["seniority"] == "Senior"
    assert parsed["experience_range"] == {"min": 5, "max": 9}

def test_generate_reasoning():
    feat = {
        "current_title": "AI Engineer",
        "years_of_experience": 6.0,
        "top_career_note": "AI Engineer at Initech",
        "has_production_evidence": True,
        "growth_potential": "HIGH",
        "top_skill_name": "Pinecone"
    }
    r = generate_reasoning(feat, rank=5, score=0.85, sem=0.9)
    assert "AI Engineer" in r
    assert "Pinecone" in r
    assert "6-year" in r
    assert "Growth Potential: High" in r

def test_redrob_signals_none():
    cand = {
        "candidate_id": "CAND_NONE_SIGNALS",
        "profile": {"years_of_experience": 5.0, "current_title": "ML Engineer"},
        "career_history": [{"company": "Google", "duration_months": 60, "title": "ML Engineer", "company_size": "100-500", "description": "built model in production"}],
        "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 36, "endorsements": 10}],
        "redrob_signals": None
    }
    # Should not raise AttributeError
    invalid, reason = is_invalid(cand)
    assert not invalid
    
    feats = extract_all_features(cand)
    assert feats["behavioral_score"] > 0.0
    assert feats["trust_score"] > 0.0

def test_generate_reasoning_with_concern():
    feat = {
        "current_title": "AI Engineer",
        "years_of_experience": 6.0,
        "top_career_note": "AI Engineer at Initech",
        "has_production_evidence": True,
        "growth_potential": "HIGH",
        "top_skill_name": "Pinecone",
        "notice_period_days": 120  # will trigger concern
    }
    r = generate_reasoning(feat, rank=50, score=0.75, sem=0.8)
    assert "Growth Potential: High" in r
    assert "Note: notice period 120 days" in r

