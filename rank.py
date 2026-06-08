#!/usr/bin/env python3
"""
rank.py — TalentMind AI v2 runtime ranking.
Usage: python rank.py --candidates candidates.jsonl --out submission.csv
Requires: data/ directory from precompute.py
Constraints: ≤5 min, ≤16GB RAM, CPU only, no network.
"""
import argparse, csv, json, pickle, time
from pathlib import Path
import numpy as np
from typing import List

from talentmind.config import (
    DATA_DIR, WEIGHT_PROFILES, DEFAULT_WEIGHT_PROFILE,
    SEMANTIC_RECALL_K, CAREER_RECALL_K, BEHAVIORAL_RECALL_K,
    LOCAL_MODEL_PATH,
)
from talentmind.embedder import embed_single, cosine_top_k
from talentmind.jd_intelligence import parse_jd, JDProfile
from talentmind.explainer import generate_reasoning, ScoreBreakdown

def compute_candidate_relevance(feat: dict, jd_profile: JDProfile) -> float:
    """Calculates how well a candidate's skills and experience satisfy the JD requirements."""
    cand_skills = set(feat.get("candidate_skills", []))
    req_skills = set(s.lower() for s in jd_profile.required_skills)
    pref_skills = set(s.lower() for s in jd_profile.preferred_skills)
    
    # Skill matching
    req_match = len(cand_skills.intersection(req_skills)) / len(req_skills) if req_skills else 1.0
    pref_match = len(cand_skills.intersection(pref_skills)) / len(pref_skills) if pref_skills else 1.0
    skill_score = 0.7 * req_match + 0.3 * pref_match
    
    # Experience matching
    cand_yoe = feat.get("years_of_experience", 0.0)
    exp_min = jd_profile.experience_range.get("min", 0)
    exp_max = jd_profile.experience_range.get("max", 100)
    if exp_min <= cand_yoe <= exp_max:
        exp_score = 1.0
    else:
        distance = min(abs(cand_yoe - exp_min), abs(cand_yoe - exp_max))
        exp_score = max(0.0, 1.0 - distance / 5.0)
        
    return float(0.6 * skill_score + 0.4 * exp_score)

def compute_jd_satisfaction(feat: dict, jd_profile: JDProfile) -> float:
    """Calculates how well the job aligns with the candidate's career trajectory."""
    spec_score = feat.get("specialization_consistency", 0.5)
    domain_score = feat.get("domain_focus_score", 0.5)
    
    # Seniority level match
    from talentmind.career_trajectory import _seniority_level
    cand_sen = _seniority_level(feat.get("current_title", ""))
    target_sen = _seniority_level(jd_profile.seniority_level)
    if cand_sen >= target_sen:
        sen_score = 1.0
    else:
        sen_score = max(0.2, 1.0 - (target_sen - cand_sen) * 0.2)
        
    return float(0.4 * spec_score + 0.3 * domain_score + 0.3 * sen_score)

def get_percentile(val: float, all_vals: List[float]) -> float:
    """Calculates the percentile ranking of a value in a list of values."""
    if not all_vals:
        return 0.0
    sorted_vals = sorted(all_vals)
    import bisect
    count = bisect.bisect_right(sorted_vals, val)
    return (count / len(all_vals)) * 100.0

def main():
    assert Path(LOCAL_MODEL_PATH).exists(), \
        f"Local model not found at {LOCAL_MODEL_PATH}. Run precompute.py first."
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    t0 = time.time()

    print("[rank] Loading precomputed data...")
    embs = np.load(str(DATA_DIR / "candidate_embeddings.npy"))
    with open(DATA_DIR / "candidate_features.pkl", "rb") as f:
        feats = pickle.load(f)
    with open(DATA_DIR / "jd_structured.json") as f:
        jd_struct = json.load(f)
    print(f"  Pool: {len(feats)} candidates | Embeddings: {embs.shape}")

    print("[rank] Embedding JD...")
    jd_text = Path("job_description.txt").read_text(encoding="utf-8")
    jd_emb = embed_single(jd_text)

    weights = WEIGHT_PROFILES.get(
        jd_struct.get("weight_profile", DEFAULT_WEIGHT_PROFILE),
        WEIGHT_PROFILES[DEFAULT_WEIGHT_PROFILE]
    )

    # Validate weight sum equals 1.0 (Task 3)
    assert abs(sum(weights.values()) - 1.0) < 1e-9, f"Weights sum to {sum(weights.values())}, not 1.0!"

    # Reconstruct JDProfile dataclass (Task 5)
    profile_keys = {
        "required_skills", "preferred_skills", "seniority_level", "experience_range",
        "business_objective", "team_context", "industry", "responsibility_list", "implicit_signals",
        "team_structure", "responsibility_hierarchy", "industry_context", "implicit_seniority_signals",
        "hard_requirements", "nice_to_have", "leadership_signals", "responsibility_categories"
    }
    profile_args = {k: jd_struct[k] for k in profile_keys if k in jd_struct}
    jd_profile = JDProfile(**profile_args)

    print("[rank] Multi-path recall...")
    N = len(feats)
    sem_k = min(SEMANTIC_RECALL_K, N)
    car_k = min(CAREER_RECALL_K, N)
    beh_k = min(BEHAVIORAL_RECALL_K, N)

    sem_idx, _ = cosine_top_k(jd_emb, embs, sem_k)
    recall = set(sem_idx.tolist())

    career_arr = np.array([f["career_score"] for f in feats])
    recall.update(np.argpartition(career_arr, -car_k)[-car_k:].tolist())

    beh_arr = np.array([f["behavioral_score"] for f in feats])
    recall.update(np.argpartition(beh_arr, -beh_k)[-beh_k:].tolist())

    print(f"  Recall set: {len(recall)} candidates")

    print("[rank] Reranking...")
    idx_list = list(recall)
    recall_embs = embs[idx_list]
    cos_sims = recall_embs @ jd_emb

    results = []
    all_sems, all_careers, all_skills, all_exps, all_behs, all_trusts, all_growths = [], [], [], [], [], [], []
    for i, idx in enumerate(idx_list):
        f = feats[idx]
        sem = float((cos_sims[i] + 1.0) / 2.0)
        
        # Symmetric Matching (Task 4)
        candidate_relevance = compute_candidate_relevance(f, jd_profile)
        jd_satisfaction = compute_jd_satisfaction(f, jd_profile)
        combined_match = 0.5 * candidate_relevance + 0.5 * jd_satisfaction
        
        blended_skill = float(max(0.0, min(1.0, 0.7 * f["skill_score"] + 0.3 * combined_match)))
        blended_experience = float(max(0.0, min(1.0, 0.7 * f["experience_score"] + 0.3 * combined_match)))
        
        # Growth Score normalization (Task 3)
        growth_val = float(max(0.0, min(1.0, f.get("growth_score", 0.20))))

        raw_signals = {
            "semantic": sem,
            "career": f["career_score"],
            "skill": blended_skill,
            "experience": blended_experience,
            "behavioral": f["behavioral_score"],
            "trust": f["trust_score"],
            "growth": growth_val
        }

        weighted_signals = {
            "semantic": weights["semantic"] * sem,
            "career": weights["career"] * f["career_score"],
            "skill": weights["skill"] * blended_skill,
            "experience": weights["experience"] * blended_experience,
            "behavioral": weights["behavioral"] * f["behavioral_score"],
            "trust": weights["trust"] * f["trust_score"],
            "growth": weights["growth"] * growth_val
        }

        score = sum(weighted_signals.values())
        f["raw_signals"] = raw_signals
        f["weighted_signals"] = weighted_signals

        all_sems.append(sem)
        all_careers.append(f["career_score"])
        all_skills.append(blended_skill)
        all_exps.append(blended_experience)
        all_behs.append(f["behavioral_score"])
        all_trusts.append(f["trust_score"])
        all_growths.append(growth_val)

        results.append((score, f, sem))

    results.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    top100 = results[:100]

    scores = [round(r[0], 6) for r in top100]
    assert all(scores[i] >= scores[i+1] for i in range(99)), \
        "CRITICAL: Score monotonicity violated"

    print(f"[rank] Writing {args.out}...")
    seen = set()
    with open(args.out, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for pos, (score, feat, sem) in enumerate(top100, 1):
            # Compute percentiles
            sem_pct = get_percentile(feat["raw_signals"]["semantic"], all_sems)
            career_pct = get_percentile(feat["raw_signals"]["career"], all_careers)
            skill_pct = get_percentile(feat["raw_signals"]["skill"], all_skills)
            exp_pct = get_percentile(feat["raw_signals"]["experience"], all_exps)
            beh_pct = get_percentile(feat["raw_signals"]["behavioral"], all_behs)
            trust_pct = get_percentile(feat["raw_signals"]["trust"], all_trusts)
            growth_pct = get_percentile(feat["raw_signals"]["growth"], all_growths)
            
            # Compute contributions
            total_weighted = sum(feat["weighted_signals"].values())
            sem_contrib = (feat["weighted_signals"]["semantic"] / total_weighted * 100.0) if total_weighted > 0 else 0.0
            career_contrib = (feat["weighted_signals"]["career"] / total_weighted * 100.0) if total_weighted > 0 else 0.0
            skill_contrib = (feat["weighted_signals"]["skill"] / total_weighted * 100.0) if total_weighted > 0 else 0.0
            exp_contrib = (feat["weighted_signals"]["experience"] / total_weighted * 100.0) if total_weighted > 0 else 0.0
            beh_contrib = (feat["weighted_signals"]["behavioral"] / total_weighted * 100.0) if total_weighted > 0 else 0.0
            trust_contrib = (feat["weighted_signals"]["trust"] / total_weighted * 100.0) if total_weighted > 0 else 0.0
            growth_contrib = (feat["weighted_signals"]["growth"] / total_weighted * 100.0) if total_weighted > 0 else 0.0
            
            # Extract evidence
            matched_skills = [s for s in feat.get("candidate_skills", []) if any(rs.lower() in s for rs in jd_profile.required_skills) or any(ps.lower() in s for ps in jd_profile.preferred_skills)]
            
            titles_chron = list(reversed(feat.get("candidate_titles", [])))
            unique_titles = []
            for t in titles_chron:
                t_cap = t.title()
                if not unique_titles or unique_titles[-1] != t_cap:
                    unique_titles.append(t_cap)
            promotion_evidence = unique_titles
            
            leadership_evidence = []
            if feat.get("leadership_scope", 0.0) > 0:
                for title in feat.get("candidate_titles", []):
                    if any(w in title.lower() for w in ["lead", "manager", "head", "director", "vp", "chief", "principal"]):
                        leadership_evidence.append(f"Title: {title.title()}")
                        break

            breakdown = ScoreBreakdown(
                candidate_id=feat["candidate_id"],
                semantic=feat["raw_signals"]["semantic"],
                career=feat["raw_signals"]["career"],
                skill=feat["raw_signals"]["skill"],
                experience=feat["raw_signals"]["experience"],
                behavioral=feat["raw_signals"]["behavioral"],
                trust=feat["raw_signals"]["trust"],
                growth=feat["raw_signals"]["growth"],
                semantic_pct=sem_pct,
                career_pct=career_pct,
                skill_pct=skill_pct,
                experience_pct=exp_pct,
                behavioral_pct=beh_pct,
                trust_pct=trust_pct,
                growth_pct=growth_pct,
                semantic_contrib=sem_contrib,
                career_contrib=career_contrib,
                skill_contrib=skill_contrib,
                experience_contrib=exp_contrib,
                behavioral_contrib=beh_contrib,
                trust_contrib=trust_contrib,
                growth_contrib=growth_contrib,
                matched_skills=matched_skills,
                promotion_evidence=promotion_evidence,
                leadership_evidence=leadership_evidence
            )
            
            r = generate_reasoning(feat, pos, score, breakdown=breakdown)
            if r in seen:
                r = r + f" [#{pos}]"
            seen.add(r)
            writer.writerow([feat["candidate_id"], pos, round(score, 6), r])

    print(f"[rank] Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
