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

from talentmind.config import (
    DATA_DIR, WEIGHT_PROFILES, DEFAULT_WEIGHT_PROFILE,
    SEMANTIC_RECALL_K, CAREER_RECALL_K, BEHAVIORAL_RECALL_K,
    LOCAL_MODEL_PATH,
)
from talentmind.embedder import embed_single, cosine_top_k
from talentmind.jd_intelligence import parse_jd
from talentmind.explainer import generate_reasoning

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
    for i, idx in enumerate(idx_list):
        f = feats[idx]
        sem = float((cos_sims[i] + 1.0) / 2.0)
        score = (
            weights["semantic"]   * sem                    +
            weights["career"]     * f["career_score"]      +
            weights["skill"]      * f["skill_score"]       +
            weights["experience"] * f["experience_score"]  +
            weights["behavioral"] * f["behavioral_score"]  +
            weights["trust"]      * f["trust_score"]
        )
        results.append((score, f, sem))

    results.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    top100 = results[:100]

    scores = [round(r[0], 6) for r in top100]
    assert all(scores[i] >= scores[i+1] for i in range(99)), \
        "CRITICAL: Score monotonicity violated"

    print(f"[rank] Writing {args.out}...")
    seen = set()
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for pos, (score, feat, sem) in enumerate(top100, 1):
            r = generate_reasoning(feat, pos, score, sem)
            if r in seen:
                r = r + f" [#{pos}]"
            seen.add(r)
            writer.writerow([feat["candidate_id"], pos, round(score, 6), r])

    print(f"[rank] Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
