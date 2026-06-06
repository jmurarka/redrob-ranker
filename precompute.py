#!/usr/bin/env python3
"""
precompute.py — Offline feature and embedding generation.
Usage: python precompute.py --candidates candidates.jsonl [--jd job_description.txt]
Output: data/ directory with all precomputed artifacts.
"""
import argparse, gzip, json, pickle, time
from pathlib import Path
import numpy as np

from talentmind.config import DATA_DIR, LOCAL_MODEL_PATH
from talentmind.qa_filter import is_invalid
from talentmind.feature_extractor import extract_all_features
from talentmind.text_builder import build_candidate_text
from talentmind.embedder import embed_texts, _get_model
from talentmind.jd_intelligence import parse_jd

def load_jsonl(path):
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--jd", default="job_description.txt")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print("[precompute] Parsing JD...")
    jd_text = Path(args.jd).read_text(encoding="utf-8")
    parse_jd(jd_text, output_path=str(DATA_DIR / "jd_structured.json"))

    print("[precompute] QA filter + feature extraction...")
    ids, feats, texts = [], [], []
    total = rejected = 0

    for c in load_jsonl(args.candidates):
        total += 1
        if total % 10000 == 0:
            print(f"  {total} processed ({time.time()-t0:.0f}s)")
        invalid, _ = is_invalid(c)
        if invalid:
            rejected += 1
            continue
        ids.append(c["candidate_id"])
        feats.append(extract_all_features(c))
        texts.append(build_candidate_text(c))

    print(f"  Total={total} | Rejected={rejected} | Clean={len(ids)}")

    print("[precompute] Generating embeddings...")
    embs = embed_texts(texts, batch_size=512, show_progress=True)

    print("[precompute] Saving model locally...")
    if not (Path(LOCAL_MODEL_PATH) / "model.safetensors").exists():
        _get_model().save(LOCAL_MODEL_PATH)

    print("[precompute] Persisting artifacts...")
    np.save(str(DATA_DIR / "candidate_embeddings.npy"), embs)
    with open(DATA_DIR / "candidate_features.pkl", "wb") as f:
        pickle.dump(feats, f, protocol=4)
    with open(DATA_DIR / "clean_candidate_ids.json", "w") as f:
        json.dump(ids, f)

    print(f"[precompute] Done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
