# app.py — HuggingFace Spaces sandbox using TalentMind AI v2
import gradio as gr
import json
import csv
import io
from pathlib import Path
import numpy as np

from talentmind.qa_filter import is_invalid
from talentmind.feature_extractor import extract_all_features
from talentmind.text_builder import build_candidate_text
from talentmind.embedder import embed_texts, embed_single
from talentmind.explainer import generate_reasoning
from talentmind.jd_intelligence import parse_jd
from talentmind.config import WEIGHT_PROFILES, DEFAULT_WEIGHT_PROFILE

def rank_sample(json_text: str, jd_text: str) -> str:
    if not jd_text.strip():
        return "Error: Job Description cannot be empty."
    
    # Parse JD to select profile
    jd_struct = parse_jd(jd_text)
    weights = WEIGHT_PROFILES.get(
        jd_struct.get("weight_profile", DEFAULT_WEIGHT_PROFILE),
        WEIGHT_PROFILES[DEFAULT_WEIGHT_PROFILE]
    )
    
    # Embed JD
    jd_emb = embed_single(jd_text)
    
    candidates = []
    for line in json_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            candidates.append(json.loads(line))
        except Exception as e:
            return f"Error parsing JSON line: {line}\nException: {e}"
            
    scored = []
    texts = []
    feats = []
    
    for c in candidates:
        invalid, reason = is_invalid(c)
        if invalid:
            # Skip honeypots/invalid candidates
            continue
        feat = extract_all_features(c)
        text = build_candidate_text(c)
        feats.append(feat)
        texts.append(text)
        
    if not feats:
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        return output.getvalue()
        
    # Generate candidate embeddings
    candidate_embs = embed_texts(texts, show_progress=False)
    cos_sims = candidate_embs @ jd_emb
    
    for i, feat in enumerate(feats):
        sem = float((cos_sims[i] + 1.0) / 2.0)
        score = (
            weights["semantic"]   * sem                    +
            weights["career"]     * feat["career_score"]   +
            weights["skill"]      * feat["skill_score"]    +
            weights["experience"] * feat["experience_score"] +
            weights["behavioral"] * feat["behavioral_score"] +
            weights["trust"]      * feat["trust_score"]
        )
        scored.append((score, feat, sem))
        
    # Sort by score descending, then candidate_id ascending
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    top100 = scored[:100]
    
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    
    seen = set()
    for pos, (score, feat, sem) in enumerate(top100, 1):
        r = generate_reasoning(feat, pos, score, sem)
        if r in seen:
            r = r + f" [#{pos}]"
        seen.add(r)
        writer.writerow([feat["candidate_id"], pos, round(score, 6), r])
        
    return output.getvalue()

# Load default JD
jd_file = Path("job_description.txt")
default_jd = jd_file.read_text(encoding="utf-8") if jd_file.exists() else ""

demo = gr.Interface(
    fn=rank_sample,
    inputs=[
        gr.Textbox(label="Paste candidate JSONL (one JSON per line, ≤100 candidates)", lines=10),
        gr.Textbox(label="Job Description", value=default_jd, lines=10)
    ],
    outputs=gr.Textbox(label="Ranked CSV output"),
    title="TalentMind AI v2 — Demo",
    description="A CPU-only Explainable Multi-Signal Candidate Matcher utilizing SentenceTransformers."
)

if __name__ == "__main__":
    demo.launch()
