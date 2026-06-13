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
from talentmind.jd_intelligence import parse_jd, JDProfile
from talentmind.config import WEIGHT_PROFILES, DEFAULT_WEIGHT_PROFILE
from rank import compute_candidate_relevance, compute_jd_satisfaction

# Rich Custom CSS for a Premium, Dark-Themed Dashboard
custom_css = """
body {
    background: radial-gradient(circle at top right, #0f172a, #020617) !important;
    color: #f8fafc !important;
    font-family: 'Outfit', 'Inter', sans-serif !important;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 40px auto !important;
    background: rgba(15, 23, 42, 0.75) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 24px !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;
    padding: 40px !important;
}
.header-container {
    text-align: center;
    margin-bottom: 40px;
}
.header-title {
    background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4f46e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.025em;
    margin-bottom: 10px !important;
}
.header-subtitle {
    color: #94a3b8 !important;
    font-size: 1.15rem !important;
    font-weight: 400;
}
.tab-nav {
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    margin-bottom: 25px !important;
}
.tab-nav button {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    padding: 12px 24px !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.3s ease !important;
}
.tab-nav button.selected {
    color: #818cf8 !important;
    border-bottom: 3px solid #6366f1 !important;
}
.primary-btn {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4) !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.6) !important;
}
.primary-btn:active {
    transform: translateY(0) !important;
}
.copilot-card {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    margin-top: 20px !important;
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-high {
    background-color: rgba(16, 185, 129, 0.15) !important;
    color: #10b981 !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
}
.badge-medium {
    background-color: rgba(245, 158, 11, 0.15) !important;
    color: #f59e0b !important;
    border: 1px solid rgba(245, 158, 11, 0.3) !important;
}
.badge-low {
    background-color: rgba(239, 68, 68, 0.15) !important;
    color: #ef4444 !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
}
.score-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
}
.score-table th, .score-table td {
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.score-table th {
    color: #94a3b8;
    font-weight: 600;
}
.score-table tr:hover {
    background: rgba(255, 255, 255, 0.02);
}
"""


def _jd_profile_from_struct(jd_struct: dict) -> JDProfile:
    profile_keys = {
        "required_skills", "preferred_skills", "seniority_level", "experience_range",
        "business_objective", "team_context", "industry", "responsibility_list",
        "implicit_signals", "team_structure", "responsibility_hierarchy",
        "industry_context", "implicit_seniority_signals", "hard_requirements",
        "nice_to_have", "leadership_signals", "responsibility_categories",
    }
    profile_args = {key: jd_struct[key] for key in profile_keys if key in jd_struct}
    return JDProfile(**profile_args)


def rank_sample(json_text: str, jd_text: str) -> str:
    if not jd_text.strip():
        return "Error: Job Description cannot be empty."
    if not json_text.strip():
        return "Error: Candidate JSONL input cannot be empty."
    
    # Parse JD to select weight profile
    jd_struct = parse_jd(jd_text)
    jd_profile = _jd_profile_from_struct(jd_struct)
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
        candidate_relevance = compute_candidate_relevance(feat, jd_profile)
        jd_satisfaction = compute_jd_satisfaction(feat, jd_profile)
        combined_match = 0.5 * candidate_relevance + 0.5 * jd_satisfaction
        blended_skill = float(max(0.0, min(1.0, 0.7 * feat["skill_score"] + 0.3 * combined_match)))
        blended_experience = float(max(0.0, min(1.0, 0.7 * feat["experience_score"] + 0.3 * combined_match)))
        growth_val = float(max(0.0, min(1.0, feat.get("growth_score", 0.20))))
        score = (
            weights["semantic"]   * sem                    +
            weights["career"]     * feat["career_score"]   +
            weights["skill"]      * blended_skill          +
            weights["experience"] * blended_experience     +
            weights["behavioral"] * feat["behavioral_score"] +
            weights["trust"]      * feat["trust_score"]    +
            weights["growth"]     * growth_val             +
            weights["logistics"]  * feat.get("logistics_score", 0.60)
        )
        feat["raw_signals"] = {
            "semantic": sem,
            "career": feat["career_score"],
            "skill": blended_skill,
            "experience": blended_experience,
            "behavioral": feat["behavioral_score"],
            "trust": feat["trust_score"],
            "growth": growth_val,
            "logistics": feat.get("logistics_score", 0.60),
        }
        feat["weighted_signals"] = {
            key: weights[key] * feat["raw_signals"][key]
            for key in feat["raw_signals"]
        }
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

def explain_comparison(cand_id_a: str, cand_id_b: str, json_text: str, jd_text: str) -> tuple[str, str]:
    if not cand_id_a.strip() or not cand_id_b.strip():
        return "Error: Both Candidate IDs must be specified.", ""
    if not jd_text.strip():
        return "Error: Job Description cannot be empty.", ""
    if not json_text.strip():
        return "Error: Candidate JSONL input cannot be empty.", ""

    # Search candidates in the JSONL database
    cand_a_raw = None
    cand_b_raw = None
    for line in json_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
            if c.get("candidate_id") == cand_id_a.strip():
                cand_a_raw = c
            if c.get("candidate_id") == cand_id_b.strip():
                cand_b_raw = c
        except Exception:
            continue

    if not cand_a_raw:
        return f"Error: Candidate '{cand_id_a}' not found in the input JSONL database.", ""
    if not cand_b_raw:
        return f"Error: Candidate '{cand_id_b}' not found in the input JSONL database.", ""

    # QA Filters check
    invalid_a, reason_a = is_invalid(cand_a_raw)
    invalid_b, reason_b = is_invalid(cand_b_raw)

    if invalid_a or invalid_b:
        errs = []
        if invalid_a:
            errs.append(f"Candidate {cand_id_a} failed QA verification (Reason: {reason_a})")
        if invalid_b:
            errs.append(f"Candidate {cand_id_b} failed QA verification (Reason: {reason_b})")
        return "\n".join(errs), ""

    # Feature extraction
    feat_a = extract_all_features(cand_a_raw)
    feat_b = extract_all_features(cand_b_raw)

    # Embed and calculate semantic similarity
    jd_emb = embed_single(jd_text)
    text_a = build_candidate_text(cand_a_raw)
    text_b = build_candidate_text(cand_b_raw)
    embs = embed_texts([text_a, text_b], show_progress=False)
    
    sem_a = float((np.dot(embs[0], jd_emb) + 1.0) / 2.0)
    sem_b = float((np.dot(embs[1], jd_emb) + 1.0) / 2.0)

    # Calculate weight profile
    jd_struct = parse_jd(jd_text)
    jd_profile = _jd_profile_from_struct(jd_struct)
    weights = WEIGHT_PROFILES.get(
        jd_struct.get("weight_profile", DEFAULT_WEIGHT_PROFILE),
        WEIGHT_PROFILES[DEFAULT_WEIGHT_PROFILE]
    )

    # Scores calculation
    a_match = 0.5 * compute_candidate_relevance(feat_a, jd_profile) + 0.5 * compute_jd_satisfaction(feat_a, jd_profile)
    b_match = 0.5 * compute_candidate_relevance(feat_b, jd_profile) + 0.5 * compute_jd_satisfaction(feat_b, jd_profile)
    skill_a = float(max(0.0, min(1.0, 0.7 * feat_a["skill_score"] + 0.3 * a_match)))
    skill_b = float(max(0.0, min(1.0, 0.7 * feat_b["skill_score"] + 0.3 * b_match)))
    exp_a = float(max(0.0, min(1.0, 0.7 * feat_a["experience_score"] + 0.3 * a_match)))
    exp_b = float(max(0.0, min(1.0, 0.7 * feat_b["experience_score"] + 0.3 * b_match)))
    score_a = (
        weights["semantic"]   * sem_a                    +
        weights["career"]     * feat_a["career_score"]   +
        weights["skill"]      * skill_a                  +
        weights["experience"] * exp_a                    +
        weights["behavioral"] * feat_a["behavioral_score"] +
        weights["trust"]      * feat_a["trust_score"]    +
        weights["growth"]     * feat_a.get("growth_score", 0.20) +
        weights["logistics"]  * feat_a.get("logistics_score", 0.60)
    )

    score_b = (
        weights["semantic"]   * sem_b                    +
        weights["career"]     * feat_b["career_score"]   +
        weights["skill"]      * skill_b                  +
        weights["experience"] * exp_b                    +
        weights["behavioral"] * feat_b["behavioral_score"] +
        weights["trust"]      * feat_b["trust_score"]    +
        weights["growth"]     * feat_b.get("growth_score", 0.20) +
        weights["logistics"]  * feat_b.get("logistics_score", 0.60)
    )

    # Growth potentials
    growth_a = feat_a.get("growth_potential", "MEDIUM")
    growth_b = feat_b.get("growth_potential", "MEDIUM")

    # Perform comparison
    if score_a > score_b:
        winner_id, loser_id = cand_id_a, cand_id_b
        w_feat, l_feat = feat_a, feat_b
        w_sem, l_sem = sem_a, sem_b
    else:
        winner_id, loser_id = cand_id_b, cand_id_a
        w_feat, l_feat = feat_b, feat_a
        w_sem, l_sem = sem_b, sem_a

    reasons = []
    # Semantic
    diff = w_sem - l_sem
    if diff > 0.01:
        reasons.append(f"stronger semantic alignment with the JD (+{diff:.2f})")
    # Career
    diff = w_feat["career_score"] - l_feat["career_score"]
    if diff > 0.01:
        reasons.append(f"stronger career trajectory (+{diff:.2f})")
    # Skill
    diff = w_feat["skill_score"] - l_feat["skill_score"]
    if diff > 0.01:
        reasons.append(f"more relevant expert skills (+{diff:.2f})")
    # Experience
    diff = w_feat["experience_score"] - l_feat["experience_score"]
    if diff > 0.01:
        reasons.append(f"closer experience bracket fit (+{diff:.2f})")
    # Behavioral
    diff = w_feat["behavioral_score"] - l_feat["behavioral_score"]
    if diff > 0.01:
        reasons.append(f"stronger behavioral/activity signals (+{diff:.2f})")
    # Trust
    diff = w_feat["trust_score"] - l_feat["trust_score"]
    if diff > 0.01:
        reasons.append(f"higher trust/profile verification (+{diff:.2f})")

    reasons_str = ", ".join(reasons) if reasons else "marginal score differences across all features"

    # Template-based explanation string
    explanation = f"Candidate {winner_id} ranks higher because of: {reasons_str}. Growth Potential — {cand_id_a}: {growth_a.capitalize()} | {cand_id_b}: {growth_b.capitalize()}."

    # Build premium HTML Scorecard comparison table
    badge_class_a = f"badge badge-{growth_a.lower()}"
    badge_class_b = f"badge badge-{growth_b.lower()}"

    html_scorecard = f"""
    <div class="copilot-card">
        <h3 style="margin-top:0; color:#818cf8; font-size:1.3rem;">⚡ Comparative Scorecard</h3>
        <table class="score-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Candidate A: {cand_id_a}</th>
                    <th>Candidate B: {cand_id_b}</th>
                    <th>Weight</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Overall Weighted Score</strong></td>
                    <td><strong style="color:#a5b4fc; font-size:1.15rem;">{score_a:.4f}</strong></td>
                    <td><strong style="color:#a5b4fc; font-size:1.15rem;">{score_b:.4f}</strong></td>
                    <td>100%</td>
                </tr>
                <tr>
                    <td><strong>Growth Potential</strong></td>
                    <td><span class="{badge_class_a}">{growth_a}</span></td>
                    <td><span class="{badge_class_b}">{growth_b}</span></td>
                    <td>{weights.get("growth", 0)*100:.0f}%</td>
                </tr>
                <tr>
                    <td>Semantic Similarity</td>
                    <td>{sem_a:.4f}</td>
                    <td>{sem_b:.4f}</td>
                    <td>{weights.get("semantic", 0)*100:.0f}%</td>
                </tr>
                <tr>
                    <td>Career Score</td>
                    <td>{feat_a["career_score"]:.4f}</td>
                    <td>{feat_b["career_score"]:.4f}</td>
                    <td>{weights.get("career", 0)*100:.0f}%</td>
                </tr>
                <tr>
                    <td>Skill Score</td>
                    <td>{feat_a["skill_score"]:.4f}</td>
                    <td>{feat_b["skill_score"]:.4f}</td>
                    <td>{weights.get("skill", 0)*100:.0f}%</td>
                </tr>
                <tr>
                    <td>Experience Score</td>
                    <td>{feat_a["experience_score"]:.4f}</td>
                    <td>{feat_b["experience_score"]:.4f}</td>
                    <td>{weights.get("experience", 0)*100:.0f}%</td>
                </tr>
                <tr>
                    <td>Behavioral Score</td>
                    <td>{feat_a["behavioral_score"]:.4f}</td>
                    <td>{feat_b["behavioral_score"]:.4f}</td>
                    <td>{weights.get("behavioral", 0)*100:.0f}%</td>
                </tr>
                <tr>
                    <td>Trust Score</td>
                    <td>{feat_a["trust_score"]:.4f}</td>
                    <td>{feat_b["trust_score"]:.4f}</td>
                    <td>{weights.get("trust", 0)*100:.0f}%</td>
                </tr>
                <tr>
                    <td>Logistics Fit</td>
                    <td>{feat_a.get("logistics_score", 0.60):.4f}</td>
                    <td>{feat_b.get("logistics_score", 0.60):.4f}</td>
                    <td>{weights.get("logistics", 0)*100:.0f}%</td>
                </tr>
            </tbody>
        </table>
    </div>
    """

    return explanation, html_scorecard

# Load default JD
jd_file = Path("job_description.txt")
default_jd = jd_file.read_text(encoding="utf-8") if jd_file.exists() else ""

# Load sample candidates snippet if available to pre-fill
sample_candidates_file = Path("candidates.jsonl")
default_jsonl = ""
if sample_candidates_file.exists():
    # Load first 5 lines as sample candidates to make copy-pasting easier for the user
    try:
        with open(sample_candidates_file, "r", encoding="utf-8") as f:
            lines = [f.readline().strip() for _ in range(5)]
            default_jsonl = "\n".join(lines)
    except Exception:
        pass

# Build the blocks-based UI
with gr.Blocks(css=custom_css, title="TalentMind AI v2 Dashboard") as demo:
    # Header Banner
    with gr.Group(elem_classes="header-container"):
        gr.HTML('<h1 class="header-title">🧠 TalentMind AI v2</h1>')
        gr.HTML('<p class="header-subtitle">Explainable Multi-Signal Candidate Matcher & Recruiter Sandbox</p>')
        
    with gr.Tabs(elem_classes="tabs"):
        # Tab 1: Rank Candidates
        with gr.Tab("Rank Candidates"):
            gr.Markdown("Rank a list of candidates against the job description and output a formatted, ranked CSV.")
            with gr.Row():
                with gr.Column(scale=1):
                    jd_input = gr.Textbox(
                        label="Job Description",
                        value=default_jd,
                        lines=12,
                        placeholder="Paste the Job Description here..."
                    )
                with gr.Column(scale=1):
                    jsonl_input = gr.Textbox(
                        label="Candidate Profiles (JSONL)",
                        value=default_jsonl,
                        lines=12,
                        placeholder="Paste candidates here (one candidate JSON per line)..."
                    )
            
            rank_btn = gr.Button("Rank Candidates", elem_classes="primary-btn")
            csv_output = gr.Textbox(label="Ranked Output (CSV format)", lines=15)
            
            rank_btn.click(
                fn=rank_sample,
                inputs=[jsonl_input, jd_input],
                outputs=[csv_output]
            )

        # Tab 2: Recruiter Copilot
        with gr.Tab("Recruiter Copilot"):
            gr.Markdown("Compare two candidates side-by-side to understand the score differences and see Growth Potential details.")
            with gr.Row():
                with gr.Column(scale=1):
                    cand_a_input = gr.Textbox(
                        label="Candidate A ID", 
                        placeholder="e.g. CAND_001",
                        value="CAND_001"
                    )
                with gr.Column(scale=1):
                    cand_b_input = gr.Textbox(
                        label="Candidate B ID", 
                        placeholder="e.g. CAND_002",
                        value="CAND_002"
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    jd_copilot = gr.Textbox(
                        label="Job Description",
                        value=default_jd,
                        lines=10,
                        placeholder="Paste the Job Description here..."
                    )
                with gr.Column(scale=1):
                    jsonl_copilot = gr.Textbox(
                        label="Candidate Profiles (JSONL)",
                        value=default_jsonl,
                        lines=10,
                        placeholder="Paste candidate database here (one JSON per line)..."
                    )
            
            explain_btn = gr.Button("Explain ↗", elem_classes="primary-btn")
            
            with gr.Column():
                explanation_output = gr.Textbox(label="Template-Based Comparative Explanation", lines=4)
                scorecard_output = gr.HTML(label="Detailed Comparison Card")
                
            explain_btn.click(
                fn=explain_comparison,
                inputs=[cand_a_input, cand_b_input, jsonl_copilot, jd_copilot],
                outputs=[explanation_output, scorecard_output]
            )

if __name__ == "__main__":
    demo.launch()
