import re
import json
from pathlib import Path

ROLE_PATTERNS = {
    "ML_ENGINEER": [
        r"\b(machine learning|ml engineer|ai engineer|llm|nlp|ranking|retrieval|recommendation|search engineer)\b"
    ],
    "DATA_ENGINEER": [r"\b(data engineer|etl|pipeline|spark|airflow|dbt|warehouse)\b"],
    "BACKEND_ENGINEER": [r"\b(backend|api|microservice|rest|grpc|distributed systems)\b"],
}
SENIORITY_PATTERNS = {
    "Principal": [r"\bprincipal\b", r"\bstaff\b"],
    "Senior":    [r"\bsenior\b", r"\bsr\b", r"\blead\b"],
    "Mid":       [r"\bmid.level\b"],
    "Junior":    [r"\bjunior\b", r"\bentry.level\b"],
}
EXP_RE = re.compile(r"(\d+)[–\-—to]+\s*(\d+)\s*years?", re.IGNORECASE)
SKILL_ONTOLOGY = {
    "embeddings":      ["embedding", "sentence-transformer", "dense vector"],
    "vector database": ["vector db", "faiss", "qdrant", "pinecone", "weaviate", "milvus",
                        "opensearch", "elasticsearch"],
    "hybrid search":   ["bm25", "hybrid retrieval", "sparse dense"],
    "rag":             ["retrieval-augmented", "rag pipeline"],
    "ranking eval":    ["ndcg", "mrr", "learning to rank", "ltr"],
    "python":          ["python"],
    "pytorch":         ["pytorch", "torch"],
    "fine-tuning":     ["fine-tun", "lora", "qlora", "peft"],
    "llm":             ["llm", "large language model"],
    "a/b testing":     ["a/b test", "ab test", "online experiment"],
}

def parse_jd(jd_text: str, output_path: str = None) -> dict:
    tl = jd_text.lower()

    role_category = "ML_ENGINEER"
    for cat, patterns in ROLE_PATTERNS.items():
        if any(re.search(p, tl) for p in patterns):
            role_category = cat
            break

    seniority = "Mid"
    for level, patterns in SENIORITY_PATTERNS.items():
        if any(re.search(p, tl) for p in patterns):
            seniority = level
            break

    m = EXP_RE.search(jd_text)
    exp_range = {"min": int(m.group(1)), "max": int(m.group(2))} if m else {"min": 3, "max": 10}

    required, preferred = [], []
    for skill, aliases in SKILL_ONTOLOGY.items():
        hits = [(tl.find(skill))] + [tl.find(a) for a in aliases]
        pos = next((p for p in hits if p != -1), -1)
        if pos == -1:
            continue
        ctx = tl[max(0, pos - 200): pos + 200]
        if any(w in ctx for w in ["preferred", "nice to have", "bonus", "plus"]):
            preferred.append(skill)
        else:
            required.append(skill)

    disqualifying = []
    if "consulting" in tl or "services company" in tl:
        disqualifying.append("consulting_only")

    result = {
        "role_category": role_category, "seniority": seniority,
        "experience_range": exp_range, "required_skills": required,
        "preferred_skills": preferred, "disqualifying_signals": disqualifying,
        "industry_domain": "AI / HR-Tech", "weight_profile": role_category,
    }
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result
