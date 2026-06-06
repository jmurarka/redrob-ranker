from datetime import date
from pathlib import Path

SUBMISSION_DATE = date(2026, 6, 5)
DATA_DIR = Path("data")

WEIGHT_PROFILES = {
    "ML_ENGINEER":      {"semantic": 0.20, "career": 0.25, "skill": 0.20,
                         "experience": 0.10, "behavioral": 0.10, "trust": 0.10},
    "DATA_ENGINEER":    {"semantic": 0.15, "career": 0.25, "skill": 0.25,
                         "experience": 0.10, "behavioral": 0.10, "trust": 0.15},
    "BACKEND_ENGINEER": {"semantic": 0.15, "career": 0.30, "skill": 0.20,
                         "experience": 0.15, "behavioral": 0.10, "trust": 0.10},
}
DEFAULT_WEIGHT_PROFILE = "ML_ENGINEER"

TIER_A_SKILLS = frozenset({
    "embeddings", "embedding", "sentence-transformers", "sentence transformers",
    "vector database", "vector db", "vector search", "hybrid search",
    "faiss", "qdrant", "pinecone", "weaviate", "milvus", "opensearch",
    "elasticsearch", "retrieval", "retrieval-augmented", "rag",
    "ranking system", "information retrieval", "ndcg", "mrr",
    "learning to rank", "ltr", "dense retrieval", "bm25", "reranking",
    "reranker", "cross-encoder", "bi-encoder", "bge", "e5",
    "vector index", "embedding drift", "index refresh",
})
TIER_B_SKILLS = frozenset({
    "python", "pytorch", "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "mlops", "model serving", "inference optimization", "recommendation system",
    "recsys", "a/b testing", "ab testing", "xgboost", "lightgbm",
    "mlflow", "weights & biases", "wandb", "hugging face", "huggingface",
    "transformers", "nlp", "bert", "production deployment", "deployed",
})
TIER_C_SKILLS = frozenset({
    "kubeflow", "airflow", "distributed systems", "fastapi", "flask",
    "docker", "kubernetes", "aws", "gcp", "azure", "data pipelines",
    "scikit-learn", "numpy", "pandas", "redis", "kafka",
})

CONSULTING_COMPANIES = frozenset({
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies", "tech mahindra",
    "mphasis", "hexaware", "niit technologies", "persistent systems",
    "l&t infotech", "zensar", "mastech", "kpit",
})

INDIA_PREFERRED = frozenset({
    "pune", "noida", "gurgaon", "gurugram", "faridabad", "delhi",
    "new delhi", "hyderabad", "mumbai", "bengaluru", "bangalore", "ncr",
})

# QA filter thresholds
MAX_YOE_DELTA = 3.0
MAX_EXPERT_ZERO_EVIDENCE_SKILLS = 8
MAX_SINGLE_ROLE_MONTHS = 240
MAX_SALARY_LPA = 200

# Recall sizes
SEMANTIC_RECALL_K = 2000
CAREER_RECALL_K = 500
BEHAVIORAL_RECALL_K = 250

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
LOCAL_MODEL_PATH = str(DATA_DIR / "miniLM_model")
