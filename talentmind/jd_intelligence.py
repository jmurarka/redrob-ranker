import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

# Ontologies for Role Category Identification
ROLE_ONTOLOGY = {
    "ML_ENGINEER": [
        "machine learning", "ml engineer", "ai engineer", "llm", "nlp", 
        "ranking", "retrieval", "recommendation", "search engineer", "deep learning"
    ],
    "DATA_ENGINEER": [
        "data engineer", "etl", "pipeline", "spark", "airflow", "dbt", "warehouse", "hadoop"
    ],
    "BACKEND_ENGINEER": [
        "backend", "api", "microservice", "rest", "grpc", "distributed systems", "database"
    ]
}

# Skill Ontology
SKILL_ONTOLOGY = {
    "embeddings": ["embedding", "sentence-transformer", "dense vector", "cross-encoder"],
    "vector database": ["vector db", "faiss", "qdrant", "pinecone", "weaviate", "milvus", "opensearch", "elasticsearch"],
    "hybrid search": ["bm25", "hybrid retrieval", "sparse dense", "hybrid search"],
    "rag": ["retrieval-augmented", "rag pipeline", "llamaindex"],
    "ranking eval": ["ndcg", "mrr", "learning to rank", "ltr"],
    "python": ["python"],
    "pytorch": ["pytorch", "torch"],
    "fine-tuning": ["fine-tun", "lora", "qlora", "peft"],
    "llm": ["llm", "large language model", "gpt", "claude", "llama"],
    "a/b testing": ["a/b test", "ab test", "online experiment"],
    "mlops": ["mlops", "model serving", "triton", "tfserving", "bentoml"]
}

# Responsibility Ontology
RESPONSIBILITY_ONTOLOGY = {
    "design": ["design", "architect", "prototype", "define"],
    "build": ["build", "develop", "implement", "create", "write"],
    "deploy": ["deploy", "production", "scale", "optimize", "maintain", "serve"],
    "collaborate": ["collaborate", "partner", "work with", "liaise"]
}

# Leadership Ontology
LEADERSHIP_ONTOLOGY = {
    "mentorship": ["mentor", "coach", "guide", "train"],
    "ownership": ["roadmap", "own", "strategy", "vision", "decision"],
    "management": ["manage", "lead", "team size", "direct reports", "hire"]
}

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
DOMAINS = ["fintech", "saas", "healthcare", "hr-tech", "adtech", "ai", "e-commerce"]
COMPLIANCE = ["gdpr", "hipaa", "soc2", "pci", "iso27001"]
NEGATIVE_SECTION_MARKERS = (
    "things we explicitly do not want",
    "things we explicitly don't want",
    "do not want",
    "disqualifiers",
    "not a fit",
)
PREFERRED_SECTION_MARKERS = (
    "things we'd like you to have",
    "things we would like you to have",
    "nice to have",
    "preferred",
    "bonus",
    "plus",
)
REQUIRED_SECTION_MARKERS = (
    "things you absolutely need",
    "requirements",
    "must have",
    "what you need",
    "core expertise",
)

@dataclass
class JDProfile:
    """Dataclass mirroring candidate profile structure for symmetric matching."""
    required_skills: List[str]
    preferred_skills: List[str]
    seniority_level: str
    experience_range: Dict[str, int]
    business_objective: str
    team_context: str
    industry: str
    responsibility_list: List[str]
    implicit_signals: List[str]
    
    # Task 1 Fields
    team_structure: str = ""
    responsibility_hierarchy: List[str] = field(default_factory=list)
    industry_context: str = ""
    implicit_seniority_signals: List[str] = field(default_factory=list)
    hard_requirements: List[str] = field(default_factory=list)
    nice_to_have: List[str] = field(default_factory=list)
    leadership_signals: List[str] = field(default_factory=list)
    responsibility_categories: List[str] = field(default_factory=list)


def _to_sentences(text: str) -> List[str]:
    """Tokenize raw text into sentences using basic delimiters."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def extract_business_objective(jd_text: str) -> str:
    """Extracts business goals/objectives from JD text using markers."""
    sentences = _to_sentences(jd_text)
    patterns = [
        r"\b(mission is to|we are on a mission to|help us scale|join us to)\b",
        r"\b(build|scale|optimize|develop|design|architect)\s+\w+\s+(to|for)\b",
    ]
    for p in patterns:
        for s in sentences:
            if re.search(p, s.lower()):
                return s
    # Fallback to sentences mentioning seeking / looking for
    for s in sentences:
        if any(kw in s.lower() for kw in ["seeking", "looking for"]):
            return s
    return "Build, scale, and optimize machine learning and software engineering systems."


def extract_leadership_signals(jd_text: str) -> List[str]:
    """Scans text for keywords in the leadership ontology."""
    tl = jd_text.lower()
    signals = []
    for category, kws in LEADERSHIP_ONTOLOGY.items():
        if any(kw in tl for kw in kws):
            signals.append(category)
    return signals


def extract_team_context(jd_text: str) -> str:
    """Extracts reporting structure and collaborative context from the text."""
    sentences = _to_sentences(jd_text)
    team_info = []
    for s in sentences:
        sl = s.lower()
        if any(kw in sl for kw in ["reports to", "reporting", "report to"]):
            team_info.append(s)
        elif any(kw in sl for kw in ["collaborate with", "partner with", "work closely with", "cross-functional"]):
            team_info.append(s)
    if team_info:
        return " ".join(team_info)
    
    # Generic reporting structure heuristics as fallback
    reporting_line = "Engineering Lead"
    if "reports to" in jd_text.lower():
        rep_match = re.search(r"reports\s+to\s+([^,\n.]+)", jd_text.lower())
        if rep_match:
            reporting_line = rep_match.group(1).strip().title()
    partners = []
    for term in ["product manager", "designer", "data platform", "frontend", "devops", "qa"]:
        if term in jd_text.lower():
            partners.append(term.title())
    return f"Reports to {reporting_line}. Collaborates with {', '.join(partners) if partners else 'cross-functional team'}."


def extract_requirements(jd_text: str) -> tuple[List[str], List[str]]:
    """Partitions skills in skill ontology into hard and preferred requirements."""
    lines = [line.strip().lower() for line in jd_text.split('\n') if line.strip()]
    hard = []
    pref = []
    
    in_preferred_section = False
    in_negative_section = False
    
    for line in lines:
        if "final note for the participants" in line:
            in_negative_section = True
            in_preferred_section = False
            continue
        if any(marker in line for marker in NEGATIVE_SECTION_MARKERS):
            in_negative_section = True
            in_preferred_section = False
            continue
        if in_negative_section and any(marker in line for marker in ["skills inventory", "things you absolutely need", "on location", "the vibe check", "how to read between the lines"]):
            in_negative_section = False
        if in_negative_section:
            continue

        if any(h in line for h in PREFERRED_SECTION_MARKERS):
            in_preferred_section = True
        elif any(h in line for h in REQUIRED_SECTION_MARKERS):
            in_preferred_section = False
            
        for skill, aliases in SKILL_ONTOLOGY.items():
            matched_aliases = []
            if skill in line:
                matched_aliases.append(skill)
            for a in aliases:
                if a in line:
                    matched_aliases.append(a)
            if not matched_aliases:
                continue
                
            is_pref = in_preferred_section or any(w in line for w in ["preferred", "nice to have", "plus", "bonus", "optional", "desired", "helpful", "won't reject"])
            if is_pref:
                pref.append(skill)
                pref.extend(matched_aliases)
            else:
                hard.append(skill)
                hard.extend(matched_aliases)
    pref_set = set(pref)
    hard_set = set(hard) - pref_set
    return list(hard_set), list(pref_set)


def extract_responsibility_categories(jd_text: str) -> List[str]:
    """Scans text for keywords in the responsibility ontology."""
    tl = jd_text.lower()
    cats = []
    for cat, kws in RESPONSIBILITY_ONTOLOGY.items():
        if any(kw in tl for kw in kws):
            cats.append(cat)
    return cats


def extract_responsibility_list(jd_text: str) -> List[str]:
    """Extracts bulleted or direct responsibility list items."""
    lines = [line.strip().lstrip("-*• ") for line in jd_text.split('\n') if len(line.strip()) > 10]
    res_list = []
    for line in lines:
        if any(line.lower().startswith(kw) for kw in ["design", "build", "develop", "implement", "deploy", "scale", "optimize", "maintain", "collaborate", "lead", "work"]):
            res_list.append(line)
    return res_list[:5] if res_list else ["Design and build scalable engineering systems."]


def extract_industry_context(jd_text: str) -> str:
    """Finds matching industry domains in the text."""
    tl = jd_text.lower()
    matched = [d.title() for d in DOMAINS if d in tl]
    if matched:
        return ", ".join(matched)
    return "AI & Software Engineering"


def extract_implicit_signals(jd_text: str) -> List[str]:
    """Infers implicit signals like mentorship and autonomy."""
    tl = jd_text.lower()
    signals = []
    if any(kw in tl for kw in ["mentor", "coach", "guide", "train"]):
        signals.append("Mentorship required")
    if any(kw in tl for kw in ["roadmap", "own", "strategy", "vision"]):
        signals.append("Roadmap ownership")
    if any(kw in tl for kw in ["autonomous", "self-starter", "independent"]):
        signals.append("High autonomy")
    if not signals:
        signals.append("Standard professional execution")
    return signals


def extract_jd_deep_profile(jd_text: str) -> JDProfile:
    """
    Deterministic rule-based recruiter ontology extraction returning a typed JDProfile.
    """
    tl = jd_text.lower()

    # Determine seniority
    title_match = re.search(r"job description:\s*([^\n]+)", jd_text, re.IGNORECASE)
    title_text = title_match.group(1).lower() if title_match else "\n".join(jd_text.splitlines()[:8]).lower()
    seniority = "Mid"
    if re.search(r"\b(senior|sr\.?|lead)\b", title_text):
        seniority = "Senior"
    elif re.search(r"\b(principal|staff)\b", title_text):
        seniority = "Principal"
    else:
        positive_text = []
        in_negative = False
        for line in jd_text.splitlines():
            ll = line.lower()
            if any(marker in ll for marker in NEGATIVE_SECTION_MARKERS):
                in_negative = True
            elif in_negative and any(marker in ll for marker in ["on location", "the vibe check", "how to read between the lines", "final note"]):
                in_negative = False
            if not in_negative:
                positive_text.append(line)
        positive_blob = "\n".join(positive_text).lower()
        for level, patterns in SENIORITY_PATTERNS.items():
            if any(re.search(p, positive_blob) for p in patterns):
                seniority = level
                break

    # Determine experience range
    m = EXP_RE.search(jd_text)
    exp_range = {"min": int(m.group(1)), "max": int(m.group(2))} if m else {"min": 3, "max": 10}

    # Extract all components deterministically
    hard_req, pref_req = extract_requirements(jd_text)
    business_obj = extract_business_objective(jd_text)
    team_ctx = extract_team_context(jd_text)
    industry = extract_industry_context(jd_text)
    resp_list = extract_responsibility_list(jd_text)
    impl_signals = extract_implicit_signals(jd_text)
    lead_signals = extract_leadership_signals(jd_text)
    resp_cats = extract_responsibility_categories(jd_text)

    return JDProfile(
        required_skills=hard_req,
        preferred_skills=pref_req,
        seniority_level=seniority,
        experience_range=exp_range,
        business_objective=business_obj,
        team_context=team_ctx,
        industry=industry,
        responsibility_list=resp_list,
        implicit_signals=impl_signals,
        team_structure=team_ctx,
        responsibility_hierarchy=resp_list,
        industry_context=industry,
        implicit_seniority_signals=impl_signals,
        hard_requirements=hard_req,
        nice_to_have=pref_req,
        leadership_signals=lead_signals,
        responsibility_categories=resp_cats
    )


def parse_jd(jd_text: str, output_path: str = None) -> dict:
    """
    Parses JD and returns a dictionary compatible with the legacy precompute/rank modules.
    """
    profile = extract_jd_deep_profile(jd_text)
    
    tl = jd_text.lower()
    role_category = "ML_ENGINEER"
    for cat, aliases in ROLE_ONTOLOGY.items():
        if any(a in tl for a in aliases):
            role_category = cat
            break

    disqualifying = []
    if "consulting" in tl or "services company" in tl:
        disqualifying.append("consulting_only")

    # Combine dataclass fields and extra keys into output dict
    result = asdict(profile)
    result.update({
        "role_category": role_category,
        "seniority": profile.seniority_level,
        "disqualifying_signals": disqualifying,
        "industry_domain": profile.industry,
        "weight_profile": role_category,
    })

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result
