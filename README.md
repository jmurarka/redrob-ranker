# TalentMind AI v2: Candidate Ranking & Explainability Engine

TalentMind AI v2 is a production-grade, CPU-only, offline multi-signal candidate ranking system designed to rank 100,000 candidates against a Job Description under a 5-minute runtime constraint. The system is engineered to run entirely offline with zero network calls at runtime, satisfying strict hackathon sandboxing requirements.

---

## 1. Upgrades & Walkthrough

TalentMind AI v2 has been upgraded to close five core implementation gaps:

1.  **Ontology-Based JD Intelligence**: Uses deterministic Recruiter Ontologies (`ROLE_ONTOLOGY`, `SKILL_ONTOLOGY`, `RESPONSIBILITY_ONTOLOGY`, `LEADERSHIP_ONTOLOGY`) to parse business goals, reporting context, responsibilities, and partition skills into hard requirements vs preferred nice-to-haves without high-latency LLM calls.
2.  **Career Trajectory Intelligence**: Evaluates vertical movement rate (`promotion_velocity`), technical career focus (`specialization_consistency`), scope/autonomy (`leadership_growth`), longevity (`tenure_stability`), and industry domain exposure (`domain_focus_score`) to produce a weighted `career_intelligence_score`.
3.  **Growth Score Integration**: Integrates normalized `growth_score` (representing promotion/skill velocity over timeline) with a 5% ranking weight. Weight vectors are validated at runtime to sum to exactly 1.0.
4.  **Symmetric Matching**: Implements two-way candidate relevance (skills/YOE fit to JD) and job satisfaction (career specialization/seniority alignment with JD) scores. The resulting `combined_match` is blended directly into existing skills and experience signals (30% blending weight).
5.  **Evidence-Based Explainability**: Tracks candidate signals through the typed `ScoreBreakdown` class, computes relative candidate-pool percentiles (using `bisect` insertion indexing), and outputs 100% auditable summaries citing exact promotion paths and matched skills with standard casing.

---

## 2. System Architecture & Data Flow

```
                                  PHASE 1: OFFLINE PRECOMPUTE
                              (Allowed: Network, Time, GPU/CPU)
                                               │
  [100K Candidates JSONL]                      ▼
             │               ┌───────────────────────────────────┐
             ├──────────────►│ Layer 1: Candidate Quality Filter │► [Reject Honeypots (13,606)]
             │               └─────────────────┬─────────────────┘
             ▼                                 │
  [86,394 Clean Pool]                          ▼
             │               ┌───────────────────────────────────┐
             ├──────────────►│ Career Trajectory, Skills, YOE,   │
             │               │ Behavioral & Trust Feature Extr.  │
             │               └─────────────────┬─────────────────┘
             ▼                                 │
  [MiniLM-L6-v2 Embedder]                      ▼
             │               ┌───────────────────────────────────┐
             └──────────────►│ Serialize Embeddings & Features   │► [data/candidate_features.pkl]
                             └───────────────────────────────────┘  [data/candidate_embeddings.npy]
                                                                    [data/clean_candidate_ids.json]
                                               │
                                               ▼
                                  PHASE 2: ONLINE RANKING RUNTIME
                               (Constraints: CPU-only, Offline, ≤5m)
                                               │
  [Job Description Text]                       ▼
             │               ┌───────────────────────────────────┐
             ├──────────────►│ Layer 7: Recruiter Ontology JD    │► [JDProfile Dataclass Object]
             │               │ Extraction Engine                 │
             │               └─────────────────┬─────────────────┘
             ▼                                 │
  [Local MiniLM Embedder]                      ▼
             │               ┌───────────────────────────────────┐
             ├──────────────►│ Layer 8: Multi-Path Recall        │► Path 1: Top 2000 Semantic
             │               │ (Union & Deduplicate candidates)  │  Path 2: Top 500 Career
             │               └─────────────────┬─────────────────┘  Path 3: Top 250 Behavioral
             ▼                                 │
  [2,456 Recalled Subset]                      ▼
             │               ┌───────────────────────────────────┐
             ├──────────────►│ Layer 9: Symmetric Reranking &    │► Blends Skills/YOE Matching
             │               │ Score Percentiles computation     │  Asserts Weight sum == 1.0
             │               └─────────────────┬─────────────────┘  Dual-key Monotonic Sorting
             ▼                                 │
  [Top 100 Ranked Candidates]                  ▼
                             ┌───────────────────────────────────┐
                             │ Layer 10: Evidence-based Explainer│► [submission.csv]
                             └───────────────────────────────────┘
```

---

## 3. How It Works (Component Design)

### 3.1 Recruiter Ontology Engine (`talentmind/jd_intelligence.py`)
Parses the JD deterministically at runtime.
*   **Business Goal**: Scans sentences for mission markers (e.g. `mission is to`, `help us scale`).
*   **Requirements Partitioning**: Scans skills against `SKILL_ONTOLOGY`. If the line or header contains preference words (`preferred`, `plus`, `nice to have`), it maps it as preferred; otherwise, it is hard-required.
*   **Industry & Team Context**: Scans reporting boundaries (`reports to VP of Eng`) and domains (`fintech`, `saas`).

### 3.2 Career Trajectory Analyzer (`talentmind/career_trajectory.py`)
Computes advanced progression signals:
*   **Promotion Velocity**: Promotions count divided by career years:
    $$PV = \frac{\text{Promotions}}{\max(0.5, \text{Total Career Years})}$$
*   **Tenure Stability**: Evaluates company longevity, penalizing averages $< 12$ months and rewarding product company tenures $> 24$ months.
*   **Specialization Consistency**: Calculates the percentage of historical titles aligning with core engineering keywords.

### 3.3 Symmetric Matcher (`rank.py`)
*   **JD $\to$ Candidate**: Calculates exact skill requirement intersection (hard requirements weighted at 70%, preferred at 30%) and target YOE alignment.
*   **Candidate $\to$ JD**: Calculates matching seniority alignment (candidate current level vs target level) and career consistency.
*   **Score Blending**: Blends the average of the two matches (50/50 combined) into skill and experience scores at a 30% blending factor.

---

## 4. Setup & Running Instructions

### 4.1 Setup
1.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 4.2 Run Precompute (Phase 1)
Parses the candidate database and pre-caches features and MiniLM embeddings:
```bash
python precompute.py --candidates candidates.jsonl
```

### 4.3 Run Reranking (Phase 2)
Parses the JD text, retrieves candidates, scores, and writes ranked list to CSV:
```bash
python rank.py --candidates candidates.jsonl --out submission.csv
```

---

## 5. Running Tests

To run the complete test suite (18 test cases checking legacy logic, Recruiter Ontologies, career metrics, weight validations, and explanation formatting):
```bash
python -m pytest tests/
```

---

## 6. Known Limitations & Failure Modes

Every deterministic system has boundaries. TalentMind AI v2 behaves as follows under edge cases:

1.  **Ontology Keyword Limitations (Symmetric Match)**:
    - *Failure Mode*: Since symmetric matching utilizes exact token intersections on extracted skills, a candidate listing `"dense vectors"` might not match a JD requiring `"embeddings"` unless mapped under the same key in `SKILL_ONTOLOGY` aliases.
    - *Mitigation*: The `all-MiniLM` semantic embedding recall path acts as a safety net, pulling in candidates who express similar skills through different vocabulary.
2.  **Unstructured JDs**:
    - *Failure Mode*: If the job description is a single block of text without clear headings, the ontology cannot partition "Preferred" vs "Required" sections, fallback to treating all skills as required.
3.  **Very Brief Career Histories**:
    - *Failure Mode*: For candidates with $< 6$ months of total career history, promotion velocity calculations and specialization consistency scores can fluctuate erratically.
