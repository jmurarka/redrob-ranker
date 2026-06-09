# Redrob Hackathon Readiness Audit

## Current Position

TalentMind AI v2 is an offline, CPU-oriented candidate ranking system for the
Senior AI Engineer founding-team JD. The implementation combines semantic
retrieval, career trajectory scoring, skill/experience fit, behavioral
availability, trust, growth, and location/logistics fit.

The system is designed to satisfy the hackathon constraints:

- ranking step runs without hosted LLM/API calls
- ranking step uses local precomputed artifacts
- scoring weights sum to exactly 1.0
- output CSV follows the official schema
- reasoning is generated from candidate/profile features, not free-form LLM text

## Fixes Applied After Review

- Removed the unused `jd_match` weight and aligned all scoring documentation with
  the actual score vector.
- Added a 5% `logistics_score` so India/Pune/Noida and relocation constraints
  influence the ranking.
- Reworked JD parsing to avoid reading seniority/skills from negative sections.
  The official JD now parses as `Senior` with `5-9` years.
- Preferred-section skills such as LoRA/QLoRA/PEFT/fine-tuning now stay
  preferred instead of being promoted to hard requirements.
- Updated the sandbox app to use the same symmetric skill/experience blending,
  growth, and logistics signals as the batch ranker.
- Cleaned explanation text to avoid mojibake and awkward casing in manual review.
- Replaced fake metadata values with explicit TODO fields that must be filled
  before portal submission.

## Remaining Submission Checklist

- Fill real team/contact information in `submission_metadata.yaml`.
- Replace `TODO_HOSTED_SANDBOX_URL` with the actual HuggingFace/Streamlit/Replit
  sandbox link.
- Install dependencies in a clean environment and run:

```bash
python precompute.py --candidates candidates.jsonl --jd job_description.txt
python rank.py --candidates candidates.jsonl --jd job_description.txt --out submission.csv
python ../India_runs_data_and_ai_challenge/validate_submission.py submission.csv
python -m pytest tests
```

- Re-sample the regenerated top 10 and top 100 for location, relocation,
  services-only backgrounds, low response rate, and long notice period.

## Interview Defense Notes

The strongest defense is that the ranker is not a keyword counter. It uses
semantic recall for text variation, career scoring for production ML/search
evidence, behavioral/trust signals for hireability, and logistics for the
real-world constraints in the JD. The most important tradeoff to acknowledge is
that deterministic parsing is fast and reproducible, but needs carefully scoped
section handling to avoid false positives from negative or explanatory prose.
