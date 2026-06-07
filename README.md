# Redrob Candidate Ranker

A production-grade, CPU-only, fast candidate ranking system designed to rank 100,000 candidates against a Senior AI Engineer JD (Redrob AI, Series A, Pune/Noida).

## Setup & Installation

1. Ensure Python 3.11+ is installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Reproducing the Submission

> [!IMPORTANT]
> precompute.py must be run before rank.py. The data/ directory is a required runtime dependency.

To run the candidate ranker on the candidate pool and generate the final ranked CSV, run the following command from the repository root:

```bash
python rank.py --candidates ../India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv
```

## System Architecture

The ranking pipeline consists of a 4-stage funnel:
1. **Stage 0: Honeypot Filter**: Removes logically impossible profiles (e.g. experience mismatches, duration exceeding company age, fake expert skills).
2. **Stage 1: Rapid Triage**: Performs fast binary pass/fail constraints check on titles, text keywords, and relocation/logistics.
3. **Stage 2: Multi-Signal Deep Scoring**: Evaluates surviving candidates across 5 weighted components (Skills, Trajectory, Experience target bracket, Location, and Education) multiplied by a platform activity Behavioral Multiplier.
4. **Stage 3: Deep Re-Ranking & Reasoning**: Refines top-500 candidate scores, solves tie-breaks deterministically using candidate ID, and generates a unique, non-templated reasoning string for each of the top 100 candidates.

## Running Tests

To verify the pipeline logic, execute the following command:
```bash
python -m pytest tests/
```
