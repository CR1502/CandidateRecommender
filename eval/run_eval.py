"""
Offline ranking evaluation.

Ranks every resume in eval/data/resumes against every job in eval/data/jobs
using the same engine and text cleaning as the API, then compares the order
with the graded labels in eval/data/qrels.json.

    uv run python eval/run_eval.py
    EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 uv run python eval/run_eval.py
    uv run python eval/run_eval.py --json results.json --min-ndcg 0.8

Metrics
  NDCG@5 / NDCG@10  graded ranking quality (gain 2^grade - 1), 1.0 is perfect
  MRR               1 / rank of the first top-graded resume, averaged over jobs
  P@5               share of the top 5 with grade >= 2
  Junk@5            share of the top 5 with grade 0 (lower is better)

Calibration looks at the absolute scores, which NDCG ignores: the mean score
per grade, and how many grade-0 pairs land in a "Good" tier or better.

The dataset is small and synthetic (29 resumes x 6 jobs), so treat it as a
regression check, not a benchmark. Differences of a few hundredths are noise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean

from loguru import logger

from candidate_recommender.config import get_settings
from candidate_recommender.core.embeddings import EmbeddingEngine
from candidate_recommender.core.text_cleaner import TextCleaner

DATA = Path(__file__).parent / "data"
GOOD_TIER_PCT = 50  # "Good Candidate" or better


def load_dataset() -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    jobs = {p.stem: p.read_text() for p in sorted((DATA / "jobs").glob("*.txt"))}
    resumes = {p.stem: p.read_text() for p in sorted((DATA / "resumes").glob("*.txt"))}
    qrels = {
        k: v
        for k, v in json.loads((DATA / "qrels.json").read_text()).items()
        if not k.startswith("_")
    }

    for job, labels in qrels.items():
        if job not in jobs:
            sys.exit(f"qrels.json names unknown job {job!r}")
        unknown = set(labels) - set(resumes)
        if unknown:
            sys.exit(f"qrels.json names unknown resumes for {job!r}: {sorted(unknown)}")
    return jobs, resumes, qrels


def dcg(grades: list[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at(ranked_grades: list[int], all_grades: list[int], k: int) -> float:
    ideal = dcg(sorted(all_grades, reverse=True)[:k])
    return dcg(ranked_grades[:k]) / ideal if ideal else 0.0


def evaluate(engine: EmbeddingEngine) -> dict:
    jobs, resumes, qrels = load_dataset()
    cleaner = TextCleaner()

    # Same preparation as api/services/pipeline.py
    candidates = [
        {
            "filename": rid,
            "candidate_name": rid,
            "raw_text": text,
            "text": cleaner.prepare_for_embedding(text),
        }
        for rid, text in resumes.items()
    ]

    per_job = {}
    pairs = []  # (grade, percentage_score) for calibration
    start = time.perf_counter()
    for job_id, jd in jobs.items():
        labels = qrels.get(job_id, {})
        ranked = engine.rank_candidates(
            cleaner.prepare_for_embedding(jd), [dict(c) for c in candidates], top_k=len(candidates)
        )
        order = [r["filename"] for r in ranked]
        grades = [labels.get(rid, 0) for rid in order]
        top_grade = max(labels.values(), default=0)
        first_top = next((i for i, g in enumerate(grades) if g == top_grade), None)

        per_job[job_id] = {
            "ndcg@5": ndcg_at(grades, grades, 5),
            "ndcg@10": ndcg_at(grades, grades, 10),
            "mrr": 1 / (first_top + 1) if first_top is not None else 0.0,
            "p@5": sum(g >= 2 for g in grades[:5]) / 5,
            "junk@5": sum(g == 0 for g in grades[:5]) / 5,
            "top5": [
                (rid, labels.get(rid, 0), r["percentage_score"])
                for rid, r in zip(order[:5], ranked[:5], strict=True)
            ],
            "ranks": {rid: i + 1 for i, rid in enumerate(order)},
        }
        pairs.extend((labels.get(r["filename"], 0), r["percentage_score"]) for r in ranked)
    elapsed = time.perf_counter() - start

    metrics = ["ndcg@5", "ndcg@10", "mrr", "p@5", "junk@5"]
    summary = {m: mean(j[m] for j in per_job.values()) for m in metrics}
    by_grade = {g: [pct for grade, pct in pairs if grade == g] for g in range(4)}
    calibration = {
        "mean_pct_by_grade": {g: round(mean(v), 1) for g, v in by_grade.items() if v},
        "grade0_in_good_tier": sum(pct >= GOOD_TIER_PCT for pct in by_grade[0]),
        "grade0_pairs": len(by_grade[0]),
        "grade3_below_good_tier": sum(pct < GOOD_TIER_PCT for pct in by_grade[3]),
        "grade3_pairs": len(by_grade[3]),
    }
    return {
        "model": engine.model_name,
        "summary": summary,
        "per_job": per_job,
        "calibration": calibration,
        "seconds": round(elapsed, 2),
    }


def print_report(result: dict) -> None:
    print(f"\nModel: {result['model']}   ({result['seconds']}s to rank all jobs)\n")
    header = f"{'job':<20}{'NDCG@5':>8}{'NDCG@10':>9}{'MRR':>6}{'P@5':>6}{'Junk@5':>8}"
    print(header)
    print("-" * len(header))
    for job, m in result["per_job"].items():
        print(
            f"{job:<20}{m['ndcg@5']:>8.3f}{m['ndcg@10']:>9.3f}{m['mrr']:>6.2f}{m['p@5']:>6.2f}{m['junk@5']:>8.2f}"
        )
    s = result["summary"]
    print("-" * len(header))
    print(
        f"{'MEAN':<20}{s['ndcg@5']:>8.3f}{s['ndcg@10']:>9.3f}{s['mrr']:>6.2f}{s['p@5']:>6.2f}{s['junk@5']:>8.2f}"
    )

    c = result["calibration"]
    print("\nCalibration (composite %)")
    print(
        "  mean score by grade: "
        + ", ".join(f"g{g}={v}" for g, v in c["mean_pct_by_grade"].items())
    )
    print(
        f"  grade-0 pairs scored Good tier or better: {c['grade0_in_good_tier']}/{c['grade0_pairs']}"
    )
    print(
        f"  grade-3 pairs scored below Good tier:     {c['grade3_below_good_tier']}/{c['grade3_pairs']}"
    )

    print("\nTop 5 per job (resume, grade, score)")
    for job, m in result["per_job"].items():
        print(f"  {job}")
        for rid, grade, pct in m["top5"]:
            print(f"    g{grade}  {pct:5.1f}  {rid}")

    ranks = {job: m["ranks"] for job, m in result["per_job"].items()}
    tech_jobs = [j for j in ranks if j != "marketing_manager"]
    stuffer = min(ranks[j]["r24_keyword_stuffer_retail"] for j in tech_jobs)
    print("\nTrap checks")
    print(f"  keyword-stuffed retail resume, best rank on any tech job: {stuffer} (want > 5)")
    print(
        f"  long resume w/ skills past ~512 tokens, rank on backend_python: {ranks['backend_python']['r25_long_manager_skills_at_end']} (want <= 2)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", type=Path, help="also write full results to this file")
    parser.add_argument(
        "--min-ndcg", type=float, help="exit non-zero if mean NDCG@5 falls below this"
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    engine = EmbeddingEngine.from_settings(get_settings())
    result = evaluate(engine)
    print_report(result)

    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    if args.min_ndcg is not None and result["summary"]["ndcg@5"] < args.min_ndcg:
        sys.exit(f"\nFAIL: mean NDCG@5 {result['summary']['ndcg@5']:.3f} < {args.min_ndcg}")


if __name__ == "__main__":
    main()
