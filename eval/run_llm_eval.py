"""
LLM assessment evaluation (needs a running Ollama with the model pulled).

For each job, ranks all eval resumes with the embedding engine, then has the
LLM assess the top N — the candidates a recruiter would actually read — and
compares its recommendations with the graded labels in data/qrels.json.

    uv run python eval/run_llm_eval.py                       # OLLAMA_MODEL from settings
    OLLAMA_MODEL=qwen3:8b uv run python eval/run_llm_eval.py --top 3

Metrics
  Exact        recommendation matches the grade (Strong Yes=3, Yes=2, Maybe=1, No=0)
  Within 1     off by at most one step
  False yes    "Yes"/"Strong Yes" for a grade 0–1 resume (lower is better)
  Missed good  "No" for a grade 2–3 resume (lower is better)
  Ungrounded   share of LLM-reported skills that don't appear in the resume and
               were dropped (a hallucination rate)
  Fallbacks    assessments that failed and used the template instead

Slow: each assessment is one LLM call (~10–30s on a laptop GPU).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from run_eval import load_dataset  # noqa: E402

from candidate_recommender.config import get_settings  # noqa: E402
from candidate_recommender.core.embeddings import EmbeddingEngine  # noqa: E402
from candidate_recommender.core.summarizer import CandidateSummarizer  # noqa: E402
from candidate_recommender.core.text_cleaner import TextCleaner  # noqa: E402

REC_VALUE = {"Strong Yes": 3, "Yes": 2, "Maybe": 1, "No": 0}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--top", type=int, default=5, help="assess the top N candidates per job")
    parser.add_argument("--json", type=Path, help="also write full results to this file")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    settings = get_settings()
    summarizer = CandidateSummarizer.from_settings(settings)
    if not summarizer.llm_available():
        sys.exit(
            f"Ollama model '{settings.ollama_model}' is not available at {settings.ollama_base_url}"
        )

    # Record the raw replies to measure how many reported skills were ungrounded.
    raw_replies: list[dict] = []
    generate = summarizer.client.generate_json

    def recording_generate(*a, **kw):
        reply = generate(*a, **kw)
        raw_replies.append(reply)
        return reply

    summarizer.client.generate_json = recording_generate

    engine = EmbeddingEngine.from_settings(settings)
    cleaner = TextCleaner()
    jobs, resumes, qrels = load_dataset()
    rows = []
    for job_id, jd in jobs.items():
        clean_jd = cleaner.prepare_for_embedding(jd)
        candidates = [
            {
                "filename": r,
                "candidate_name": r,
                "raw_text": t,
                "text": cleaner.prepare_for_embedding(t),
            }
            for r, t in resumes.items()
        ]
        ranked = engine.rank_candidates(clean_jd, candidates, top_k=args.top)
        jd_skills = {s.lower() for s in cleaner.extract_key_skills(clean_jd)}
        for c in ranked:
            c["matching_skills"] = [
                s for s in cleaner.extract_key_skills(c["raw_text"]) if s.lower() in jd_skills
            ]
            c["contact"] = cleaner.extract_contact_details(c["raw_text"])
            before = len(raw_replies)
            start = time.perf_counter()
            summarizer.batch_assess([c], clean_jd)
            seconds = time.perf_counter() - start
            reported = (
                raw_replies[-1].get("matching_skills", []) if len(raw_replies) > before else []
            )
            grounded = {s.lower() for s in c["matching_skills"]}
            canon = cleaner.canonicalize_skills(reported)
            rows.append(
                {
                    "job": job_id,
                    "resume": c["filename"],
                    "grade": qrels.get(job_id, {}).get(c["filename"], 0),
                    "score": c["percentage_score"],
                    "recommendation": c["recommendation"],
                    "source": c["summary_source"],
                    "seconds": round(seconds, 1),
                    "reported_skills": len(canon),
                    "ungrounded_skills": sum(s.lower() not in grounded for s in canon),
                    "summary": c["fit_summary"],
                    "gaps": c["gaps"],
                }
            )
            r = rows[-1]
            print(f"{job_id:<18} g{r['grade']} {r['score']:5.1f}% -> {str(r['recommendation']):<10} "
                  f"{r['seconds']:5.1f}s  {r['resume']}", flush=True)  # fmt: skip

    llm_rows = [r for r in rows if r["source"] == "llm"]
    diffs = [abs(REC_VALUE[r["recommendation"]] - r["grade"]) for r in llm_rows]
    times = sorted(r["seconds"] for r in rows)
    reported = sum(r["reported_skills"] for r in llm_rows)
    summary = {
        "model": settings.ollama_model,
        "assessments": len(rows),
        "exact": sum(d == 0 for d in diffs) / max(len(diffs), 1),
        "within_1": sum(d <= 1 for d in diffs) / max(len(diffs), 1),
        "false_yes": sum(r["grade"] <= 1 and REC_VALUE[r["recommendation"]] >= 2 for r in llm_rows),
        "missed_good": sum(r["grade"] >= 2 and r["recommendation"] == "No" for r in llm_rows),
        "ungrounded_skill_rate": sum(r["ungrounded_skills"] for r in llm_rows) / max(reported, 1),
        "fallbacks": len(rows) - len(llm_rows),
        "seconds_p50": statistics.median(times),
        "seconds_p90": times[int(0.9 * (len(times) - 1))],
    }
    print(f"\nModel {summary['model']}: {summary['assessments']} assessments")
    print(f"  exact {summary['exact']:.0%}  within 1 {summary['within_1']:.0%}  "
          f"false yes {summary['false_yes']}  missed good {summary['missed_good']}")  # fmt: skip
    print(
        f"  ungrounded skills {summary['ungrounded_skill_rate']:.0%}  fallbacks {summary['fallbacks']}"
    )
    print(f"  latency p50 {summary['seconds_p50']:.1f}s  p90 {summary['seconds_p90']:.1f}s")
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
