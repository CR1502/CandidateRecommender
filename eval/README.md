# Ranking evaluation

A small labelled dataset and a script that measure how well the engine ranks
resumes, so scoring changes are judged by numbers rather than by eye.

```bash
uv run python eval/run_eval.py                      # report
uv run python eval/run_eval.py --min-ndcg 0.93      # fail below a threshold (CI does this)
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 SEMANTIC_FLOOR=0.5 uv run python eval/run_eval.py
```

Any setting in `src/candidate_recommender/config.py` can be overridden with an
environment variable for an experiment.

## Dataset

- `data/jobs/`: 6 job descriptions (backend, ML, frontend, SRE, data engineering, marketing).
- `data/resumes/`: 29 synthetic resumes. Contact details are fake (`example.com`).
- `data/qrels.json`: a grade for every job/resume pair. 3 = would interview,
  2 = plausible with gaps, 1 = related but weak, 0 = not relevant (the default for
  unlisted pairs). Grades were set before running any model.

The resumes include deliberate hard cases:

| Resume | Tests |
|---|---|
| `r24_keyword_stuffer_retail` | A retail manager with a long buzzword skills list and no technical work |
| `r25_long_manager_skills_at_end` | A strong backend engineer whose hands-on experience starts past the model's ~512-token window |
| `r02_mid_django_dates_only` | Experience given only as date ranges, never "N years" |
| `r03_java_spring_backend` | A near miss: a strong backend engineer in the wrong language |
| `r27`–`r29` | Long, realistic resumes (one relevant, one with key experience late, one irrelevant) |

**Limitations.** The dataset is small, synthetic, and was written by the same
person who tuned the scoring, so it risks overfitting. Differences below about
0.01 NDCG are noise. The most valuable next step is adding real, anonymised
resumes graded by the people who hire for those roles.

## Metrics

- **NDCG@5 / NDCG@10**: ranking quality using the graded labels (1.0 is perfect).
- **MRR**: 1 / rank of the first top-graded resume.
- **P@5**: share of the top 5 with grade ≥ 2.
- **Junk@5**: share of the top 5 with grade 0 (lower is better).
- **Calibration**: the mean score for each grade, and how many grade-0 pairs
  score in the "Good Candidate" tier (≥ 50%) or higher. NDCG ignores absolute
  scores, but recruiters read them.

## Results (Phase 3, September 2026)

All runs use the final 29-resume dataset.

| Configuration | NDCG@5 | NDCG@10 | Junk@5 | Grade-0 pairs ≥ Good | Mean score, g0 / g3 |
|---|---|---|---|---|---|
| Original scoring (`main` before Phase 3) | 0.908 | 0.907 | 0.27 | 45 / 138 | 46.1 / 81.8 |
| + calibrated semantic score, missing components renormalised, relevance-gated experience | 0.935 | 0.932 | 0.20 | 3 / 138 | 24.5 / 84.2 |
| + resume chunking (250 words, `max_mean`) | 0.943 | 0.944 | 0.20 | 3 / 138 | 25.1 / 85.8 |
| + experience from employment dates, skill evidence and nice-to-have weighting (**current default**) | **0.948** | **0.957** | **0.20** | **2 / 138** | **25.4 / 86.3** |

### Tried and not adopted

| Variant | NDCG@5 | Why not |
|---|---|---|
| `BAAI/bge-base-en-v1.5` (+ chunking) | 0.945 | No better than `bge-small`, 2.5× slower, 3× larger |
| `BAAI/bge-m3` (8k context, no chunking) | 0.924 | Worse, about 16× slower, 2.2GB, and ranked the keyword stuffer #2 |
| Dense + BM25 (20% weight) | 0.952 | Within noise. BM25 is normalised within the uploaded batch, so a single resume always scores 1.0 |
| `bge-reranker-base` cross-encoder, 50/50 with dense | 0.953 | Within noise, 1.1GB, about 3s per job on CPU. On its own it scored 26 grade-0 pairs as Good or better |
| `bge-reranker-v2-m3` cross-encoder, 50/50 with dense | 0.954 | Within noise, 2.2GB, about 4s per job on CPU |

Semantic calibration was swept over floor ∈ {0.40, 0.45, 0.50, 0.55} and ceiling
∈ {0.80, 0.85, 0.90} on the first 26 resumes, before chunking. NDCG stayed flat (0.906–0.935). 0.45 / 0.85 gave the
cleanest tier separation without pushing any grade-3 pair below "Good".
Changing the embedding model changes its cosine range, so re-sweep these values
(`SEMANTIC_FLOOR`, `SEMANTIC_CEILING`) when you switch models.

### Known remaining errors

- The keyword stuffer still ranks #5 for the data engineering job (score 44, "Okay").
- For the ML job, the MLOps engineer (grade 1) ranks #2, ahead of the NLP data scientist (grade 2).
- `r25` ranks #3 for the backend job, behind a grade-2 Go engineer.
