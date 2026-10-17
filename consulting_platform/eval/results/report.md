# Evaluation report
_Generated 2026-10-17 14:08 | LLM: No LLM (extractive mode)_

## 1. Routing accuracy
Test set: 52 labelled requests (7 cross-domain).

| Strategy | Primary acc. | Exact set match | Mean Jaccard | Macro F1 |
|---|---|---|---|---|
| lexicon | 96.2% | 92.3% | 0.946 | 0.958 |
| prototype | 100.0% | 94.2% | 0.974 | 0.979 |
| hybrid | 100.0% | 94.2% | 0.974 | 0.979 |

**Hold-out set** (23 requests written after the router was tuned, 3 cross-domain):

| Strategy | Primary acc. | Exact set match | Mean Jaccard | Macro F1 |
|---|---|---|---|---|
| lexicon | 100.0% | 91.3% | 0.957 | 0.966 |
| prototype | 100.0% | 91.3% | 0.957 | 0.966 |
| hybrid | 100.0% | 91.3% | 0.957 | 0.966 |

## 2. Retrieval accuracy (agent retrieval pipelines)

| Domain | n | Hit@1 | Hit@3 | Hit@k | MRR@k | Source precision@k |
|---|---|---|---|---|---|---|
| csr | 10 | 60.0% | 90.0% | 100.0% | 0.753 | 90.0% |
| sec | 10 | 100.0% | 100.0% | 100.0% | 1.000 | 100.0% |
| clinical | 12 | 91.7% | 100.0% | 100.0% | 0.958 | 48.6% |
| technology | 5 | 100.0% | 100.0% | 100.0% | 1.000 | 58.3% |
| **overall** | 37 | 86.5% | 97.3% | 100.0% | 0.920 | 75.0% |

## 3. Recommendation relevance

**Market basket (leave-one-out, 5 random 80/20 splits):** Hit@5 = 13.7% (+/-1.5%), popularity baseline = 13.2%, random = 13.2%, MRR = 0.062, precision@5 = 0.027.

**Clinical trial recommender:** P@5 = 97.5%, nDCG@5 = 0.984
**CSR investment recommender:** Hit@1 = 100.0%, Hit@3 = 100.0%
**Technology fault diagnosis:** Hit@1 = 100.0%, Hit@2 = 100.0%

## 4. Response quality (No LLM (extractive mode))

| Citation coverage | Citation validity | Grounding rate | Query coverage | Actionable |
|---|---|---|---|---|
| 72.2% | 100.0% | 98.3% | 58.9% | 70.0% |

### Caveats (read before quoting numbers)
- Test sets are small and were written by the project team; the CSR-investment and fault-diagnosis cases (8 each) are sanity checks, not benchmarks. The routing hold-out set is the fairest routing number.
- Market basket: Hit@5 is close to both popularity and random baselines, i.e. the 1,000 transactions contain only weak co-purchase structure. The recommender therefore reports lift/support with each suggestion and abstains when no rule meets the support and lift thresholds.
- Without an LLM, answers are extractive (evidence sentences), so grounding is high by construction; re-run with `--with-llm` to measure generated answers.

### Metric definitions
- *Primary acc.*: top routed domain is (one of) the gold domains; *Exact set match*: routed set == gold set.
- *Hit@k*: a passage from the expected source containing an expected answer phrase is in the top-k.
- *Citation validity*: cited ids exist in the evidence; *Grounding rate*: cited sentences share >=30% of their content words with the cited passage.
- *Market basket Hit@5*: hide one item of each held-out basket, recommend from the rest.