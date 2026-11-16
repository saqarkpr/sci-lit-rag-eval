# Results summary

Seed 42, 5-fold stratified CV, k=5, negation cues = `generic`. Rates are followed by their count and a 95% percentile-bootstrap confidence interval (2000 resamples).

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | QA acc. [95% CI] | ECE | Brier | Hallucination rate | False-abstention rate | Evidence-grounding rate |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.984 | 0.984 | 0.984 | 0.984 | 0.506 [0.39, 0.62] | 0.167 | 0.272 | 0.188 (3/16) [0.00, 0.38] | 0.197 (12/61) [0.10, 0.30] | 1.000 (49/49) [1.00, 1.00] |
| tfidf_lsa | 0.869 | 0.967 | 0.984 | 0.919 | 0.494 [0.39, 0.61] | 0.163 | 0.281 | 0.312 (5/16) [0.12, 0.56] | 0.279 (17/61) [0.18, 0.39] | 0.909 (40/44) [0.82, 0.98] |

## Accuracy by answer type

Extractive items are scored two ways: *fuzzy* (prediction and gold are substrings of each other, case-insensitive; used in the headline table) and *EM* (exact match after case/hyphen/whitespace normalisation).

| Retriever | yes/no (n) | extractive fuzzy (n) | extractive EM | unanswerable (n) | overall fuzzy | overall EM |
|---|---|---|---|---|---|---|
| bm25 | 0.500 (34) | 0.333 (27) | 0.111 | 0.812 (16) | 0.506 | 0.429 |
| tfidf_lsa | 0.588 (34) | 0.259 (27) | 0.074 | 0.688 (16) | 0.494 | 0.429 |

## Efficiency (CPU-only)

Timings are machine-dependent; treat the *ratio* between retrievers as the result, not the absolute values.

| Retriever | Build time (s) | Mean latency (ms) | P95 latency (ms) | Approx. memory (KB) |
|---|---|---|---|---|
| bm25 | 0.0025 | 0.110 | 0.156 | 473.0 |
| tfidf_lsa | 0.0234 | 0.841 | 1.032 | 197.6 |

## CV-seed stability (10 fold-split seeds, mean +- std [min, max])

| Retriever | QA acc. | ECE | Hallucination rate | False-abstention rate |
|---|---|---|---|---|
| bm25 | 0.534 +- 0.042 [0.47, 0.60] | 0.190 +- 0.028 [0.16, 0.26] | 0.169 +- 0.040 [0.12, 0.25] | 0.174 +- 0.020 [0.15, 0.21] |
| tfidf_lsa | 0.477 +- 0.036 [0.42, 0.53] | 0.187 +- 0.023 [0.13, 0.21] | 0.350 +- 0.041 [0.31, 0.44] | 0.302 +- 0.030 [0.26, 0.36] |

## Negation-cue ablation (seed 42)

`generic` is a domain-agnostic cue list written without looking at the benchmark (headline). `tuned` is the original list that was assembled by reading the benchmark's yes/no questions and contains phrases that each single out one gold-"No" item; it is reported only to quantify how much that leakage inflated the numbers.

| Cue set | Retriever | QA acc. | yes/no acc. | Hallucination rate | False-abstention rate | ECE |
|---|---|---|---|---|---|---|
| generic | bm25 | 0.506 | 0.500 | 0.188 | 0.197 | 0.167 |
| generic | tfidf_lsa | 0.494 | 0.588 | 0.312 | 0.279 | 0.163 |
| tuned | bm25 | 0.571 | 0.588 | 0.125 | 0.180 | 0.173 |
| tuned | tfidf_lsa | 0.494 | 0.618 | 0.312 | 0.295 | 0.162 |

## Robustness probe 1: negation flip test (BM25 features, full-data yes/no model)

- Questions probed: 34
- Flipped prediction on negated paraphrase: 12
- Negation flip rate: 0.353 (higher = more logically consistent; 1.0 = always flips as it should)

## Robustness probe 2: retrieval under query perturbation

Each answerable question is perturbed 20 times per setting (`drop` = tokens removed, `typo` = adjacent-character swap inside tokens); Recall@1 is the mean over perturbations (+- std).

| Retriever | clean R@1 | drop 10% | drop 20% | drop 30% | typo 10% | typo 20% | typo 30% |
|---|---|---|---|---|---|---|---|
| bm25 | 0.984 | 0.973 +- 0.013 | 0.958 +- 0.015 | 0.945 +- 0.024 | 0.970 +- 0.013 | 0.957 +- 0.016 | 0.952 +- 0.018 |
| tfidf_lsa | 0.869 | 0.839 +- 0.027 | 0.793 +- 0.016 | 0.789 +- 0.031 | 0.820 +- 0.030 | 0.813 +- 0.030 | 0.783 +- 0.030 |
