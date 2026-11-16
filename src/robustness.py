"""Robustness probes. Both are *diagnostics* of the fitted system, not
held-out generalization estimates (see README, Limitations).

1. negation_flip_probe: train the Stage-C (yes vs. no) classifier on all
   yes/no items, then negate each question's logical polarity
   ("Is it NOT the case that ...") and check whether the predicted answer
   flips. A logically sound classifier should flip on (nearly) every item.

2. query_perturbation_probe: how much does retrieval Recall@1 / MRR degrade
   when a fraction of the query's tokens is randomly dropped or corrupted
   with a character-level typo? This approximates organic, imperfect user
   queries, which the hand-written benchmark questions (lexically close to
   the abstracts) do not represent.
"""
import random
import re

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.generator import extract_features

_WORD_RE = re.compile(r"\S+")


def negate_question(question: str) -> str:
    stripped = question.rstrip("?").strip()
    for aux in ["Does ", "Is ", "Did ", "Has ", "Do ", "Are ", "Can ", "Was "]:
        if stripped.startswith(aux):
            stripped = stripped[len(aux):]
            break
    return f"Is it NOT the case that {stripped[0].lower()}{stripped[1:]}?"


def negation_flip_probe(qa_items, retriever, docs_by_id, k=5):
    yesno_items = [q for q in qa_items if q["answer_type"] == "yes_no"]

    def featurize(question):
        retrieved = retriever.retrieve(question, k=k)
        feats, _ = extract_features(question, retrieved, docs_by_id)
        return feats

    X = np.array([featurize(q["question"]) for q in yesno_items])
    y = np.array([1 if q["gold_answer"] == "Yes" else 0 for q in yesno_items])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    yes_col = list(clf.classes_).index(1)

    flips, per_item = 0, []
    for q, x in zip(yesno_items, X):
        p_orig = clf.predict_proba(x[None, :])[0][yes_col]
        negated = negate_question(q["question"])
        p_neg = clf.predict_proba(np.array([featurize(negated)]))[0][yes_col]
        flipped = (p_orig >= 0.5) != (p_neg >= 0.5)
        flips += int(flipped)
        per_item.append({"qid": q["qid"], "p_yes_original": float(p_orig),
                         "p_yes_negated": float(p_neg), "flipped": bool(flipped)})

    return {
        "n_probed": len(yesno_items),
        "n_flipped_on_negation": flips,
        "negation_flip_rate": flips / len(yesno_items),
        "per_item": per_item,
        "note": (
            "Diagnostic probe: the yes/no classifier is trained on all "
            f"{len(yesno_items)} yes/no items (not held out) and then asked the "
            "polarity-negated version of each question. A logically sound "
            "classifier should flip on (nearly) every item, since negating the "
            "question negates the correct answer; a low flip rate means it is "
            "tracking surface lexical cues rather than semantic polarity."
        ),
    }


def _perturb(question: str, rate: float, rng: random.Random, mode: str) -> str:
    words = _WORD_RE.findall(question)
    n_hit = max(1, int(round(rate * len(words))))
    idx = rng.sample(range(len(words)), min(n_hit, len(words)))
    out = list(words)
    for i in idx:
        if mode == "drop":
            out[i] = ""
        else:  # typo: swap two adjacent characters inside the word
            w = out[i]
            if len(w) >= 3:
                j = rng.randrange(0, len(w) - 1)
                w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            out[i] = w
    return " ".join(w for w in out if w)


def query_perturbation_probe(qa_items, retriever, rates=(0.1, 0.2, 0.3),
                             n_perturbations=20, seed=0, k=5):
    answerable = [q for q in qa_items if q["answer_type"] != "unanswerable"]

    def score(questions_golds):
        hits, rr = 0, []
        for question, gold in questions_golds:
            ids = [rid for rid, _ in retriever.retrieve(question, k=k)]
            rank = ids.index(gold) + 1 if gold in ids else None
            hits += int(rank == 1)
            rr.append(1.0 / rank if rank else 0.0)
        return hits / len(questions_golds), float(np.mean(rr))

    clean_r1, clean_mrr = score([(q["question"], q["gold_arxiv_id"]) for q in answerable])
    results = {"clean": {"recall@1": clean_r1, "mrr": clean_mrr}, "perturbed": []}
    for mode in ("drop", "typo"):
        for rate in rates:
            r1s, mrrs = [], []
            for rep in range(n_perturbations):
                rng = random.Random(seed * 10_000 + rep)
                batch = [(_perturb(q["question"], rate, rng, mode), q["gold_arxiv_id"]) for q in answerable]
                r1, mrr = score(batch)
                r1s.append(r1)
                mrrs.append(mrr)
            results["perturbed"].append({
                "mode": mode, "rate": rate, "n_perturbations": n_perturbations,
                "recall@1_mean": float(np.mean(r1s)), "recall@1_std": float(np.std(r1s)),
                "mrr_mean": float(np.mean(mrrs)), "mrr_std": float(np.std(mrrs)),
            })
    results["note"] = (
        "Retrieval robustness to noisy queries: for each answerable question, "
        "a fraction `rate` of its tokens is either dropped or given a "
        "character-swap typo, repeated n_perturbations times with fixed seeds; "
        "Recall@1 / MRR are averaged over repetitions."
    )
    return results
