#!/usr/bin/env python3
"""Fit the cascaded classifier-based generator under stratified k-fold
cross-validation for one retriever, and dump out-of-fold predictions +
retrieval metrics + efficiency numbers to results/.

Usage:
    python train.py --retriever bm25
    python train.py --retriever tfidf_lsa --seed 42
    python train.py --retriever bm25 --negation-cues tuned   # ablation arm

Defaults come from configs/default.yaml; any flag given overrides it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.data_utils import load_corpus, load_qa, corpus_index_by_id, load_config
from src.retrieval import build_retriever
from src.retrieval_eval import evaluate_retriever
from src.efficiency import benchmark_retriever
from src.generator import extract_features, predict_extractive_span, set_negation_cues

ROOT = Path(__file__).resolve().parent


def build_dataset(qa_items, retriever, docs_by_id, k=5):
    """Pre-compute retrieval + features once per item (retrieval isn't
    fit on QA labels, so this is leakage-free w.r.t. the CV below)."""
    rows = []
    for q in qa_items:
        retrieved = retriever.retrieve(q["question"], k=k)
        feats, top1_id = extract_features(q["question"], retrieved, docs_by_id)
        rows.append({
            "qid": q["qid"],
            "question": q["question"],
            "gold_answer_type": q["answer_type"],
            "gold_answer": q["gold_answer"],
            "gold_arxiv_id": q["gold_arxiv_id"],
            "features": feats,
            "top1_id": top1_id,
            "top1_text": docs_by_id[top1_id]["text"],
        })
    return rows


def run_cv(rows, seed=42, n_splits=5):
    """One shared StratifiedKFold split drives all three cascade stages, so
    a Stage-A error propagates into B/C exactly as it would at inference."""
    X = np.array([r["features"] for r in rows])
    y_unanswerable = np.array([1 if r["gold_answer_type"] == "unanswerable" else 0 for r in rows])
    y_extractive = np.array([1 if r["gold_answer_type"] == "extractive" else 0 for r in rows])
    y_is_yes = np.array([1 if r["gold_answer"] == "Yes" else 0 for r in rows])
    is_answerable = (y_unanswerable == 0)
    is_yes_no = np.array([r["gold_answer_type"] == "yes_no" for r in rows])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strata = np.array([r["gold_answer_type"] for r in rows])

    predictions = [None] * len(rows)

    for train_idx, test_idx in skf.split(X, strata):
        # Stage A: answerable vs unanswerable
        clf_a = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf_a.fit(X[train_idx], y_unanswerable[train_idx])
        p_unanswerable = clf_a.predict_proba(X[test_idx])[:, list(clf_a.classes_).index(1)]
        pred_unanswerable = p_unanswerable >= 0.5

        # Stage B: trained only on truly-answerable TRAIN items
        train_answerable_idx = train_idx[is_answerable[train_idx]]
        clf_b = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf_b.fit(X[train_answerable_idx], y_extractive[train_answerable_idx])
        classes_b = list(clf_b.classes_)

        # Stage C: trained only on truly-yes/no TRAIN items
        train_yesno_idx = train_idx[is_yes_no[train_idx]]
        clf_c = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf_c.fit(X[train_yesno_idx], y_is_yes[train_yesno_idx])
        classes_c = list(clf_c.classes_)

        for local_i, global_i in enumerate(test_idx):
            r = rows[global_i]
            p_una = float(p_unanswerable[local_i])
            if pred_unanswerable[local_i]:
                predictions[global_i] = {
                    **r, "predicted_answer_type": "unanswerable",
                    "predicted_value": None, "confidence": p_una,
                }
                continue

            proba_b = clf_b.predict_proba(X[global_i:global_i + 1])[0]
            p_extractive = float(proba_b[classes_b.index(1)]) if 1 in classes_b else 0.0
            pred_extractive = p_extractive >= 0.5
            p_answerable = 1.0 - p_una

            if pred_extractive:
                span = predict_extractive_span(r["top1_text"])
                predictions[global_i] = {
                    **r, "predicted_answer_type": "extractive",
                    "predicted_value": span, "confidence": p_answerable * p_extractive,
                }
            else:
                proba_c = clf_c.predict_proba(X[global_i:global_i + 1])[0]
                p_yes = float(proba_c[classes_c.index(1)]) if 1 in classes_c else 0.5
                pred_yes = p_yes >= 0.5
                conf = p_answerable * (1.0 - p_extractive) * (p_yes if pred_yes else 1.0 - p_yes)
                predictions[global_i] = {
                    **r, "predicted_answer_type": "yes_no",
                    "predicted_value": "Yes" if pred_yes else "No", "confidence": conf,
                }

    return predictions


def strip_predictions(predictions):
    """Drop large / non-serializable fields before writing JSON."""
    return [{k: v for k, v in p.items() if k not in ("top1_text", "features")} for p in predictions]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--retriever", choices=["bm25", "tfidf_lsa"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--k", type=int, help="evidence passages retrieved per question")
    ap.add_argument("--n-splits", type=int)
    ap.add_argument("--negation-cues", choices=["generic", "tuned"],
                    help="generic (default, headline numbers) or tuned (ablation arm)")
    ap.add_argument("--out-suffix", default="",
                    help="appended to result filenames, e.g. '_tunedcues' for ablation runs")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    retriever_name = args.retriever or cfg["retriever"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    k = args.k if args.k is not None else cfg["top_k"]
    n_splits = args.n_splits if args.n_splits is not None else cfg["n_splits"]
    cue_mode = set_negation_cues(args.negation_cues or cfg["negation_cues"])

    docs = load_corpus(ROOT / cfg["data"]["corpus"])
    qa = load_qa(ROOT / cfg["data"]["qa_benchmark"])
    docs_by_id = corpus_index_by_id(docs)

    retriever = build_retriever(retriever_name, docs, n_components=cfg["tfidf_lsa"]["n_components"])

    retrieval_metrics, per_query_ranks = evaluate_retriever(retriever, qa, ks=(1, 3, max(5, k)))
    eff = benchmark_retriever(retriever, [q["question"] for q in qa], k=k)

    rows = build_dataset(qa, retriever, docs_by_id, k=k)
    predictions = run_cv(rows, seed=seed, n_splits=n_splits)
    clean_predictions = strip_predictions(predictions)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    tag = f"{retriever_name}{args.out_suffix}"
    run_meta = {"retriever": retriever_name, "seed": seed, "top_k": k, "n_splits": n_splits,
                "negation_cues": cue_mode, "n_components": cfg["tfidf_lsa"]["n_components"]}
    with open(results_dir / f"records_{tag}.json", "w") as f:
        json.dump({"run": run_meta, "records": clean_predictions}, f, indent=2)
    with open(results_dir / f"retrieval_metrics_{tag}.json", "w") as f:
        json.dump({"run": run_meta, "metrics": retrieval_metrics, "per_query": per_query_ranks}, f, indent=2)
    with open(results_dir / f"efficiency_{tag}.json", "w") as f:
        json.dump({"run": run_meta, **eff}, f, indent=2)

    print(f"[{tag}] retrieval:", retrieval_metrics)
    print(f"[{tag}] efficiency:", eff)
    print(f"[{tag}] wrote {len(clean_predictions)} out-of-fold predictions (negation cues: {cue_mode})")


if __name__ == "__main__":
    main()
