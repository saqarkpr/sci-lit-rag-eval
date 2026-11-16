"""End-to-end smoke tests: retrievers, CV cascade, metrics and probes all
run on the real (small) data in a few seconds."""
import pytest

from src.retrieval import build_retriever, RETRIEVERS
from src.retrieval_eval import evaluate_retriever
from src.efficiency import benchmark_retriever
from src.robustness import negate_question, negation_flip_probe, query_perturbation_probe
from train import build_dataset, run_cv
from evaluate import is_correct, metrics_from_records


@pytest.mark.parametrize("name", sorted(RETRIEVERS))
def test_retriever_contract(name, docs, qa):
    r = build_retriever(name, docs, n_components=8)
    out = r.retrieve("retrieval-augmented generation survey", k=3)
    assert len(out) == 3
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(i, str) for i, _ in out)
    assert r.approx_memory_bytes() > 0 and r.build_time_s >= 0
    metrics, per_query = evaluate_retriever(r, qa)
    assert metrics["n_queries"] == 61 and len(per_query) == 61
    assert 0 < metrics["recall@1"] <= metrics["recall@3"] <= metrics["recall@5"] <= 1
    eff = benchmark_retriever(r, [q["question"] for q in qa[:5]], n_repeats=1)
    assert eff["n_queries_timed"] == 5


def test_cv_cascade_is_out_of_fold_and_deterministic(docs, docs_by_id, qa):
    r = build_retriever("bm25", docs)
    rows = build_dataset(qa, r, docs_by_id, k=5)
    preds1 = run_cv(rows, seed=1, n_splits=5)
    preds2 = run_cv(rows, seed=1, n_splits=5)
    assert len(preds1) == len(qa) and all(p is not None for p in preds1)
    assert [p["predicted_value"] for p in preds1] == [p["predicted_value"] for p in preds2]
    for p in preds1:
        assert 0.0 <= p["confidence"] <= 1.0
        assert p["predicted_answer_type"] in ("unanswerable", "yes_no", "extractive")
        if p["predicted_answer_type"] == "extractive":
            # every produced span is a literal substring of the evidence
            assert p["predicted_value"] == "" or p["predicted_value"] in p["top1_text"]
    m = metrics_from_records(preds1)
    assert 0 <= m["accuracy"] <= 1 and m["accuracy_ci95"][0] <= m["accuracy"] <= m["accuracy_ci95"][1]
    assert set(m["accuracy_by_type"]) == {"yes_no", "extractive", "unanswerable"}


def test_is_correct_rules():
    assert is_correct({"gold_answer_type": "unanswerable", "predicted_answer_type": "unanswerable"})
    assert not is_correct({"gold_answer_type": "yes_no", "predicted_answer_type": "extractive",
                           "gold_answer": "Yes", "predicted_value": "Yes"})
    rec = {"gold_answer_type": "extractive", "predicted_answer_type": "extractive",
           "gold_answer": "Dense Passage Retrieval (DPR)", "predicted_value": "DPR"}
    assert is_correct(rec) and not is_correct(rec, strict=True)


def test_probes(docs, docs_by_id, qa):
    assert negate_question("Does the model use DPO?") == "Is it NOT the case that the model use DPO?"
    r = build_retriever("bm25", docs)
    neg = negation_flip_probe(qa, r, docs_by_id)
    assert neg["n_probed"] == 34 and 0 <= neg["negation_flip_rate"] <= 1
    pert = query_perturbation_probe(qa, r, rates=(0.2,), n_perturbations=2)
    assert pert["clean"]["recall@1"] >= pert["perturbed"][0]["recall@1_mean"] - 0.05
    assert {p["mode"] for p in pert["perturbed"]} == {"drop", "typo"}
