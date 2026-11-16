import numpy as np
import pytest

from src.evaluate_trust import (
    expected_calibration_error, brier_score, hallucination_rate,
    false_abstention_rate, evidence_grounding_rate, bootstrap_ci,
)


def test_ece_perfectly_calibrated_is_zero():
    conf = np.array([0.25] * 4 + [0.75] * 4)
    corr = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    ece, bins = expected_calibration_error(conf, corr, n_bins=4)
    assert ece == pytest.approx(0.0)
    assert sum(b["count"] for b in bins) == 8


def test_ece_overconfident():
    ece, _ = expected_calibration_error([0.9, 0.9], [0, 0], n_bins=10)
    assert ece == pytest.approx(0.9)


def test_ece_includes_confidence_one_in_last_bin():
    ece, bins = expected_calibration_error([1.0], [1], n_bins=10)
    assert bins[-1]["count"] == 1 and ece == 0.0


def test_brier():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def _rec(gold_type, pred_type, gold_id="a", top1="a"):
    return {"gold_answer_type": gold_type, "predicted_answer_type": pred_type,
            "gold_arxiv_id": gold_id, "top1_id": top1}


def test_hallucination_and_false_abstention():
    recs = [_rec("unanswerable", "yes_no", None), _rec("unanswerable", "unanswerable", None),
            _rec("yes_no", "unanswerable"), _rec("yes_no", "yes_no")]
    assert hallucination_rate(recs) == (0.5, 1, 2)
    assert false_abstention_rate(recs) == (0.5, 1, 2)
    assert hallucination_rate([_rec("yes_no", "yes_no")]) == (None, 0, 0)


def test_evidence_grounding_only_counts_answered_items():
    recs = [_rec("yes_no", "yes_no", "a", "a"), _rec("yes_no", "yes_no", "a", "b"),
            _rec("yes_no", "unanswerable", "a", "b"), _rec("unanswerable", "yes_no", None, "b")]
    assert evidence_grounding_rate(recs) == (0.5, 1, 2)


def test_bootstrap_ci_contains_point_estimate_and_is_deterministic():
    vals = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    est, lo, hi = bootstrap_ci(vals, seed=1)
    est2, lo2, hi2 = bootstrap_ci(vals, seed=1)
    assert est == pytest.approx(0.7) and lo <= est <= hi
    assert (lo, hi) == (lo2, hi2)
    assert bootstrap_ci([]) == (None, None, None)
