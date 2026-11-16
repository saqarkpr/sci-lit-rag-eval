import pytest

from src import generator as G
from src.generator import (
    extract_candidate_spans, predict_extractive_span, fuzzy_match, exact_match,
    extract_features, FEATURE_NAMES, set_negation_cues,
)


def test_acronym_extraction_prefers_frequent_method_names():
    text = ("We propose MixPE, a mixed-precision engine. MixPE outperforms LLM baselines; "
            "the LLM baseline uses GPU kernels. fDPO is also mentioned once.")
    ranked = extract_candidate_spans(text)
    assert ranked[0] == "MixPE"
    assert "LLM" not in ranked and "GPU" not in ranked  # stoplisted
    assert "fDPO" in ranked
    assert predict_extractive_span("nothing acronym-like here") == ""


def test_match_functions():
    assert fuzzy_match("DPR", "Dense Passage Retrieval (DPR)")
    assert not exact_match("DPR", "Dense Passage Retrieval (DPR)")
    assert exact_match("I-LLM", "i llm")
    assert not fuzzy_match("", "x") and not exact_match("x", "")


def test_extract_features_shape_and_negation(docs, docs_by_id):
    retrieved = [(docs[0]["arxiv_id"], 10.0), (docs[1]["arxiv_id"], 4.0), (docs[2]["arxiv_id"], 1.0)]
    feats, top1 = extract_features("Does this paper use only sparse retrieval?", retrieved, docs_by_id)
    assert top1 == docs[0]["arxiv_id"]
    assert len(feats) == len(FEATURE_NAMES)
    f = dict(zip(FEATURE_NAMES, feats))
    assert f["margin"] == pytest.approx(6.0)
    assert f["has_negation"] == 1 and f["starts_aux"] == 1 and f["q_len"] == 7


def test_set_negation_cues_swaps_in_place():
    try:
        set_negation_cues("tuned")
        assert "single response" in G.NEGATION_CUES
        set_negation_cues("generic")
        assert "single response" not in G.NEGATION_CUES and "not " in G.NEGATION_CUES
        with pytest.raises(ValueError):
            set_negation_cues("bogus")
    finally:
        set_negation_cues("generic")
