"""Integrity checks on the corpus and benchmark files."""
from collections import Counter


def test_corpus_shape(docs):
    assert len(docs) == 34
    ids = [d["arxiv_id"] for d in docs]
    assert len(set(ids)) == len(ids), "duplicate arXiv ids"
    for d in docs:
        assert d["title"] and d["abstract"]
        assert isinstance(d["is_survey"], bool)


def test_benchmark_shape(qa, docs_by_id):
    assert len(qa) == 77
    assert Counter(q["answer_type"] for q in qa) == {"yes_no": 34, "extractive": 27, "unanswerable": 16}
    assert len({q["qid"] for q in qa}) == 77
    for q in qa:
        if q["answer_type"] == "unanswerable":
            assert q["gold_answer"] is None and q["gold_arxiv_id"] is None
        else:
            assert q["gold_arxiv_id"] in docs_by_id, q["qid"]
        if q["answer_type"] == "yes_no":
            assert q["gold_answer"] in ("Yes", "No")


def test_extractive_gold_is_grounded_in_gold_doc(qa, docs_by_id):
    """Gold spans must be recoverable from the gold document. Four are
    written in a reordered form (e.g. 'RAG (Retrieval-Augmented Generation)'
    vs. the abstract's 'Retrieval-Augmented Generation (RAG)'), so we check
    that every alphanumeric word of the gold span appears in the document."""
    import re
    for q in qa:
        if q["answer_type"] != "extractive":
            continue
        text = docs_by_id[q["gold_arxiv_id"]]["text"].lower()
        words = re.findall(r"[a-z0-9]+", q["gold_answer"].lower())
        assert words and all(w in text for w in words), (q["qid"], q["gold_answer"])


def test_extractive_only_for_non_surveys(qa, docs_by_id):
    for q in qa:
        if q["answer_type"] == "extractive":
            assert not docs_by_id[q["gold_arxiv_id"]]["is_survey"], q["qid"]
