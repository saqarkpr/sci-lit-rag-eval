"""Classifier-based answer generator (no LLM / no GPU required).

Given a question and a retriever's top-k evidence documents, this module:
  1. builds a small feature vector describing the retrieval result and the
     question's surface form,
  2. (via classifiers trained in train.py) predicts whether the question is
     answerable at all, and if so whether it expects a yes/no or an
     extractive answer, and which value (yes/no, or a text span),
  3. extracts candidate answer spans for extractive questions using a
     regex-based acronym/method-name detector run over the retrieved text
     -- deliberately NOT a generative model, so any produced span is
     guaranteed to be a literal substring of retrieved evidence (this is
     discussed as a faithfulness/limitation trade-off in the README).
"""
import re
from collections import Counter

from src.data_utils import tokenize

# Two negation-cue lists, selectable at run time (see set_negation_cues):
#
# GENERIC_NEGATION_CUES is a domain-agnostic list written WITHOUT looking at
# the benchmark. It is the default and is what the headline numbers in
# results/report.md use.
#
# BENCHMARK_TUNED_NEGATION_CUES is the original list from the first version
# of this project. It was assembled by reading the benchmark's yes/no
# questions and adding phrases ("single response", "entire long",
# "full before", ...) that each pick out exactly one gold-"No" question. That
# is feature engineering on the evaluation set -- a mild form of test-set
# leakage -- so it is kept ONLY as an ablation arm (results/report.md,
# "Negation-cue ablation") to quantify how much of the yes/no accuracy it
# was buying, and is never used for the headline results.
GENERIC_NEGATION_CUES = [
    "not ", "n't", "without", "rather than", "exclusively", "only", "never",
    "lack", "no longer", "instead of", "neither", "nor ",
]

BENCHMARK_TUNED_NEGATION_CUES = [
    "not ", "n't", "without", "rather than", "exclusively", "only",
    "identical", "lacks", "lack of", "never", "no longer", "restrict",
    "regardless of", "ignoring", "before the", "single response",
    "single reasoning", "entire long", "full before",
]

NEGATION_CUE_SETS = {
    "generic": GENERIC_NEGATION_CUES,
    "tuned": BENCHMARK_TUNED_NEGATION_CUES,
}

# Mutable module-level list consulted by extract_features(); swap its contents
# via set_negation_cues() rather than rebinding the name, so every module that
# imported it sees the change.
NEGATION_CUES = list(GENERIC_NEGATION_CUES)


def set_negation_cues(mode: str = "generic"):
    """Select which negation-cue list extract_features() uses."""
    if mode not in NEGATION_CUE_SETS:
        raise ValueError(f"unknown negation-cue set {mode!r}; choose from {sorted(NEGATION_CUE_SETS)}")
    NEGATION_CUES[:] = NEGATION_CUE_SETS[mode]
    return mode


AUX_START_RE = re.compile(r"^(does|is|did|has|do|isn't|doesn't|didn't)\b", re.I)

# Tokens that look like acronyms but are generic field vocabulary rather than
# the name of a method a paper introduces. NOTE: a few entries ("tnp", "tnps",
# "ptnp", "ptnps", "pro", "gd", "np", "nps") were added after inspecting
# corpus abstracts, i.e. the stoplist is tuned to this corpus. The ablation in
# results/report.md shows that removing them changes nothing for BM25 and
# only one extractive item for TF-IDF+LSA, but the tuning is disclosed here
# rather than hidden.
ACRONYM_STOPLIST = {
    "llm", "llms", "ai", "nlp", "rag", "gpu", "gpus", "rl", "ppo", "rlhf",
    "ptq", "sota", "qa", "icl", "api", "bert", "id", "ir", "lm", "lms",
    "tf", "idf", "sft", "gd", "np", "nps", "tnp", "tnps", "ptnp", "ptnps",
    "moe", "mmlu", "pro",
}

CANDIDATE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")

FEATURE_NAMES = [
    "top1_score", "margin", "top1_z", "overlap", "has_negation",
    "starts_aux", "q_len", "n_candidates",
]


def _looks_like_acronym(token: str) -> bool:
    if token.lower() in ACRONYM_STOPLIST:
        return False
    if len(token) < 2:
        return False
    upper_count = sum(1 for c in token if c.isupper())
    if token.isupper() and 2 <= len(token) <= 8:
        return True
    if upper_count >= 2:
        return True
    if re.match(r"^[a-z][A-Z]", token):  # e.g. "fDPO"
        return True
    return False


def extract_candidate_spans(text: str):
    tokens = CANDIDATE_RE.findall(text)
    candidates = [t for t in tokens if _looks_like_acronym(t)]
    counts = Counter(candidates)
    # stable order: by frequency desc, then first-occurrence position asc
    first_pos = {}
    for i, t in enumerate(candidates):
        first_pos.setdefault(t, i)
    ranked = sorted(counts.keys(), key=lambda t: (-counts[t], first_pos[t]))
    return ranked


def predict_extractive_span(top1_text: str):
    ranked = extract_candidate_spans(top1_text)
    return ranked[0] if ranked else ""


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("-", " ").split())


def exact_match(pred: str, gold: str) -> bool:
    """Strict scoring: case/whitespace/hyphen-insensitive string equality."""
    if not pred or not gold:
        return False
    return _norm(pred) == _norm(gold)


def fuzzy_match(pred: str, gold: str) -> bool:
    """Lenient scoring: either string is a substring of the other (case-
    insensitive). This is deliberately generous -- e.g. a predicted "DPR"
    matches gold "Dense Passage Retrieval (DPR)" -- so exact_match() is
    reported alongside it in results/report.md."""
    if not pred or not gold:
        return False
    p, g = pred.lower(), gold.lower()
    return p in g or g in p


def extract_features(question: str, retrieved, docs_by_id):
    """retrieved: list of (arxiv_id, score), already sorted desc, len>=1."""
    top1_id, top1_score = retrieved[0]
    top2_score = retrieved[1][1] if len(retrieved) > 1 else 0.0
    scores = [s for _, s in retrieved]
    mean_s = sum(scores) / len(scores)
    var_s = sum((s - mean_s) ** 2 for s in scores) / len(scores)
    std_s = var_s ** 0.5
    top1_z = (top1_score - mean_s) / (std_s + 1e-6)

    q_tokens = set(tokenize(question))
    top1_text = docs_by_id[top1_id]["text"]
    d_tokens = set(tokenize(top1_text))
    overlap = len(q_tokens & d_tokens) / (len(q_tokens | d_tokens) + 1e-6)

    q_lower = question.lower()
    has_negation = int(any(cue in q_lower for cue in NEGATION_CUES))
    starts_aux = int(bool(AUX_START_RE.match(question.strip())))
    q_len = len(question.split())
    n_candidates = len(extract_candidate_spans(top1_text))

    feats = {
        "top1_score": top1_score,
        "margin": top1_score - top2_score,
        "top1_z": top1_z,
        "overlap": overlap,
        "has_negation": has_negation,
        "starts_aux": starts_aux,
        "q_len": q_len,
        "n_candidates": n_candidates,
    }
    return [feats[name] for name in FEATURE_NAMES], top1_id
