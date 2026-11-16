"""Loading and basic preprocessing for the corpus and QA benchmark."""
import json
import re
from pathlib import Path

STOPWORDS = set("""
a an the of to in on for and or is are was were be been being this that these those
with as by at from into over under between among it its it's their his her our your
we you they he she which what who whom does do did doesn't don't didn't not no nor
""".split())

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")


def tokenize(text: str, drop_stopwords: bool = True):
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def load_corpus(path):
    with open(path) as f:
        docs = json.load(f)
    for d in docs:
        d["text"] = d["title"] + ". " + d["abstract"]
    return docs


def load_qa(path):
    with open(path) as f:
        return json.load(f)


def corpus_index_by_id(docs):
    return {d["arxiv_id"]: d for d in docs}


def load_config(path):
    """Read configs/default.yaml (or another YAML file). CLI flags in
    train.py / evaluate.py override whatever is set here."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("retriever", "bm25")
    cfg.setdefault("seed", 42)
    cfg.setdefault("top_k", 5)
    cfg.setdefault("n_splits", 5)
    cfg.setdefault("ece_bins", 10)
    cfg.setdefault("negation_cues", "generic")
    cfg.setdefault("tfidf_lsa", {}).setdefault("n_components", 16)
    cfg.setdefault("robustness", {})
    cfg["robustness"].setdefault("seeds", list(range(10)))
    cfg["robustness"].setdefault("perturbation_rates", [0.1, 0.2, 0.3])
    cfg["robustness"].setdefault("n_perturbations", 20)
    cfg.setdefault("data", {})
    cfg["data"].setdefault("corpus", "data/corpus.json")
    cfg["data"].setdefault("qa_benchmark", "data/qa_benchmark.json")
    return cfg


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    docs = load_corpus(root / "data" / "corpus.json")
    qa = load_qa(root / "data" / "qa_benchmark.json")
    print(f"{len(docs)} documents, {len(qa)} QA items")
    by_type = {}
    for q in qa:
        by_type[q["answer_type"]] = by_type.get(q["answer_type"], 0) + 1
    print("QA type breakdown:", by_type)
