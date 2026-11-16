import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import load_corpus, load_qa, corpus_index_by_id  # noqa: E402


@pytest.fixture(scope="session")
def docs():
    return load_corpus(ROOT / "data" / "corpus.json")


@pytest.fixture(scope="session")
def qa():
    return load_qa(ROOT / "data" / "qa_benchmark.json")


@pytest.fixture(scope="session")
def docs_by_id(docs):
    return corpus_index_by_id(docs)
