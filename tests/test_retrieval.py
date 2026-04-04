"""Tests for TF-IDF retrieval policy."""

import json
from pathlib import Path

import polars as pl
import pytest

from src.policies.retrieval import RetrievalPolicy


_CORPUS_PATH = Path("tests/fixtures/retrieval_corpus.json")


def _load_corpus() -> dict:
    return json.loads(_CORPUS_PATH.read_text())


def _make_train_data(corpus: dict) -> pl.DataFrame:
    """Build a pseudo-training DataFrame from the corpus for fit()."""
    rows = []
    for doc in corpus["documents"]:
        rows.append({
            "doc_id": doc["id"],
            "title": doc["title"],
            "text": doc["text"],
        })
    return pl.DataFrame(rows)


def test_retrieval_is_base_policy():
    """RetrievalPolicy extends BasePolicy."""
    from src.policies.base import BasePolicy
    assert issubclass(RetrievalPolicy, BasePolicy)


def test_fit_builds_tfidf_matrix():
    """fit() creates a TF-IDF matrix from the corpus."""
    corpus = _load_corpus()
    train = _make_train_data(corpus)
    policy = RetrievalPolicy().fit(train)
    assert policy._tfidf_matrix is not None
    assert policy._tfidf_matrix.shape[0] == len(corpus["documents"])


def test_score_returns_sorted_descending():
    """score() returns (doc_id, score) sorted descending."""
    corpus = _load_corpus()
    train = _make_train_data(corpus)
    policy = RetrievalPolicy().fit(train)

    doc_ids = [d["id"] for d in corpus["documents"]]
    scored = policy.score(doc_ids, context={"query": "machine learning"})

    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)
    assert len(scored) == len(doc_ids)


def test_score_relevant_docs_rank_higher():
    """Highly relevant documents should rank near the top."""
    corpus = _load_corpus()
    train = _make_train_data(corpus)
    policy = RetrievalPolicy().fit(train)

    doc_ids = [d["id"] for d in corpus["documents"]]

    # Query about gradient boosting — doc 12 should rank high
    scored = policy.score(doc_ids, context={"query": "gradient boosting XGBoost LightGBM"})
    top_5_ids = [doc_id for doc_id, _ in scored[:5]]
    assert 12 in top_5_ids


def test_score_without_query_raises():
    """score() without a query in context raises ValueError."""
    corpus = _load_corpus()
    train = _make_train_data(corpus)
    policy = RetrievalPolicy().fit(train)

    with pytest.raises(ValueError, match="query"):
        policy.score([1, 2, 3], context=None)


def test_evaluate_returns_metrics():
    """evaluate() returns NDCG, MRR, HitRate metrics."""
    corpus = _load_corpus()
    train = _make_train_data(corpus)
    policy = RetrievalPolicy(corpus_path=_CORPUS_PATH).fit(train)
    metrics = policy.evaluate(pl.DataFrame(), k=10)

    assert "ndcg@10" in metrics
    assert "mrr" in metrics
    assert "hit_rate@10" in metrics
    # Scores should be in realistic range, not trivially 0
    assert 0.3 < metrics["ndcg@10"] <= 1.0
    assert 0.3 < metrics["mrr"] <= 1.0


def test_unknown_doc_ids_get_zero_score():
    """Documents not in the corpus get score 0."""
    corpus = _load_corpus()
    train = _make_train_data(corpus)
    policy = RetrievalPolicy().fit(train)

    scored = policy.score([1, 9999], context={"query": "machine learning"})
    scores_dict = dict(scored)
    assert scores_dict[9999] == 0.0
    assert scores_dict[1] > 0.0
