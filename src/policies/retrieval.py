"""TF-IDF retrieval policy — document ranking by query relevance.

Demonstrates that the BasePolicy interface, evaluation metrics, and CI
regression gates generalize across decision domains. Same harness,
different domain.
"""

import json
from pathlib import Path

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.policies.base import BasePolicy
from src.evaluation.naive import graded_ndcg_at_k, mrr, hit_rate_at_k

_DEFAULT_CORPUS = Path("tests/fixtures/retrieval_corpus.json")


class RetrievalPolicy(BasePolicy):
    """TF-IDF document retrieval with cosine similarity scoring."""

    def __init__(self, corpus_path: Path | None = None):
        self._corpus_path = corpus_path or _DEFAULT_CORPUS
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._doc_ids: list[int] = []
        self._doc_id_to_idx: dict[int, int] = {}

    def fit(self, train_data: pl.DataFrame) -> "RetrievalPolicy":
        """Build TF-IDF matrix from document corpus.

        Args:
            train_data: DataFrame with doc_id, title, text columns.
                        Used for the TF-IDF vocabulary.
        """
        doc_ids = train_data["doc_id"].to_list()
        titles = train_data["title"].to_list()
        texts = train_data["text"].to_list()

        # Combine title and text for richer TF-IDF representation
        combined = [f"{t} {txt}" for t, txt in zip(titles, texts)]

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = self._vectorizer.fit_transform(combined)
        self._doc_ids = doc_ids
        self._doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        return self

    def score(
        self, items: list[int], context: dict | None = None,
    ) -> list[tuple[int, float]]:
        """Rank documents by cosine similarity to query.

        Args:
            items: Document IDs to rank.
            context: Must contain {"query": str}.

        Returns:
            List of (doc_id, similarity_score) sorted descending.
        """
        if not context or "query" not in context:
            raise ValueError("context must contain 'query' key for retrieval")

        query = context["query"]
        query_vec = self._vectorizer.transform([query])
        sim = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        scored = []
        for doc_id in items:
            idx = self._doc_id_to_idx.get(doc_id)
            score = float(sim[idx]) if idx is not None else 0.0
            scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def evaluate(self, test_data: pl.DataFrame, k: int = 10) -> dict[str, float]:
        """Evaluate retrieval quality using corpus queries with graded relevance.

        Ignores test_data — uses the queries and relevance judgments from
        the corpus fixture. This is standard for retrieval evaluation
        (queries are the test set, not user interactions).
        """
        corpus = json.loads(self._corpus_path.read_text())
        queries = corpus["queries"]
        all_doc_ids = [d["id"] for d in corpus["documents"]]

        ndcg_scores = []
        mrr_scores = []
        hit_scores = []

        for q in queries:
            # Graded relevance for NDCG, binary set for MRR/HitRate
            grades = {int(doc_id): grade for doc_id, grade in q["relevant"].items()}
            relevant = set(grades.keys())
            ranked = self.score(all_doc_ids, context={"query": q["text"]})
            ranked_ids = [doc_id for doc_id, _ in ranked]

            ndcg_scores.append(graded_ndcg_at_k(ranked_ids, grades, k))
            mrr_scores.append(mrr(ranked_ids, relevant))
            hit_scores.append(hit_rate_at_k(ranked_ids, relevant, k))

        return {
            f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores),
            "mrr": sum(mrr_scores) / len(mrr_scores),
            f"hit_rate@{k}": sum(hit_scores) / len(hit_scores),
        }
