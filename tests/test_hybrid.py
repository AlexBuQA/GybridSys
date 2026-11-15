from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import build_user_item_matrix
from src.models.base import BaseRecommender
from src.models.content import ContentBasedRecommender
from src.models.hybrid import HybridRecommender, HybridVariant
from src.models.popularity import PopularityRecommender


class FixedModel(BaseRecommender):
    def __init__(self, matrix, offset):
        self.train_matrix = matrix
        self.offset = offset

    def recommend(self, user_id: int, n: int = 5):
        seen = self.seen_books(user_id)
        return [(book, float(n - i + self.offset)) for i, book in enumerate(
            book for book in range(1, 7) if book not in seen
        )][:n]


def build_hybrid():
    train = pd.DataFrame({"user_id": [1, 1, 2, 2, 3], "book_id": [1, 2, 2, 3, 4], "rating": [5, 4, 4, 3, 5]})
    matrix = build_user_item_matrix(train, 3, 6)
    popularity = PopularityRecommender(min_ratings=1).fit(train, matrix)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(["fantasy magic", "fantasy dragon", "space ship", "space mars", "history war", "history rome"])
    content = ContentBasedRecommender(vectorizer, csr_matrix(tfidf), np.arange(1, 7), popularity)
    content.train_matrix = matrix
    models = {name: FixedModel(matrix, i) for i, name in enumerate(["popularity", "content", "item_cf", "svd", "two_tower"])}
    models["popularity"] = popularity
    return HybridRecommender(models, popularity, content, candidate_limit=10, diversity_lambda=0.7)


def test_candidate_union_and_mmr_are_valid() -> None:
    hybrid = build_hybrid()
    recommendations = hybrid.recommend(1, 3)
    ids = [book for book, _ in recommendations]
    assert len(ids) == len(set(ids)) == 3
    assert not {1, 2}.intersection(ids)
    candidates = hybrid.candidates(1)
    assert set(candidates.normalized_scores) == {"popularity", "content", "item_cf", "svd", "two_tower"}
    assert np.all((candidates.normalized_scores["content"] >= 0) & (candidates.normalized_scores["content"] <= 1))


def test_segments_ablation_and_cold_start() -> None:
    hybrid = build_hybrid()
    assert hybrid.segment_for_user(1) == "new"
    assert hybrid.segment_for_user(999) == "cold_start"
    assert hybrid.segment_for_user(999) == "cold_start"
    assert hybrid.recommend(999, 2, seed_book_id=1)[0][0] != 1
    variant = HybridVariant(hybrid, include_neural=False, diversify=False)
    assert len(variant.recommend(1, 3)) == 3
    hybrid.clear_cache()
    assert not hybrid._candidate_cache
