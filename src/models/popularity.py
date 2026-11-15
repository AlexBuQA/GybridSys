"""Неперсонализированный бейзлайн популярности."""

from __future__ import annotations

import pandas as pd
from scipy.sparse import csr_matrix

from .base import BaseRecommender, Recommendations


class PopularityRecommender(BaseRecommender):
    """Top-N по среднему рейтингу среди книг с достаточным числом оценок."""

    def __init__(self, min_ratings: int = 50) -> None:
        if min_ratings < 1:
            raise ValueError("min_ratings должно быть >= 1")
        self.min_ratings = min_ratings
        self.ranked_book_ids: list[int] = []
        self.scores: dict[int, float] = {}
        self.rating_counts: dict[int, int] = {}

    def fit(self, train: pd.DataFrame, train_matrix: csr_matrix | None = None) -> "PopularityRecommender":
        stats = train.groupby("book_id")["rating"].agg(["mean", "count"])
        eligible = stats[stats["count"] >= self.min_ratings].copy()
        eligible = eligible.sort_index(kind="mergesort").sort_values(
            ["mean", "count"], ascending=[False, False], kind="mergesort"
        )
        self.ranked_book_ids = eligible.index.astype(int).tolist()
        self.scores = {int(k): float(v) for k, v in eligible["mean"].items()}
        self.rating_counts = {int(k): int(v) for k, v in stats["count"].items()}
        self.train_matrix = train_matrix
        return self

    def recommend(self, user_id: int, n: int = 5) -> Recommendations:
        self.validate_n(n)
        if not self.ranked_book_ids:
            raise RuntimeError("Модель Popularity не обучена")
        return self.take_unseen(self.ranked_book_ids, self.scores, self.seen_books(user_id), n)

    def fill(self, recommendations: Recommendations, user_id: int, n: int) -> Recommendations:
        """Добавляет популярные книги, если персональная модель выдала мало результатов."""
        if len(recommendations) >= n:
            return recommendations[:n]
        selected = {book_id for book_id, _ in recommendations}
        seen = self.seen_books(user_id) | selected
        result = list(recommendations)
        for book_id in self.ranked_book_ids:
            if book_id not in seen:
                result.append((int(book_id), float(self.scores[book_id])))
                if len(result) == n:
                    break
        return result
