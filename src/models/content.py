"""Контентные рекомендации на TF-IDF-профилях книг."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import BaseRecommender, Recommendations
from .popularity import PopularityRecommender


def build_book_profiles(
    books: pd.DataFrame,
    book_tags: pd.DataFrame,
    tags: pd.DataFrame,
    max_tags: int = 30,
) -> pd.DataFrame:
    """Профиль = original_title + до max_tags популярных пользовательских тегов.

    book_tags.goodreads_book_id соответствует books.book_id, а не внутреннему id.
    """
    top_tags = book_tags.sort_values(
        ["goodreads_book_id", "count"], ascending=[True, False], kind="mergesort"
    ).groupby("goodreads_book_id", sort=False, as_index=False).head(max_tags)
    top_tags = top_tags.merge(tags[["tag_id", "tag_name"]], on="tag_id", how="left")
    top_tags = top_tags.dropna(subset=["tag_name"])

    def join_tags(group: pd.DataFrame) -> str:
        names = group["tag_name"].astype(str)
        return " ".join(re.sub(r"\s+", "-", name.strip()) for name in names if name.strip())

    tag_profiles = (
        top_tags.groupby("goodreads_book_id", sort=False)[["tag_name"]]
        .apply(join_tags, include_groups=False)
        .rename("tag_profile")
        .reset_index()
    )
    result = books[["id", "book_id", "original_title", "title"]].merge(
        tag_profiles, left_on="book_id", right_on="goodreads_book_id", how="left"
    )
    titles = result["original_title"].fillna(result["title"]).fillna("").astype(str)
    tag_text = result["tag_profile"].fillna("").astype(str)
    # Повторение названия повышает его вес относительно многочисленных полок.
    result["profile"] = (titles + " " + titles + " " + tag_text).str.strip()
    return result[["id", "profile"]].sort_values("id").reset_index(drop=True)


class ContentBasedRecommender(BaseRecommender):
    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        tfidf_matrix: spmatrix,
        book_ids: np.ndarray,
        popularity: PopularityRecommender | None = None,
    ) -> None:
        self.vectorizer = vectorizer
        self.tfidf_matrix = csr_matrix(tfidf_matrix, dtype=np.float32)
        self.book_ids = np.asarray(book_ids, dtype=np.int32)
        self.id_to_index = {int(book_id): i for i, book_id in enumerate(self.book_ids)}
        self.popularity = popularity

    @classmethod
    def fit(
        cls,
        books: pd.DataFrame,
        book_tags: pd.DataFrame,
        tags: pd.DataFrame,
        train_matrix: csr_matrix | None = None,
        popularity: PopularityRecommender | None = None,
        max_features: int = 30_000,
    ) -> "ContentBasedRecommender":
        profiles = build_book_profiles(books, book_tags, tags)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w[\w-]+\b",
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(profiles["profile"])
        model = cls(vectorizer, matrix, profiles["id"].to_numpy(), popularity=popularity)
        model.train_matrix = train_matrix
        return model

    def similar_books(self, book_id: int, n: int = 5) -> Recommendations:
        self.validate_n(n)
        if book_id not in self.id_to_index:
            raise KeyError(f"Книга с id={book_id} не найдена")
        index = self.id_to_index[book_id]
        scores = (self.tfidf_matrix @ self.tfidf_matrix[index].T).toarray().ravel()
        scores[index] = -np.inf
        nearest = np.argsort(-scores, kind="stable")[:n]
        return [(int(self.book_ids[i]), float(scores[i])) for i in nearest if np.isfinite(scores[i])]

    def _history_scores(self, history: dict[int, float]) -> np.ndarray | None:
        positive = [
            (self.id_to_index[int(book_id)], float(rating))
            for book_id, rating in history.items()
            if rating >= 4 and int(book_id) in self.id_to_index
        ]
        if not positive:
            return None
        indices = np.array([item[0] for item in positive], dtype=np.int32)
        weights = np.array([item[1] for item in positive], dtype=np.float32)
        profile = self.tfidf_matrix[indices].multiply(weights[:, None]).sum(axis=0)
        profile = normalize(csr_matrix(profile), norm="l2", axis=1)
        return (self.tfidf_matrix @ profile.T).toarray().ravel()

    def recommend(self, user_id: int, n: int = 5) -> Recommendations:
        self.validate_n(n)
        if self.popularity is None:
            raise RuntimeError("Для персональных контентных рекомендаций нужен Popularity fallback")
        history: dict[int, float] = {}
        if self.train_matrix is not None and 1 <= user_id <= self.train_matrix.shape[0]:
            row = self.train_matrix.getrow(user_id - 1)
            history = {int(i + 1): float(v) for i, v in zip(row.indices, row.data)}
        scores = self._history_scores(history)
        if scores is None:
            return self.popularity.recommend(user_id, n)

        seen = set(history)
        order = np.argsort(-scores, kind="stable")
        result: Recommendations = []
        for index in order:
            book_id = int(self.book_ids[index])
            if book_id not in seen:
                result.append((book_id, float(scores[index])))
                if len(result) == n:
                    break
        return self.popularity.fill(result, user_id, n)
