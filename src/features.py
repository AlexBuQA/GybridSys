"""Расширенные признаки пользователей, книг и взаимодействий.

Все статистики оценок рассчитываются строго по train. Метаданные книг и теги
не содержат test-оценок и используются как внешнее описание каталога.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

USER_FEATURES = [
    "log_rating_count",
    "mean_rating",
    "rating_std",
    "positive_ratio",
    "negative_ratio",
    "rating_entropy",
    "activity_score",
]

TOPIC_PATTERNS = {
    "topic_fiction": r"fiction|literary|literature|novel",
    "topic_fantasy": r"fantasy|magic|dragon|wizard",
    "topic_science_fiction": r"science-fiction|sci-fi|space|cyberpunk|dystopian",
    "topic_romance": r"romance|romantic|love",
    "topic_mystery": r"mystery|thriller|crime|detective|suspense",
    "topic_history": r"history|historical|war|politics|biography|memoir",
    "topic_children": r"children|kids|picture-book|middle-grade",
    "topic_young_adult": r"young-adult|ya-|teen|coming-of-age",
    "topic_nonfiction": r"non-fiction|nonfiction|science|philosophy|psychology|self-help|business",
    "topic_comics": r"comic|graphic-novel|manga|superhero",
}
TOPIC_FEATURES = list(TOPIC_PATTERNS)
BOOK_FEATURES = [
    "log_rating_count",
    "bayesian_mean",
    "rating_std",
    "positive_ratio",
    "rating_entropy",
    "popularity_score",
    "global_log_ratings_count",
    "publication_year",
    *TOPIC_FEATURES,
]


def _rating_entropy(ratings: pd.DataFrame, key: str, size: int) -> np.ndarray:
    """Нормированная энтропия распределения оценок 1–5."""
    result = np.zeros(size, dtype=np.float32)
    counts = ratings.groupby([key, "rating"], observed=True).size()
    totals = counts.groupby(level=0).sum()
    probabilities = counts / totals
    entropy = -(probabilities * np.log2(probabilities)).groupby(level=0).sum() / np.log2(5)
    ids = entropy.index.to_numpy(dtype=np.int32) - 1
    valid = (ids >= 0) & (ids < size)
    result[ids[valid]] = entropy.to_numpy(dtype=np.float32)[valid]
    return result


def build_user_features(train: pd.DataFrame, n_users: int) -> pd.DataFrame:
    stats = train.groupby("user_id")["rating"].agg(["count", "mean", "std"]).reindex(
        range(1, n_users + 1)
    )
    counts = stats["count"].fillna(0).to_numpy(dtype=np.float32)
    means = stats["mean"].fillna(float(train["rating"].mean())).to_numpy(dtype=np.float32)
    stds = stats["std"].fillna(0).to_numpy(dtype=np.float32)
    positive = (
        train.assign(flag=train["rating"] >= 4)
        .groupby("user_id")["flag"].mean().reindex(range(1, n_users + 1)).fillna(0).to_numpy()
    )
    negative = (
        train.assign(flag=train["rating"] <= 2)
        .groupby("user_id")["flag"].mean().reindex(range(1, n_users + 1)).fillna(0).to_numpy()
    )
    max_count = max(float(counts.max()), 1.0)
    result = pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1, dtype=np.int32),
            "rating_count": counts.astype(np.int32),
            "log_rating_count": np.log1p(counts),
            "mean_rating": means,
            "rating_std": stds,
            "positive_ratio": positive,
            "negative_ratio": negative,
            "rating_entropy": _rating_entropy(train, "user_id", n_users),
            "activity_score": np.log1p(counts) / np.log1p(max_count),
        }
    )
    result["segment"] = pd.cut(
        result["rating_count"], bins=[-1, 0, 2, 9, np.inf],
        labels=["cold_start", "new", "warm", "active"],
    ).astype(str)
    return result


def build_book_features(
    train: pd.DataFrame,
    books: pd.DataFrame,
    book_tags: pd.DataFrame,
    tags: pd.DataFrame,
    n_items: int,
    shrinkage: float = 20.0,
) -> pd.DataFrame:
    """Статистики популярности/разнообразия и тематические доли тегов."""
    ids = pd.Index(range(1, n_items + 1), name="id")
    stats = train.groupby("book_id")["rating"].agg(["count", "mean", "std"]).reindex(ids)
    counts = stats["count"].fillna(0).to_numpy(dtype=np.float32)
    global_mean = float(train["rating"].mean())
    sums = train.groupby("book_id")["rating"].sum().reindex(ids).fillna(0).to_numpy()
    bayesian = (sums + shrinkage * global_mean) / (counts + shrinkage)
    positive = (
        train.assign(flag=train["rating"] >= 4)
        .groupby("book_id")["flag"].mean().reindex(ids).fillna(0).to_numpy()
    )
    max_count = max(float(counts.max()), 1.0)
    catalog = books.drop_duplicates("id").set_index("id").reindex(ids)
    publication_year = pd.to_numeric(catalog["original_publication_year"], errors="coerce")
    publication_year = publication_year.fillna(publication_year.median()).fillna(0).to_numpy()
    global_counts = pd.to_numeric(catalog["ratings_count"], errors="coerce").fillna(0).to_numpy()

    result = pd.DataFrame(
        {
            "book_id": np.arange(1, n_items + 1, dtype=np.int32),
            "rating_count": counts.astype(np.int32),
            "log_rating_count": np.log1p(counts),
            "bayesian_mean": bayesian,
            "rating_std": stats["std"].fillna(0).to_numpy(dtype=np.float32),
            "positive_ratio": positive,
            "rating_entropy": _rating_entropy(train, "book_id", n_items),
            "popularity_score": np.log1p(counts) / np.log1p(max_count),
            "global_log_ratings_count": np.log1p(global_counts),
            "publication_year": publication_year,
        }
    )

    # Внешний Goodreads ID связывается с books.book_id, внутренний — с books.id.
    tagged = book_tags.merge(
        books[["book_id", "id"]], left_on="goodreads_book_id", right_on="book_id", how="inner"
    ).merge(tags[["tag_id", "tag_name"]], on="tag_id", how="inner")
    names = tagged["tag_name"].fillna("").astype(str).str.lower()
    counts_by_topic: dict[str, pd.Series] = {}
    for topic, pattern in TOPIC_PATTERNS.items():
        matched = tagged.loc[names.str.contains(pattern, regex=True, na=False)]
        counts_by_topic[topic] = matched.groupby("id")["count"].sum()
    for topic, values in counts_by_topic.items():
        result[topic] = result["book_id"].map(values).fillna(0).astype(np.float32)
    topic_sum = result[TOPIC_FEATURES].sum(axis=1)
    # Сглаживание сохраняет нулевой профиль у книги без тематических тегов.
    result[TOPIC_FEATURES] = result[TOPIC_FEATURES].div(topic_sum + 1.0, axis=0)
    return result


def scale_feature_arrays(
    user_features: pd.DataFrame, book_features: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Стандартизация без утечки: параметры берутся из train-признаков."""
    user = user_features[USER_FEATURES].to_numpy(dtype=np.float32)
    book = book_features[BOOK_FEATURES].to_numpy(dtype=np.float32)
    user_mean, book_mean = user.mean(axis=0), book.mean(axis=0)
    user_std = np.where(user.std(axis=0) < 1e-6, 1.0, user.std(axis=0))
    book_std = np.where(book.std(axis=0) < 1e-6, 1.0, book.std(axis=0))
    scaler = {
        "user_mean": user_mean.tolist(), "user_std": user_std.tolist(),
        "book_mean": book_mean.tolist(), "book_std": book_std.tolist(),
        "user_columns": USER_FEATURES, "book_columns": BOOK_FEATURES,
    }
    return (user - user_mean) / user_std, (book - book_mean) / book_std, scaler


@dataclass
class InteractionFeatureBuilder:
    """Признаки пары пользователь–кандидат для анализа и объяснений."""

    train_matrix: csr_matrix
    user_features: pd.DataFrame
    book_features: pd.DataFrame

    def build(
        self,
        user_id: int,
        book_ids: list[int],
        content_scores: dict[int, float] | None = None,
        item_cf_scores: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        content_scores = content_scores or {}
        item_cf_scores = item_cf_scores or {}
        topics = self.book_features.set_index("book_id")[TOPIC_FEATURES]
        history: dict[int, float] = {}
        if 1 <= user_id <= self.train_matrix.shape[0]:
            row = self.train_matrix.getrow(user_id - 1)
            history = {int(i + 1): float(v) for i, v in zip(row.indices, row.data)}
        positive = [(book, rating) for book, rating in history.items() if rating >= 4]
        if not positive:
            positive = list(history.items())
        affinity = np.zeros(len(book_ids), dtype=np.float32)
        if positive:
            weights = np.array([rating for _, rating in positive], dtype=np.float32)
            profile = np.average(
                topics.reindex([book for book, _ in positive]).fillna(0).to_numpy(),
                axis=0, weights=weights,
            )
            affinity = topics.reindex(book_ids).fillna(0).to_numpy() @ profile
        user_mean = float(self.user_features.iloc[user_id - 1]["mean_rating"]) if 1 <= user_id <= len(self.user_features) else 0.0
        return pd.DataFrame(
            {
                "user_id": user_id,
                "book_id": book_ids,
                "content_history_similarity": [content_scores.get(book, 0.0) for book in book_ids],
                "item_cf_score": [item_cf_scores.get(book, 0.0) for book in book_ids],
                "topic_affinity": affinity,
                "user_mean_rating": user_mean,
                "book_log_popularity": self.book_features.set_index("book_id")["log_rating_count"].reindex(book_ids).fillna(0).to_numpy(),
            }
        )
