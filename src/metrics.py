"""Корректные ranking-метрики, разнообразие и сегментная оценка."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true и y_pred должны быть непустыми массивами одинаковой формы")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k < 1:
        raise ValueError("k должно быть положительным")
    hits = len(set(dict.fromkeys(recommended[:k])).intersection(relevant))
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k < 1:
        raise ValueError("k должно быть положительным")
    if not relevant:
        return 0.0
    hits = len(set(dict.fromkeys(recommended[:k])).intersection(relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Бинарный nDCG@K с IDCG по min(|Relevant|, K) книгам."""
    if k < 1:
        raise ValueError("k должно быть положительным")
    unique = list(dict.fromkeys(recommended[:k]))
    dcg = sum(1.0 / math.log2(rank + 2) for rank, book in enumerate(unique) if book in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def ranking_metrics(recommended: list[int], relevant: set[int], k: int) -> dict[str, float]:
    unique = list(dict.fromkeys(recommended[:k]))
    return {
        "hits": float(len(set(unique).intersection(relevant))),
        "precision_at_k": precision_at_k(unique, relevant, k),
        "recall_at_k": recall_at_k(unique, relevant, k),
        "ndcg_at_k": ndcg_at_k(unique, relevant, k),
    }


def intra_list_diversity(book_ids: list[int], content: object) -> float:
    """Среднее 1-cosine для всех пар рекомендаций; 0 для списка <2."""
    ids = list(dict.fromkeys(book_ids))
    if len(ids) < 2:
        return 0.0
    indices = [content.id_to_index[book] for book in ids if book in content.id_to_index]
    if len(indices) < 2:
        return 0.0
    matrix = content.tfidf_matrix[indices]
    similarity = (matrix @ matrix.T).toarray()
    distances = 1.0 - np.clip(similarity, 0.0, 1.0)
    return float(distances[np.triu_indices(len(indices), k=1)].mean())


def segment_for_count(count: int) -> str:
    if count == 0:
        return "cold_start"
    if count <= 2:
        return "new"
    if count <= 9:
        return "warm"
    return "active"


def evaluate_recommenders(
    models: dict[str, object],
    test: pd.DataFrame,
    train: pd.DataFrame,
    k: int = 10,
    max_users: int = 600,
    seed: int = 42,
    content: object | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Одинаковая выборка пользователей для всех моделей и макроусреднение."""
    relevant = (
        test[test["rating"] >= 4].groupby("user_id")["book_id"]
        .agg(lambda values: set(int(v) for v in values)).to_dict()
    )
    eligible = np.array(sorted(relevant), dtype=np.int32)
    if not len(eligible):
        raise ValueError("В test нет пользователей с релевантными оценками")
    rng = np.random.default_rng(seed)
    if len(eligible) > max_users:
        eligible = np.sort(rng.choice(eligible, size=max_users, replace=False))
    counts = train.groupby("user_id").size().to_dict()
    popularity = train.groupby("book_id").size().to_dict()
    total = max(len(train), 1)

    summary_rows: list[dict] = []
    all_details: list[dict] = []
    for name, model in models.items():
        print(f"Оценка {name}: {len(eligible)} пользователей", flush=True)
        rows: list[dict] = []
        catalog: set[int] = set()
        for user_id in eligible:
            user_id = int(user_id)
            recommendations = model.recommend(user_id=user_id, n=k)
            ids = [int(book) for book, _ in recommendations]
            values = ranking_metrics(ids, relevant[user_id], k)
            diversity = intra_list_diversity(ids, content) if content is not None else np.nan
            novelty = float(np.mean([-np.log2((popularity.get(book, 0) + 1) / (total + 1)) for book in ids])) if ids else 0.0
            row = {
                "model": name, "user_id": user_id,
                "segment": segment_for_count(int(counts.get(user_id, 0))),
                **values, "ild_at_k": diversity, "novelty_at_k": novelty,
            }
            rows.append(row)
            catalog.update(ids[:k])
        all_details.extend(rows)
        details = pd.DataFrame(rows)
        std = float(details["ndcg_at_k"].std(ddof=1)) if len(details) > 1 else 0.0
        margin = 1.96 * std / np.sqrt(len(details))
        summary_rows.append(
            {
                "model": name,
                "precision_at_k": float(details["precision_at_k"].mean()),
                "recall_at_k": float(details["recall_at_k"].mean()),
                "ndcg_at_k": float(details["ndcg_at_k"].mean()),
                "ndcg_ci95_low": max(0.0, float(details["ndcg_at_k"].mean() - margin)),
                "ndcg_ci95_high": min(1.0, float(details["ndcg_at_k"].mean() + margin)),
                "ild_at_k": float(details["ild_at_k"].mean()),
                "novelty_at_k": float(details["novelty_at_k"].mean()),
                "catalog_coverage": len(catalog) / (content.tfidf_matrix.shape[0] if content is not None else 1),
                "evaluated_users": len(eligible), "k": k,
                "rmse": getattr(model, "rmse", np.nan),
            }
        )
    details_frame = pd.DataFrame(all_details)
    segment_metrics = (
        details_frame.groupby(["model", "segment"], as_index=False)
        .agg(
            precision_at_k=("precision_at_k", "mean"), recall_at_k=("recall_at_k", "mean"),
            ndcg_at_k=("ndcg_at_k", "mean"), ild_at_k=("ild_at_k", "mean"),
            users=("user_id", "count"),
        )
    )
    return pd.DataFrame(summary_rows), details_frame, segment_metrics
