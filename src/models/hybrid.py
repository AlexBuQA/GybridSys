"""Кандидатный гибрид, сегментные веса, validation-тюнинг и MMR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import intra_list_diversity, ranking_metrics, segment_for_count

from .base import BaseRecommender, Recommendations
from .content import ContentBasedRecommender
from .popularity import PopularityRecommender

MODEL_NAMES = ["popularity", "content", "item_cf", "svd", "two_tower"]
DEFAULT_WEIGHTS = {
    "cold_start": {"popularity": 1.0, "content": 0.0, "item_cf": 0.0, "svd": 0.0, "two_tower": 0.0},
    "new": {"popularity": 0.20, "content": 0.50, "item_cf": 0.05, "svd": 0.10, "two_tower": 0.15},
    "warm": {"popularity": 0.10, "content": 0.30, "item_cf": 0.20, "svd": 0.20, "two_tower": 0.20},
    "active": {"popularity": 0.05, "content": 0.20, "item_cf": 0.25, "svd": 0.25, "two_tower": 0.25},
}


@dataclass
class CandidateSet:
    user_id: int
    book_ids: np.ndarray
    raw_scores: dict[str, np.ndarray]
    normalized_scores: dict[str, np.ndarray]


class HybridRecommender(BaseRecommender):
    def __init__(
        self,
        models: dict[str, object],
        popularity: PopularityRecommender,
        content: ContentBasedRecommender,
        candidate_limit: int = 60,
        diversity_lambda: float = 0.85,
        mmr_candidate_limit: int = 40,
    ) -> None:
        if candidate_limit < 10:
            raise ValueError("candidate_limit должен быть >= 10")
        self.models = models
        self.popularity = popularity
        self.content = content
        self.train_matrix = content.train_matrix
        self.candidate_limit = int(candidate_limit)
        self.diversity_lambda = float(diversity_lambda)
        self.mmr_candidate_limit = int(mmr_candidate_limit)
        self.weights = {segment: values.copy() for segment, values in DEFAULT_WEIGHTS.items()}
        self._candidate_cache: dict[tuple[int, int], CandidateSet] = {}

    def segment_for_user(self, user_id: int) -> str:
        return segment_for_count(len(self.seen_books(user_id)))

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if not finite.any():
            return np.zeros_like(values, dtype=np.float32)
        low, high = float(values[finite].min()), float(values[finite].max())
        if high - low < 1e-12:
            return np.ones_like(values, dtype=np.float32)
        result = (values - low) / (high - low)
        result[~finite] = 0.0
        return result.astype(np.float32)

    def candidates(self, user_id: int, limit: int | None = None) -> CandidateSet:
        limit = int(limit or self.candidate_limit)
        key = (int(user_id), limit)
        if key in self._candidate_cache:
            return self._candidate_cache[key]
        source_recommendations: dict[str, Recommendations] = {}
        all_books: set[int] = set()
        for name, model in self.models.items():
            recommendations = model.recommend(user_id=user_id, n=limit)
            source_recommendations[name] = recommendations
            all_books.update(book for book, _ in recommendations)
        book_ids = np.array(sorted(all_books), dtype=np.int32)
        raw, normalized = {}, {}
        for name, recommendations in source_recommendations.items():
            scores = np.zeros(len(book_ids), dtype=np.float32)
            positions = {int(book): i for i, book in enumerate(book_ids)}
            returned = np.array([score for _, score in recommendations], dtype=np.float32)
            scaled = self._minmax(returned)
            for (book, _), value, raw_value in zip(recommendations, scaled, returned):
                scores[positions[int(book)]] = value
            raw_values = np.zeros(len(book_ids), dtype=np.float32)
            for book, value in recommendations:
                raw_values[positions[int(book)]] = value
            raw[name], normalized[name] = raw_values, scores
        result = CandidateSet(int(user_id), book_ids, raw, normalized)
        self._candidate_cache[key] = result
        return result

    def fuse(self, candidates: CandidateSet, include_neural: bool = True) -> np.ndarray:
        segment = self.segment_for_user(candidates.user_id)
        weights = self.weights[segment]
        score = np.zeros(len(candidates.book_ids), dtype=np.float32)
        total = 0.0
        for name, values in candidates.normalized_scores.items():
            if name == "two_tower" and not include_neural:
                continue
            weight = float(weights.get(name, 0.0))
            score += weight * values
            total += weight
        return score / total if total > 0 else score

    def _mmr(self, candidates: CandidateSet, relevance: np.ndarray, n: int) -> list[int]:
        pool_size = min(len(candidates.book_ids), max(n, self.mmr_candidate_limit))
        pool = np.argsort(-relevance, kind="stable")[:pool_size]
        selected: list[int] = []
        remaining = set(int(i) for i in pool)
        content_indices = {
            int(i): self.content.id_to_index.get(int(candidates.book_ids[i])) for i in pool
        }
        while remaining and len(selected) < n:
            best_index, best_value = None, -np.inf
            for index in sorted(remaining):
                if not selected:
                    similarity = 0.0
                else:
                    current = content_indices[index]
                    previous = [content_indices[i] for i in selected if content_indices[i] is not None]
                    if current is None or not previous:
                        similarity = 0.0
                    else:
                        values = (
                            self.content.tfidf_matrix[current]
                            @ self.content.tfidf_matrix[previous].T
                        ).toarray().ravel()
                        similarity = float(np.max(values))
                objective = self.diversity_lambda * float(relevance[index]) - (
                    1.0 - self.diversity_lambda
                ) * similarity
                if objective > best_value + 1e-12:
                    best_index, best_value = index, objective
            selected.append(int(best_index))
            remaining.remove(int(best_index))
        return selected

    def recommend(
        self, user_id: int, n: int = 5, seed_book_id: int | None = None,
        include_neural: bool = True, diversify: bool = True,
    ) -> Recommendations:
        self.validate_n(n)
        if not self.seen_books(user_id):
            if seed_book_id is not None:
                return self.content.similar_books(seed_book_id, n)
            if user_id < 1 or user_id > self.train_matrix.shape[0]:
                return self.popularity.recommend(user_id, n)
        candidates = self.candidates(user_id)
        fused = self.fuse(candidates, include_neural=include_neural)
        if diversify:
            indices = self._mmr(candidates, fused, n)
        else:
            indices = np.argsort(-fused, kind="stable")[:n].tolist()
        result = [(int(candidates.book_ids[i]), float(fused[i])) for i in indices]
        return self.popularity.fill(result, user_id, n)

    def explain(self, user_id: int, n: int = 10) -> list[dict]:
        candidates = self.candidates(user_id)
        fused = self.fuse(candidates)
        recommendations = self.recommend(user_id, n)
        positions = {int(book): i for i, book in enumerate(candidates.book_ids)}
        segment = self.segment_for_user(user_id)
        rows = []
        for rank, (book, score) in enumerate(recommendations, start=1):
            row = {
                "rank": rank, "book_id": book, "hybrid_score": score,
                "segment": segment, "diversity_lambda": self.diversity_lambda,
            }
            index = positions.get(book)
            for name in self.models:
                row[f"{name}_score"] = float(candidates.raw_scores[name][index]) if index is not None else None
            row["applied_weights"] = self.weights[segment].copy()
            rows.append(row)
        return rows

    def clear_cache(self) -> None:
        self._candidate_cache = {}


class HybridVariant:
    """Ablation-обёртка над общим engine без повторного обучения."""

    def __init__(self, engine: HybridRecommender, include_neural: bool, diversify: bool) -> None:
        self.engine = engine
        self.include_neural = include_neural
        self.diversify = diversify

    def recommend(self, user_id: int, n: int = 5) -> Recommendations:
        return self.engine.recommend(
            user_id, n, include_neural=self.include_neural, diversify=self.diversify
        )


def _top_from_candidates(candidates: CandidateSet, fused: np.ndarray, k: int) -> list[int]:
    order = np.argsort(-fused, kind="stable")[:k]
    return candidates.book_ids[order].astype(int).tolist()


def tune_hybrid_weights(
    engine: HybridRecommender,
    validation: pd.DataFrame,
    k: int = 10,
    max_users_per_segment: int = 80,
    n_trials: int = 32,
    seed: int = 42,
) -> pd.DataFrame:
    """Ненормированные score сначала min-max, затем Dirichlet random search весов.

    Подбор выполняется только по nDCG@K на validation. Test здесь не доступен.
    """
    relevant = (
        validation[validation["rating"] >= 4].groupby("user_id")["book_id"]
        .agg(lambda values: set(int(v) for v in values)).to_dict()
    )
    rng = np.random.default_rng(seed + 101)
    selected: dict[str, list[int]] = {}
    for segment in ["new", "warm", "active"]:
        users = np.array(
            [user for user in sorted(relevant) if engine.segment_for_user(int(user)) == segment],
            dtype=np.int32,
        )
        if len(users) > max_users_per_segment:
            users = np.sort(rng.choice(users, max_users_per_segment, replace=False))
        selected[segment] = users.astype(int).tolist()
        for user in selected[segment]:
            engine.candidates(user)

    rows: list[dict] = []
    for segment, users in selected.items():
        if not users:
            continue
        candidates = [engine.candidates(user) for user in users]
        trials = [DEFAULT_WEIGHTS[segment]]
        trials.extend(
            {name: float(value) for name, value in zip(MODEL_NAMES, values)}
            for values in rng.dirichlet(np.ones(len(MODEL_NAMES)), size=n_trials)
        )
        # One-hot варианты позволяют честно обнулить слабые источники.
        for index, name in enumerate(MODEL_NAMES):
            trials.append({key: float(index == i) for i, key in enumerate(MODEL_NAMES)})
        for trial, weights in enumerate(trials):
            engine.weights[segment] = weights
            metrics = []
            for user, candidate_set in zip(users, candidates):
                ids = _top_from_candidates(candidate_set, engine.fuse(candidate_set), k)
                metrics.append(ranking_metrics(ids, relevant[user], k))
            row = {
                "segment": segment, "trial": trial, "users": len(users),
                "precision_at_k": float(np.mean([m["precision_at_k"] for m in metrics])),
                "recall_at_k": float(np.mean([m["recall_at_k"] for m in metrics])),
                "ndcg_at_k": float(np.mean([m["ndcg_at_k"] for m in metrics])),
                **{f"weight_{name}": weights[name] for name in MODEL_NAMES},
            }
            rows.append(row)
        segment_rows = [row for row in rows if row["segment"] == segment]
        best = max(segment_rows, key=lambda row: (row["ndcg_at_k"], row["recall_at_k"]))
        engine.weights[segment] = {name: best[f"weight_{name}"] for name in MODEL_NAMES}
        print(
            f"Веса {segment}: nDCG@{k}={best['ndcg_at_k']:.5f}, "
            + ", ".join(f"{name}={engine.weights[segment][name]:.3f}" for name in MODEL_NAMES),
            flush=True,
        )
    return pd.DataFrame(rows)


def tune_diversity_lambda(
    engine: HybridRecommender,
    validation: pd.DataFrame,
    k: int = 10,
    max_users: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Multi-objective выбор lambda: nDCG@K + 0.20 * ILD@K на validation.

    Коэффициент 0.20 задан заранее как продуктовый компромисс, а не подобран
    по test. При lambda=1 разнообразие не влияет на выдачу.
    """
    relevant = (
        validation[validation["rating"] >= 4].groupby("user_id")["book_id"]
        .agg(lambda values: set(int(v) for v in values)).to_dict()
    )
    users = np.array(sorted(relevant), dtype=np.int32)
    rng = np.random.default_rng(seed + 202)
    if len(users) > max_users:
        users = np.sort(rng.choice(users, max_users, replace=False))
    if not len(users):
        return pd.DataFrame(
            columns=["diversity_lambda", "ndcg_at_k", "ild_at_k", "validation_objective", "users"]
        )
    rows = []
    for value in [0.60, 0.70, 0.80, 0.85, 0.90, 1.0]:
        engine.diversity_lambda = value
        ndcg_values, diversity_values = [], []
        for user in users:
            ids = [book for book, _ in engine.recommend(int(user), k, diversify=True)]
            ndcg_values.append(ranking_metrics(ids, relevant[int(user)], k)["ndcg_at_k"])
            diversity_values.append(intra_list_diversity(ids, engine.content))
        ndcg = float(np.mean(ndcg_values))
        ild = float(np.mean(diversity_values))
        rows.append(
            {"diversity_lambda": value, "ndcg_at_k": ndcg, "ild_at_k": ild,
             "validation_objective": ndcg + 0.20 * ild, "users": len(users)}
        )
    result = pd.DataFrame(rows)
    best = result.loc[result["validation_objective"].idxmax()]
    engine.diversity_lambda = float(best["diversity_lambda"])
    print(f"MMR lambda={engine.diversity_lambda:.2f} (validation objective={best['validation_objective']:.5f})", flush=True)
    return result
