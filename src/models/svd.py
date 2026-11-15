"""SVD из scikit-learn на смещённых остатках явных оценок."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .base import BaseRecommender, Recommendations
from .popularity import PopularityRecommender


class SVDRecommender(BaseRecommender):
    """TruncatedSVD + регуляризованные пользовательские и книжные смещения.

    Разложению передаются остатки только для наблюдаемых оценок, а не исходные
    нули CSR как будто это реальные нулевые рейтинги.
    """

    def __init__(
        self,
        global_mean: float,
        user_means: np.ndarray,
        item_biases: np.ndarray,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        known_users: np.ndarray,
        known_items: np.ndarray,
        train_matrix: csr_matrix,
        popularity: PopularityRecommender,
        params: dict | None = None,
        rmse: float | None = None,
    ) -> None:
        self.global_mean = float(global_mean)
        self.user_means = user_means.astype(np.float32)
        self.item_biases = item_biases.astype(np.float32)
        self.user_factors = user_factors.astype(np.float32)
        self.item_factors = item_factors.astype(np.float32)
        self.known_users = known_users.astype(bool)
        self.known_items = known_items.astype(bool)
        self.train_matrix = train_matrix
        self.popularity = popularity
        self.params = params or {}
        self.rmse = rmse

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        test: pd.DataFrame,
        train_matrix: csr_matrix,
        popularity: PopularityRecommender,
        n_factors: int = 32,
        n_iter: int = 7,
        shrinkage: float = 10.0,
        seed: int = 42,
        verbose: bool = True,
    ) -> "SVDRecommender":
        n_users, n_items = train_matrix.shape
        user_idx = train["user_id"].to_numpy(dtype=np.int32) - 1
        item_idx = train["book_id"].to_numpy(dtype=np.int32) - 1
        values = train["rating"].to_numpy(dtype=np.float32)
        global_mean = float(values.mean())

        user_counts = np.bincount(user_idx, minlength=n_users).astype(np.float32)
        user_sums = np.bincount(user_idx, weights=values, minlength=n_users).astype(np.float32)
        user_means = (user_sums + shrinkage * global_mean) / (user_counts + shrinkage)
        known_users = user_counts > 0

        user_residuals = values - user_means[user_idx]
        item_counts = np.bincount(item_idx, minlength=n_items).astype(np.float32)
        item_residual_sums = np.bincount(
            item_idx, weights=user_residuals, minlength=n_items
        ).astype(np.float32)
        item_biases = item_residual_sums / (item_counts + shrinkage)
        known_items = item_counts > 0

        residuals = values - user_means[user_idx] - item_biases[item_idx]
        residual_matrix = csr_matrix(
            (residuals, (user_idx, item_idx)), shape=(n_users, n_items), dtype=np.float32
        )
        components = max(1, min(n_factors, n_users, n_items - 1))
        if verbose:
            print(f"SVD: randomized TruncatedSVD, {components} факторов", flush=True)
        transformer = TruncatedSVD(
            n_components=components, algorithm="randomized", n_iter=n_iter, random_state=seed
        )
        user_factors = transformer.fit_transform(residual_matrix).astype(np.float32)
        item_factors = transformer.components_.T.astype(np.float32)

        params = {
            "library": "scikit-learn TruncatedSVD",
            "n_factors": int(components),
            "n_iter": n_iter,
            "shrinkage": shrinkage,
            "seed": seed,
            "explained_variance_ratio_sum": float(transformer.explained_variance_ratio_.sum()),
        }
        model = cls(
            global_mean,
            user_means,
            item_biases,
            user_factors,
            item_factors,
            known_users,
            known_items,
            train_matrix,
            popularity,
            params=params,
        )
        predictions = model.predict_many(test["user_id"].to_numpy(), test["book_id"].to_numpy())
        model.rmse = float(np.sqrt(np.mean((test["rating"].to_numpy() - predictions) ** 2)))
        return model

    def _score_all(self, user_id: int) -> np.ndarray:
        user_index = user_id - 1
        if user_id < 1 or user_id > len(self.known_users) or not self.known_users[user_index]:
            return np.array([], dtype=np.float32)
        estimates = (
            self.user_means[user_index]
            + self.item_biases
            + self.item_factors @ self.user_factors[user_index]
        )
        return np.clip(estimates, 1.0, 5.0).astype(np.float32)

    def predict_many(self, user_ids: np.ndarray, book_ids: np.ndarray) -> np.ndarray:
        user_idx = np.asarray(user_ids, dtype=np.int32) - 1
        item_idx = np.asarray(book_ids, dtype=np.int32) - 1
        if user_idx.shape != item_idx.shape:
            raise ValueError("user_ids и book_ids должны иметь одинаковую форму")
        if np.any(item_idx < 0) or np.any(item_idx >= len(self.item_biases)):
            raise ValueError("book_id вне диапазона каталога")
        valid_users = (user_idx >= 0) & (user_idx < len(self.known_users)) & self.known_users[
            np.clip(user_idx, 0, len(self.known_users) - 1)
        ]
        base = np.full(len(user_idx), self.global_mean, dtype=np.float32)
        base += self.item_biases[item_idx]
        if valid_users.any():
            base[valid_users] = self.user_means[user_idx[valid_users]] + self.item_biases[
                item_idx[valid_users]
            ]
            latent = np.einsum(
                "ij,ij->i", self.user_factors[user_idx[valid_users]], self.item_factors[item_idx[valid_users]]
            )
            base[valid_users] += latent.astype(np.float32)
        return np.clip(base, 1.0, 5.0)

    def predict_rating(self, user_id: int, book_id: int) -> float:
        return float(self.predict_many(np.array([user_id]), np.array([book_id]))[0])

    def recommend(self, user_id: int, n: int = 5) -> Recommendations:
        self.validate_n(n)
        scores = self._score_all(user_id)
        if not len(scores):
            return self.popularity.recommend(user_id, n)
        seen = self.seen_books(user_id)
        if seen:
            scores[np.array(list(seen), dtype=np.int32) - 1] = -np.inf
        order = np.argsort(-scores, kind="stable")[:n]
        result = [(int(index + 1), float(scores[index])) for index in order if np.isfinite(scores[index])]
        return self.popularity.fill(result, user_id, n)
