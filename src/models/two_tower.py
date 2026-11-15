"""Two-Tower нейросеть на NumPy: embeddings + dense/tanh, BPR и Adam.

Реализация не требует PyTorch/TensorFlow. Градиенты по обеим башням и
embedding-таблицам вычисляются явно, поэтому архитектуру удобно проверять.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import expit

from .base import BaseRecommender, Recommendations
from .popularity import PopularityRecommender


class TwoTowerRecommender(BaseRecommender):
    def __init__(
        self,
        train_matrix: csr_matrix,
        user_features: np.ndarray,
        item_features: np.ndarray,
        popularity: PopularityRecommender,
        embedding_dim: int = 16,
        output_dim: int = 32,
        seed: int = 42,
    ) -> None:
        self.train_matrix = train_matrix
        self.user_features = np.asarray(user_features, dtype=np.float32)
        self.item_features = np.asarray(item_features, dtype=np.float32)
        self.popularity = popularity
        self.embedding_dim = int(embedding_dim)
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        n_users, n_items = train_matrix.shape
        if self.user_features.shape[0] != n_users or self.item_features.shape[0] != n_items:
            raise ValueError("Размеры feature-матриц не совпадают с train_matrix")
        rng = np.random.default_rng(seed)
        self.user_embeddings = rng.normal(0, 0.05, (n_users, embedding_dim)).astype(np.float32)
        self.item_embeddings = rng.normal(0, 0.05, (n_items, embedding_dim)).astype(np.float32)
        self.W_user = rng.normal(
            0, np.sqrt(2 / (embedding_dim + self.user_features.shape[1])),
            (embedding_dim + self.user_features.shape[1], output_dim),
        ).astype(np.float32)
        self.W_item = rng.normal(
            0, np.sqrt(2 / (embedding_dim + self.item_features.shape[1])),
            (embedding_dim + self.item_features.shape[1], output_dim),
        ).astype(np.float32)
        self.b_user = np.zeros(output_dim, dtype=np.float32)
        self.b_item = np.zeros(output_dim, dtype=np.float32)
        self.params: dict[str, float | int] = {}
        self.history: list[dict[str, float]] = []
        self._item_vectors_cache: np.ndarray | None = None

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        train_matrix: csr_matrix,
        user_features: np.ndarray,
        item_features: np.ndarray,
        popularity: PopularityRecommender,
        embedding_dim: int = 16,
        output_dim: int = 32,
        epochs: int = 4,
        batch_size: int = 2048,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-6,
        seed: int = 42,
        verbose: bool = True,
    ) -> "TwoTowerRecommender":
        if epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("epochs/batch_size должны быть положительными, learning_rate > 0")
        model = cls(
            train_matrix, user_features, item_features, popularity,
            embedding_dim=embedding_dim, output_dim=output_dim, seed=seed,
        )
        positive = train[train["rating"] >= 4]
        if positive.empty:
            raise ValueError("Для BPR нужна хотя бы одна train-оценка >= 4")
        users = positive["user_id"].to_numpy(dtype=np.int32) - 1
        items = positive["book_id"].to_numpy(dtype=np.int32) - 1
        model._train_bpr(
            users, items, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=weight_decay, verbose=verbose,
        )
        model.params = {
            "architecture": "Two-Tower: trainable ID embeddings + dense/tanh feature projection",
            "objective": "Bayesian Personalized Ranking (BPR)",
            "optimizer": "Adam", "embedding_dim": embedding_dim, "output_dim": output_dim,
            "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "seed": seed,
            "positive_interactions": int(len(positive)),
        }
        return model

    def _user_forward(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([self.user_embeddings[indices], self.user_features[indices]], axis=1)
        output = np.tanh(x @ self.W_user + self.b_user)
        return output, x

    def _item_forward(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([self.item_embeddings[indices], self.item_features[indices]], axis=1)
        output = np.tanh(x @ self.W_item + self.b_item)
        return output, x

    def _sample_negatives(self, users: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        negatives = rng.integers(0, self.train_matrix.shape[1], size=len(users), dtype=np.int32)
        # Отрицательным примером не может быть любая train-оценка пользователя.
        for _ in range(20):
            seen = self.train_matrix[users, negatives].A1 > 0
            if not seen.any():
                break
            negatives[seen] = rng.integers(
                0, self.train_matrix.shape[1], size=int(seen.sum()), dtype=np.int32
            )
        return negatives

    def _train_bpr(
        self, users: np.ndarray, positives: np.ndarray, epochs: int, batch_size: int,
        learning_rate: float, weight_decay: float, verbose: bool,
    ) -> None:
        rng = np.random.default_rng(self.seed + 17)
        names = ["user_embeddings", "item_embeddings", "W_user", "W_item", "b_user", "b_item"]
        first_moment = {name: np.zeros_like(getattr(self, name)) for name in names}
        second_moment = {name: np.zeros_like(getattr(self, name)) for name in names}
        step = 0
        dim = self.embedding_dim
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(users))
            losses, accuracies = [], []
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                u, p = users[batch], positives[batch]
                n = self._sample_negatives(u, rng)
                up, xu = self._user_forward(u)
                vp, xp = self._item_forward(p)
                vn, xn = self._item_forward(n)
                difference = np.sum(up * vp, axis=1) - np.sum(up * vn, axis=1)
                loss = float(np.mean(np.logaddexp(0.0, -difference)))
                losses.append(loss)
                accuracies.append(float(np.mean(difference > 0)))
                batch_scale = 1.0 / len(batch)
                g = (expit(difference) - 1.0).astype(np.float32) * batch_scale

                d_up = g[:, None] * (vp - vn)
                d_vp = g[:, None] * up
                d_vn = -g[:, None] * up
                dz_u = d_up * (1.0 - up * up)
                dz_p = d_vp * (1.0 - vp * vp)
                dz_n = d_vn * (1.0 - vn * vn)
                # Все веса используются до Adam-обновления.
                grad_user_embedding = dz_u @ self.W_user[:dim].T
                grad_item_positive = dz_p @ self.W_item[:dim].T
                grad_item_negative = dz_n @ self.W_item[:dim].T
                gradients = {
                    "W_user": xu.T @ dz_u + weight_decay * self.W_user,
                    "W_item": xp.T @ dz_p + xn.T @ dz_n + weight_decay * self.W_item,
                    "b_user": dz_u.sum(axis=0), "b_item": dz_p.sum(axis=0) + dz_n.sum(axis=0),
                }
                grad_user_embedding += weight_decay * self.user_embeddings[u]
                grad_item_positive += weight_decay * self.item_embeddings[p]
                grad_item_negative += weight_decay * self.item_embeddings[n]
                user_embedding_update = np.zeros_like(self.user_embeddings)
                item_embedding_update = np.zeros_like(self.item_embeddings)
                np.add.at(user_embedding_update, u, grad_user_embedding)
                np.add.at(item_embedding_update, p, grad_item_positive)
                np.add.at(item_embedding_update, n, grad_item_negative)
                gradients["user_embeddings"] = user_embedding_update
                gradients["item_embeddings"] = item_embedding_update

                step += 1
                for name in names:
                    gradient = np.asarray(gradients[name], dtype=np.float32)
                    first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradient
                    second_moment[name] = 0.999 * second_moment[name] + 0.001 * gradient * gradient
                    corrected_m = first_moment[name] / (1.0 - 0.9 ** step)
                    corrected_v = second_moment[name] / (1.0 - 0.999 ** step)
                    parameter = getattr(self, name)
                    parameter -= learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)
                self._item_vectors_cache = None
            row = {
                "epoch": epoch, "train_bpr_loss": float(np.mean(losses)),
                "train_bpr_accuracy": float(np.mean(accuracies)),
            }
            self.history.append(row)
            if verbose:
                print(
                    f"Two-Tower epoch {epoch}/{epochs}: loss={row['train_bpr_loss']:.5f}, "
                    f"BPR accuracy={row['train_bpr_accuracy']:.4f}", flush=True,
                )

    def item_vectors(self) -> np.ndarray:
        if self._item_vectors_cache is None:
            vectors, _ = self._item_forward(np.arange(self.train_matrix.shape[1], dtype=np.int32))
            self._item_vectors_cache = vectors.astype(np.float32)
        return self._item_vectors_cache

    def user_vector(self, user_id: int) -> np.ndarray:
        if 1 <= user_id <= self.train_matrix.shape[0]:
            index = np.array([user_id - 1], dtype=np.int32)
            return self._user_forward(index)[0][0]
        # Для неизвестного ID: нулевой embedding и средние (стандартизованные) признаки.
        x = np.concatenate(
            [np.zeros(self.embedding_dim, dtype=np.float32), np.zeros(self.user_features.shape[1], dtype=np.float32)]
        )[None, :]
        return np.tanh(x @ self.W_user + self.b_user)[0]

    def score_all(self, user_id: int) -> np.ndarray:
        return self.item_vectors() @ self.user_vector(user_id)

    def recommend(self, user_id: int, n: int = 5) -> Recommendations:
        self.validate_n(n)
        if not 1 <= user_id <= self.train_matrix.shape[0]:
            return self.popularity.recommend(user_id, n)
        scores = self.score_all(user_id)
        seen = self.seen_books(user_id)
        if seen:
            scores[np.array(list(seen), dtype=np.int32) - 1] = -np.inf
        order = np.argsort(-scores, kind="stable")[:n]
        result = [(int(i + 1), float(scores[i])) for i in order if np.isfinite(scores[i])]
        return self.popularity.fill(result, user_id, n)
