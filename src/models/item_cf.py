"""Item-Based Collaborative Filtering с разреженным top-K сходством."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .base import BaseRecommender, Recommendations
from .popularity import PopularityRecommender


class ItemCFRecommender(BaseRecommender):
    """Косинусная близость книг по векторам явных оценок пользователей.

    Хранится только n_neighbors соседей на книгу, а не плотная матрица B x B.
    """

    def __init__(
        self,
        similarity: csr_matrix,
        train_matrix: csr_matrix,
        popularity: PopularityRecommender,
        n_neighbors: int = 50,
    ) -> None:
        self.similarity = similarity.tocsr().astype(np.float32)
        self.train_matrix = train_matrix
        self.popularity = popularity
        self.n_neighbors = n_neighbors
        self.global_mean = float(train_matrix.data.mean()) if train_matrix.nnz else 3.0

    @classmethod
    def fit(
        cls,
        train_matrix: csr_matrix,
        popularity: PopularityRecommender,
        n_neighbors: int = 50,
        chunk_size: int = 128,
        verbose: bool = True,
    ) -> "ItemCFRecommender":
        n_items = train_matrix.shape[1]
        if not 1 <= n_neighbors < n_items:
            raise ValueError("n_neighbors должно быть в диапазоне [1, n_items)")
        item_user = normalize(train_matrix.T.tocsr(), norm="l2", axis=1, copy=True)
        item_user_t = item_user.T.tocsr()
        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        for start in range(0, n_items, chunk_size):
            end = min(start + chunk_size, n_items)
            block = (item_user[start:end] @ item_user_t).toarray().astype(np.float32)
            block[np.arange(end - start), np.arange(start, end)] = 0.0
            candidate_cols = np.argpartition(block, -n_neighbors, axis=1)[:, -n_neighbors:]
            for local_row in range(end - start):
                cols = candidate_cols[local_row]
                values = block[local_row, cols]
                positive = values > 1e-12
                cols, values = cols[positive], values[positive]
                if len(cols):
                    order = np.lexsort((cols, -values))
                    cols, values = cols[order], values[order]
                    row_parts.append(np.full(len(cols), start + local_row, dtype=np.int32))
                    col_parts.append(cols.astype(np.int32))
                    value_parts.append(values.astype(np.float32))
            if verbose and (start // chunk_size) % 20 == 0:
                print(f"Item-CF similarity: обработано {end}/{n_items} книг", flush=True)

        rows = np.concatenate(row_parts) if row_parts else np.array([], dtype=np.int32)
        cols = np.concatenate(col_parts) if col_parts else np.array([], dtype=np.int32)
        values = np.concatenate(value_parts) if value_parts else np.array([], dtype=np.float32)
        similarity = csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
        similarity.sort_indices()
        return cls(similarity, train_matrix, popularity, n_neighbors=n_neighbors)

    def predict_rating(self, user_id: int, book_id: int, k: int | None = None) -> float:
        """Взвешенное среднее оценок пользователя по K похожим книгам."""
        if not 1 <= user_id <= self.train_matrix.shape[0]:
            return self.popularity.scores.get(book_id, self.global_mean)
        if not 1 <= book_id <= self.train_matrix.shape[1]:
            raise KeyError(f"Книга с id={book_id} не найдена")
        limit = min(k or self.n_neighbors, self.n_neighbors)
        if limit < 1:
            raise ValueError("k должно быть положительным")

        neighbors = self.similarity.getrow(book_id - 1)
        user = self.train_matrix.getrow(user_id - 1)
        user_ratings = {int(index): float(value) for index, value in zip(user.indices, user.data)}
        weighted_sum, weight_sum, used = 0.0, 0.0, 0
        pairs = sorted(zip(neighbors.indices, neighbors.data), key=lambda x: -x[1])
        for neighbor, similarity in pairs:
            if int(neighbor) in user_ratings:
                weighted_sum += float(similarity) * user_ratings[int(neighbor)]
                weight_sum += abs(float(similarity))
                used += 1
                if used == limit:
                    break
        if weight_sum > 0:
            return weighted_sum / weight_sum
        return self.popularity.scores.get(book_id, self.global_mean)

    def recommend(self, user_id: int, n: int = 5) -> Recommendations:
        self.validate_n(n)
        if not 1 <= user_id <= self.train_matrix.shape[0]:
            return self.popularity.recommend(user_id, n)
        user = self.train_matrix.getrow(user_id - 1)
        if user.nnz == 0:
            return self.popularity.recommend(user_id, n)

        numerator = (user @ self.similarity.T).toarray().ravel()
        binary = user.copy()
        binary.data = np.ones_like(binary.data)
        denominator = (binary @ self.similarity.T).toarray().ravel()
        scores = np.full(self.train_matrix.shape[1], -np.inf, dtype=np.float32)
        np.divide(numerator, denominator, out=scores, where=denominator > 1e-12)
        scores[user.indices] = -np.inf
        valid = np.flatnonzero(np.isfinite(scores))
        if len(valid):
            order = valid[np.argsort(-scores[valid], kind="stable")[:n]]
            result = [(int(index + 1), float(scores[index])) for index in order]
        else:
            result = []
        return self.popularity.fill(result, user_id, n)
