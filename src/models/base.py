"""Общие функции рекомендателей."""

from __future__ import annotations

from scipy.sparse import csr_matrix

Recommendations = list[tuple[int, float]]


class BaseRecommender:
    """Хранит train-матрицу и исключает уже оценённые книги из выдачи."""

    train_matrix: csr_matrix | None = None

    def set_train_matrix(self, matrix: csr_matrix) -> None:
        self.train_matrix = matrix

    def seen_books(self, user_id: int) -> set[int]:
        if self.train_matrix is None or user_id < 1 or user_id > self.train_matrix.shape[0]:
            return set()
        start, end = self.train_matrix.indptr[user_id - 1], self.train_matrix.indptr[user_id]
        return set((self.train_matrix.indices[start:end] + 1).tolist())

    @staticmethod
    def validate_n(n: int) -> None:
        if n < 1:
            raise ValueError("N должно быть положительным")

    @staticmethod
    def take_unseen(
        ranked_book_ids: list[int] | tuple[int, ...], scores: dict[int, float], seen: set[int], n: int
    ) -> Recommendations:
        result: Recommendations = []
        for book_id in ranked_book_ids:
            if book_id not in seen:
                result.append((int(book_id), float(scores.get(book_id, 0.0))))
                if len(result) == n:
                    break
        return result
