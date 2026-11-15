"""Загрузка, очистка и разбиение данных Goodbooks-10k."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

RATING_COLUMNS = ["user_id", "book_id", "rating"]


def load_source_data(data_dir: str | Path, include_to_read: bool = False) -> dict[str, pd.DataFrame]:
    """Читает исходные CSV-файлы без изменения исходников."""
    data_dir = Path(data_dir)
    required = ["ratings", "books", "tags", "book_tags"]
    if include_to_read:
        required.append("to_read")

    missing = [name for name in required if not (data_dir / f"{name}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Не найдены файлы в {data_dir}: {', '.join(missing)}")

    data = {
        "ratings": pd.read_csv(
            data_dir / "ratings.csv",
            dtype={"user_id": np.int32, "book_id": np.int32, "rating": np.int8},
        ),
        "books": pd.read_csv(data_dir / "books.csv", low_memory=False),
        "tags": pd.read_csv(
            data_dir / "tags.csv", dtype={"tag_id": np.int32, "tag_name": "string"}
        ),
        "book_tags": pd.read_csv(
            data_dir / "book_tags.csv",
            dtype={"goodreads_book_id": np.int32, "tag_id": np.int32, "count": np.int32},
        ),
    }
    if include_to_read:
        data["to_read"] = pd.read_csv(
            data_dir / "to_read.csv", dtype={"user_id": np.int32, "book_id": np.int32}
        )
    return data


def clean_ratings(ratings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Удаляет дубли user_id/book_id до split, чтобы исключить утечку в тест.

    В файле нет временных меток, поэтому детерминированно сохраняется последняя
    строка из исходного порядка. Оценки не усредняются: шкала 1–5 порядковая.
    """
    if list(ratings.columns) != RATING_COLUMNS:
        ratings = ratings[RATING_COLUMNS]
    if ratings[RATING_COLUMNS].isna().any().any():
        raise ValueError("В ratings.csv есть пропуски в user_id, book_id или rating")
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Оценки должны находиться в диапазоне от 1 до 5")

    raw_count = len(ratings)
    unique_users = int(ratings["user_id"].nunique())
    unique_books = int(ratings["book_id"].nunique())
    cleaned = ratings.drop_duplicates(["user_id", "book_id"], keep="last").reset_index(drop=True)
    cleaned = cleaned.astype({"user_id": np.int32, "book_id": np.int32, "rating": np.int8})
    stats = {
        "raw_ratings": int(raw_count),
        "clean_ratings": int(len(cleaned)),
        "duplicates_removed": int(raw_count - len(cleaned)),
        "users": unique_users,
        "rated_books": unique_books,
    }
    return cleaned, stats


def user_holdout_split(
    ratings: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Случайный holdout внутри каждого пользователя; минимум 1 оценка в test.

    У каждого пользователя остаётся хотя бы одна строка в train. Разбиение
    воспроизводимо для фиксированного порядка строк и seed.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size должен быть между 0 и 1")
    if ratings.empty:
        raise ValueError("Нельзя разбить пустой набор оценок")

    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(ratings), dtype=bool)
    groups = ratings.groupby("user_id", sort=False).indices
    for indices in groups.values():
        n_test = min(len(indices) - 1, max(1, int(np.ceil(len(indices) * test_size))))
        if n_test <= 0:
            continue
        chosen = rng.permutation(indices)[:n_test]
        test_mask[chosen] = True

    train = ratings.loc[~test_mask].reset_index(drop=True)
    test = ratings.loc[test_mask].reset_index(drop=True)
    if set(map(tuple, train[["user_id", "book_id"]].to_numpy())).intersection(
        map(tuple, test[["user_id", "book_id"]].to_numpy())
    ):
        raise RuntimeError("Обнаружено пересечение пар user_id/book_id между train и test")
    return train, test


def build_user_item_matrix(
    ratings: pd.DataFrame, n_users: int, n_items: int = 10_000
) -> csr_matrix:
    """Создаёт CSR-матрицу явных оценок; отсутствующие значения равны нулю."""
    if ratings.duplicated(["user_id", "book_id"]).any():
        raise ValueError("Матрица требует уникальные пары user_id/book_id")
    if ratings.empty:
        return csr_matrix((n_users, n_items), dtype=np.float32)
    if ratings["user_id"].max() > n_users or ratings["book_id"].max() > n_items:
        raise ValueError("Идентификатор выходит за размеры матрицы")

    rows = ratings["user_id"].to_numpy(dtype=np.int32) - 1
    cols = ratings["book_id"].to_numpy(dtype=np.int32) - 1
    values = ratings["rating"].to_numpy(dtype=np.float32)
    matrix = csr_matrix((values, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    matrix.sort_indices()
    return matrix


def get_user_history(matrix: csr_matrix, user_id: int) -> dict[int, float]:
    """Возвращает историю в внешних ID книг (1..10000)."""
    if user_id < 1 or user_id > matrix.shape[0]:
        return {}
    row = matrix.getrow(user_id - 1)
    return {int(book + 1): float(value) for book, value in zip(row.indices, row.data)}
