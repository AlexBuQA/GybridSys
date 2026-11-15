"""Разбиение holdout на validation и test без пересечений."""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_validation_test(
    holdout: pd.DataFrame, validation_fraction: float = 0.5, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит исходный holdout внутри пользователя.

    Train не изменяется, поэтому можно безопасно переиспользовать классические
    модели из ДЗ. При одной holdout-оценке она остаётся в test: такой пользователь
    не участвует в подборе весов, но может участвовать в финальной оценке.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction должен быть между 0 и 1")
    required = {"user_id", "book_id", "rating"}
    if not required.issubset(holdout.columns):
        raise ValueError(f"Нужны колонки {sorted(required)}")

    rng = np.random.default_rng(seed + 1)
    validation_mask = np.zeros(len(holdout), dtype=bool)
    for indices in holdout.groupby("user_id", sort=False).indices.values():
        if len(indices) < 2:
            continue
        n_validation = min(
            len(indices) - 1, max(1, int(np.ceil(len(indices) * validation_fraction)))
        )
        chosen = rng.permutation(indices)[:n_validation]
        validation_mask[chosen] = True

    validation = holdout.loc[validation_mask].reset_index(drop=True)
    test = holdout.loc[~validation_mask].reset_index(drop=True)
    train_pairs = set(map(tuple, validation[["user_id", "book_id"]].to_numpy()))
    test_pairs = set(map(tuple, test[["user_id", "book_id"]].to_numpy()))
    if train_pairs.intersection(test_pairs):
        raise RuntimeError("Validation и test пересекаются")
    return validation, test


def user_segment(user_id: int, train: pd.DataFrame) -> str:
    """Возвращает сегмент по числу train-взаимодействий."""
    count = int((train["user_id"] == user_id).sum())
    if count == 0:
        return "cold_start"
    if count <= 2:
        return "new"
    if count <= 9:
        return "warm"
    return "active"
