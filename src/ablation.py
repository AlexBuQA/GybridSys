"""Изолированная оценка вклада пользовательских и книжных признаков Two-Tower."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.metrics import evaluate_recommenders
from src.models.two_tower import TwoTowerRecommender

ID_ONLY = "Two-Tower ID only"
USER_ONLY = "Two-Tower + user features"
ITEM_ONLY = "Two-Tower + item features"
FULL = "Two-Tower full features"
ABLATION_MODELS = [ID_ONLY, USER_ONLY, ITEM_ONLY, FULL]


def make_feature_variants(
    user_features: np.ndarray, item_features: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Возвращает маски признаков с неизменными размерностями.

    Нули в стандартизованных признаках означают среднее значение по train.
    Одинаковые размерности сохраняют исходную инициализацию весов для одного seed.
    """
    user = np.asarray(user_features, dtype=np.float32)
    item = np.asarray(item_features, dtype=np.float32)
    if user.ndim != 2 or item.ndim != 2:
        raise ValueError("Feature-матрицы должны быть двумерными")
    zero_user, zero_item = np.zeros_like(user), np.zeros_like(item)
    return {
        ID_ONLY: (zero_user, zero_item),
        USER_ONLY: (user.copy(), zero_item),
        ITEM_ONLY: (zero_user, item.copy()),
        FULL: (user.copy(), item.copy()),
    }


def run_feature_ablation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    full_model: TwoTowerRecommender,
    content: object,
    k: int = 10,
    max_users: int = 600,
    seed: int = 42,
) -> dict[str, pd.DataFrame | dict]:
    """Обучает три маски и сравнивает их с уже обученной полной моделью.

    Полная модель не переобучается: она та же, что используется в гибриде.
    Параметры всех вариантов, test-пользователи и relevance threshold совпадают.
    """
    if not isinstance(full_model, TwoTowerRecommender):
        raise TypeError("full_model должен быть TwoTowerRecommender")
    variants = make_feature_variants(full_model.user_features, full_model.item_features)
    params = full_model.params
    models: dict[str, TwoTowerRecommender] = {}
    history_rows: list[dict] = []
    train_matrix: csr_matrix = full_model.train_matrix

    for name in ABLATION_MODELS:
        user_array, item_array = variants[name]
        if name == FULL:
            model = full_model
        else:
            print(f"Feature ablation: обучение {name}", flush=True)
            model = TwoTowerRecommender.fit(
                train, train_matrix, user_array, item_array, full_model.popularity,
                embedding_dim=full_model.embedding_dim,
                output_dim=full_model.output_dim,
                epochs=int(params.get("epochs", 4)),
                batch_size=int(params.get("batch_size", 2048)),
                learning_rate=float(params.get("learning_rate", 0.001)),
                weight_decay=float(params.get("weight_decay", 1e-6)),
                seed=seed, verbose=True,
            )
        models[name] = model
        history_rows.extend({"model": name, **row} for row in model.history)

    metrics, per_user, segments = evaluate_recommenders(
        models, test, train, k=k, max_users=max_users, seed=seed, content=content,
    )
    history = pd.DataFrame(history_rows)
    configuration = {
        "seed": seed,
        "k": k,
        "evaluated_users": int(metrics["evaluated_users"].iloc[0]),
        "relevance_threshold": 4,
        "same_test_users_for_all_variants": True,
        "full_model_reused_from_hybrid": True,
        "masking": "zero standardized features (train mean); dimensions and initialization preserved",
        "models": ABLATION_MODELS,
        "parameters": {
            key: params.get(key) for key in [
                "embedding_dim", "output_dim", "epochs", "batch_size",
                "learning_rate", "weight_decay", "positive_interactions",
            ]
        },
        "note": "Analytical ablation only; hybrid weights are not retuned on these test results.",
    }
    return {
        "metrics": metrics, "per_user": per_user, "segments": segments,
        "history": history, "config": configuration,
    }


def save_feature_ablation(results: dict, artifacts_dir: str | Path) -> None:
    output = Path(artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    results["metrics"].to_csv(output / "feature_ablation_metrics.csv", index=False, float_format="%.8f")
    results["per_user"].to_csv(output / "feature_ablation_per_user.csv.gz", index=False, compression="gzip")
    results["segments"].to_csv(output / "feature_ablation_segment_metrics.csv", index=False, float_format="%.8f")
    results["history"].to_csv(output / "feature_ablation_history.csv", index=False, float_format="%.8f")
    with (output / "feature_ablation_config.json").open("w", encoding="utf-8") as file:
        json.dump(results["config"], file, ensure_ascii=False, indent=2)
