"""Smoke-проверка сохранённых моделей, метрик и примеров."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    artifacts = ROOT / "artifacts"
    required = ["hybrid.joblib", "two_tower.joblib", "metrics.csv", "segment_metrics.csv", "config.json", "examples.json"]
    missing = [name for name in required if not (artifacts / name).exists()]
    if missing:
        raise FileNotFoundError(f"Отсутствуют артефакты: {missing}")
    hybrid = joblib.load(artifacts / "hybrid.joblib")
    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv.gz")
    metrics = pd.read_csv(artifacts / "metrics.csv")
    expected = {"Popularity", "Content-Based", "Item-Based CF", "SVD", "Two-Tower", "Hybrid-Classical", "Hybrid (no MMR)", "Hybrid"}
    assert set(metrics["model"]) == expected
    assert metrics["evaluated_users"].nunique() == 1
    user_id = int(train["user_id"].iloc[0])
    result = hybrid.recommend(user_id, 10)
    assert len(result) == 10
    seen = set(train.loc[train["user_id"] == user_id, "book_id"])
    assert not seen.intersection(book for book, _ in result)
    config = json.loads((artifacts / "config.json").read_text(encoding="utf-8"))
    assert config["split"]["train_rows"] == len(train)
    ablation_path = artifacts / "feature_ablation_metrics.csv"
    if ablation_path.exists():
        required_ablation = [
            "feature_ablation_metrics.csv", "feature_ablation_per_user.csv.gz",
            "feature_ablation_segment_metrics.csv", "feature_ablation_history.csv",
            "feature_ablation_config.json",
        ]
        missing_ablation = [name for name in required_ablation if not (artifacts / name).is_file()]
        assert not missing_ablation, f"Неполный feature-ablation: {missing_ablation}"
        ablation = pd.read_csv(ablation_path)
        details = pd.read_csv(artifacts / "feature_ablation_per_user.csv.gz")
        segments_ablation = pd.read_csv(artifacts / "feature_ablation_segment_metrics.csv")
        history = pd.read_csv(artifacts / "feature_ablation_history.csv")
        ablation_config = json.loads((artifacts / "feature_ablation_config.json").read_text(encoding="utf-8"))
        expected_ablation = {
            "Two-Tower ID only", "Two-Tower + user features",
            "Two-Tower + item features", "Two-Tower full features",
        }
        expected_users = int(metrics["evaluated_users"].iloc[0])
        assert set(ablation["model"]) == expected_ablation
        assert (ablation["evaluated_users"] == expected_users).all()
        assert ablation_config["seed"] == config["seed"]
        assert ablation_config["k"] == config["evaluation"]["k"]
        assert ablation_config["evaluated_users"] == expected_users
        assert ablation_config["same_test_users_for_all_variants"]
        assert ablation_config["full_model_reused_from_hybrid"]

        # Одна и та же evaluation-выборка и ровно одна строка на пользователя/вариант.
        assert not details.duplicated(["model", "user_id"]).any()
        assert set(details["model"]) == expected_ablation
        user_sets = [set(group["user_id"]) for _, group in details.groupby("model")]
        assert all(users == user_sets[0] for users in user_sets[1:])
        assert len(user_sets[0]) == expected_users
        assert len(details) == len(expected_ablation) * expected_users

        # Сводные метрики действительно являются макроусреднением per-user строк.
        metric_columns = ["precision_at_k", "recall_at_k", "ndcg_at_k", "ild_at_k", "novelty_at_k"]
        macro = details.groupby("model")[metric_columns].mean().reindex(ablation["model"])
        np.testing.assert_allclose(ablation[metric_columns], macro, rtol=1e-6, atol=1e-7)

        # Полная модель обязана совпадать с Two-Tower, уже интегрированной в гибрид.
        original = metrics.set_index("model").loc["Two-Tower"]
        full = ablation.set_index("model").loc["Two-Tower full features"]
        comparison_columns = metric_columns + ["catalog_coverage", "evaluated_users", "k"]
        np.testing.assert_allclose(full[comparison_columns], original[comparison_columns], rtol=0, atol=1e-8)

        segment_users = segments_ablation.groupby("model")["users"].sum()
        assert set(segment_users.index) == expected_ablation
        assert (segment_users == expected_users).all()
        epochs = int(ablation_config["parameters"]["epochs"])
        assert len(history) == len(expected_ablation) * epochs
        for _, group in history.groupby("model"):
            assert group["epoch"].tolist() == list(range(1, epochs + 1))
        print(
            f"OK: feature ablation, {len(ablation)} варианта, {expected_users} общих пользователей, "
            f"{len(details)} per-user строк, {len(history)} строк history; full = исходной Two-Tower."
        )
    print(f"OK: {len(metrics)} моделей, {len(result)} рекомендаций, пользователь {user_id}, прочитанное исключено.")


if __name__ == "__main__":
    main()
