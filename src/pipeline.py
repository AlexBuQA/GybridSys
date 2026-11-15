"""Сквозной pipeline: данные → признаки → модели → гибрид → оценка."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data import build_user_item_matrix, clean_ratings, load_source_data, user_holdout_split
from src.eda import run_eda
from src.features import (
    BOOK_FEATURES, USER_FEATURES, InteractionFeatureBuilder, build_book_features,
    build_user_features, scale_feature_arrays,
)
from src.metrics import evaluate_recommenders, rmse
from src.models import (
    ContentBasedRecommender, HybridRecommender, HybridVariant, ItemCFRecommender,
    PopularityRecommender, SVDRecommender, TwoTowerRecommender,
)
from src.models.hybrid import tune_diversity_lambda, tune_hybrid_weights
from src.reporting import generate_report
from src.split import split_validation_test
from src.visualization import plot_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-users", type=int, default=600)
    parser.add_argument("--validation-users-per-segment", type=int, default=80)
    parser.add_argument("--weight-trials", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-ratings", type=int, default=50)
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--n-factors", type=int, default=32)
    parser.add_argument("--n-iter", type=int, default=7)
    parser.add_argument("--neural-epochs", type=int, default=4)
    parser.add_argument("--neural-embedding", type=int, default=16)
    parser.add_argument("--neural-hidden", type=int, default=32)
    parser.add_argument("--neural-batch-size", type=int, default=2048)
    parser.add_argument("--neural-learning-rate", type=float, default=0.001)
    parser.add_argument("--candidate-limit", type=int, default=60)
    parser.add_argument("--reuse-classical", action="store_true", help="Загрузить классические модели из artifacts/classical")
    parser.add_argument("--quick", action="store_true", help="Быстрая проверка pipeline на малой evaluation-выборке")
    parser.add_argument("--feature-ablation", action="store_true", help="Сравнить ID-only, user/item features и полную Two-Tower")
    return parser.parse_args()


def _same_matrix(left: csr_matrix, right: csr_matrix) -> bool:
    return (
        left.shape == right.shape and left.nnz == right.nnz
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.allclose(left.data, right.data)
    )


def _load_or_train_classical(
    args: argparse.Namespace, train: pd.DataFrame, holdout: pd.DataFrame,
    train_matrix: csr_matrix, data: dict[str, pd.DataFrame], classical_dir: Path,
) -> dict[str, object]:
    if args.reuse_classical:
        print("Загрузка проверенных классических моделей из ДЗ", flush=True)
        models = {
            "popularity": joblib.load(classical_dir / "popularity.joblib"),
            "content": joblib.load(classical_dir / "content.joblib"),
            "item_cf": joblib.load(classical_dir / "item_cf.joblib"),
            "svd": joblib.load(classical_dir / "svd.joblib"),
        }
        for name, model in models.items():
            if not _same_matrix(model.train_matrix, train_matrix):
                raise ValueError(
                    f"Артефакт {name} обучен на другом train. Запустите без --reuse-classical."
                )
        return models

    print("Обучение Popularity и Content-Based", flush=True)
    popularity = PopularityRecommender(min_ratings=args.min_ratings).fit(train, train_matrix)
    content = ContentBasedRecommender.fit(
        data["books"], data["book_tags"], data["tags"],
        train_matrix=train_matrix, popularity=popularity,
    )
    print("Обучение Item-Based CF", flush=True)
    item_cf = ItemCFRecommender.fit(
        train_matrix, popularity, n_neighbors=args.neighbors, chunk_size=args.chunk_size,
    )
    print("Обучение SVD", flush=True)
    svd = SVDRecommender.fit(
        train, holdout, train_matrix, popularity,
        n_factors=args.n_factors, n_iter=args.n_iter, seed=args.seed,
    )
    models = {"popularity": popularity, "content": content, "item_cf": item_cf, "svd": svd}
    classical_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, classical_dir / f"{name}.joblib", compress=3)
    return models


def main() -> None:
    args = parse_args()
    if args.quick:
        args.eval_users = min(args.eval_users, 100)
        args.validation_users_per_segment = min(args.validation_users_per_segment, 30)
        args.weight_trials = min(args.weight_trials, 12)
        args.neural_epochs = min(args.neural_epochs, 2)
        args.neighbors = min(args.neighbors, 20)
        args.n_factors = min(args.n_factors, 8)
        args.n_iter = min(args.n_iter, 3)
    started = time.perf_counter()
    np.random.seed(args.seed)
    artifacts_dir, reports_dir = Path(args.artifacts_dir), Path(args.reports_dir)
    processed_dir = Path(args.processed_dir)
    for directory in [artifacts_dir, reports_dir / "figures", processed_dir, artifacts_dir / "classical"]:
        directory.mkdir(parents=True, exist_ok=True)

    print("1/10. Загрузка, очистка и train/validation/test split", flush=True)
    data = load_source_data(args.data_dir)
    ratings, cleaning_stats = clean_ratings(data["ratings"])
    train, holdout = user_holdout_split(ratings, test_size=0.2, seed=args.seed)
    validation, test = split_validation_test(holdout, seed=args.seed)
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        frame.to_csv(processed_dir / f"{name}.csv.gz", index=False, compression="gzip")
    n_users, n_items = int(ratings["user_id"].max()), int(data["books"]["id"].max())
    train_matrix = build_user_item_matrix(train, n_users=n_users, n_items=n_items)
    print(f"train={len(train)}, validation={len(validation)}, test={len(test)}", flush=True)

    print("2/10. EDA", flush=True)
    eda_summary = run_eda(
        ratings, data["books"], data["tags"], data["book_tags"], cleaning_stats, reports_dir,
    )

    print("3/10. Классические модели из ДЗ", flush=True)
    classical = _load_or_train_classical(
        args, train, holdout, train_matrix, data, artifacts_dir / "classical"
    )
    popularity = classical["popularity"]
    content = classical["content"]

    print("4/10. Расширенные признаки", flush=True)
    user_features = build_user_features(train, n_users)
    book_features = build_book_features(
        train, data["books"], data["book_tags"], data["tags"], n_items
    )
    user_features.to_csv(processed_dir / "user_features.csv.gz", index=False, compression="gzip")
    book_features.to_csv(processed_dir / "book_features.csv.gz", index=False, compression="gzip")
    user_array, item_array, scaler = scale_feature_arrays(user_features, book_features)
    print(f"Признаки: users={user_array.shape}, books={item_array.shape}", flush=True)

    print("5/10. Two-Tower + BPR + Adam", flush=True)
    two_tower = TwoTowerRecommender.fit(
        train, train_matrix, user_array, item_array, popularity,
        embedding_dim=args.neural_embedding, output_dim=args.neural_hidden,
        epochs=args.neural_epochs, batch_size=args.neural_batch_size,
        learning_rate=args.neural_learning_rate, seed=args.seed,
    )
    neural_history = pd.DataFrame(two_tower.history)
    neural_history.to_csv(artifacts_dir / "neural_history.csv", index=False, float_format="%.8f")
    joblib.dump(two_tower, artifacts_dir / "two_tower.joblib", compress=3)

    print("6/10. Подбор сегментных весов на validation", flush=True)
    models = {**classical, "two_tower": two_tower}
    hybrid = HybridRecommender(
        models, popularity, content, candidate_limit=args.candidate_limit,
    )
    weight_search = tune_hybrid_weights(
        hybrid, validation, k=args.top_k,
        max_users_per_segment=args.validation_users_per_segment,
        n_trials=args.weight_trials, seed=args.seed,
    )
    weight_search.to_csv(artifacts_dir / "validation_weight_search.csv", index=False, float_format="%.8f")

    print("7/10. Настройка MMR на validation", flush=True)
    diversity_search = tune_diversity_lambda(
        hybrid, validation, k=args.top_k,
        max_users=min(120, args.validation_users_per_segment * 2), seed=args.seed,
    )
    diversity_search.to_csv(artifacts_dir / "validation_diversity_search.csv", index=False, float_format="%.8f")

    print("8/10. Финальная оценка на test", flush=True)
    classical["svd"].rmse = rmse(
        test["rating"].to_numpy(),
        classical["svd"].predict_many(test["user_id"].to_numpy(), test["book_id"].to_numpy()),
    )
    evaluation_models = {
        "Popularity": popularity, "Content-Based": content,
        "Item-Based CF": classical["item_cf"], "SVD": classical["svd"],
        "Two-Tower": two_tower,
        "Hybrid-Classical": HybridVariant(hybrid, include_neural=False, diversify=False),
        "Hybrid (no MMR)": HybridVariant(hybrid, include_neural=True, diversify=False),
        "Hybrid": hybrid,
    }
    metrics, per_user, segment_metrics = evaluate_recommenders(
        evaluation_models, test, train, k=args.top_k, max_users=args.eval_users,
        seed=args.seed, content=content,
    )
    metrics.to_csv(artifacts_dir / "metrics.csv", index=False, float_format="%.8f")
    per_user.to_csv(artifacts_dir / "evaluation_per_user.csv.gz", index=False, compression="gzip")
    segment_metrics.to_csv(artifacts_dir / "segment_metrics.csv", index=False, float_format="%.8f")

    feature_ablation_metrics = None
    if args.feature_ablation:
        from src.ablation import run_feature_ablation, save_feature_ablation
        from src.visualization import plot_feature_ablation

        print("Дополнительно. Feature-ablation: ID / user / item / full", flush=True)
        ablation_results = run_feature_ablation(
            train, test, two_tower, content,
            k=args.top_k, max_users=args.eval_users, seed=args.seed,
        )
        save_feature_ablation(ablation_results, artifacts_dir)
        plot_feature_ablation(
            ablation_results["metrics"], ablation_results["history"], reports_dir / "figures"
        )
        feature_ablation_metrics = ablation_results["metrics"]

    print("9/10. Графики, примеры и аналитический отчёт", flush=True)
    plot_all(
        metrics, segment_metrics, weight_search, diversity_search,
        neural_history, reports_dir / "figures",
    )
    counts = train.groupby("user_id").size()
    positive_users = test.loc[test["rating"] >= 4, "user_id"].unique()
    example_user = next((int(u) for u in positive_users if counts.get(u, 0) >= 10), int(positive_users[0]))
    example = hybrid.explain(example_user, args.top_k)
    interaction_builder = InteractionFeatureBuilder(train_matrix, user_features, book_features)
    candidate_set = hybrid.candidates(example_user)
    interaction_example = interaction_builder.build(
        example_user, [row["book_id"] for row in example],
        content_scores={int(book): float(candidate_set.raw_scores["content"][i]) for i, book in enumerate(candidate_set.book_ids)},
        item_cf_scores={int(book): float(candidate_set.raw_scores["item_cf"][i]) for i, book in enumerate(candidate_set.book_ids)},
    )
    interaction_example.to_csv(artifacts_dir / "interaction_features_example.csv", index=False, float_format="%.8f")
    titles = data["books"].set_index("id")[["title", "authors"]].to_dict("index")
    for row in example:
        row.update(titles.get(row["book_id"], {"title": "", "authors": ""}))
    with (artifacts_dir / "examples.json").open("w", encoding="utf-8") as file:
        json.dump({"user_id": example_user, "recommendations": example}, file, ensure_ascii=False, indent=2)

    config = {
        "project": "hybrid-book-recommender", "version": "2.0.0",
        "author": "Александра Бужор", "seed": args.seed,
        "data": {"dataset": "Goodbooks-10k", "n_users": n_users, "n_items": n_items, **cleaning_stats},
        "split": {
            "method": "random per-user train/holdout, then per-user validation/test",
            "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
            "temporal": False,
        },
        "features": {"user": USER_FEATURES, "book": BOOK_FEATURES, "scaler": scaler},
        "models": {
            "popularity": {"min_ratings": args.min_ratings},
            "content": {"vectorizer": "TF-IDF", "max_features": 30000, "max_tags_per_book": 30},
            "item_cf": {"similarity": "cosine", "n_neighbors": int(classical["item_cf"].n_neighbors)},
            "svd": classical["svd"].params,
            "two_tower": two_tower.params,
        },
        "hybrid": {
            "candidate_limit": args.candidate_limit, "weights": hybrid.weights,
            "diversity_lambda": hybrid.diversity_lambda,
            "weight_optimization": "validation nDCG@K; default + Dirichlet + one-hot",
            "diversity_optimization": "validation nDCG@K + 0.20*ILD@K (coefficient fixed a priori)",
            "reused_classical_artifacts": bool(args.reuse_classical),
        },
        "evaluation": {
            "k": args.top_k, "users": int(metrics["evaluated_users"].iloc[0]),
            "relevance_threshold": 4, "aggregation": "macro average over identical users",
            "candidate_catalog": "full catalog; train items excluded",
        },
        "eda": eda_summary,
    }
    with (artifacts_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    generate_report(
        metrics, segment_metrics, hybrid, config, reports_dir / "REPORT.md",
        feature_ablation=feature_ablation_metrics,
    )

    hybrid.clear_cache()
    joblib.dump(hybrid, artifacts_dir / "hybrid.joblib", compress=3)
    elapsed = time.perf_counter() - started
    best = metrics.loc[metrics["ndcg_at_k"].idxmax()]
    summary = {
        "status": "success", "elapsed_seconds": elapsed,
        "best_model_by_ndcg": best["model"], "best_ndcg_at_k": float(best["ndcg_at_k"]),
        "hybrid_lambda": hybrid.diversity_lambda, "evaluated_users": int(best["evaluated_users"]),
    }
    with (artifacts_dir / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("10/10. Готово", flush=True)
    print(metrics[["model", "precision_at_k", "recall_at_k", "ndcg_at_k", "ild_at_k"]].to_string(index=False), flush=True)
    print(f"Время: {elapsed / 60:.1f} мин. Отчёт: {reports_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
