"""Воспроизводимый EDA и сохранение графиков."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLOR = "#6C5CE7"
SECONDARY = "#00A8A8"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _distribution_plot(counts: pd.Series, title: str, xlabel: str, path: Path) -> None:
    distribution = counts.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.scatter(distribution.index, distribution.values, s=18, alpha=0.65, color=COLOR)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Количество наблюдений (log)")
    ax.grid(True, which="both", alpha=0.25)
    _save(fig, path)


def run_eda(
    ratings: pd.DataFrame,
    books: pd.DataFrame,
    tags: pd.DataFrame,
    book_tags: pd.DataFrame,
    cleaning_stats: dict,
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    user_activity = ratings.groupby("user_id").size()
    book_popularity = ratings.groupby("book_id").size()
    goodreads_popularity = books.set_index("id")["ratings_count"].astype(float)
    n_users = int(ratings["user_id"].nunique())
    n_items = int(books["id"].nunique())
    sparsity = 1.0 - len(ratings) / (n_users * n_items)
    rating_distribution = ratings["rating"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(rating_distribution.index.astype(str), rating_distribution.values, color=COLOR)
    ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=9)
    ax.set_title("Распределение оценок Goodbooks-10k", fontsize=14, weight="bold")
    ax.set_xlabel("Оценка")
    ax.set_ylabel("Количество оценок")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, figures_dir / "01_rating_distribution.png")

    _distribution_plot(
        user_activity,
        "Активность пользователей: длинный хвост",
        "Количество оценок на пользователя (log)",
        figures_dir / "02_user_activity.png",
    )
    _distribution_plot(
        book_popularity,
        "Оценки на книгу: в срезе действует верхний предел ≈ 100",
        "Количество оценок на книгу в ratings.csv (log)",
        figures_dir / "03_book_popularity.png",
    )

    top_tags = (
        book_tags.merge(tags[["tag_id", "tag_name"]], on="tag_id", how="left")
        .dropna(subset=["tag_name"])
        .groupby("tag_name")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_tags.index.astype(str), top_tags.values, color=SECONDARY)
    ax.set_title("20 самых частых тегов по суммарному count", fontsize=14, weight="bold")
    ax.set_xlabel("Суммарное число применений тега")
    ax.set_ylabel("Тег")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, figures_dir / "04_top_tags.png")

    # В присланном срезе почти у всех книг ровно 100 оценок. Поэтому реальную
    # неоднородность популярности дополнительно показываем через метаданные Goodreads.
    top_books = books[["id", "original_title", "title", "authors", "ratings_count"]].copy()
    top_books["dataset_ratings"] = top_books["id"].map(book_popularity).fillna(0).astype(int)
    top_books["label"] = top_books["original_title"].fillna(top_books["title"]).str.slice(0, 48)
    top_books = top_books.sort_values("ratings_count", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(11, 7))
    plot_data = top_books.sort_values("ratings_count")
    ax.barh(plot_data["label"], plot_data["ratings_count"], color=COLOR)
    ax.set_title("15 самых популярных книг по Goodreads ratings_count", fontsize=14, weight="bold")
    ax.set_xlabel("Глобальное количество оценок из метаданных Goodreads")
    ax.set_ylabel("Книга")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, figures_dir / "05_top_books.png")

    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    summary = {
        **{k: int(v) for k, v in cleaning_stats.items()},
        "books_in_catalog": n_items,
        "sparsity": float(sparsity),
        "sparsity_percent": float(sparsity * 100),
        "mean_rating": float(ratings["rating"].mean()),
        "median_rating": float(ratings["rating"].median()),
        "share_ratings_ge_4_percent": float((ratings["rating"] >= 4).mean() * 100),
        "rating_distribution": {str(int(k)): int(v) for k, v in rating_distribution.items()},
        "user_activity_quantiles": {str(q): float(user_activity.quantile(q)) for q in quantiles},
        "book_popularity_quantiles": {str(q): float(book_popularity.quantile(q)) for q in quantiles},
        "goodreads_popularity_quantiles": {
            str(q): float(goodreads_popularity.quantile(q)) for q in quantiles
        },
        "books_with_100_ratings_in_sample": int((book_popularity == 100).sum()),
        "books_with_100_ratings_in_sample_percent": float((book_popularity == 100).mean() * 100),
        "active_users_at_or_above_p99": int((user_activity >= user_activity.quantile(0.99)).sum()),
        "users_with_at_most_2_ratings": int((user_activity <= 2).sum()),
        "books_with_at_most_5_ratings": int((book_popularity <= 5).sum()),
        "top_tags": [
            {"tag": str(tag), "count": int(count)}
            for tag, count in top_tags.sort_values(ascending=False).items()
        ],
        "top_books": [
            {
                "book_id": int(row.id),
                "title": str(row.label),
                "authors": str(row.authors),
                "ratings_in_dataset": int(row.dataset_ratings),
                "goodreads_ratings_count": int(row.ratings_count),
            }
            for row in top_books.itertuples()
        ],
    }
    with (output_dir / "eda_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary
