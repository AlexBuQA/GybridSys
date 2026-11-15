"""Визуализация фактических результатов без ручного редактирования чисел."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_all(
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    weight_search: pd.DataFrame,
    diversity_search: pd.DataFrame,
    neural_history: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(metrics), 3)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, metric, title in zip(
        axes, ["precision_at_k", "recall_at_k", "ndcg_at_k"],
        ["Precision@10", "Recall@10", "nDCG@10"],
    ):
        values = metrics[metric]
        axis.barh(metrics["model"], values, color=colors)
        axis.set_title(title)
        axis.set_xlabel("Макроусреднённое значение")
        for i, value in enumerate(values):
            axis.text(value, i, f" {value:.4f}", va="center", fontsize=9)
        axis.set_xlim(0, max(float(values.max()) * 1.22, 0.01))
    fig.suptitle("Сравнение моделей на непересекающемся test", fontsize=16)
    _save(fig, output / "06_model_comparison.png")

    pivot = segments.pivot(index="model", columns="segment", values="ndcg_at_k").fillna(0)
    pivot = pivot.reindex(columns=[c for c in ["new", "warm", "active"] if c in pivot.columns])
    fig, axis = plt.subplots(figsize=(13, 7))
    pivot.plot(kind="bar", ax=axis, width=0.82)
    axis.set_title("nDCG@10 по сегментам пользователей")
    axis.set_ylabel("nDCG@10")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(title="Сегмент")
    _save(fig, output / "07_segment_ndcg.png")

    fig, axis = plt.subplots(figsize=(11, 6))
    if not weight_search.empty:
        for segment, group in weight_search.groupby("segment"):
            axis.scatter(group["trial"], group["ndcg_at_k"], s=22, alpha=0.55, label=segment)
            best = group.loc[group["ndcg_at_k"].idxmax()]
            axis.scatter(best["trial"], best["ndcg_at_k"], s=130, marker="*", edgecolor="black")
    axis.set_title("Поиск весов на validation (звезда — лучший вариант)")
    axis.set_xlabel("Номер конфигурации")
    axis.set_ylabel("nDCG@10")
    axis.legend(title="Сегмент")
    _save(fig, output / "08_weight_search.png")

    fig, axis = plt.subplots(figsize=(10, 6))
    axis2 = axis.twinx()
    axis.plot(diversity_search["diversity_lambda"], diversity_search["ndcg_at_k"], "o-", color="#2266aa", label="nDCG@10")
    axis2.plot(diversity_search["diversity_lambda"], diversity_search["ild_at_k"], "s--", color="#cc6633", label="ILD@10")
    axis.set_xlabel("lambda: вес релевантности в MMR")
    axis.set_ylabel("nDCG@10", color="#2266aa")
    axis2.set_ylabel("Внутрисписочное разнообразие", color="#cc6633")
    axis.set_title("Validation: компромисс релевантности и разнообразия")
    _save(fig, output / "09_diversity_tradeoff.png")

    fig, axis = plt.subplots(figsize=(10, 7))
    for i, row in metrics.iterrows():
        axis.scatter(row["ild_at_k"], row["ndcg_at_k"], s=100, color=colors[i])
        axis.annotate(row["model"], (row["ild_at_k"], row["ndcg_at_k"]), xytext=(6, 5), textcoords="offset points")
    axis.set_title("Test: качество ранжирования и разнообразие")
    axis.set_xlabel("ILD@10 (выше — разнообразнее)")
    axis.set_ylabel("nDCG@10 (выше — релевантнее)")
    _save(fig, output / "10_quality_diversity_map.png")

    fig, axis = plt.subplots(figsize=(10, 6))
    axis2 = axis.twinx()
    axis.plot(neural_history["epoch"], neural_history["train_bpr_loss"], "o-", label="BPR loss")
    axis2.plot(neural_history["epoch"], neural_history["train_bpr_accuracy"], "s--", color="#dd7755", label="BPR accuracy")
    axis.set_xlabel("Эпоха")
    axis.set_ylabel("BPR loss")
    axis2.set_ylabel("Доля score(positive) > score(negative)")
    axis.set_title("Обучение Two-Tower на train")
    axis.set_xticks(neural_history["epoch"])
    _save(fig, output / "11_neural_training.png")


def plot_feature_ablation(
    metrics: pd.DataFrame, history: pd.DataFrame, output_dir: str | Path
) -> None:
    """Графики изолированного вклада пользовательских и книжных признаков."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    labels = {
        "Two-Tower ID only": "ID only",
        "Two-Tower + user features": "+ user features",
        "Two-Tower + item features": "+ item features",
        "Two-Tower full features": "Full features",
    }
    ordered = metrics.set_index("model").reindex(labels).reset_index()
    names = [labels[model] for model in ordered["model"]]
    colors = ["#9aa5b1", "#4c9f70", "#4c78a8", "#e45756"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for axis, metric, title in zip(
        axes, ["precision_at_k", "recall_at_k", "ndcg_at_k"],
        ["Precision@10", "Recall@10", "nDCG@10"],
    ):
        bars = axis.bar(names, ordered[metric], color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylim(bottom=0)
        for bar, value in zip(bars, ordered[metric]):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Feature ablation: одинаковые train, seed и 600 test-пользователей", fontsize=15)
    _save(fig, output / "12_feature_ablation.png")

    fig, axis = plt.subplots(figsize=(10, 6))
    for name, group in history.groupby("model"):
        axis.plot(group["epoch"], group["train_bpr_loss"], "o-", label=labels.get(name, name))
    axis.set_title("Feature ablation: динамика BPR loss на train")
    axis.set_xlabel("Эпоха")
    axis.set_ylabel("BPR loss")
    axis.set_xticks(sorted(history["epoch"].unique()))
    axis.legend()
    _save(fig, output / "13_feature_ablation_training.png")
