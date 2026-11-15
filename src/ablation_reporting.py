"""Текстовый раздел feature-ablation для отчёта."""

from __future__ import annotations

import pandas as pd

from src.ablation import FULL, ID_ONLY, ITEM_ONLY, USER_ONLY


def _table(metrics: pd.DataFrame) -> str:
    base = float(metrics.set_index("model").loc[ID_ONLY, "ndcg_at_k"])
    lines = [
        "| Вариант | Precision@10 | Recall@10 | nDCG@10 | Δ nDCG к ID-only | ILD@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['model']} | {row['precision_at_k']:.4f} | {row['recall_at_k']:.4f} | "
            f"{row['ndcg_at_k']:.4f} | {row['ndcg_at_k'] - base:+.4f} | {row['ild_at_k']:.4f} |"
        )
    return "\n".join(lines)


def build_feature_ablation_section(metrics: pd.DataFrame | None) -> str:
    if metrics is None or metrics.empty:
        return (
            "\n## 5.1. Feature-ablation эксперимент\n\n"
            "Эксперимент в этом запуске не выполнялся. Команда: "
            "`python scripts/feature_ablation.py` или `python scripts/train.py --feature-ablation`.\n"
        )
    indexed = metrics.set_index("model")
    base = float(indexed.loc[ID_ONLY, "ndcg_at_k"])
    full = float(indexed.loc[FULL, "ndcg_at_k"])
    user = float(indexed.loc[USER_ONLY, "ndcg_at_k"])
    item = float(indexed.loc[ITEM_ONLY, "ndcg_at_k"])
    best_name = indexed["ndcg_at_k"].idxmax()
    best_value = float(indexed.loc[best_name, "ndcg_at_k"])
    full_delta = full - base
    user_delta = user - base
    item_delta = item - base
    user_given_items_delta = full - item
    relative_text = f" ({full_delta / base:+.1%})" if base > 0 else ""
    if full_delta > 0:
        conclusion = (
            f"Полный набор улучшил nDCG@10 относительно ID-only на **{full_delta:+.4f}**{relative_text}. "
            "Это подтверждает положительный суммарный вклад признаков в текущем эксперименте."
        )
    elif full_delta < 0:
        conclusion = (
            f"Полный набор изменил nDCG@10 относительно ID-only на **{full_delta:+.4f}**{relative_text}. "
            "Следовательно, автоматическое улучшение от объединения всех признаков не подтверждено; "
            "их следует отбирать по validation, а не объявлять полезными заранее."
        )
    else:
        conclusion = "Полный набор показал тот же nDCG@10, что и ID-only: отдельного выигрыша признаков не выявлено."

    return f"""
## 5.1. Feature-ablation: вклад расширенных признаков

Для изоляции вклада признаков сравниваются четыре варианта Two-Tower: только ID-embeddings, ID + пользовательские признаки, ID + книжные признаки и полный набор. Используются одинаковые train, seed, размерности сети, BPR-параметры и {int(metrics['evaluated_users'].iloc[0])} test-пользователей. Полная модель — та же, что интегрирована в гибрид; она не переобучается для ablation.

Стандартизованные признаки в отключённой группе заменяются нулями, то есть средними train-значениями. Размерности сети и исходная инициализация сохраняются. Веса гибрида по этим test-результатам не перенастраиваются.

{_table(metrics)}

Изолированный эффект пользовательских признаков относительно ID-only: **{user_delta:+.4f} nDCG@10**. Эффект книжных признаков: **{item_delta:+.4f} nDCG@10**. Добавление пользовательских признаков к книжным изменило nDCG@10 на **{user_given_items_delta:+.4f}**. Лучший вариант по nDCG@10 — **{best_name}** ({best_value:.4f}).

{conclusion}

Результат не означает, что каждая группа признаков полезна отдельно: эффект пользовательских признаков зависит от комбинации с книжными. Доверительные интервалы сохранены в CSV; выбор признаков для следующей итерации следует подтверждать на validation и нескольких seed, а не перенастраивать модель по этому test.

Эти различия описывают текущий dataset, split и бюджет обучения, но не доказывают универсальный эффект. Признаки взаимодействий в Two-Tower не подаются: они используются для анализа и объяснений. Графики — `reports/figures/12_feature_ablation.png` и `13_feature_ablation_training.png`; исходные per-user результаты — в `artifacts/feature_ablation_per_user.csv.gz`.
"""
