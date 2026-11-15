"""Генерация аналитического отчёта только по рассчитанным артефактам."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _metrics_table(metrics: pd.DataFrame) -> str:
    lines = ["| Модель | Precision@10 | Recall@10 | nDCG@10 | ILD@10 | Coverage |", "|---|---:|---:|---:|---:|---:|"]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['model']} | {row['precision_at_k']:.4f} | {row['recall_at_k']:.4f} | "
            f"{row['ndcg_at_k']:.4f} | {row['ild_at_k']:.4f} | {row['catalog_coverage']:.4f} |"
        )
    return "\n".join(lines)


def generate_report(
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    hybrid: object,
    config: dict,
    path: str | Path,
    feature_ablation: pd.DataFrame | None = None,
) -> None:
    from src.ablation_reporting import build_feature_ablation_section

    indexed = metrics.set_index("model")
    classical_names = ["Popularity", "Content-Based", "Item-Based CF", "SVD"]
    best_classical_name = indexed.loc[classical_names, "ndcg_at_k"].idxmax()
    best_classical = float(indexed.loc[best_classical_name, "ndcg_at_k"])
    final = indexed.loc["Hybrid"]
    neural = indexed.loc["Two-Tower"]
    no_neural = indexed.loc["Hybrid-Classical"]
    no_mmr = indexed.loc["Hybrid (no MMR)"]
    delta = (float(final["ndcg_at_k"]) - best_classical) / best_classical if best_classical > 0 else np.nan
    # Чистый вклад нейросети сравнивается без MMR; иначе эффекты смешиваются.
    neural_delta = float(no_mmr["ndcg_at_k"]) - float(no_neural["ndcg_at_k"])
    mmr_ild_delta = float(final["ild_at_k"]) - float(no_mmr["ild_at_k"])
    mmr_ndcg_delta = float(final["ndcg_at_k"]) - float(no_mmr["ndcg_at_k"])
    feature_ablation_section = build_feature_ablation_section(feature_ablation)
    leaders = []
    for segment, group in segments.groupby("segment"):
        leader = group.loc[group["ndcg_at_k"].idxmax()]
        leaders.append(f"- **{segment}**: лидер {leader['model']}, nDCG@10 = {leader['ndcg_at_k']:.4f} на {int(leader['users'])} пользователях.")
    weights = []
    for segment in ["new", "warm", "active"]:
        values = ", ".join(f"{key}={value:.3f}" for key, value in hybrid.weights[segment].items())
        weights.append(f"- **{segment}**: {values}")

    text = f"""# Итоговый проект: гибридная система рекомендаций книг

**Автор:** Александра Бужор · [@Angry_C_F_O](https://t.me/Angry_C_F_O)  
**Датасет:** Goodbooks-10k · seed = {config['seed']} · K = {config['evaluation']['k']}

## 1. Результаты

Все значения ниже рассчитаны автоматически на test после завершения подбора параметров. Релевантна книга с test-оценкой ≥ 4; метрики сначала считаются для пользователя, затем макроусредняются по одной и той же выборке из {config['evaluation']['users']} пользователей.

{_metrics_table(metrics)}

ILD@10 — среднее `1 − cosine` между TF-IDF-профилями книг внутри списка. Coverage — доля каталога, встретившаяся в выдачах оцениваемых пользователей. Novelty и доверительные интервалы сохранены в `artifacts/metrics.csv`.

**Лучшая отдельная классическая модель:** {best_classical_name} (nDCG@10 = {best_classical:.4f}). Относительное изменение финального Hybrid: {delta:+.2%}. Абсолютный вклад нейросети в nDCG Hybrid без MMR: {neural_delta:+.4f}. MMR изменил ILD на {mmr_ild_delta:+.4f}, а nDCG на {mmr_ndcg_delta:+.4f}. Это фактический компромисс текущего запуска, а не заранее обещанное улучшение.

## 1.1. Развитие работы «Построение прототипа книжного рекомендательного сервиса»

Предыдущая работа была ориентирована на **работающий сервис с базовыми моделями и Streamlit-интерфейсом**. Итоговый проект является **исследовательской ML-системой**: классические модели стали основой для обучаемого гибрида, нейросетевого канала и явного управления разнообразием.

| Компонент | Прототип рекомендательного сервиса | Итоговый проект |
|---|---|---|
| Цель | Построить сервис и сравнить базовые алгоритмы | Исследовать гибридизацию, нейросеть и relevance/diversity |
| Классические модели | Popularity, Content-Based, Item-Based CF, SVD | Те же проверенные модели и артефакты переиспользованы |
| Нейросеть | Отсутствовала | Two-Tower с embeddings, dense + tanh, BPR-loss и Adam |
| Гибрид | Переключение Popularity → Content → SVD по правилам | Общий пул кандидатов и взвешенное объединение score |
| Веса | Фиксированные правила выбора одной модели | Сегментные веса new/warm/active на validation |
| Признаки | Оценки, названия и теги | Дополнительно активность, энтропия, популярность, тематики и признаки взаимодействий |
| Разнообразие | Не контролировалось явно | MMR с настройкой lambda на validation |
| Разбиение | Train/test | Train/validation/test без подбора параметров по test |
| Сравнение | Отдельные модели и простой Hybrid | Дополнительно ablation нейросети, MMR и групп признаков Two-Tower |
| Метрики | Precision, Recall, nDCG, RMSE | Добавлены ILD, novelty, coverage, доверительные интервалы и сегменты |
| Интерфейс | Streamlit-приложение | CLI и выполненный Notebook; Streamlit не входит в этот вариант |
| Фокус | Пользовательский прототип сервиса | Воспроизводимый ML-pipeline и экспериментальный анализ |

### Унаследованные компоненты

Из домашней работы сохранены Goodbooks-10k, очистка и дедупликация, исходный train/holdout, четыре классические модели и их артефакты, TF-IDF-профили книг, разреженная Item-CF, SVD со смещениями, базовые ranking-метрики, фильтрация прочитанного и EDA. Таким образом, итоговый проект развивает предыдущую работу, а не заменяет её независимой реализацией.

### Изменение логики рекомендаций и практический эффект

Раньше модели преимущественно заменяли друг друга в зависимости от длины истории. Теперь они работают совместно: формируют общий пул кандидатов, проходят per-model нормализацию score и объединяются сегментными весами. MMR дополнительно снижает тематическую избыточность выдачи.

Текущий эксперимент не показывает превосходства гибрида по всем метрикам: {best_classical_name} остаётся лидером по nDCG. Чистый вклад нейросети без MMR составляет {neural_delta:+.4f} nDCG, а MMR меняет ILD на {mmr_ild_delta:+.4f} и nDCG на {mmr_ndcg_delta:+.4f}. Поэтому финальный Hybrid следует выбирать ради компромисса персонализации и разнообразия, а при единственной цели nDCG — ориентироваться на лидера сравнения.

Прямое сопоставление старых и новых численных значений некорректно: изменились validation/test-протокол, evaluation-выборка и архитектура гибрида.

## 2. Протокол без утечки данных

1. Дубли пар `user_id—book_id` удалены до разбиения.
2. Воспроизводимый per-user split сохраняет минимум одну оценку в train. Исходный holdout дополнительно разделён на validation и test.
3. Строки: train = {config['split']['train_rows']}, validation = {config['split']['validation_rows']}, test = {config['split']['test_rows']}. Пересечений пар нет.
4. Все пользовательские и книжные статистики рассчитаны только по train. Веса гибрида и lambda MMR выбраны по validation. Test использован один раз для итоговой таблицы и RMSE.
5. У файла нет временных меток, поэтому split случайный, а не temporal. Это ограничение: реальное качество на будущих взаимодействиях может отличаться.

Формулы: `Precision@K = hits/K`, `Recall@K = hits/|Relevant|`, `DCG@K = Σ rel_i/log2(rank_i+1)`, `nDCG@K = DCG/IDCG`. Повторы удаляются до подсчёта hits. IDCG содержит `min(|Relevant|, K)` единиц.

## 3. Расширенные признаки

**Пользователи:** логарифм числа оценок, средний рейтинг, стандартное отклонение, доли оценок ≥4 и ≤2, энтропия распределения, нормированная активность и сегмент.

**Книги:** логарифм популярности, байесовски сглаженный средний рейтинг, стандартное отклонение и энтропия оценок, доля положительных оценок, глобальная популярность Goodreads, год публикации и 10 тематических долей тегов (fiction, fantasy, science fiction, romance, mystery, history, children, young adult, nonfiction, comics).

**Взаимодействия:** сходство TF-IDF-кандидата с положительной историей, Item-CF score, тематическая близость к истории, средний рейтинг пользователя и популярность кандидата. Пример сохранён в `artifacts/interaction_features_example.csv`.

Таблицы `data/processed/user_features.csv.gz` и `book_features.csv.gz` позволяют проверить каждый признак. Отсутствующие признаки заполнены нейтральными значениями до стандартизации; статистики scaler сохранены в конфигурации.

## 4. Гибридная архитектура

Каждая модель формирует до {config['hybrid']['candidate_limit']} кандидатов. Пули объединяются, train-книги исключаются. Сырые score несопоставимы, поэтому внутри списка каждой модели выполняется min-max нормализация; для отсутствующего кандидата нормированный score равен нулю. Затем применяются сегментные веса:

{chr(10).join(weights)}

Веса неотрицательны и суммируются в 1. Поиск включает стартовую конфигурацию, Dirichlet random search и one-hot варианты; цель — validation nDCG@10. Холодный старт без истории использует Popularity, а при указанной понравившейся книге — Content-Similar.

Для балансировки применён Maximal Marginal Relevance:

`MMR(i) = λ · relevance(i) − (1−λ) · max similarity(i, уже выбранные)`.

Выбрано **λ = {hybrid.diversity_lambda:.2f}** по validation-функции `nDCG@10 + 0.20·ILD@10` (коэффициент 0.20 задан заранее, не подобран по test). При λ=1 получается чистое ранжирование, при меньших значениях усиливается разнообразие. Абляции Hybrid-Classical и Hybrid (no MMR) показывают вклад каждого компонента.

## 5. Two-Tower нейросетевая модель

Архитектура реализована на NumPy: обучаемые ID-embeddings пользователя и книги объединяются со стандартизованными признаками соответствующей башни, проходят dense+tanh и дают векторы размерности {config['models']['two_tower']['output_dim']}. Score — скалярное произведение векторов. Обучение BPR использует пары «положительная train-оценка ≥4 / случайная непросмотренная книга». Градиенты вычисляются явно; оптимизатор — Adam.

BPR ближе к задаче top-N, чем RMSE: модель учит порядок пар, а не точное значение рейтинга. Нейросеть получила nDCG@10 = {neural['ndcg_at_k']:.4f}, Recall@10 = {neural['recall_at_k']:.4f}. Этот результат сравнивается с классическими моделями на тех же пользователях. Loss и pairwise accuracy по эпохам находятся в `artifacts/neural_history.csv` и на графике `reports/figures/11_neural_training.png`.

{feature_ablation_section}

## 6. Сегментный анализ

{chr(10).join(leaders)}

Сегменты: new = 1–2 train-оценки, warm = 3–9, active ≥10. Холодный старт без train-истории отдельно проверяется inference-тестом; в текущем split у всех известных пользователей остаётся хотя бы одна train-оценка.

## 7. Сильные и слабые стороны

- **Popularity:** простой и устойчивый fallback, но не персонализирован и усиливает popularity bias. В предоставленном срезе у многих книг искусственно одинаковое число оценок, поэтому бейзлайн нельзя экстраполировать на весь Goodreads.
- **Content-Based:** работает при малой истории и для новых книг, объясним тегами. Слабости — шумные пользовательские полки, синонимы и риск чрезмерной узости рекомендаций.
- **Item-Based CF:** учитывает совместные оценки и интерпретируется соседями. Чувствителен к разреженности; построение попарного сходства дороже, чем обращение к готовой top-K матрице.
- **SVD:** восстанавливает скрытые факторы и даёт прогноз рейтинга. RMSE не гарантирует высокий nDCG, поскольку pointwise-ошибка и качество первых позиций — разные цели.
- **Two-Tower:** BPR оптимизирует относительный порядок, а feature-ablation показывает фактический вклад каждой группы признаков. Эффект не обязан быть положительным для каждой группы отдельно; uniform negative sampling упрощает реальное распределение, а короткое обучение и отсутствие времени ограничивают качество.
- **Hybrid:** переключает веса по активности, снижает зависимость от одного сигнала и поддерживает fallback. Цена — больше вычислений, необходимость validation и сложнее диагностика.
- **MMR:** явно управляет разнообразием, но может пожертвовать релевантностью; фактическое направление эффекта указано в разделе результатов.

## 8. Выводы и рекомендации

Для production-прототипа следует сохранить Popularity как аварийный fallback, Content-Based для холодного старта и гибрид для пользователей с историей. Выбор между {best_classical_name}, Two-Tower и Hybrid нужно делать по сегменту и бизнес-ограничению: если важен только nDCG — по лидеру таблицы, если важны тематическое покрытие и discovery — совместно по nDCG, ILD, coverage и novelty.

Следующий эксперимент: актуальный датасет с временными метками, temporal split, более информативные hard negatives и отдельный A/B-тест с CTR, добавлением в список чтения и дочитыванием. Офлайн-метрики не доказывают причинный эффект на бизнес.

## 9. Воспроизведение

- `python scripts/train.py` — полный pipeline с обучением классических моделей.
- `python scripts/train.py --reuse-classical` — повторное использование проверенных классических артефактов при неизменном train.
- `python scripts/recommend.py --user-id 1 --n 10` — проверка персональной выдачи.
- `pytest -q` — unit- и integration-тесты.

Все графики построены по CSV текущего запуска. Подробные параметры и число строк зафиксированы в `artifacts/config.json`.
"""
    Path(path).write_text(text, encoding="utf-8")
