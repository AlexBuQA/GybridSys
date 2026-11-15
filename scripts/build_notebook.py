"""Создание и выполнение итогового Notebook по сохранённым артефактам."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    })
    cells = []

    def md(text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text: str) -> None:
        cells.append(nbf.v4.new_code_cell(text))

    md("""# Итоговый проект: гибридная система рекомендаций книг

**Автор:** Александра Бужор · [@Angry_C_F_O](https://t.me/Angry_C_F_O)

Проект объединяет Popularity, Content-Based, Item-Based CF и SVD из домашней работы с Two-Tower нейросетью, сегментным взвешиванием и MMR-диверсификацией. Notebook воспроизводит анализ зафиксированного pipeline; обучение выполняется командой `python scripts/train.py`.""")
    code("""from pathlib import Path
import json
import joblib
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / 'artifacts').exists():
    ROOT = ROOT.parent
ARTIFACTS = ROOT / 'artifacts'
REPORTS = ROOT / 'reports'
config = json.loads((ARTIFACTS / 'config.json').read_text(encoding='utf-8'))
metrics = pd.read_csv(ARTIFACTS / 'metrics.csv')
segments = pd.read_csv(ARTIFACTS / 'segment_metrics.csv')
print(config['project'], config['version'])""")

    md("""## 1. Протокол и контроль утечки

Дубли удаляются до split. Модели обучаются на train; веса и MMR-параметр выбираются на validation; test используется для финального сравнения. Релевантна оценка ≥4. Метрики макроусредняются по одной и той же выборке пользователей, train-книги исключаются.""")
    code("""pd.DataFrame([
    {'split': key, 'rows': value}
    for key, value in config['split'].items() if key.endswith('_rows')
])""")

    md("""## 2. Расширенные признаки

Пользовательские признаки описывают активность и распределение оценок. Книжные — популярность, байесовский рейтинг, разнообразие оценок и тематические теги. Все рейтинговые статистики получены только из train. Для пары пользователь–книга дополнительно рассчитываются content similarity, Item-CF score и тематическая близость к истории.""")
    code("""users = pd.read_csv(ROOT / 'data/processed/user_features.csv.gz')
books = pd.read_csv(ROOT / 'data/processed/book_features.csv.gz')
print('User features:', config['features']['user'])
print('Book features:', config['features']['book'])
display(users.head())
display(books.head())
display(pd.read_csv(ARTIFACTS / 'interaction_features_example.csv'))""")

    md("""## 3. Метрики

$Precision@K = |TopK \\cap Relevant|/K$

$Recall@K = |TopK \\cap Relevant|/|Relevant|$

$nDCG@K = DCG@K/IDCG@K$, где релевантность бинарна, а $IDCG$ соответствует идеальному порядку. ILD измеряет среднее попарное `1 − cosine` внутри списка; Coverage — покрытие каталога.""")
    code("""columns = ['model', 'precision_at_k', 'recall_at_k', 'ndcg_at_k', 'ild_at_k', 'novelty_at_k', 'catalog_coverage']
metrics[columns].sort_values('ndcg_at_k', ascending=False).reset_index(drop=True)""")
    code("display(Image(filename=str(REPORTS / 'figures/06_model_comparison.png')))")

    md("""## 4. Two-Tower: архитектура и обучение

Обучаемые ID-embeddings и непрерывные признаки каждой сущности проходят отдельную dense+tanh башню. Score — dot product. BPR максимизирует разницу `score(user, positive) − score(user, sampled_negative)`. Градиенты и Adam реализованы на NumPy. График показывает фактические train loss и pairwise accuracy; это не validation-метрики.""")
    code("""history = pd.read_csv(ARTIFACTS / 'neural_history.csv')
display(history)
display(Image(filename=str(REPORTS / 'figures/11_neural_training.png')))""")

    md("""## 4.1. Feature ablation: действительно ли помогают признаки

Сравниваются четыре варианта: только ID, ID + пользовательские признаки, ID + книжные признаки и полный набор. Train, seed, размерности сети, BPR-параметры и 600 test-пользователей совпадают. Отключённые стандартизованные признаки заменяются нулями (средними train-значениями), поэтому архитектура и инициализация остаются сопоставимыми. Полная модель используется та же, что и в гибриде. Веса гибрида по этим результатам не перенастраиваются.""")
    code("""ablation_path = ARTIFACTS / 'feature_ablation_metrics.csv'
if ablation_path.exists():
    ablation = pd.read_csv(ablation_path)
    base = ablation.set_index('model').loc['Two-Tower ID only', 'ndcg_at_k']
    ablation['delta_ndcg_vs_id_only'] = ablation['ndcg_at_k'] - base
    display(ablation[['model', 'precision_at_k', 'recall_at_k', 'ndcg_at_k', 'delta_ndcg_vs_id_only', 'ild_at_k']])
    display(Image(filename=str(REPORTS / 'figures/12_feature_ablation.png')))
    display(Image(filename=str(REPORTS / 'figures/13_feature_ablation_training.png')))
    indexed_ablation = ablation.set_index('model')
    full = indexed_ablation.loc['Two-Tower full features', 'ndcg_at_k']
    user = indexed_ablation.loc['Two-Tower + user features', 'ndcg_at_k']
    item = indexed_ablation.loc['Two-Tower + item features', 'ndcg_at_k']
    print(f'Полный набор vs ID-only, Δ nDCG@10: {full - base:+.8f}')
    relative = f'{(full - base) / base:+.2%}' if base else 'не определён'
    print('Относительный эффект полного набора:', relative)
    print(f'Пользовательские признаки vs ID-only, Δ nDCG@10: {user - base:+.8f}')
    print(f'Книжные признаки vs ID-only, Δ nDCG@10: {item - base:+.8f}')
    print('Лидер по nDCG@10:', indexed_ablation['ndcg_at_k'].idxmax())
    print('Вывод относится к текущим split и seed; веса гибрида по test не перенастраиваются.')
else:
    print('Feature ablation ещё не запущен: python scripts/feature_ablation.py')""")

    md("""## 5. Гибрид: кандидаты, веса и разнообразие

Модели формируют общий пул кандидатов. Score каждой модели min-max нормализуется, затем применяются сегментные веса. MMR использует `λ·relevance − (1−λ)·max_similarity`. Коэффициент diversity objective равен 0.20 и задан до просмотра test. Hybrid-Classical и Hybrid (no MMR) — ablation-контроли.""")
    code("""weights = pd.DataFrame(config['hybrid']['weights']).T
weights.index.name = 'segment'
print('MMR lambda:', config['hybrid']['diversity_lambda'])
display(weights)
display(pd.read_csv(ARTIFACTS / 'validation_diversity_search.csv'))
display(Image(filename=str(REPORTS / 'figures/09_diversity_tradeoff.png')))""")
    code("display(Image(filename=str(REPORTS / 'figures/10_quality_diversity_map.png')))")

    md("""## 6. Сегментный анализ

New: 1–2 train-оценки; warm: 3–9; active: ≥10. У new недостаточно validation-наблюдений после split, поэтому стартовые веса content-oriented заданы как холодно-стартовая эвристика. Warm/active веса оптимизируются только на validation.""")
    code("""segments.sort_values(['segment', 'ndcg_at_k'], ascending=[True, False])""")
    code("display(Image(filename=str(REPORTS / 'figures/07_segment_ndcg.png')))")

    md("""## 7. Проверка персональной выдачи

Ниже загружается сериализованный гибрид и формируется новый список. `hybrid_score` — взвешенный нормированный score; исходные score моделей приведены в собственных шкалах и не должны сравниваться напрямую.""")
    code("""hybrid = joblib.load(ARTIFACTS / 'hybrid.joblib')
example = json.loads((ARTIFACTS / 'examples.json').read_text(encoding='utf-8'))
user_id = int(example['user_id'])
recommendations = hybrid.recommend(user_id=user_id, n=10)
catalog = pd.read_csv(ROOT / 'data/raw/books.csv', low_memory=False).set_index('id')
rows = []
for rank, (book_id, score) in enumerate(recommendations, 1):
    rows.append({'rank': rank, 'book_id': book_id, 'title': catalog.loc[book_id, 'title'],
                 'authors': catalog.loc[book_id, 'authors'], 'score': score})
print('User:', user_id, '| segment:', hybrid.segment_for_user(user_id))
pd.DataFrame(rows)""")

    md("""## 8. Выводы

Лидер определяется по фактическому test nDCG, а не по предположению о преимуществе нейросети. Вклад Two-Tower оценивается разницей Hybrid (no MMR) и Hybrid-Classical без влияния MMR, вклад MMR — разницей Hybrid и Hybrid (no MMR). Для выбора модели учитываются одновременно nDCG, ILD, novelty, coverage и сегмент. Полный критический разбор и ограничения находятся в `reports/REPORT.md`.""")
    code("""indexed = metrics.set_index('model')
best = indexed['ndcg_at_k'].idxmax()
print('Лидер по nDCG@10:', best, indexed.loc[best, 'ndcg_at_k'])
print('Вклад нейросети в nDCG без MMR:', indexed.loc['Hybrid (no MMR)', 'ndcg_at_k'] - indexed.loc['Hybrid-Classical', 'ndcg_at_k'])
print('Изменение ILD от MMR:', indexed.loc['Hybrid', 'ild_at_k'] - indexed.loc['Hybrid (no MMR)', 'ild_at_k'])
print('Изменение nDCG от MMR:', indexed.loc['Hybrid', 'ndcg_at_k'] - indexed.loc['Hybrid (no MMR)', 'ndcg_at_k'])""")

    md("""## 9. Воспроизведение

- `python scripts/train.py` — обучение всего pipeline;
- `python scripts/train.py --reuse-classical` — пересчёт новых компонентов;
- `pytest -q` — тесты;
- `python scripts/validate_artifacts.py` — inference smoke-test.

Ограничения: случайный split без времени, шумные теги, uniform negative sampling, offline-метрики вместо A/B-теста. Следующий шаг — актуальные временные данные и оценка влияния на добавление в список чтения и дочитывание.""")
    notebook.cells = cells
    path = ROOT / "notebooks" / "final_project.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.write(notebook, path)
    print(f"Выполненный Notebook сохранён: {path}")


if __name__ == "__main__":
    main()
