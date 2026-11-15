# Гибридная система рекомендаций книг с Two-Tower

**Итоговый проект · Goodbooks-10k**  
**Автор:** Александра Бужор · Telegram: [@Angry_C_F_O](https://t.me/Angry_C_F_O)

Самостоятельный репозиторий для локального запуска в VS Code и публикации на GitHub. Развивает модели из домашней работы: Popularity, Content-Based, Item-Based CF и SVD. Добавлены расширенные признаки, кандидатный гибрид с весами по сегментам, MMR-диверсификация, Two-Tower нейросеть с BPR и Adam, validation-подбор параметров, сквозной pipeline, аналитический отчёт, выполненный Notebook и тесты.

## Чем итоговый проект отличается от работы «Построение прототипа книжного рекомендательного сервиса»

Предыдущая работа — это **работающий сервис с базовыми моделями и Streamlit-интерфейсом**. Итоговый проект — **исследовательская ML-система**, в которой классические модели стали основой для обучаемого гибрида, нейросетевого канала и явного управления разнообразием.

| Компонент | Прототип рекомендательного сервиса | Итоговый проект |
|---|---|---|
| Цель | Построить сервис и сравнить базовые алгоритмы | Исследовать гибридизацию, нейросеть и баланс relevance/diversity |
| Классические модели | Popularity, Content-Based, Item-Based CF, SVD | Те же проверенные модели и артефакты переиспользованы |
| Нейросеть | Отсутствовала | Two-Tower с embeddings, dense + tanh, BPR-loss и Adam |
| Логика гибрида | Переключение Popularity → Content → SVD по правилам | Общий пул кандидатов от всех моделей и взвешенное объединение score |
| Веса моделей | Фиксированные правила выбора одной модели | Сегментные веса new/warm/active, подбираемые на validation |
| Признаки | Оценки, названия и теги | Дополнительно активность, разброс и энтропия оценок, популярность, тематики и признаки взаимодействий |
| Разнообразие | Не контролировалось явно | MMR с настройкой lambda на validation |
| Разбиение | Train/test | Train/validation/test без подбора параметров по test |
| Сравнение | Отдельные модели и простой Hybrid | Дополнительно Hybrid-Classical, Hybrid без MMR и финальный Hybrid |
| Метрики | Precision, Recall, nDCG, RMSE | Добавлены ILD, novelty, coverage, доверительные интервалы и сегментный анализ |
| Интерфейс | Streamlit-приложение | CLI и выполненный Notebook; Streamlit не входит в этот вариант |
| Фокус | Пользовательский прототип сервиса | Воспроизводимый ML-pipeline и ablation-анализ |

### Что унаследовано из домашней работы

Сохранены Goodbooks-10k, очистка и дедупликация, исходный train/holdout, четыре классические модели и их обученные артефакты, TF-IDF-профили книг, разреженная Item-CF, SVD со смещениями, базовые ranking-метрики, фильтрация прочитанного и EDA. Поэтому итоговый проект является развитием домашней работы, а не независимой переписанной версией.

### Практический эффект изменений

Раньше модели в основном заменяли друг друга в зависимости от длины истории. Теперь они формируют общий пул кандидатов и работают совместно: score каждой модели нормализуется, взвешивается по сегменту пользователя, после чего MMR может снизить тематическую избыточность списка.

По итогам текущего эксперимента Content-Based остаётся лучшей отдельной моделью по nDCG@10 (0,1404). Добавление Two-Tower повышает nDCG гибрида без MMR на 0,0005. MMR увеличивает ILD с 0,8518 до 0,9083, но снижает nDCG до 0,1218. Следовательно, финальный Hybrid выбирается ради компромисса персонализации и разнообразия, а не потому, что он автоматически превосходит все модели по каждой метрике.

Старые и новые числовые значения нельзя сравнивать напрямую: изменились validation/test-протокол, evaluation-выборка и архитектура гибрида.

## Что реализовано

| Требование | Где проверить |
|---|---|
| Гибрид моделей из ДЗ | `src/models/hybrid.py`, `artifacts/metrics.csv` |
| Признаки пользователей, книг, взаимодействий | `src/features.py`, `data/processed/*features*` |
| Баланс персонализации и разнообразия | сегментные веса + MMR, `validation_diversity_search.csv` |
| Precision@K, Recall@K, nDCG@K | `src/metrics.py`, `tests/test_metrics.py` |
| Сравнение отдельных моделей и гибрида | `reports/REPORT.md`, `artifacts/metrics.csv` |
| Сегментный и критический анализ | `artifacts/segment_metrics.csv`, разделы 6–8 отчёта |
| Нейросетевая архитектура | `src/models/two_tower.py` |
| Интеграция нейросети | Two-Tower — пятый источник кандидатов гибрида |
| Сквозной воспроизводимый процесс | `src/pipeline.py`, `scripts/train.py` |
| Визуализации | `reports/figures/01…13_*.png` |
| Доказательство вклада признаков | `src/ablation.py`, `artifacts/feature_ablation_metrics.csv` |
| Документирование | этот README, отчёт и выполненный Notebook |

## 1. Быстрый запуск в VS Code на Windows

1. Откройте в VS Code именно папку `hybrid-book-recommender`.
2. Установите Python 3.11 и при установке включите команду `py`.
3. В терминале VS Code выполните:

```bat
setup.bat
run_demo.bat
run_tests.bat
```

`setup.bat` создаёт `.venv` и устанавливает зависимости. Обучение для демонстрации **не требуется**: данные, обученные модели, результаты метрик, графики и Notebook включены.

Выберите интерпретатор `.venv\Scripts\python.exe`: `Ctrl+Shift+P` → `Python: Select Interpreter`. Готовые launch-конфигурации находятся в разделе **Run and Debug**.

### Ручная установка

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/recommend.py --user-id 1 --n 10
pytest -q
```

macOS / Linux:

```bash
chmod +x setup.sh run_demo.sh run_tests.sh
./setup.sh
./run_demo.sh
./run_tests.sh
```

## 2. Проверка рекомендаций

Персональная выдача финального гибрида:

```bash
python scripts/recommend.py --user-id 1 --n 10
```

Отдельная модель:

```bash
python scripts/recommend.py --user-id 1 --model two_tower --n 10
python scripts/recommend.py --user-id 1 --model content --n 10
```

Ablation без нейросети или без MMR:

```bash
python scripts/recommend.py --user-id 1 --no-neural
python scripts/recommend.py --user-id 1 --no-mmr
```

Холодный старт по одной понравившейся книге:

```bash
python scripts/recommend.py --user-id 999999 --seed-book-id 1 --n 10
```

Smoke-проверка сохранённых артефактов:

```bash
python scripts/validate_artifacts.py
```

## 3. Сквозной pipeline

```text
CSV → очистка и дедупликация → per-user train/holdout → validation/test
    → EDA → классические модели из ДЗ
    → train-only признаки пользователей и книг
    → Two-Tower + BPR + Adam
    → объединение кандидатов → min-max нормализация score
    → подбор сегментных весов на validation
    → подбор lambda MMR на validation
    → финальные метрики на test → CSV, графики, отчёт и примеры
```

Полный пересчёт с обучением всех классических моделей:

```bash
python scripts/train.py
```

Повторный расчёт признаков, нейросети и гибрида с переиспользованием классических моделей:

```bash
python scripts/train.py --reuse-classical
```

Ускоренная проверка:

```bash
python scripts/train.py --quick --reuse-classical
```

`--reuse-classical` допустим только при неизменном train: pipeline сравнивает разреженные матрицы и останавливается при несовпадении. После `--quick` артефакты соответствуют уменьшенной выборке; для итоговых значений повторите обычную команду.

Основные параметры можно изменить через CLI, например:

```bash
python scripts/train.py --neural-epochs 8 --weight-trials 64 --eval-users 1500 --reuse-classical
```

Список всех параметров: `python scripts/train.py --help`.

## 4. Данные и честная оценка

Используются `ratings.csv`, `books.csv`, `tags.csv`, `book_tags.csv`. После удаления 2 278 дублей остаётся 979 478 оценок, 53 424 пользователя и 10 000 книг.

- train: 761 605 оценок;
- validation: 104 958 оценок — подбор весов гибрида и lambda MMR;
- test: 112 915 оценок — финальное сравнение и RMSE;
- релевантная книга: test rating ≥ 4;
- одинаковая случайная выборка пользователей для всех моделей, seed = 42, K = 10;
- уже оценённые в train книги исключены из выдачи;
- пользовательские и книжные статистики рассчитываются только по train.

В исходном файле нет временных меток, поэтому используется воспроизводимый случайный per-user split, а не временной. У новых пользователей с 1–2 train-оценками в validation недостаточно наблюдений: их стартовые content-oriented веса являются эвристикой холодного старта; веса warm/active подбираются на validation.

## 5. Модели и признаки

### Классические модели

- **Popularity:** средний train-рейтинг при достаточном числе оценок, fallback для холодного старта.
- **Content-Based:** TF-IDF названия и до 30 тегов; профиль пользователя строится по книгам с оценками ≥4.
- **Item-Based CF:** косинусное сходство книг, разреженная top-K матрица соседей.
- **SVD:** TruncatedSVD остатков после регуляризованных пользовательских и книжных смещений.

### Расширенные признаки

Пользователи: активность, число оценок, средний рейтинг, разброс, доли положительных и отрицательных оценок, энтропия.

Книги: популярность, байесовски сглаженный рейтинг, разброс и энтропия оценок, год, глобальная популярность и 10 тематических категорий тегов.

Взаимодействия: сходство кандидата с историей, Item-CF score, тематическая близость к предпочтениям, средний рейтинг пользователя и популярность книги.

### Two-Tower

Каждая башня объединяет обучаемый ID-embedding и стандартизованные признаки, затем применяет dense + tanh. Скалярное произведение пользовательского и книжного векторов — score. Обучение идёт на положительных train-оценках ≥4 с BPR-loss и случайными непросмотренными отрицательными примерами.

Нейросеть реализована на NumPy с явным backpropagation и Adam. PyTorch/TensorFlow не нужны: это упрощает установку, а `tests/test_two_tower.py` проверяет обучение и inference.

### Feature ablation

Для изолированной проверки вклада признаков реализованы четыре варианта Two-Tower: только ID, ID + пользовательские признаки, ID + книжные признаки и полный набор. Во всех вариантах совпадают train, seed, размерности архитектуры, BPR-параметры и 600 test-пользователей. Отключённые стандартизованные признаки заменяются нулями — средними train-значениями, поэтому размерности сети и инициализация остаются сопоставимыми. Полная модель та же, что используется в гибриде.

Фактические результаты эксперимента:

| Вариант Two-Tower | Precision@10 | Recall@10 | nDCG@10 | Δ nDCG к ID-only | ILD@10 |
|---|---:|---:|---:|---:|---:|
| ID only | 0,0007 | 0,0044 | 0,0038 | +0,0000 | 0,9667 |
| + user features | 0,0008 | 0,0043 | 0,0027 | −0,0012 | 0,9628 |
| + item features | 0,0033 | 0,0148 | 0,0072 | +0,0034 | 0,9374 |
| **Full features** | **0,0030** | **0,0143** | **0,0075** | **+0,0037** | 0,9418 |

Полный набор улучшил nDCG@10 относительно ID-only на **0,003684, или 96,2%**, и оказался лучшим из четырёх вариантов. Основной вклад дали книжные признаки (+0,003355). Пользовательские признаки отдельно снизили nDCG на 0,001173, но в сочетании с книжными дали небольшой прирост +0,000329. Следовательно, полезность зависит от комбинации признаков: вывод относится к текущим split, seed и бюджету обучения, а не является универсальной гарантией.

Повторить эксперимент без переобучения гибрида:

```bash
python scripts/feature_ablation.py
```

Или выполнить его внутри общего pipeline:

```bash
python scripts/train.py --reuse-classical --feature-ablation
```

Результаты сохраняются в `artifacts/feature_ablation_*.csv`, графики — `12_feature_ablation.png` и `13_feature_ablation_training.png`. Веса гибрида по этим test-результатам не перенастраиваются. Признаки взаимодействий используются для анализа и объяснений и не подаются в Two-Tower.

### Гибрид и MMR

Каждая модель вносит кандидатов в общий пул. Score каждой модели нормализуются отдельно, поскольку рейтинг, косинус и dot product несопоставимы. Веса зависят от сегмента: new, warm, active. Для нового пользователя без истории работает Popularity или Content-Similar по seed-книге.

MMR выбирает книги по формуле:

```text
lambda * relevance - (1 - lambda) * max_similarity_to_selected
```

Lambda выбирается на validation по заранее заданной функции `nDCG@10 + 0.20 * ILD@10`. Финальный отчёт отдельно показывает изменение nDCG и разнообразия, включая случаи, когда компромисс не улучшает все метрики одновременно.

## 6. Результаты и Notebook

Фактические числа полного запуска на 600 одинаковых test-пользователях:

| Модель | Precision@10 | Recall@10 | nDCG@10 | ILD@10 |
|---|---:|---:|---:|---:|
| Popularity | 0,0002 | 0,0004 | 0,0002 | 0,8283 |
| Content-Based | **0,0337** | **0,2093** | **0,1404** | 0,8299 |
| Item-Based CF | 0,0042 | 0,0301 | 0,0141 | 0,9364 |
| SVD | 0,0003 | 0,0006 | 0,0004 | 0,8348 |
| Two-Tower | 0,0030 | 0,0143 | 0,0075 | **0,9418** |
| Hybrid-Classical | 0,0308 | 0,1989 | 0,1349 | 0,8512 |
| Hybrid (no MMR) | 0,0310 | 0,1977 | 0,1354 | 0,8518 |
| Hybrid | 0,0263 | 0,1727 | 0,1218 | 0,9083 |

Content-Based лидирует по ranking-метрикам. Добавление Two-Tower без MMR повысило nDCG гибрида на 0,0005; MMR увеличил ILD на 0,0565 при снижении nDCG на 0,0135. Таким образом, финальный гибрид полезен для компромисса relevance/diversity, но не заменяет Content-Based, если единственная цель — nDCG. SVD RMSE на test = 0,8514; это несопоставимая с ranking-метриками задача.

Детальные результаты:

- сводная таблица: `artifacts/metrics.csv`;
- сегменты: `artifacts/segment_metrics.csv`;
- веса: `artifacts/validation_weight_search.csv` и `artifacts/config.json`;
- история нейросети: `artifacts/neural_history.csv`;
- feature-ablation: `artifacts/feature_ablation_metrics.csv`, `feature_ablation_history.csv` и `feature_ablation_per_user.csv.gz`;
- графики ablation: `reports/figures/12_feature_ablation.png` и `13_feature_ablation_training.png`;
- аналитические выводы: [reports/REPORT.md](reports/REPORT.md);
- выполненный Notebook: [notebooks/final_project.ipynb](notebooks/final_project.ipynb).

Notebook не перетренировывает модели: он читает зафиксированные результаты, показывает формулы, признаки, архитектуру, метрики, графики и пример выдачи. Для пересоздания outputs:

```bash
python scripts/build_notebook.py
```

Для интерактивной работы:

```bash
jupyter notebook notebooks/final_project.ipynb
```

## 7. Структура репозитория

```text
hybrid-book-recommender/
├── src/
│   ├── data.py                 # загрузка, очистка, исходный holdout
│   ├── split.py                # validation/test без пересечений
│   ├── features.py             # расширенные признаки
│   ├── ablation.py             # обучение и оценка четырёх feature-вариантов
│   ├── ablation_reporting.py   # выводы ablation по фактическим метрикам
│   ├── metrics.py              # ranking, ILD, novelty, сегменты
│   ├── pipeline.py             # единый end-to-end процесс
│   ├── reporting.py            # отчёт по фактическим результатам
│   ├── visualization.py        # сравнительные графики
│   └── models/
│       ├── popularity.py
│       ├── content.py
│       ├── item_cf.py
│       ├── svd.py
│       ├── two_tower.py         # нейросеть, BPR, backprop, Adam
│       └── hybrid.py            # candidate union, веса, MMR, ablation
├── scripts/
│   ├── train.py
│   ├── feature_ablation.py     # отдельный запуск без переобучения гибрида
│   ├── recommend.py
│   ├── validate_artifacts.py
│   └── build_notebook.py
├── tests/                      # unit + integration tests, включая feature masks
├── notebooks/final_project.ipynb
├── artifacts/                  # обученные модели, метрики, примеры
├── data/                       # Goodbooks-10k + processed features/splits
├── reports/                    # REPORT.md и 13 графиков
├── .vscode/                    # интерпретатор, запуск, тесты
├── .github/workflows/ci.yml    # GitHub Actions
└── requirements.txt
```

## 8. Публикация на GitHub

Создайте пустой репозиторий на GitHub, затем в корне проекта:

```bash
git init
git add .
git commit -m "Hybrid book recommender with Two-Tower and MMR"
git branch -M main
git remote add origin https://github.com/ВАШ_ЛОГИН/hybrid-book-recommender.git
git push -u origin main
```

`.venv`, кэши и временные файлы исключены через `.gitignore`. Данные и артефакты намеренно включены для запуска «из коробки». После push GitHub Actions установит зависимости, выполнит тесты и проверит сохранённые модели.

## 9. Ограничения

Это учебный offline-прототип, а не доказательство production-эффекта. Случайный split не моделирует будущее; sample evaluation добавляет статистическую неопределённость; пользовательские теги содержат шум; uniform negatives упрощают негативный опыт; метрики не измеряют дочитывания и бизнес-ценность. Следующий этап — temporal split на актуальных данных и A/B-тест.

## Лицензии

Код — MIT, см. `LICENSE`. Goodbooks-10k предоставлен под CC BY-SA 4.0; источник и особенности ID описаны в `data/README.md`. Проект не является официальным сервисом Goodreads.
