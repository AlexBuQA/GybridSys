# Данные Goodbooks-10k

Источник: присланный `archive.zip` с Goodbooks-10k. Описание набора и актуальная расширенная версия указаны автором датасета на странице проекта: https://github.com/zygmuntz/goodbooks-10k

Лицензия исходного набора по приложенному описанию: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**: https://creativecommons.org/licenses/by-sa/4.0/

## Файлы

- `ratings.csv` — явные оценки 1–5: `book_id,user_id,rating`.
- `books.csv` — каталог и метаданные; внутренний `id` = 1..10000.
- `book_tags.csv` — применение тегов: `goodreads_book_id,tag_id,count`.
- `tags.csv` — расшифровка `tag_id` в `tag_name`.
- `to_read.csv` — пары пользователь/книга из списка чтения; в текущих моделях не используется.
- `sample_book.xml` — пример исходных метаданных; для запуска не требуется.

## Важные связи ID

`ratings.book_id` и `books.id` — внутренний ID 1..10000. `book_tags.goodreads_book_id` соответствует `books.book_id` (Goodreads ID), а не `books.id`. Неверная связь испортит контентные профили.

## Особенности присланной версии

- Присланная версия содержит 2 278 дублей пар `user_id—book_id`. Pipeline сохраняет последнюю строку до split.
- После очистки остаётся 979 478 оценок, 53 424 пользователя и 10 000 книг.
- 6 248 книг (62,48%) имеют ровно 100 оценок после очистки; верхний предел среза ограничивает естественный длинный хвост популярности. Для дополнительного анализа используется `books.ratings_count`.
- Временных меток в `ratings.csv` нет, поэтому реализован случайный per-user holdout, а не временной split.
- `books.ratings_count` и `books.average_rating` — глобальные метаданные Goodreads. Они используются в EDA/UI, но Popularity обучается только по train-оценкам, чтобы не создавать утечку.

Исходные CSV сохранены без изменений. Очищенные фиксированные train/test наборы находятся в `data/processed/` в gzip-формате. При использовании данных соблюдайте условия CC BY-SA 4.0 и указывайте Goodbooks-10k.
