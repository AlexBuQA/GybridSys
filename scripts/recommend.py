"""Печать рекомендаций и вклада моделей для выбранного пользователя."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model", default="hybrid", choices=["hybrid", "popularity", "content", "item_cf", "svd", "two_tower"])
    parser.add_argument("--seed-book-id", type=int, default=None, help="Понравившаяся книга для нового пользователя")
    parser.add_argument("--no-mmr", action="store_true")
    parser.add_argument("--no-neural", action="store_true")
    parser.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    args = parser.parse_args()
    artifacts = Path(args.artifacts_dir)
    hybrid = joblib.load(artifacts / "hybrid.joblib")
    books = pd.read_csv(Path(args.data_dir) / "books.csv", low_memory=False).set_index("id")
    if args.model == "hybrid":
        recommendations = hybrid.recommend(
            args.user_id, args.n, seed_book_id=args.seed_book_id,
            include_neural=not args.no_neural, diversify=not args.no_mmr,
        )
        print(f"Сегмент: {hybrid.segment_for_user(args.user_id)}; MMR lambda={hybrid.diversity_lambda:.2f}")
    else:
        if args.seed_book_id is not None:
            parser.error("--seed-book-id доступен только для --model hybrid")
        recommendations = hybrid.models[args.model].recommend(args.user_id, args.n)
    rows = []
    for rank, (book_id, score) in enumerate(recommendations, start=1):
        book = books.loc[book_id]
        rows.append({"rank": rank, "book_id": book_id, "title": book["title"], "authors": book["authors"], "score": round(score, 6)})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nПользователь: {args.user_id}; модель: {args.model}; рекомендаций: {len(rows)}")


if __name__ == "__main__":
    main()
