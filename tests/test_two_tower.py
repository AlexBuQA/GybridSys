from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import build_user_item_matrix
from src.models.popularity import PopularityRecommender
from src.models.two_tower import TwoTowerRecommender


def test_two_tower_trains_and_excludes_seen_books() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for user in range(1, 5):
        for book in range(1, 7):
            if (user + book) % 3 != 0:
                rows.append((user, book, 4 + int((user + book) % 2)))
    rows.extend([(1, 7, 2), (2, 7, 5), (3, 8, 5)])
    train = pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])
    matrix = build_user_item_matrix(train, n_users=4, n_items=8)
    popularity = PopularityRecommender(min_ratings=1).fit(train, matrix)
    model = TwoTowerRecommender.fit(
        train, matrix, rng.normal(size=(4, 3)).astype(np.float32),
        rng.normal(size=(8, 5)).astype(np.float32), popularity,
        embedding_dim=4, output_dim=6, epochs=2, batch_size=8, seed=42, verbose=False,
    )
    assert len(model.history) == 2
    assert np.isfinite([row["train_bpr_loss"] for row in model.history]).all()
    assert model.item_vectors().shape == (8, 6)
    recommendations = model.recommend(1, 3)
    assert len(recommendations) == 3
    seen = {book for user, book, _ in rows if user == 1}
    assert not seen.intersection(book for book, _ in recommendations)
