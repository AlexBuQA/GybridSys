from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import build_user_item_matrix
from src.features import (
    InteractionFeatureBuilder, build_book_features, build_user_features, scale_feature_arrays,
)


def small_data():
    train = pd.DataFrame(
        {"user_id": [1, 1, 1, 2, 2, 3], "book_id": [1, 2, 3, 1, 4, 2],
         "rating": [5, 4, 2, 3, 5, 4]}
    )
    books = pd.DataFrame(
        {"id": [1, 2, 3, 4], "book_id": [101, 102, 103, 104],
         "original_publication_year": [2000, 2001, np.nan, 1990],
         "ratings_count": [100, 200, 50, 500]}
    )
    tags = pd.DataFrame({"tag_id": [1, 2], "tag_name": ["fantasy", "science-fiction"]})
    book_tags = pd.DataFrame(
        {"goodreads_book_id": [101, 102, 104], "tag_id": [1, 2, 1], "count": [10, 8, 4]}
    )
    return train, books, tags, book_tags


def test_extended_features_are_meaningful() -> None:
    train, books, tags, book_tags = small_data()
    users = build_user_features(train, 3)
    items = build_book_features(train, books, book_tags, tags, 4)
    assert users.loc[0, "rating_count"] == 3
    assert users.loc[0, "mean_rating"] == pytest.approx(11 / 3)
    assert users.loc[0, "segment"] == "warm"
    assert items.loc[0, "rating_count"] == 2
    assert items.loc[0, "topic_fantasy"] > 0
    assert items.loc[1, "topic_science_fiction"] > 0
    user_array, item_array, scaler = scale_feature_arrays(users, items)
    assert np.isfinite(user_array).all() and np.isfinite(item_array).all()
    assert len(scaler["user_columns"]) == user_array.shape[1]


def test_interaction_features_use_history() -> None:
    train, books, tags, book_tags = small_data()
    users = build_user_features(train, 3)
    items = build_book_features(train, books, book_tags, tags, 4)
    matrix = build_user_item_matrix(train, n_users=3, n_items=4)
    builder = InteractionFeatureBuilder(matrix, users, items)
    result = builder.build(1, [1, 4], content_scores={1: 0.7}, item_cf_scores={4: 0.2})
    assert result.loc[0, "content_history_similarity"] == 0.7
    assert result.loc[1, "item_cf_score"] == 0.2
    assert np.isfinite(result["topic_affinity"]).all()
