from __future__ import annotations

import pandas as pd

from src.split import split_validation_test


def test_validation_test_are_disjoint_and_reproducible() -> None:
    holdout = pd.DataFrame(
        {"user_id": [1] * 4 + [2] * 2 + [3], "book_id": [1, 2, 3, 4, 1, 5, 6],
         "rating": [5, 4, 3, 2, 5, 4, 4]}
    )
    validation, test = split_validation_test(holdout, seed=42)
    validation_again, test_again = split_validation_test(holdout, seed=42)
    pd.testing.assert_frame_equal(validation, validation_again)
    pd.testing.assert_frame_equal(test, test_again)
    left = set(map(tuple, validation[["user_id", "book_id"]].to_numpy()))
    right = set(map(tuple, test[["user_id", "book_id"]].to_numpy()))
    assert not left.intersection(right)
    assert len(validation) + len(test) == len(holdout)
    assert set(validation.user_id) == {1, 2}
    assert 3 in set(test.user_id)
