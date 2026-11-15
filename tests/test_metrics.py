from __future__ import annotations

import math

import pytest

from src.metrics import ndcg_at_k, precision_at_k, recall_at_k, ranking_metrics


def test_precision_recall_ndcg_manual_example() -> None:
    recommended, relevant = [1, 2, 3, 4], {2, 4, 5}
    assert precision_at_k(recommended, relevant, 4) == 0.5
    assert recall_at_k(recommended, relevant, 4) == pytest.approx(2 / 3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(recommended, relevant, 4) == pytest.approx(dcg / idcg)


def test_ndcg_rewards_relevant_item_at_the_top() -> None:
    relevant = {1, 2}
    assert ndcg_at_k([1, 3, 2], relevant, 3) > ndcg_at_k([3, 2, 1], relevant, 3)
    assert ndcg_at_k([1, 2], relevant, 2) == 1.0
    assert ndcg_at_k([3, 4], relevant, 2) == 0.0


def test_duplicates_do_not_inflate_hits() -> None:
    values = ranking_metrics([1, 1, 1, 2], {1, 2}, 4)
    assert values["hits"] == 2
    assert values["precision_at_k"] == 0.5


def test_empty_relevant_set_and_invalid_k() -> None:
    assert recall_at_k([1], set(), 1) == 0.0
    assert ndcg_at_k([1], set(), 1) == 0.0
    with pytest.raises(ValueError):
        precision_at_k([1], {1}, 0)
