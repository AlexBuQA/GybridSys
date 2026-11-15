from __future__ import annotations

import numpy as np

from src.ablation import ABLATION_MODELS, FULL, ID_ONLY, ITEM_ONLY, USER_ONLY, make_feature_variants


def test_feature_masks_preserve_shapes_and_information() -> None:
    user = np.arange(12, dtype=np.float32).reshape(4, 3)
    item = np.arange(20, dtype=np.float32).reshape(5, 4)
    variants = make_feature_variants(user, item)
    assert list(variants) == ABLATION_MODELS
    for user_array, item_array in variants.values():
        assert user_array.shape == user.shape
        assert item_array.shape == item.shape
    assert not variants[ID_ONLY][0].any() and not variants[ID_ONLY][1].any()
    np.testing.assert_array_equal(variants[USER_ONLY][0], user)
    assert not variants[USER_ONLY][1].any()
    assert not variants[ITEM_ONLY][0].any()
    np.testing.assert_array_equal(variants[ITEM_ONLY][1], item)
    np.testing.assert_array_equal(variants[FULL][0], user)
    np.testing.assert_array_equal(variants[FULL][1], item)
