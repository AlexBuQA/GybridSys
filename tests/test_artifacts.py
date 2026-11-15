from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


def test_saved_pipeline_returns_recommendations() -> None:
    root = Path(__file__).resolve().parents[1]
    hybrid_path = root / "artifacts" / "hybrid.joblib"
    if not hybrid_path.exists():
        return
    hybrid = joblib.load(hybrid_path)
    train = pd.read_csv(root / "data" / "processed" / "train.csv.gz")
    user_id = int(train["user_id"].iloc[0])
    recommendations = hybrid.recommend(user_id, 10)
    assert len(recommendations) == 10
    assert len({book for book, _ in recommendations}) == 10
    seen = set(train.loc[train["user_id"] == user_id, "book_id"])
    assert not seen.intersection(book for book, _ in recommendations)
    config = json.loads((root / "artifacts" / "config.json").read_text(encoding="utf-8"))
    assert config["evaluation"]["k"] == 10
    assert config["models"]["two_tower"]["objective"].startswith("Bayesian")
