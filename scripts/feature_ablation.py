"""Запуск feature-ablation по уже обученной полной Two-Tower."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation import run_feature_ablation, save_feature_ablation
from src.reporting import generate_report
from src.visualization import plot_feature_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-users", type=int, default=600)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    artifacts = ROOT / "artifacts"
    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv.gz")
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv.gz")
    full_model = joblib.load(artifacts / "two_tower.joblib")
    content = joblib.load(artifacts / "classical" / "content.joblib")
    results = run_feature_ablation(
        train, test, full_model, content,
        k=args.top_k, max_users=args.eval_users, seed=args.seed,
    )
    save_feature_ablation(results, artifacts)
    plot_feature_ablation(results["metrics"], results["history"], ROOT / "reports" / "figures")

    # Обновляем отчёт без повторного обучения гибрида.
    hybrid = joblib.load(artifacts / "hybrid.joblib")
    config = json.loads((artifacts / "config.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(artifacts / "metrics.csv")
    segments = pd.read_csv(artifacts / "segment_metrics.csv")
    generate_report(
        metrics, segments, hybrid, config, ROOT / "reports" / "REPORT.md",
        feature_ablation=results["metrics"],
    )
    print(results["metrics"][["model", "precision_at_k", "recall_at_k", "ndcg_at_k", "ild_at_k"]].to_string(index=False), flush=True)
    print(f"Feature-ablation сохранён в {artifacts}", flush=True)


if __name__ == "__main__":
    main()
