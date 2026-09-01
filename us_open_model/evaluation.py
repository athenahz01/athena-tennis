from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .features import FeatureBuilder
from .model import TourModel


def binary_metrics(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, float]:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, int)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (
            p < edges[index + 1] if index < bins - 1 else p <= edges[index + 1]
        )
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return {
        "n": int(len(y)),
        "log_loss": float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "ece_10": float(ece),
    }


def clustered_bootstrap(
    model_loss: np.ndarray,
    baseline_loss: np.ndarray,
    dates: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {"date": pd.to_datetime(dates).date, "difference": model_loss - baseline_loss}
    )
    clusters = [group["difference"].to_numpy(float) for _, group in frame.groupby("date")]
    cluster_sums = np.asarray([cluster.sum() for cluster in clusters], dtype=float)
    cluster_counts = np.asarray([len(cluster) for cluster in clusters], dtype=float)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(clusters), (n_resamples, len(clusters)))
    means = cluster_sums[chosen].sum(axis=1) / cluster_counts[chosen].sum(axis=1)
    delta = float(frame["difference"].mean())
    return {
        "model_minus_baseline_log_loss": delta,
        "ci_95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "one_sided_p_model_not_better": float(np.mean(means >= 0)),
        "ship": bool(np.quantile(means, 0.975) < 0 and delta < -0.001),
        "clusters": len(clusters),
        "resamples": n_resamples,
    }


def walk_forward_backtest(
    history: pd.DataFrame,
    years: list[int],
    *,
    n_resamples: int = 10_000,
) -> dict[str, Any]:
    builder = FeatureBuilder()
    x, y, meta = builder.build(history)
    output: dict[str, Any] = {"protocol": "expanding annual walk-forward", "folds": []}
    for year in years:
        for tour in ("ATP", "WTA"):
            train_mask = (
                meta["tour"].eq(tour) & (pd.to_datetime(meta["date"]).dt.year < year)
            ).to_numpy()
            test_mask = (
                meta["tour"].eq(tour) & (pd.to_datetime(meta["date"]).dt.year == year)
            ).to_numpy()
            if train_mask.sum() < 500 or test_mask.sum() < 20:
                continue
            model = TourModel.fit(
                x[train_mask], y[train_mask], meta.loc[train_mask].reset_index(drop=True)
            )
            test_meta = meta.loc[test_mask].reset_index(drop=True)
            components = model.component_probabilities_many(x[test_mask], test_meta)
            model_p = components[model.champion]
            elo_p = test_meta["elo_probability"].to_numpy(float)
            outcome = y[test_mask]
            model_loss = -(outcome * np.log(np.clip(model_p, 1e-9, 1)) +
                           (1 - outcome) * np.log(np.clip(1 - model_p, 1e-9, 1)))
            elo_loss = -(outcome * np.log(np.clip(elo_p, 1e-9, 1)) +
                         (1 - outcome) * np.log(np.clip(1 - elo_p, 1e-9, 1)))
            output["folds"].append(
                {
                    "year": year,
                    "tour": tour,
                    "selected_component": model.champion,
                    "model": binary_metrics(outcome, model_p),
                    "components": {
                        name: binary_metrics(outcome, probability)
                        for name, probability in components.items()
                    },
                    "elo": binary_metrics(outcome, elo_p),
                    "paired_gate_vs_elo": clustered_bootstrap(
                        model_loss,
                        elo_loss,
                        test_meta["date"].to_numpy(),
                        n_resamples=n_resamples,
                        seed=year + (0 if tour == "ATP" else 100),
                    ),
                }
            )
    return output


def save_backtest(result: dict[str, Any], path: Path | None = None) -> Path:
    path = path or config.OUTPUTS / "walk_forward_backtest.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path
