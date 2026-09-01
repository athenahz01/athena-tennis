from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURE_NAMES, FeatureBuilder
from .schema import MatchContext
from .scoring import clip_probability, logit


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


@dataclass
class TourModel:
    base: Pipeline
    stack: LogisticRegression
    champion: str
    validation: dict[str, Any]

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> "TourModel":
        if len(y) < 100:
            raise ValueError("At least 100 chronological matches are required to train a tour model")
        split = max(80, int(len(y) * 0.80))
        split = min(split, len(y) - 20)
        base = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.35, max_iter=2000, random_state=42)),
            ]
        )
        base.fit(x[:split], y[:split])
        p_ml = base.predict_proba(x[split:])[:, 1]
        p_elo = meta.iloc[split:]["elo_probability"].to_numpy(float)
        p_point = meta.iloc[split:]["point_probability"].to_numpy(float)
        stack_x = np.column_stack(
            [
                [logit(p) for p in p_ml],
                [logit(p) for p in p_elo],
                [logit(p) for p in p_point],
            ]
        )
        stack = LogisticRegression(C=0.5, max_iter=1000, random_state=43)
        stack.fit(stack_x, y[split:])
        p_stack = stack.predict_proba(stack_x)[:, 1]
        candidates = {"elo": p_elo, "point": p_point, "ml": p_ml, "stack": p_stack}
        metrics = {
            name: {
                "log_loss": _log_loss(y[split:], probability),
                "brier": _brier(y[split:], probability),
                "accuracy": float(np.mean((probability >= 0.5) == y[split:])),
            }
            for name, probability in candidates.items()
        }
        champion = min(metrics, key=lambda key: metrics[key]["log_loss"])
        validation = {
            "n_train": split,
            "n_calibration": len(y) - split,
            "start_date": str(meta.iloc[split:]["date"].min().date()),
            "end_date": str(meta.iloc[split:]["date"].max().date()),
            "candidates": metrics,
            "selected": champion,
        }
        return cls(base=base, stack=stack, champion=champion, validation=validation)

    def probabilities(self, x: np.ndarray, baselines: dict[str, float]) -> dict[str, Any]:
        p_ml = float(self.base.predict_proba(x.reshape(1, -1))[0, 1])
        p_elo = float(baselines["elo_probability"])
        p_point = float(baselines["point_probability"])
        stack_x = np.asarray([[logit(p_ml), logit(p_elo), logit(p_point)]])
        p_stack = float(self.stack.predict_proba(stack_x)[0, 1])
        all_p = {"elo": p_elo, "point": p_point, "ml": p_ml, "stack": p_stack}
        selected = all_p[self.champion]
        return {
            "elo": round(p_elo, 6),
            "point_model": round(p_point, 6),
            "machine_learning": round(p_ml, 6),
            "stack": round(p_stack, 6),
            "final": round(clip_probability(selected, 0.01, 0.99), 6),
            "champion_component": self.champion,
        }

    def component_probabilities_many(
        self, x: np.ndarray, meta: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        p_ml = self.base.predict_proba(x)[:, 1]
        p_elo = meta["elo_probability"].to_numpy(float)
        p_point = meta["point_probability"].to_numpy(float)
        stack_x = np.column_stack(
            [
                [logit(p) for p in p_ml],
                [logit(p) for p in p_elo],
                [logit(p) for p in p_point],
            ]
        )
        p_stack = self.stack.predict_proba(stack_x)[:, 1]
        return {
            "elo": p_elo,
            "point": p_point,
            "ml": p_ml,
            "stack": p_stack,
        }

    def predict_many(self, x: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        return self.component_probabilities_many(x, meta)[self.champion]


@dataclass
class TennisPredictor:
    builder: FeatureBuilder
    models: dict[str, TourModel]
    trained_at: str
    cutoff: str
    data_rows: int
    model_version: str = config.MODEL_VERSION

    @classmethod
    def train(cls, history: pd.DataFrame, cutoff: str) -> "TennisPredictor":
        builder = FeatureBuilder()
        x, y, meta = builder.build(history)
        models = {}
        for tour in ("ATP", "WTA"):
            mask = meta["tour"].eq(tour).to_numpy()
            models[tour] = TourModel.fit(
                x[mask], y[mask], meta.loc[mask].reset_index(drop=True)
            )
        return cls(
            builder=builder,
            models=models,
            trained_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            cutoff=cutoff,
            data_rows=len(history),
        )

    def matchup(
        self,
        player1: str,
        player2: str,
        context: MatchContext,
    ) -> dict[str, Any]:
        tour = context.tour.upper()
        date = pd.Timestamp(context.match_date)
        p1_state = self.builder.get_state(tour, player1)
        p2_state = self.builder.get_state(tour, player2)
        vector, baselines, p1, p2 = self.builder.matchup_features(
            p1_state, p2_state, context, date
        )
        probabilities = self.models[tour].probabilities(vector, baselines)
        # Frozen before the 2026 US Open: the WTA stack failed the pre-declared
        # date-clustered log-loss promotion gate on results through August 30.
        # Keep every component visible, but deploy the safer Elo baseline for this regime.
        if tour == "WTA" and date.year == 2026:
            probabilities["final"] = probabilities["elo"]
            probabilities["champion_component"] = "elo_2026_safety_fallback"
        components = ("elo", "point_model", "machine_learning")
        disagreement = max(probabilities[key] for key in components) - min(
            probabilities[key] for key in components
        )
        quality = sorted(set(p1.data_quality + p2.data_quality))
        if disagreement > 0.15:
            quality.append("large_model_disagreement")
        if tour == "WTA" and date.year == 2026:
            quality.append("2026_wta_stack_failed_gate_using_elo_fallback")
        return {
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "training_cutoff": self.cutoff,
            "context": context.__dict__,
            "player1": p1.to_dict(),
            "player2": p2.to_dict(),
            "probabilities": probabilities,
            "serve_matchup": {
                "player1_service_point_win": round(baselines["p1_serve_probability"], 6),
                "player2_service_point_win": round(baselines["p2_serve_probability"], 6),
            },
            "model_disagreement": round(disagreement, 4),
            "data_quality": quality or ["ok"],
            "feature_values": dict(zip(FEATURE_NAMES, np.round(vector, 6).tolist())),
        }

    def save(self, path: Path = config.ARTIFACTS / "uso_hybrid_v1.joblib") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        report = {
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "cutoff": self.cutoff,
            "data_rows": self.data_rows,
            "features": FEATURE_NAMES,
            "tour_validation": {
                tour: model.validation for tour, model in self.models.items()
            },
        }
        path.with_suffix(".json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(
        cls, path: Path = config.ARTIFACTS / "uso_hybrid_v1.joblib"
    ) -> "TennisPredictor":
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}. Run `us-open train` first.")
        return joblib.load(path)
