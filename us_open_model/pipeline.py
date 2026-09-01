from __future__ import annotations

from typing import Any

from .model import TennisPredictor
from .schema import MatchContext, PlayerProfile
from .scoring import logit, expit, match_probability
from .simulator import run_simulation


def _align_serve_probabilities(
    p1_serve: float, p2_serve: float, best_of: int, target_match_p: float
) -> tuple[float, float]:
    """Move both serve matchups symmetrically until simulation physics matches the champion."""
    low, high = -1.2, 1.2
    for _ in range(36):
        middle = (low + high) / 2
        adjusted1 = expit(logit(p1_serve) + middle)
        adjusted2 = expit(logit(p2_serve) - middle)
        probability = match_probability(adjusted1, adjusted2, best_of)
        if probability < target_match_p:
            low = middle
        else:
            high = middle
    shift = (low + high) / 2
    return expit(logit(p1_serve) + shift), expit(logit(p2_serve) - shift)


def predict_full(
    predictor: TennisPredictor,
    player1: str,
    player2: str,
    context: MatchContext,
    *,
    n_sims: int = 20_000,
    seed: int = 20260901,
) -> dict[str, Any]:
    forecast = predictor.matchup(player1, player2, context)
    p1 = PlayerProfile(**forecast["player1"])
    p2 = PlayerProfile(**forecast["player2"])
    raw1 = forecast["serve_matchup"]["player1_service_point_win"]
    raw2 = forecast["serve_matchup"]["player2_service_point_win"]
    aligned1, aligned2 = _align_serve_probabilities(
        raw1, raw2, context.best_of, forecast["probabilities"]["final"]
    )
    simulation = run_simulation(
        p1,
        p2,
        aligned1,
        aligned2,
        best_of=context.best_of,
        n_sims=n_sims,
        seed=seed,
    )
    forecast["serve_matchup"]["simulation_aligned_player1"] = round(aligned1, 6)
    forecast["serve_matchup"]["simulation_aligned_player2"] = round(aligned2, 6)
    forecast["simulation"] = simulation
    forecast["winner"] = {
        "name": player1 if forecast["probabilities"]["final"] >= 0.5 else player2,
        "probability": round(
            max(forecast["probabilities"]["final"], 1 - forecast["probabilities"]["final"]),
            4,
        ),
    }
    return forecast

