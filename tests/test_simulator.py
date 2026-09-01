import numpy as np
import pytest

from us_open_model.pipeline import _align_serve_probabilities
from us_open_model.schema import PlayerProfile
from us_open_model.scoring import match_probability
from us_open_model.simulator import run_simulation, simulate_once


def _profile(name: str) -> PlayerProfile:
    return PlayerProfile(name=name, player_id=name, tour="ATP")


def test_alignment_hits_target_probability() -> None:
    p1, p2 = _align_serve_probabilities(0.64, 0.62, 5, 0.73)
    assert match_probability(p1, p2, 5) == pytest.approx(0.73, abs=1e-8)


def test_simulated_match_obeys_score_and_stat_invariants() -> None:
    result = simulate_once(
        np.random.default_rng(12), _profile("A"), _profile("B"), 0.65, 0.61, 5
    )
    assert result.winner in (1, 2)
    assert 3 <= len(result.sets) <= 5
    assert max(result.set_winners.count(1), result.set_winners.count(2)) == 3
    assert result.total_points == result.p1.service_points + result.p2.service_points
    for side in (result.p1, result.p2):
        assert side.service_points_won <= side.service_points
        assert side.first_serves_in <= side.service_points
        assert side.break_points_saved <= side.break_points_faced
    for games1, games2 in result.sets:
        assert max(games1, games2) >= 6
        assert games1 != games2


def test_monte_carlo_probability_tracks_requested_physics() -> None:
    summary = run_simulation(
        _profile("A"),
        _profile("B"),
        0.66,
        0.60,
        best_of=3,
        n_sims=1_200,
        seed=44,
    )
    analytical = match_probability(0.66, 0.60, 3)
    assert summary["raw_simulation_p1_win"] == pytest.approx(analytical, abs=0.04)
    assert sum(summary["set_score_distribution"].values()) == pytest.approx(1, abs=0.002)
    assert summary["player1_stats"]["aces"]["p90"] >= summary["player1_stats"]["aces"]["p10"]
