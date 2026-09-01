import pytest

from us_open_model.scoring import (
    match_probability,
    race_probability,
    service_game_probability,
)


def test_service_game_known_reference_and_symmetry() -> None:
    assert service_game_probability(0.5) == pytest.approx(0.5)
    assert service_game_probability(0.65) == pytest.approx(0.8296446445)
    assert service_game_probability(0.65) == pytest.approx(
        1 - service_game_probability(0.35)
    )


def test_tennis_races_and_matches_are_symmetric() -> None:
    assert race_probability(0.5, 7) == pytest.approx(0.5)
    for best_of in (3, 5):
        even = match_probability(0.62, 0.62, best_of)
        assert even == pytest.approx(0.5, abs=1e-10)
        p = match_probability(0.67, 0.59, best_of)
        reverse = match_probability(0.59, 0.67, best_of)
        assert p > 0.5
        assert p == pytest.approx(1 - reverse, abs=1e-10)


def test_longer_match_compounds_player_edge() -> None:
    best_of_three = match_probability(0.66, 0.60, 3)
    best_of_five = match_probability(0.66, 0.60, 5)
    assert best_of_five > best_of_three > 0.5
