from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


def clip_probability(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return float(np.clip(value, low, high))


def logit(p: float) -> float:
    p = clip_probability(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1 - p))


def expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(x, -35, 35))))


def service_game_probability(p: float) -> float:
    """Probability server wins an advantage game under iid service points."""
    p = clip_probability(p)
    q = 1 - p
    before_deuce = p**4 * (1 + 4 * q + 10 * q * q)
    reach_deuce = 20 * p**3 * q**3
    from_deuce = p * p / (p * p + q * q)
    return clip_probability(before_deuce + reach_deuce * from_deuce)


def race_probability(p: float, target: int) -> float:
    """Win a first-to-target, win-by-two race with iid point probability p."""
    p = clip_probability(p)
    q = 1 - p
    out = 0.0
    for loser_points in range(target - 1):
        out += math.comb(target - 1 + loser_points, loser_points) * p**target * q**loser_points
    reach_deuce = math.comb(2 * target - 2, target - 1) * (p * q) ** (target - 1)
    from_deuce = p * p / (p * p + q * q)
    return clip_probability(out + reach_deuce * from_deuce)


def tiebreak_probability(p1_serve: float, p2_serve: float, target: int = 7) -> float:
    # Alternating serve matters less than preserving the two players' point-strength edge.
    # The detailed simulator below uses the exact 1-2-2 service sequence.
    p1_neutral_point = 0.5 * (p1_serve + (1 - p2_serve))
    return race_probability(p1_neutral_point, target)


def set_probability(p1_serve: float, p2_serve: float, first_server: int = 1) -> float:
    p1_holds = service_game_probability(p1_serve)
    p1_breaks = 1 - service_game_probability(p2_serve)
    tb = tiebreak_probability(p1_serve, p2_serve, target=7)

    @lru_cache(maxsize=None)
    def solve(g1: int, g2: int, server: int) -> float:
        if (g1 >= 6 or g2 >= 6) and abs(g1 - g2) >= 2:
            return float(g1 > g2)
        if g1 == 6 and g2 == 6:
            return tb
        p_game = p1_holds if server == 1 else p1_breaks
        return p_game * solve(g1 + 1, g2, 3 - server) + (1 - p_game) * solve(
            g1, g2 + 1, 3 - server
        )

    return clip_probability(solve(0, 0, first_server))


def match_probability(p1_serve: float, p2_serve: float, best_of: int) -> float:
    """Fast scoring-model probability used as a pre-match feature/baseline."""
    p_set = 0.5 * (
        set_probability(p1_serve, p2_serve, first_server=1)
        + set_probability(p1_serve, p2_serve, first_server=2)
    )
    needed = best_of // 2 + 1
    return clip_probability(
        sum(
            math.comb(needed - 1 + lost, lost) * p_set**needed * (1 - p_set) ** lost
            for lost in range(needed)
        )
    )

