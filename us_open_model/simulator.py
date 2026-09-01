from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schema import PlayerProfile


@dataclass
class SideStats:
    aces: int = 0
    double_faults: int = 0
    service_points: int = 0
    service_points_won: int = 0
    first_serves_in: int = 0
    first_serve_points_won: int = 0
    second_serve_points_won: int = 0
    service_games: int = 0
    breaks: int = 0
    break_points_faced: int = 0
    break_points_saved: int = 0


@dataclass
class MatchSimulation:
    sets: list[tuple[int, int]] = field(default_factory=list)
    set_winners: list[int] = field(default_factory=list)
    p1: SideStats = field(default_factory=SideStats)
    p2: SideStats = field(default_factory=SideStats)
    total_points: int = 0
    tiebreaks: int = 0
    winner: int = 0
    duration_minutes: float = 0.0


def _is_break_point(server_points: int, receiver_points: int) -> bool:
    return receiver_points >= 3 and receiver_points >= server_points + 1


def _point(
    rng: np.random.Generator,
    server: int,
    p1: PlayerProfile,
    p2: PlayerProfile,
    p1_serve_p: float,
    p2_serve_p: float,
    result: MatchSimulation,
) -> int:
    profile = p1 if server == 1 else p2
    stats = result.p1 if server == 1 else result.p2
    p_serve = p1_serve_p if server == 1 else p2_serve_p
    stats.service_points += 1
    result.total_points += 1

    # Winner is drawn from the opponent-adjusted service-point probability exactly.
    # Ace/DF/first-serve events are conditional annotations, so they cannot silently
    # move the requested point probability and de-calibrate the scoring simulator.
    server_wins = rng.random() < p_serve
    if server_wins:
        ace_conditional = min(0.50, profile.ace_rate / max(p_serve, 0.10))
        ace = rng.random() < ace_conditional
        double_fault = False
    else:
        ace = False
        double_fault = rng.random() < min(
            0.50, profile.double_fault_rate / max(1 - p_serve, 0.10)
        )
    if double_fault:
        stats.double_faults += 1
        first_in = False
    elif ace:
        stats.aces += 1
        first_in = True
    else:
        first_in = rng.random() < profile.first_serve_in
    if first_in:
        stats.first_serves_in += 1
    if server_wins:
        stats.service_points_won += 1
        if first_in:
            stats.first_serve_points_won += 1
        else:
            stats.second_serve_points_won += 1
        return server
    return 3 - server


def _game(
    rng: np.random.Generator,
    server: int,
    p1: PlayerProfile,
    p2: PlayerProfile,
    p1_serve_p: float,
    p2_serve_p: float,
    result: MatchSimulation,
) -> int:
    points = {1: 0, 2: 0}
    server_stats = result.p1 if server == 1 else result.p2
    receiver_stats = result.p2 if server == 1 else result.p1
    server_stats.service_games += 1
    while True:
        receiver = 3 - server
        was_break_point = _is_break_point(points[server], points[receiver])
        if was_break_point:
            server_stats.break_points_faced += 1
        winner = _point(rng, server, p1, p2, p1_serve_p, p2_serve_p, result)
        points[winner] += 1
        if was_break_point and winner == server:
            server_stats.break_points_saved += 1
        if points[winner] >= 4 and points[winner] - points[3 - winner] >= 2:
            if winner != server:
                receiver_stats.breaks += 1
            return winner


def _tiebreak_server(first_server: int, point_index: int) -> int:
    if point_index == 0:
        return first_server
    block = (point_index - 1) // 2
    return 3 - first_server if block % 2 == 0 else first_server


def _tiebreak(
    rng: np.random.Generator,
    first_server: int,
    target: int,
    p1: PlayerProfile,
    p2: PlayerProfile,
    p1_serve_p: float,
    p2_serve_p: float,
    result: MatchSimulation,
) -> int:
    points = {1: 0, 2: 0}
    index = 0
    result.tiebreaks += 1
    while True:
        server = _tiebreak_server(first_server, index)
        winner = _point(rng, server, p1, p2, p1_serve_p, p2_serve_p, result)
        points[winner] += 1
        index += 1
        if points[winner] >= target and points[winner] - points[3 - winner] >= 2:
            return winner


def simulate_once(
    rng: np.random.Generator,
    p1: PlayerProfile,
    p2: PlayerProfile,
    p1_serve_p: float,
    p2_serve_p: float,
    best_of: int,
) -> MatchSimulation:
    result = MatchSimulation()
    sets_needed = best_of // 2 + 1
    sets_won = {1: 0, 2: 0}
    next_server = int(rng.integers(1, 3))
    while max(sets_won.values()) < sets_needed:
        games = {1: 0, 2: 0}
        set_index = len(result.sets)
        set_first_server = next_server
        while True:
            if games[1] == 6 and games[2] == 6:
                target = 10 if set_index == best_of - 1 else 7
                winner = _tiebreak(
                    rng, next_server, target, p1, p2, p1_serve_p, p2_serve_p, result
                )
                games[winner] += 1
                next_server = 3 - next_server
                break
            winner = _game(rng, next_server, p1, p2, p1_serve_p, p2_serve_p, result)
            games[winner] += 1
            next_server = 3 - next_server
            if games[winner] >= 6 and games[winner] - games[3 - winner] >= 2:
                break
        set_winner = 1 if games[1] > games[2] else 2
        result.sets.append((games[1], games[2]))
        result.set_winners.append(set_winner)
        sets_won[set_winner] += 1
        if sum(games.values()) == 13:
            next_server = 3 - set_first_server
    result.winner = 1 if sets_won[1] > sets_won[2] else 2
    games_total = sum(a + b for a, b in result.sets)
    noise = rng.lognormal(mean=-0.005, sigma=0.10)
    result.duration_minutes = noise * (
        0.34 * result.total_points + 1.35 * games_total + 2.5 * len(result.sets)
    )
    return result


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 3),
        "p10": round(float(np.quantile(array, 0.10)), 3),
        "p50": round(float(np.quantile(array, 0.50)), 3),
        "p90": round(float(np.quantile(array, 0.90)), 3),
    }


def run_simulation(
    p1: PlayerProfile,
    p2: PlayerProfile,
    p1_serve_p: float,
    p2_serve_p: float,
    *,
    best_of: int,
    n_sims: int = 20_000,
    seed: int = 20260901,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    simulations = [
        simulate_once(rng, p1, p2, p1_serve_p, p2_serve_p, best_of)
        for _ in range(n_sims)
    ]
    p1_wins = np.fromiter((sim.winner == 1 for sim in simulations), dtype=float)
    total_games = [sum(a + b for a, b in sim.sets) for sim in simulations]
    margin_games = [sum(a - b for a, b in sim.sets) for sim in simulations]
    set_scores = Counter(
        f"{sum(w == 1 for w in sim.set_winners)}-{sum(w == 2 for w in sim.set_winners)}"
        for sim in simulations
    )
    exact_scores = Counter(
        " ".join(f"{a}-{b}" for a, b in sim.sets) for sim in simulations
    )

    def side_summary(side: int) -> dict[str, Any]:
        stats = [sim.p1 if side == 1 else sim.p2 for sim in simulations]
        return {
            "aces": _summary([s.aces for s in stats]),
            "double_faults": _summary([s.double_faults for s in stats]),
            "service_points": _summary([s.service_points for s in stats]),
            "service_points_won": _summary([s.service_points_won for s in stats]),
            "first_serves_in": _summary([s.first_serves_in for s in stats]),
            "first_serve_points_won": _summary([s.first_serve_points_won for s in stats]),
            "second_serve_points_won": _summary([s.second_serve_points_won for s in stats]),
            "service_games": _summary([s.service_games for s in stats]),
            "breaks": _summary([s.breaks for s in stats]),
            "break_points_faced": _summary([s.break_points_faced for s in stats]),
            "break_points_saved": _summary([s.break_points_saved for s in stats]),
        }

    common_lines = range(18, 51) if best_of == 5 else range(14, 36)
    totals = {
        f"over_{line}.5": round(float(np.mean(np.asarray(total_games) > line + 0.5)), 4)
        for line in common_lines
    }
    sets_needed = best_of // 2 + 1
    return {
        "n_sims": n_sims,
        "simulation_seed": seed,
        "raw_simulation_p1_win": round(float(p1_wins.mean()), 4),
        "total_games": _summary(total_games),
        "game_margin_p1": _summary(margin_games),
        "duration_minutes": _summary([sim.duration_minutes for sim in simulations]),
        "p_tiebreak": round(float(np.mean([sim.tiebreaks > 0 for sim in simulations])), 4),
        "expected_tiebreaks": round(float(np.mean([sim.tiebreaks for sim in simulations])), 3),
        "p_deciding_set": round(
            float(np.mean([len(sim.sets) == best_of for sim in simulations])), 4
        ),
        "p1_straight_sets": round(
            float(np.mean([sim.winner == 1 and len(sim.sets) == sets_needed for sim in simulations])),
            4,
        ),
        "p2_straight_sets": round(
            float(np.mean([sim.winner == 2 and len(sim.sets) == sets_needed for sim in simulations])),
            4,
        ),
        "set_score_distribution": {
            key: round(value / n_sims, 4) for key, value in set_scores.most_common()
        },
        "top_exact_scores": [
            {"score": key, "probability": round(value / n_sims, 4)}
            for key, value in exact_scores.most_common(8)
        ],
        "total_games_probabilities": totals,
        "player1_stats": side_summary(1),
        "player2_stats": side_summary(2),
    }
