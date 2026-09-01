from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import MatchContext, PlayerProfile
from .scoring import expit, logit, match_probability


TOUR_PRIORS = {
    "ATP": {
        "serve": 0.635,
        "return": 0.365,
        "ace": 0.075,
        "df": 0.032,
        "first_in": 0.625,
        "first_win": 0.735,
        "second_win": 0.515,
    },
    "WTA": {
        "serve": 0.585,
        "return": 0.415,
        "ace": 0.042,
        "df": 0.050,
        "first_in": 0.625,
        "first_win": 0.665,
        "second_win": 0.465,
    },
}

FEATURE_NAMES = [
    "elo_diff",
    "surface_elo_diff",
    "rank_log_diff",
    "serve_diff",
    "return_diff",
    "matchup_serve_diff",
    "elo_logit",
    "point_logit",
    "age_diff",
    "height_diff",
    "rest_diff",
    "matches_14d_diff",
    "minutes_14d_diff",
    "last_minutes_diff",
    "recent_win_diff",
    "surface_win_diff",
    "h2h_shrunk",
    "lefty_diff",
    "grand_slam",
    "best_of_five",
    "round_progress",
    "low_data_diff",
]

ROUND_PROGRESS = {"RR": 0.0, "R128": 0.0, "R64": 0.2, "R32": 0.4, "R16": 0.6,
                  "QF": 0.75, "SF": 0.9, "F": 1.0, "BR": 1.0}


def normalize_name(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _safe(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


@dataclass
class DecayedRate:
    successes: float
    trials: float
    last_date: pd.Timestamp | None = None
    half_life_days: float = 365.0

    def decay_to(self, date: pd.Timestamp) -> None:
        if self.last_date is not None:
            days = max(0.0, (date - self.last_date).total_seconds() / 86400)
            weight = 0.5 ** (days / self.half_life_days)
            self.successes *= weight
            self.trials *= weight
        self.last_date = date

    def add(self, successes: float, trials: float, date: pd.Timestamp) -> None:
        self.decay_to(date)
        if trials > 0 and 0 <= successes <= trials:
            self.successes += successes
            self.trials += trials

    @property
    def mean(self) -> float:
        return self.successes / self.trials if self.trials > 0 else 0.5


@dataclass
class PlayerState:
    player_id: str
    name: str
    tour: str
    hand: str = "U"
    height: float | None = None
    age: float | None = None
    rank: float | None = None
    elo: float = 1500.0
    surface_elo: dict[str, float] = field(default_factory=dict)
    rates: dict[str, DecayedRate] = field(default_factory=dict)
    hard_rates: dict[str, DecayedRate] = field(default_factory=dict)
    recent: deque = field(default_factory=lambda: deque(maxlen=30))
    wins: int = 0
    losses: int = 0
    hard_wins: int = 0
    hard_losses: int = 0
    matches_seen: int = 0
    hard_matches_seen: int = 0

    @classmethod
    def create(cls, player_id: str, name: str, tour: str) -> "PlayerState":
        priors = TOUR_PRIORS[tour]
        state = cls(player_id=player_id, name=name, tour=tour)
        equivalent_points = 220.0
        for key, mean in priors.items():
            state.rates[key] = DecayedRate(mean * equivalent_points, equivalent_points)
            state.hard_rates[key] = DecayedRate(mean * 90.0, 90.0, half_life_days=520.0)
        return state

    def blended_rate(self, key: str, surface: str) -> float:
        overall = self.rates[key].mean
        if surface.lower() != "hard":
            return overall
        surface_rate = self.hard_rates[key].mean
        observed = max(0.0, self.hard_rates[key].trials - 90.0)
        weight = min(0.65, observed / (observed + 350.0))
        return (1 - weight) * overall + weight * surface_rate

    def recent_summary(self, date: pd.Timestamp) -> dict[str, float]:
        rows = [row for row in self.recent if row[0] < date]
        within14 = [row for row in rows if (date - row[0]).days <= 14]
        last = rows[-1] if rows else None
        recent10 = rows[-10:]
        return {
            "days_rest": min(60.0, max(0.0, (date - last[0]).days)) if last else 30.0,
            "matches_14d": float(len(within14)),
            "minutes_14d": float(sum(row[2] for row in within14)),
            "last_minutes": float(last[2]) if last else 0.0,
            "recent_win": (sum(row[1] for row in recent10) + 2.0) / (len(recent10) + 4.0),
        }


def _rank_elo(rank: float | None) -> float:
    if rank is None or not np.isfinite(rank) or rank <= 0:
        return 1450.0
    return float(np.clip(1500 + 115 * math.log10(100 / rank), 1250, 1900))


def _profile(state: PlayerState, surface: str, date: pd.Timestamp) -> PlayerProfile:
    recent = state.recent_summary(date)
    surface_elo = state.surface_elo.get(surface, 1500.0)
    if state.matches_seen < 3:
        rank_prior = _rank_elo(state.rank)
        elo = 0.35 * state.elo + 0.65 * rank_prior
        surface_elo = 0.35 * surface_elo + 0.65 * rank_prior
    else:
        elo = state.elo
    quality = []
    if state.matches_seen < 5:
        quality.append("low_match_history")
    if state.hard_matches_seen < 3 and surface.lower() == "hard":
        quality.append("low_surface_history")
    if state.rank is None:
        quality.append("missing_rank")
    return PlayerProfile(
        name=state.name,
        player_id=state.player_id,
        tour=state.tour,
        hand=state.hand,
        height_cm=state.height,
        age=state.age,
        rank=state.rank,
        elo=elo,
        surface_elo=surface_elo,
        serve_point_p=state.blended_rate("serve", surface),
        return_point_p=state.blended_rate("return", surface),
        ace_rate=state.blended_rate("ace", surface),
        double_fault_rate=state.blended_rate("df", surface),
        first_serve_in=state.blended_rate("first_in", surface),
        first_serve_win=state.blended_rate("first_win", surface),
        second_serve_win=state.blended_rate("second_win", surface),
        matches_seen=state.matches_seen,
        surface_matches_seen=state.hard_matches_seen if surface.lower() == "hard" else 0,
        days_rest=recent["days_rest"],
        matches_14d=int(recent["matches_14d"]),
        minutes_14d=recent["minutes_14d"],
        last_match_minutes=recent["last_minutes"],
        data_quality=quality,
    )


def opponent_adjusted_serve(
    server: PlayerProfile, receiver: PlayerProfile, tour: str
) -> float:
    field_serve = TOUR_PRIORS[tour]["serve"]
    opponent_allowed = 1 - receiver.return_point_p
    adjusted = expit(logit(server.serve_point_p) + logit(opponent_allowed) - logit(field_serve))
    return float(np.clip(adjusted, 0.43, 0.82))


def _elo_probability(p1: PlayerProfile, p2: PlayerProfile) -> float:
    r1 = 0.55 * p1.elo + 0.45 * p1.surface_elo
    r2 = 0.55 * p2.elo + 0.45 * p2.surface_elo
    return 1 / (1 + 10 ** (-(r1 - r2) / 400))


def feature_vector(
    p1: PlayerProfile,
    p2: PlayerProfile,
    context: MatchContext,
    *,
    recent_win1: float,
    recent_win2: float,
    surface_win1: float,
    surface_win2: float,
    h2h_wins1: int,
    h2h_total: int,
) -> tuple[np.ndarray, dict[str, float]]:
    p1_serve = opponent_adjusted_serve(p1, p2, context.tour)
    p2_serve = opponent_adjusted_serve(p2, p1, context.tour)
    elo_p = _elo_probability(p1, p2)
    point_p = match_probability(p1_serve, p2_serve, context.best_of)
    rank1 = p1.rank if p1.rank and p1.rank > 0 else 500.0
    rank2 = p2.rank if p2.rank and p2.rank > 0 else 500.0
    h2h = (h2h_wins1 + 1.5) / (h2h_total + 3.0)
    values = [
        (p1.elo - p2.elo) / 400,
        (p1.surface_elo - p2.surface_elo) / 400,
        math.log(rank2) - math.log(rank1),
        p1.serve_point_p - p2.serve_point_p,
        p1.return_point_p - p2.return_point_p,
        p1_serve - p2_serve,
        logit(elo_p),
        logit(point_p),
        (_safe(p1.age, 26.0) - _safe(p2.age, 26.0)) / 10,
        (_safe(p1.height_cm, 183.0) - _safe(p2.height_cm, 183.0)) / 20,
        (p1.days_rest - p2.days_rest) / 14,
        (p1.matches_14d - p2.matches_14d) / 4,
        (p1.minutes_14d - p2.minutes_14d) / 500,
        (p1.last_match_minutes - p2.last_match_minutes) / 180,
        recent_win1 - recent_win2,
        surface_win1 - surface_win2,
        2 * h2h - 1,
        float(p1.hand == "L") - float(p2.hand == "L"),
        float(context.tournament.lower() in {"us open", "wimbledon", "roland garros", "australian open"}),
        float(context.best_of == 5),
        ROUND_PROGRESS.get(context.round.upper(), 0.25),
        min(1.0, p2.matches_seen / 15) - min(1.0, p1.matches_seen / 15),
    ]
    baselines = {
        "elo_probability": elo_p,
        "point_probability": point_p,
        "p1_serve_probability": p1_serve,
        "p2_serve_probability": p2_serve,
    }
    return np.asarray(values, dtype=float), baselines


class FeatureBuilder:
    def __init__(self) -> None:
        self.players: dict[tuple[str, str], PlayerState] = {}
        self.name_index: dict[tuple[str, str], str] = {}
        self.h2h: dict[tuple[str, str, str], list[int]] = {}

    def _key(self, tour: str, player_id: object, name: object) -> str:
        raw = str(player_id or "").strip()
        return raw if raw and raw.lower() != "nan" else normalize_name(name)

    def _state(self, tour: str, player_id: object, name: object) -> PlayerState:
        key = self._key(tour, player_id, name)
        compound = (tour, key)
        if compound not in self.players:
            self.players[compound] = PlayerState.create(key, str(name), tour)
        self.name_index[(tour, normalize_name(name))] = key
        return self.players[compound]

    def resolve(self, tour: str, name: str, surface: str, date: pd.Timestamp) -> PlayerProfile:
        normalized = normalize_name(name)
        key = self.name_index.get((tour, normalized))
        if key is None:
            # Conservative last-name fallback only when unique within a tour.
            surname = normalized.split()[-1] if normalized else ""
            candidates = [
                player_key for (candidate_tour, candidate_name), player_key in self.name_index.items()
                if candidate_tour == tour and candidate_name.split()[-1:] == [surname]
            ]
            if len(set(candidates)) == 1:
                key = candidates[0]
        if key is None:
            state = PlayerState.create(normalized or name, name, tour)
            return _profile(state, surface, date)
        return _profile(self.players[(tour, key)], surface, date)

    def get_state(self, tour: str, name: str) -> PlayerState:
        normalized = normalize_name(name)
        key = self.name_index.get((tour, normalized))
        if key is None:
            surname = normalized.split()[-1] if normalized else ""
            candidates = [
                player_key
                for (candidate_tour, candidate_name), player_key in self.name_index.items()
                if candidate_tour == tour and candidate_name.split()[-1:] == [surname]
            ]
            if len(set(candidates)) == 1:
                key = candidates[0]
        if key is None:
            return PlayerState.create(normalized or name, name, tour)
        return self.players[(tour, key)]

    def h2h_summary(self, tour: str, p1_id: str, p2_id: str) -> tuple[int, int]:
        wins, total = self.h2h.get((tour, p1_id, p2_id), [0, 0])
        return wins, total

    @staticmethod
    def _surface_win_rate(state: PlayerState, surface: str) -> float:
        if surface.lower() == "hard":
            return (state.hard_wins + 2) / (state.hard_wins + state.hard_losses + 4)
        return (state.wins + 2) / (state.wins + state.losses + 4)

    def matchup_features(
        self, p1_state: PlayerState, p2_state: PlayerState, context: MatchContext, date: pd.Timestamp
    ) -> tuple[np.ndarray, dict[str, float], PlayerProfile, PlayerProfile]:
        p1 = _profile(p1_state, context.surface, date)
        p2 = _profile(p2_state, context.surface, date)
        r1 = p1_state.recent_summary(date)["recent_win"]
        r2 = p2_state.recent_summary(date)["recent_win"]
        h2h_wins, h2h_total = self.h2h_summary(context.tour, p1_state.player_id, p2_state.player_id)
        vector, baselines = feature_vector(
            p1,
            p2,
            context,
            recent_win1=r1,
            recent_win2=r2,
            surface_win1=self._surface_win_rate(p1_state, context.surface),
            surface_win2=self._surface_win_rate(p2_state, context.surface),
            h2h_wins1=h2h_wins,
            h2h_total=h2h_total,
        )
        return vector, baselines, p1, p2

    def build(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        vectors: list[np.ndarray] = []
        labels: list[int] = []
        metadata: list[dict] = []
        for row in matches.itertuples(index=False):
            tour = str(row.tour).upper()
            date = pd.Timestamp(row.effective_date)
            surface = str(row.surface or "Hard").title()
            winner = self._state(tour, row.winner_id, row.winner_name)
            loser = self._state(tour, row.loser_id, row.loser_name)
            self._refresh_identity(winner, row, "winner")
            self._refresh_identity(loser, row, "loser")
            digest = hashlib.sha256(str(row.match_key).encode()).digest()[0]
            flipped = bool(digest % 2)
            p1_state, p2_state = (loser, winner) if flipped else (winner, loser)
            context = MatchContext(
                tour=tour,
                surface=surface,
                best_of=int(_safe(row.best_of, 3)),
                tournament=str(row.tourney_name),
                round=str(row.round),
                match_date=date.date().isoformat(),
                indoor=str(getattr(row, "indoor", "O")).upper() in {"I", "1", "TRUE"},
            )
            vector, baselines, _, _ = self.matchup_features(p1_state, p2_state, context, date)
            vectors.append(vector)
            labels.append(0 if flipped else 1)
            metadata.append(
                {
                    "match_key": row.match_key,
                    "date": date,
                    "tour": tour,
                    "surface": surface,
                    "label": labels[-1],
                    **baselines,
                }
            )
            self._update_match(row, winner, loser, date, surface)
        return np.vstack(vectors), np.asarray(labels, dtype=int), pd.DataFrame(metadata)

    @staticmethod
    def _refresh_identity(state: PlayerState, row: object, side: str) -> None:
        state.name = str(getattr(row, f"{side}_name"))
        state.hand = str(getattr(row, f"{side}_hand", "U") or "U")
        state.height = _safe(getattr(row, f"{side}_ht", None), np.nan)
        state.height = state.height if np.isfinite(state.height) else None
        state.age = _safe(getattr(row, f"{side}_age", None), np.nan)
        state.age = state.age if np.isfinite(state.age) else None
        state.rank = _safe(getattr(row, f"{side}_rank", None), np.nan)
        state.rank = state.rank if np.isfinite(state.rank) and state.rank > 0 else None

    def _update_rates(self, state: PlayerState, row: object, prefix: str, opp_prefix: str,
                      date: pd.Timestamp, surface: str) -> None:
        svpt = _safe(getattr(row, f"{prefix}_svpt", None), 0)
        first_in = _safe(getattr(row, f"{prefix}_1stIn", None), 0)
        first_won = _safe(getattr(row, f"{prefix}_1stWon", None), 0)
        second_won = _safe(getattr(row, f"{prefix}_2ndWon", None), 0)
        ace = _safe(getattr(row, f"{prefix}_ace", None), 0)
        double_fault = _safe(getattr(row, f"{prefix}_df", None), 0)
        opp_svpt = _safe(getattr(row, f"{opp_prefix}_svpt", None), 0)
        opp_service_won = _safe(getattr(row, f"{opp_prefix}_1stWon", None), 0) + _safe(
            getattr(row, f"{opp_prefix}_2ndWon", None), 0
        )
        observations = {
            "serve": (first_won + second_won, svpt),
            "return": (max(0.0, opp_svpt - opp_service_won), opp_svpt),
            "ace": (ace, svpt),
            "df": (double_fault, svpt),
            "first_in": (first_in, svpt),
            "first_win": (first_won, first_in),
            "second_win": (second_won, max(0.0, svpt - first_in)),
        }
        for key, (successes, trials) in observations.items():
            state.rates[key].add(successes, trials, date)
            if surface.lower() == "hard":
                state.hard_rates[key].add(successes, trials, date)

    def _update_match(self, row: object, winner: PlayerState, loser: PlayerState,
                      date: pd.Timestamp, surface: str) -> None:
        p_expected = 1 / (1 + 10 ** (-((0.55 * winner.elo + 0.45 * winner.surface_elo.get(surface, 1500)) -
                                        (0.55 * loser.elo + 0.45 * loser.surface_elo.get(surface, 1500))) / 400))
        level = str(getattr(row, "tourney_level", ""))
        k = 36.0 if level.upper() == "G" else 28.0
        score = str(getattr(row, "score", ""))
        games = [int(token) for token in re.findall(r"(?<!\d)(\d+)(?=-|\))", score)]
        margin = 1.0 + 0.10 * math.log1p(sum(games)) if games else 1.0
        delta = k * margin * (1 - p_expected)
        winner.elo += delta
        loser.elo -= delta
        sw = winner.surface_elo.get(surface, 1500.0)
        sl = loser.surface_elo.get(surface, 1500.0)
        surface_expected = 1 / (1 + 10 ** (-(sw - sl) / 400))
        surface_delta = k * (1 - surface_expected)
        winner.surface_elo[surface] = sw + surface_delta
        loser.surface_elo[surface] = sl - surface_delta
        self._update_rates(winner, row, "w", "l", date, surface)
        self._update_rates(loser, row, "l", "w", date, surface)
        minutes = _safe(getattr(row, "minutes", None), 90.0)
        winner.recent.append((date, 1, minutes))
        loser.recent.append((date, 0, minutes))
        winner.wins += 1
        loser.losses += 1
        winner.matches_seen += 1
        loser.matches_seen += 1
        if surface.lower() == "hard":
            winner.hard_wins += 1
            loser.hard_losses += 1
            winner.hard_matches_seen += 1
            loser.hard_matches_seen += 1
        key_w = (winner.tour, winner.player_id, loser.player_id)
        key_l = (winner.tour, loser.player_id, winner.player_id)
        winner_h2h = self.h2h.setdefault(key_w, [0, 0])
        loser_h2h = self.h2h.setdefault(key_l, [0, 0])
        winner_h2h[0] += 1
        winner_h2h[1] += 1
        loser_h2h[1] += 1
