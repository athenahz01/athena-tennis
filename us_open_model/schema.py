from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PlayerProfile:
    name: str
    player_id: str | None
    tour: str
    hand: str = "U"
    height_cm: float | None = None
    age: float | None = None
    rank: float | None = None
    elo: float = 1500.0
    surface_elo: float = 1500.0
    serve_point_p: float = 0.62
    return_point_p: float = 0.38
    ace_rate: float = 0.06
    double_fault_rate: float = 0.035
    first_serve_in: float = 0.62
    first_serve_win: float = 0.72
    second_serve_win: float = 0.51
    matches_seen: int = 0
    surface_matches_seen: int = 0
    days_rest: float = 30.0
    matches_14d: int = 0
    minutes_14d: float = 0.0
    last_match_minutes: float = 0.0
    data_quality: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatchContext:
    tour: str
    surface: str = "Hard"
    best_of: int = 5
    tournament: str = "US Open"
    round: str = "R128"
    match_date: str = "2026-09-01"
    court: str | None = None
    indoor: bool = False
    temperature_c: float | None = None
    humidity_pct: float | None = None
    wind_kph: float | None = None
    altitude_m: float = 10.0


@dataclass
class MarketQuote:
    player: str
    ticker: str
    event_ticker: str
    bid: float | None
    ask: float | None
    midpoint: float | None
    volume: float | None
    observed_at: str
