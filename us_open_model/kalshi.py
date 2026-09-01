from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from . import config
from .features import normalize_name


def _dollars(market: dict, field: str) -> float | None:
    value = market.get(f"{field}_dollars")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    value = market.get(field)
    try:
        return float(value) / 100 if value is not None else None
    except (TypeError, ValueError):
        return None


def _midpoint(market: dict) -> float | None:
    bid, ask = _dollars(market, "yes_bid"), _dollars(market, "yes_ask")
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2
    return _dollars(market, "last_price")


def fetch_markets(series_ticker: str, status: str = "open", limit: int = 1000) -> list[dict]:
    url = f"{config.KALSHI_BASE}/markets"
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "us-open-predictor/0.1"})
    rows: list[dict] = []
    cursor = None
    while True:
        params: dict[str, Any] = {
            "series_ticker": series_ticker,
            "status": status,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload.get("markets", []))
        cursor = payload.get("cursor")
        if not cursor:
            return rows


def snapshot_winner_markets(tour: str, *, status: str = "open") -> tuple[Path, list[dict]]:
    tour = tour.upper()
    series = config.KALSHI_SERIES[tour]["winner"]
    markets = fetch_markets(series, status=status)
    pulled = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {"pulled_at": pulled, "tour": tour, "series": series, "markets": markets}
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = config.SNAPSHOTS / f"kalshi_{series}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path, markets


def compare_match(
    player1: str,
    player2: str,
    markets: list[dict],
    *,
    model_p1: float,
    us_open_only: bool = True,
) -> dict[str, Any] | None:
    target = {normalize_name(player1), normalize_name(player2)}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for market in markets:
        rules = str(market.get("rules_primary", ""))
        if us_open_only and "US Open" not in rules:
            continue
        grouped[str(market.get("event_ticker"))].append(market)
    for event_ticker, event_markets in grouped.items():
        by_name = {
            normalize_name(m.get("yes_sub_title") or str(m.get("title", "")).removesuffix(" wins")): m
            for m in event_markets
        }
        if not target.issubset(by_name):
            continue
        m1, m2 = by_name[normalize_name(player1)], by_name[normalize_name(player2)]
        mid1, mid2 = _midpoint(m1), _midpoint(m2)
        if mid1 is None or mid2 is None or mid1 + mid2 <= 0:
            fair1 = None
            fair2 = None
        else:
            fair1 = mid1 / (mid1 + mid2)
            fair2 = 1 - fair1
        return {
            "event_ticker": event_ticker,
            "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "player1": {
                "name": player1,
                "ticker": m1.get("ticker"),
                "bid": _dollars(m1, "yes_bid"),
                "ask": _dollars(m1, "yes_ask"),
                "midpoint": mid1,
                "de_vig_probability": fair1,
                "volume": float(m1.get("volume_fp") or m1.get("volume") or 0),
            },
            "player2": {
                "name": player2,
                "ticker": m2.get("ticker"),
                "bid": _dollars(m2, "yes_bid"),
                "ask": _dollars(m2, "yes_ask"),
                "midpoint": mid2,
                "de_vig_probability": fair2,
                "volume": float(m2.get("volume_fp") or m2.get("volume") or 0),
            },
            "model_minus_market_p1": None if fair1 is None else round(model_p1 - fair1, 4),
            "note": "Public quote comparison only; not a trade recommendation.",
        }
    return None

