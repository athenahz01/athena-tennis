from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from . import config


@dataclass(frozen=True)
class TradingConfig:
    """Risk rules for the pre-match paper agent.

    The prediction model is a pre-match model, so this agent never treats an
    in-play price move as a new model signal. One event can open one position.
    """

    atp_min_edge_cents: float = 6.0
    wta_min_edge_cents: float = 8.0
    max_spread_cents: float = 4.0
    min_price_cents: float = 10.0
    max_price_cents: float = 90.0
    max_contracts: int = 20
    max_match_cost_cents: int = 1_000
    max_session_cost_cents: int = 10_000
    starting_bankroll_cents: int = 100_000
    kelly_fraction: float = 0.25
    fee_multiplier: float = 1.0
    blocked_quality_flags: tuple[str, ...] = (
        "large_model_disagreement",
        "low_match_history",
        "low_surface_history",
        "missing_rank",
    )


def taker_fee_cents(price_cents: float, contracts: int, multiplier: float = 1.0) -> int:
    """Conservative whole-cent estimate for Kalshi's quadratic taker fee."""
    if contracts <= 0:
        return 0
    probability = price_cents / 100.0
    return int(math.ceil(7.0 * multiplier * contracts * probability * (1.0 - probability)))


def binary_kelly_fraction(fair_probability: float, price_probability: float) -> float:
    """Full Kelly bankroll fraction for a binary contract bought at ``price``."""
    if not 0 < price_probability < 1:
        return 0.0
    return max(0.0, (fair_probability - price_probability) / (1.0 - price_probability))


def _round_cents(value: float | int | None) -> float | None:
    return None if value is None else round(float(value) * 100.0, 2)


def _quality_block(flags: Iterable[str], rules: TradingConfig) -> str | None:
    blocked = [flag for flag in flags if flag in rules.blocked_quality_flags]
    labels = {
        "large_model_disagreement": "the component models disagree",
        "low_match_history": "limited match history",
        "low_surface_history": "limited history on hard courts",
        "missing_rank": "ranking unavailable",
    }
    return None if not blocked else "Blocked: " + ", ".join(labels[flag] for flag in blocked)


def _side_view(
    *,
    side: str,
    name: str,
    ticker: str | None,
    fair_probability: float,
    market: dict[str, Any] | None,
    tour: str,
    flags: list[str],
    bankroll_cents: int,
    available_match_cents: int,
    available_session_cents: int,
    rules: TradingConfig,
) -> dict[str, Any]:
    bid = _round_cents((market or {}).get("bid"))
    ask = _round_cents((market or {}).get("ask"))
    threshold = rules.wta_min_edge_cents if tour.upper() == "WTA" else rules.atp_min_edge_cents
    view: dict[str, Any] = {
        "side": side,
        "player": name,
        "ticker": ticker,
        "fair_cents": round(fair_probability * 100.0, 2),
        "bid_cents": bid,
        "ask_cents": ask,
        "spread_cents": None if bid is None or ask is None else round(ask - bid, 2),
        "edge_cents": None,
        "min_edge_cents": threshold,
        "contracts": 0,
        "cost_cents": 0,
        "fee_cents": 0,
        "eligible": False,
        "reason": "no live two-sided quote",
    }
    if not ticker or bid is None or ask is None:
        return view
    if not (0 < bid <= ask < 100):
        view["reason"] = "invalid book"
        return view
    quality_reason = _quality_block(flags, rules)
    if quality_reason:
        view["reason"] = quality_reason
        return view
    if ask < rules.min_price_cents or ask > rules.max_price_cents:
        view["reason"] = "price outside risk band"
        return view
    if ask - bid > rules.max_spread_cents:
        view["reason"] = "spread too wide"
        return view

    one_fee = taker_fee_cents(ask, 1, rules.fee_multiplier)
    net_edge = fair_probability * 100.0 - ask - one_fee
    view["edge_cents"] = round(net_edge, 2)
    if net_edge < threshold:
        view["reason"] = "edge below entry bar"
        return view

    full_kelly = binary_kelly_fraction(fair_probability, ask / 100.0)
    kelly_budget = int(bankroll_cents * full_kelly * rules.kelly_fraction)
    budget = min(
        available_match_cents,
        available_session_cents,
        rules.max_match_cost_cents,
        kelly_budget,
    )
    if budget < ask:
        view["reason"] = "risk budget too small"
        return view
    contracts = min(rules.max_contracts, int(budget // ask))
    fee = taker_fee_cents(ask, contracts, rules.fee_multiplier)
    while contracts > 0 and math.ceil(ask * contracts) + fee > budget:
        contracts -= 1
        fee = taker_fee_cents(ask, contracts, rules.fee_multiplier)
    if contracts <= 0:
        view["reason"] = "risk budget too small"
        return view
    view.update(
        {
            "contracts": contracts,
            "cost_cents": int(math.ceil(ask * contracts)) + fee,
            "fee_cents": fee,
            "eligible": True,
            "reason": "Price and risk checks passed.",
        }
    )
    return view


def evaluate_match(
    forecast: dict[str, Any],
    market: dict[str, Any] | None,
    *,
    bankroll_cents: int,
    session_cost_cents: int,
    existing_event: bool = False,
    rules: TradingConfig | None = None,
) -> dict[str, Any]:
    """Return both side evaluations and one selected paper entry, if any."""
    rules = rules or TradingConfig()
    p1 = float(forecast["probabilities"]["final"])
    flags = list(forecast.get("data_quality") or [])
    tour = str(forecast["context"]["tour"])
    event_ticker = (market or {}).get("event_ticker")
    available_session = max(0, rules.max_session_cost_cents - session_cost_cents)
    sides = [
        _side_view(
            side="player1",
            name=forecast["player1"]["name"],
            ticker=((market or {}).get("player1") or {}).get("ticker"),
            fair_probability=p1,
            market=(market or {}).get("player1"),
            tour=tour,
            flags=flags,
            bankroll_cents=bankroll_cents,
            available_match_cents=rules.max_match_cost_cents,
            available_session_cents=available_session,
            rules=rules,
        ),
        _side_view(
            side="player2",
            name=forecast["player2"]["name"],
            ticker=((market or {}).get("player2") or {}).get("ticker"),
            fair_probability=1.0 - p1,
            market=(market or {}).get("player2"),
            tour=tour,
            flags=flags,
            bankroll_cents=bankroll_cents,
            available_match_cents=rules.max_match_cost_cents,
            available_session_cents=available_session,
            rules=rules,
        ),
    ]
    eligible = [side for side in sides if side["eligible"]]
    selected = max(eligible, key=lambda side: side["edge_cents"]) if eligible else None
    if existing_event:
        selected = None
        state = "position"
        reason = "one position already open for this match"
    elif selected:
        state = "entry"
        reason = selected["reason"]
    elif market:
        state = "watch"
        reason = max(sides, key=lambda side: side.get("edge_cents") or -999)["reason"]
    else:
        state = "market_missing"
        reason = "no matching open US Open market"
    return {
        "match_id": (
            f"{tour}:{forecast['context']['match_date']}:"
            f"{forecast['player1']['name']}:{forecast['player2']['name']}"
        ),
        "event_ticker": event_ticker,
        "tour": tour,
        "round": forecast["context"]["round"],
        "court": forecast["context"].get("court"),
        "player1": forecast["player1"]["name"],
        "player2": forecast["player2"]["name"],
        "model_p1": p1,
        "model_version": forecast["model_version"],
        "quality": flags,
        "market_observed_at": (market or {}).get("observed_at"),
        "state": state,
        "reason": reason,
        "sides": sides,
        "selected": selected,
    }


def append_journal(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_state(path: Path, rules: TradingConfig) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "ts": now,
        "mode": "paper",
        "session": {
            "started_at": now,
            "buys": 0,
            "settles": 0,
            "fees_cents": 0,
            "realized_cents": 0,
            "open_cost_cents": 0,
            "starting_bankroll_cents": rules.starting_bankroll_cents,
            "bankroll_cents": rules.starting_bankroll_cents,
        },
        "positions": [],
        "matches": [],
        "events": [],
        "rules": asdict(rules),
    }


def save_state(state: dict[str, Any], path: Path, web_path: Path | None = None) -> None:
    state["ts"] = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = json.dumps(state, indent=2, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    if web_path is not None:
        web_path.parent.mkdir(parents=True, exist_ok=True)
        web_path.write_text(payload, encoding="utf-8")


def enter_selected_positions(
    state: dict[str, Any], evaluations: list[dict[str, Any]], journal: Path
) -> None:
    session = state["session"]
    positions = state["positions"]
    events = state["events"]
    open_events = {position["event_ticker"] for position in positions if position["status"] == "open"}
    for match in evaluations:
        selected = match.get("selected")
        event_ticker = match.get("event_ticker")
        if not selected or not event_ticker or event_ticker in open_events:
            continue
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        position = {
            "status": "open",
            "opened_at": now,
            "event_ticker": event_ticker,
            "ticker": selected["ticker"],
            "tour": match["tour"],
            "player": selected["player"],
            "opponent": match["player2"] if selected["side"] == "player1" else match["player1"],
            "side": selected["side"],
            "fair_cents": selected["fair_cents"],
            "entry_cents": selected["ask_cents"],
            "edge_cents": selected["edge_cents"],
            "contracts": selected["contracts"],
            "cost_cents": selected["cost_cents"],
            "fee_cents": selected["fee_cents"],
        }
        positions.append(position)
        open_events.add(event_ticker)
        session["buys"] += 1
        session["fees_cents"] += selected["fee_cents"]
        session["open_cost_cents"] += selected["cost_cents"]
        session["bankroll_cents"] -= selected["cost_cents"]
        event = {"kind": "buy", "ts": now, **position}
        events.insert(0, event)
        del events[50:]
        append_journal(journal, event)


def fetch_market(ticker: str) -> dict[str, Any]:
    response = requests.get(
        f"{config.KALSHI_BASE}/markets/{ticker}",
        headers={"Accept": "application/json", "User-Agent": "us-open-predictor/0.2"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("market", response.json())


def settle_finalized_positions(state: dict[str, Any], journal: Path) -> None:
    session = state["session"]
    for position in state["positions"]:
        if position["status"] != "open":
            continue
        try:
            market = fetch_market(position["ticker"])
        except requests.RequestException:
            continue
        result = str(market.get("result") or "").lower()
        status = str(market.get("status") or "").lower()
        if result not in {"yes", "no"} or status not in {"determined", "finalized"}:
            continue
        payout = position["contracts"] * (100 if result == "yes" else 0)
        pnl = payout - position["cost_cents"]
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        position.update(
            {"status": "settled", "settled_at": now, "result": result, "pnl_cents": pnl}
        )
        session["settles"] += 1
        session["realized_cents"] += pnl
        session["open_cost_cents"] -= position["cost_cents"]
        session["bankroll_cents"] += payout
        event = {"kind": "settle", "ts": now, **position}
        state["events"].insert(0, event)
        del state["events"][50:]
        append_journal(journal, event)
