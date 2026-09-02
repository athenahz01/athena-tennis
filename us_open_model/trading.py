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
    entry_cutoff_minutes: int = 5
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


def pre_match_entry_gate(
    forecast: dict[str, Any],
    market: dict[str, Any] | None,
    *,
    now: dt.datetime | None = None,
    rules: TradingConfig | None = None,
) -> tuple[bool, str, str | None]:
    """Fail closed unless schedule and status prove that a match has not started."""
    rules = rules or TradingConfig()
    context = forecast.get("context") or {}
    scheduled_text = str(context.get("scheduled_start") or "").strip()
    if not scheduled_text:
        return False, "blocked: verified scheduled start unavailable", None
    try:
        scheduled = dt.datetime.fromisoformat(scheduled_text.replace("Z", "+00:00"))
    except ValueError:
        return False, "blocked: scheduled start is invalid", scheduled_text
    if scheduled.tzinfo is None or scheduled.utcoffset() is None:
        return False, "blocked: scheduled start needs a timezone", scheduled_text

    status = str(context.get("match_status") or "").strip().lower().replace("-", "_")
    allowed_statuses = {"scheduled", "pre_match", "upcoming", "not_started"}
    if status not in allowed_statuses:
        if status in {"live", "in_play", "in_progress", "suspended", "completed", "final"}:
            return (
                False,
                f"blocked: match status is {status.replace('_', ' ')}",
                scheduled.isoformat(),
            )
        return False, "blocked: verified pre-match status unavailable", scheduled.isoformat()

    market_statuses = {
        str(value).lower() for value in (market or {}).get("market_status", []) if value
    }
    if market_statuses and not market_statuses.issubset({"active", "open"}):
        return False, "blocked: Kalshi market is not open", scheduled.isoformat()

    observed = now or dt.datetime.now(dt.timezone.utc)
    if observed.tzinfo is None or observed.utcoffset() is None:
        observed = observed.replace(tzinfo=dt.timezone.utc)
    cutoff = scheduled - dt.timedelta(minutes=rules.entry_cutoff_minutes)
    if observed >= cutoff:
        return False, "blocked: entry cutoff has passed", scheduled.isoformat()
    return True, "verified pre-match window", scheduled.isoformat()


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
    now: dt.datetime | None = None,
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
    gate_open, gate_reason, scheduled_start = pre_match_entry_gate(
        forecast, market, now=now, rules=rules
    )
    if not gate_open:
        for side in sides:
            side.update(
                {
                    "eligible": False,
                    "contracts": 0,
                    "cost_cents": 0,
                    "fee_cents": 0,
                    "reason": gate_reason,
                }
            )
    eligible = [side for side in sides if side["eligible"]]
    selected = max(eligible, key=lambda side: side["edge_cents"]) if eligible else None
    if existing_event:
        selected = None
        state = "position"
        reason = "one position already open for this match"
    elif selected:
        state = "entry"
        reason = selected["reason"]
    elif not gate_open:
        state = "blocked"
        reason = gate_reason
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
        "scheduled_start": scheduled_start,
        "entry_gate": {"open": gate_open, "reason": gate_reason},
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
    open_events = {
        position["event_ticker"] for position in positions if position["status"] == "open"
    }
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
            "scheduled_start": match.get("scheduled_start"),
            "entry_gate": match.get("entry_gate"),
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


def settle_finalized_positions(
    state: dict[str, Any], journal: Path
) -> dict[str, dict[str, Any]]:
    session = state["session"]
    fetched: dict[str, dict[str, Any]] = {}
    for position in state["positions"]:
        if position["status"] != "open":
            continue
        try:
            market = fetch_market(position["ticker"])
        except requests.RequestException:
            continue
        fetched[position["ticker"]] = market
        result = str(market.get("result") or "").lower()
        status = str(market.get("status") or "").lower()
        if result not in {"yes", "no"} or status not in {
            "closed",
            "determined",
            "finalized",
            "settled",
        }:
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
    return fetched


def _market_bid_cents(market: dict[str, Any]) -> float | None:
    dollars = market.get("yes_bid_dollars")
    if dollars is not None:
        try:
            return float(dollars) * 100.0
        except (TypeError, ValueError):
            return None
    cents = market.get("yes_bid")
    try:
        return float(cents) if cents is not None else None
    except (TypeError, ValueError):
        return None


def build_performance_report(
    state: dict[str, Any], markets: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Summarize realized results and optional bid-side liquidation marks."""
    markets = markets or {}
    positions = list(state.get("positions") or [])
    settled = [position for position in positions if position.get("status") == "settled"]
    opened = [position for position in positions if position.get("status") == "open"]
    realized = sum(int(position.get("pnl_cents") or 0) for position in settled)
    marked_rows: list[dict[str, Any]] = []
    marked_pnl = 0
    marked_count = 0
    for position in opened:
        market = markets.get(str(position.get("ticker")))
        bid = _market_bid_cents(market or {})
        row = {
            "ticker": position.get("ticker"),
            "player": position.get("player"),
            "cost_cents": int(position.get("cost_cents") or 0),
            "bid_cents": None if bid is None else round(bid, 2),
            "liquidation_value_cents": None,
            "marked_pnl_cents": None,
            "market_status": (market or {}).get("status"),
        }
        if bid is not None:
            contracts = int(position.get("contracts") or 0)
            exit_fee = taker_fee_cents(bid, contracts)
            proceeds = max(0, int(math.floor(bid * contracts)) - exit_fee)
            pnl = proceeds - row["cost_cents"]
            row.update(
                {
                    "exit_fee_cents": exit_fee,
                    "liquidation_value_cents": proceeds,
                    "marked_pnl_cents": pnl,
                }
            )
            marked_pnl += pnl
            marked_count += 1
        marked_rows.append(row)

    total_cost = sum(int(position.get("cost_cents") or 0) for position in positions)
    combined_pnl = realized + marked_pnl if marked_count == len(opened) else None
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "positions": len(positions),
        "open_positions": len(opened),
        "settled_positions": len(settled),
        "wins": sum(1 for position in settled if position.get("result") == "yes"),
        "losses": sum(1 for position in settled if position.get("result") == "no"),
        "settled_hit_rate": (
            None
            if not settled
            else round(
                sum(1 for position in settled if position.get("result") == "yes")
                / len(settled),
                4,
            )
        ),
        "total_cost_cents": total_cost,
        "realized_pnl_cents": realized,
        "marked_open_pnl_cents": marked_pnl if marked_count else None,
        "combined_pnl_cents": combined_pnl,
        "combined_return": (
            None if combined_pnl is None or not total_cost else round(combined_pnl / total_cost, 6)
        ),
        "marks_available": marked_count,
        "legacy_unverified_entries": sum(
            1 for position in positions if not position.get("scheduled_start")
        ),
        "open_marks": marked_rows,
    }
