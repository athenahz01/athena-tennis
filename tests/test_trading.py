from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from us_open_model.trading import (
    TradingConfig,
    binary_kelly_fraction,
    build_performance_report,
    enter_selected_positions,
    evaluate_match,
    load_state,
    pre_match_entry_gate,
    settle_finalized_positions,
    taker_fee_cents,
)


def forecast(*, tour: str = "ATP", p1: float = 0.68, flags: list[str] | None = None) -> dict:
    return {
        "model_version": "test-v1",
        "context": {
            "tour": tour,
            "match_date": "2026-09-01",
            "round": "R128",
            "court": "Court 7",
            "scheduled_start": "2099-09-01T12:00:00-04:00",
            "match_status": "scheduled",
        },
        "player1": {"name": "Player One"},
        "player2": {"name": "Player Two"},
        "probabilities": {"final": p1},
        "data_quality": flags or ["ok"],
    }


def market(*, p1_bid: float = 0.49, p1_ask: float = 0.50) -> dict:
    return {
        "event_ticker": "EVENT",
        "observed_at": "2026-09-01T12:00:00Z",
        "player1": {
            "ticker": "EVENT-ONE",
            "bid": p1_bid,
            "ask": p1_ask,
        },
        "player2": {
            "ticker": "EVENT-TWO",
            "bid": 1 - p1_ask,
            "ask": 1 - p1_bid,
        },
    }


def test_fee_and_kelly_are_conservative() -> None:
    assert taker_fee_cents(50, 10) == 18
    assert binary_kelly_fraction(0.60, 0.50) == pytest.approx(0.2)
    assert binary_kelly_fraction(0.40, 0.50) == 0.0


def test_agent_selects_only_the_best_eligible_side() -> None:
    decision = evaluate_match(
        forecast(),
        market(),
        bankroll_cents=100_000,
        session_cost_cents=0,
    )
    assert decision["state"] == "entry"
    assert decision["selected"]["player"] == "Player One"
    assert decision["selected"]["contracts"] > 0
    assert decision["selected"]["cost_cents"] <= 1_000


def test_quality_and_spread_gates_block_entries() -> None:
    quality = evaluate_match(
        forecast(flags=["low_match_history"]),
        market(),
        bankroll_cents=100_000,
        session_cost_cents=0,
    )
    spread = evaluate_match(
        forecast(),
        market(p1_bid=0.40, p1_ask=0.50),
        bankroll_cents=100_000,
        session_cost_cents=0,
    )
    assert quality["selected"] is None
    assert all(side["reason"].startswith("Blocked:") for side in quality["sides"])
    assert spread["selected"] is None
    assert all(side["reason"] == "spread too wide" for side in spread["sides"])


def test_wta_uses_a_higher_entry_bar_and_session_cap_is_hard() -> None:
    atp = evaluate_match(
        forecast(tour="ATP", p1=0.59),
        market(),
        bankroll_cents=100_000,
        session_cost_cents=0,
    )
    wta = evaluate_match(
        forecast(tour="WTA", p1=0.59),
        market(),
        bankroll_cents=100_000,
        session_cost_cents=0,
    )
    capped = evaluate_match(
        forecast(),
        market(),
        bankroll_cents=100_000,
        session_cost_cents=TradingConfig().max_session_cost_cents,
    )
    assert atp["selected"] is not None
    assert wta["selected"] is None
    assert capped["selected"] is None


def test_entry_writer_never_opens_two_positions_for_one_event(tmp_path: Path) -> None:
    rules = TradingConfig()
    state = load_state(tmp_path / "state.json", rules)
    decision = evaluate_match(
        forecast(),
        market(),
        bankroll_cents=100_000,
        session_cost_cents=0,
    )
    journal = tmp_path / "journal.jsonl"
    enter_selected_positions(state, [decision], journal)
    enter_selected_positions(state, [decision], journal)
    assert len(state["positions"]) == 1
    assert state["session"]["buys"] == 1
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 1


def test_entry_gate_fails_closed_without_verified_schedule() -> None:
    missing = forecast()
    missing["context"]["scheduled_start"] = None
    naive = forecast()
    naive["context"]["scheduled_start"] = "2099-09-01T12:00:00"
    live = forecast()
    live["context"]["match_status"] = "live"
    as_of = dt.datetime(2026, 9, 1, 12, tzinfo=dt.timezone.utc)

    assert pre_match_entry_gate(missing, market(), now=as_of)[0] is False
    assert pre_match_entry_gate(naive, market(), now=as_of)[0] is False
    assert pre_match_entry_gate(live, market(), now=as_of)[0] is False

    decision = evaluate_match(
        missing,
        market(),
        bankroll_cents=100_000,
        session_cost_cents=0,
        now=as_of,
    )
    assert decision["state"] == "blocked"
    assert decision["selected"] is None
    assert decision["entry_gate"]["open"] is False


def test_entry_gate_closes_before_scheduled_start() -> None:
    row = forecast()
    row["context"]["scheduled_start"] = "2026-09-01T13:00:00+00:00"
    rules = TradingConfig(entry_cutoff_minutes=5)
    allowed = pre_match_entry_gate(
        row,
        market(),
        now=dt.datetime(2026, 9, 1, 12, 54, tzinfo=dt.timezone.utc),
        rules=rules,
    )
    blocked = pre_match_entry_gate(
        row,
        market(),
        now=dt.datetime(2026, 9, 1, 12, 55, tzinfo=dt.timezone.utc),
        rules=rules,
    )
    assert allowed[0] is True
    assert blocked[0] is False


def test_settlement_and_performance_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = load_state(tmp_path / "state.json", TradingConfig())
    decision = evaluate_match(
        forecast(),
        market(),
        bankroll_cents=100_000,
        session_cost_cents=0,
    )
    journal = tmp_path / "journal.jsonl"
    enter_selected_positions(state, [decision], journal)
    monkeypatch.setattr(
        "us_open_model.trading.fetch_market",
        lambda ticker: {
            "ticker": ticker,
            "status": "settled",
            "result": "yes",
            "yes_bid": 100,
        },
    )

    fetched = settle_finalized_positions(state, journal)
    report = build_performance_report(state, fetched)

    assert state["session"]["settles"] == 1
    assert report["settled_positions"] == 1
    assert report["wins"] == 1
    assert report["realized_pnl_cents"] > 0
    assert report["legacy_unverified_entries"] == 0
