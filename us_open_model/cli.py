from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from . import config
from .data import fetch_history, load_history
from .evaluation import save_backtest, walk_forward_backtest
from .kalshi import compare_match, snapshot_winner_markets
from .ledger import log_forecast, log_market
from .model import TennisPredictor
from .pipeline import predict_full
from .schema import MatchContext
from .trading import (
    TradingConfig,
    build_performance_report,
    enter_selected_positions,
    evaluate_match,
    load_state,
    save_state,
    settle_finalized_positions,
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _context(
    args: argparse.Namespace,
    *,
    tour: str | None = None,
    best_of: int | None = None,
    round_name: str | None = None,
    date: str | None = None,
    court: str | None = None,
    scheduled_start: str | None = None,
    match_status: str | None = None,
) -> MatchContext:
    resolved_tour = (tour or args.tour).upper()
    return MatchContext(
        tour=resolved_tour,
        surface=getattr(args, "surface", "Hard"),
        best_of=best_of or getattr(args, "best_of", None) or (5 if resolved_tour == "ATP" else 3),
        tournament=getattr(args, "tournament", "US Open"),
        round=round_name or getattr(args, "round", "R128"),
        match_date=date or getattr(args, "date", "2026-09-01"),
        scheduled_start=scheduled_start or getattr(args, "scheduled_start", None),
        match_status=match_status or getattr(args, "match_status", None),
        court=court or getattr(args, "court", None),
        indoor=getattr(args, "indoor", False),
        temperature_c=getattr(args, "temperature_c", None),
        humidity_pct=getattr(args, "humidity_pct", None),
        wind_kph=getattr(args, "wind_kph", None),
    )


def _optional_text(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else str(value).strip() or None


def _write_json(payload: dict[str, Any] | list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def command_fetch(args: argparse.Namespace) -> None:
    path = fetch_history(args.start_year, args.end_year, force=args.force)
    print(f"Frozen data manifest: {path}")


def command_train(args: argparse.Namespace) -> None:
    history = load_history(cutoff=args.cutoff)
    predictor = TennisPredictor.train(history, cutoff=args.cutoff)
    path = predictor.save(Path(args.artifact))
    print(f"Trained {predictor.data_rows:,} matches through {args.cutoff}: {path}")
    for tour, model in predictor.models.items():
        metrics = model.validation["candidates"][model.champion]
        print(
            f"  {tour}: {model.champion} | validation log loss {metrics['log_loss']:.4f} "
            f"| Brier {metrics['brier']:.4f} | n={model.validation['n_calibration']:,}"
        )


def command_predict(args: argparse.Namespace) -> None:
    predictor = TennisPredictor.load(Path(args.artifact))
    context = _context(args)
    forecast = predict_full(
        predictor,
        args.player1,
        args.player2,
        context,
        n_sims=args.sims,
        seed=args.seed,
    )
    prediction_id = log_forecast(forecast)
    snapshot_path = None
    market = None
    if args.kalshi:
        snapshot_path, markets = snapshot_winner_markets(context.tour)
        market = compare_match(
            args.player1,
            args.player2,
            markets,
            model_p1=forecast["probabilities"]["final"],
        )
        forecast["kalshi"] = market or {"status": "matching market not found"}
        log_market(prediction_id, market, snapshot_path)
    output = Path(args.output) if args.output else config.OUTPUTS / (
        f"{context.match_date}_{_slug(args.player1)}_vs_{_slug(args.player2)}.json"
    )
    _write_json(forecast, output)
    print(json.dumps(forecast, indent=2, ensure_ascii=False))
    print(f"Saved: {output}")


def command_slate(args: argparse.Namespace) -> None:
    predictor = TennisPredictor.load(Path(args.artifact))
    fixtures = pd.read_csv(args.fixtures)
    required = {"tour", "player1", "player2", "best_of", "round"}
    missing = required - set(fixtures.columns)
    if missing:
        raise ValueError(f"Fixture CSV missing: {sorted(missing)}")
    forecasts: list[dict[str, Any]] = []
    prediction_ids: list[str] = []
    for index, row in fixtures.iterrows():
        context = _context(
            args,
            tour=str(row.tour),
            best_of=int(row.best_of),
            round_name=str(row["round"]),
            date=str(row.get("date", args.date)),
            court=None if pd.isna(row.get("court")) else str(row.get("court")),
            scheduled_start=_optional_text(row.get("scheduled_start")),
            match_status=_optional_text(row.get("match_status")),
        )
        forecast = predict_full(
            predictor,
            str(row.player1),
            str(row.player2),
            context,
            n_sims=args.sims,
            seed=args.seed + index,
        )
        forecasts.append(forecast)
        prediction_ids.append(log_forecast(forecast))

    if args.kalshi:
        for tour in sorted(set(fixtures["tour"].str.upper())):
            snapshot_path, markets = snapshot_winner_markets(tour)
            for index, forecast in enumerate(forecasts):
                if forecast["context"]["tour"].upper() != tour:
                    continue
                market = compare_match(
                    forecast["player1"]["name"],
                    forecast["player2"]["name"],
                    markets,
                    model_p1=forecast["probabilities"]["final"],
                )
                forecast["kalshi"] = market or {"status": "matching market not found"}
                log_market(prediction_ids[index], market, snapshot_path)

    output = Path(args.output) if args.output else config.OUTPUTS / "us_open_2026_slate.json"
    _write_json(forecasts, output)
    if args.web_output:
        web_output = _write_json(forecasts, Path(args.web_output))
    else:
        web_output = None
    summary = pd.DataFrame(
        [
            {
                "tour": item["context"]["tour"],
                "date": item["context"]["match_date"],
                "player1": item["player1"]["name"],
                "player2": item["player2"]["name"],
                "p1_win": item["probabilities"]["final"],
                "prediction": item["winner"]["name"],
                "confidence": item["winner"]["probability"],
                "expected_games": item["simulation"]["total_games"]["mean"],
                "p_tiebreak": item["simulation"]["p_tiebreak"],
                "kalshi_p1": (item.get("kalshi") or {}).get("player1", {}).get(
                    "de_vig_probability"
                ),
                "model_minus_kalshi": (item.get("kalshi") or {}).get(
                    "model_minus_market_p1"
                ),
                "data_quality": ";".join(item["data_quality"]),
            }
            for item in forecasts
        ]
    )
    csv_path = output.with_suffix(".csv")
    summary.to_csv(csv_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {output} and {csv_path}")
    if web_output:
        print(f"Website data refreshed: {web_output}")


def command_backtest(args: argparse.Namespace) -> None:
    history = load_history(cutoff=args.cutoff)
    result = walk_forward_backtest(history, args.years, n_resamples=args.resamples)
    path = save_backtest(result, Path(args.output) if args.output else None)
    for fold in result["folds"]:
        print(
            f"{fold['year']} {fold['tour']} {fold['selected_component']}: "
            f"LL {fold['model']['log_loss']:.4f} vs Elo {fold['elo']['log_loss']:.4f}; "
            f"ship={fold['paired_gate_vs_elo']['ship']}"
        )
    print(f"Saved: {path}")


def _push_agent_state(state: dict[str, Any]) -> None:
    url = os.environ.get("AGENT_STATE_URL", "").strip()
    key = os.environ.get("AGENT_STATE_KEY", "").strip()
    if not url or not key:
        return
    response = requests.post(
        url,
        json=state,
        headers={"X-Agent-Key": key},
        timeout=10,
    )
    response.raise_for_status()


def _print_performance(report: dict[str, Any]) -> None:
    def money(cents: int | None) -> str:
        if cents is None:
            return "unavailable"
        sign = "-" if cents < 0 else ""
        return f"{sign}${abs(cents) / 100:,.2f}"

    print(
        "[report] "
        f"positions={report['positions']} open={report['open_positions']} "
        f"settled={report['settled_positions']} wins={report['wins']} "
        f"losses={report['losses']} realized={money(report['realized_pnl_cents'])} "
        f"marked_open={money(report['marked_open_pnl_cents'])} "
        f"combined={money(report['combined_pnl_cents'])}"
    )
    if report["legacy_unverified_entries"]:
        print(
            f"[report] warning={report['legacy_unverified_entries']} existing positions "
            "lack verified start times and are excluded from clean pre-match evaluation"
        )


def command_agent(args: argparse.Namespace) -> None:
    slate_path = Path(args.slate)
    forecasts: list[dict[str, Any]] = []
    if not args.settle_only:
        forecasts = json.loads(slate_path.read_text(encoding="utf-8"))
        if not isinstance(forecasts, list):
            raise ValueError("Agent slate must be a JSON array")
    rules = TradingConfig()
    state_path = Path(args.state)
    web_path = Path(args.web_output) if args.web_output else None
    journal = Path(args.journal) if args.journal else (
        config.DATA / "paper" / f"tennis-paper-{dt.date.today():%Y%m%d}.jsonl"
    )
    state = load_state(state_path, rules)
    stream_warned = False

    def cycle() -> None:
        nonlocal stream_warned
        settled_markets: dict[str, dict[str, Any]] = {}
        if not args.cached:
            settled_markets = settle_finalized_positions(state, journal)
        if args.settle_only:
            report = build_performance_report(state, settled_markets)
            state["performance"] = report
            state["mode"] = "paper settlement only"
            state["rules"] = asdict(rules)
            save_state(state, state_path, web_path)
            try:
                _push_agent_state(state)
            except requests.RequestException as exc:
                if not stream_warned:
                    stream_warned = True
                    print(f"[paper] website state push failed: {type(exc).__name__}")
            print(
                f"[paper] {state['ts']} settlement-only open={report['open_positions']} "
                f"settled={report['settled_positions']} state={state_path}"
            )
            if args.report:
                _print_performance(report)
                if args.report_output:
                    _write_json(report, Path(args.report_output))
                    print(f"[report] saved={args.report_output}")
            return
        markets_by_tour: dict[str, list[dict[str, Any]]] = {}
        if not args.cached:
            for tour in sorted({str(row["context"]["tour"]).upper() for row in forecasts}):
                try:
                    _, markets_by_tour[tour] = snapshot_winner_markets(tour)
                except requests.RequestException as exc:
                    print(f"[paper] {tour} market refresh failed: {type(exc).__name__}")
                    markets_by_tour[tour] = []

        open_events = {
            row["event_ticker"] for row in state.get("positions", []) if row["status"] == "open"
        }
        planned_cost = int(state["session"].get("open_cost_cents", 0))
        evaluations: list[dict[str, Any]] = []
        for forecast in forecasts:
            tour = str(forecast["context"]["tour"]).upper()
            if args.cached:
                cached_market = forecast.get("kalshi") or {}
                market = cached_market if cached_market.get("player1") else None
            else:
                market = compare_match(
                    forecast["player1"]["name"],
                    forecast["player2"]["name"],
                    markets_by_tour.get(tour, []),
                    model_p1=float(forecast["probabilities"]["final"]),
                )
            event_ticker = (market or {}).get("event_ticker")
            evaluation = evaluate_match(
                forecast,
                market,
                bankroll_cents=int(state["session"]["starting_bankroll_cents"]),
                session_cost_cents=planned_cost,
                existing_event=event_ticker in open_events,
                rules=rules,
            )
            evaluations.append(evaluation)
            if evaluation.get("selected") and not args.observe_only:
                planned_cost += int(evaluation["selected"]["cost_cents"])

        if not args.observe_only:
            enter_selected_positions(state, evaluations, journal)
        positions_by_event = {
            row["event_ticker"]: row
            for row in state.get("positions", [])
            if row["status"] == "open"
        }
        for evaluation in evaluations:
            position = positions_by_event.get(evaluation.get("event_ticker"))
            if position:
                evaluation["state"] = "position"
                evaluation["reason"] = "paper position open"
                evaluation["position"] = position
        state["matches"] = evaluations
        state["mode"] = "paper observation" if args.observe_only else "paper"
        state["rules"] = asdict(rules)
        if args.report:
            state["performance"] = build_performance_report(state, settled_markets)
        save_state(state, state_path, web_path)
        try:
            _push_agent_state(state)
        except requests.RequestException as exc:
            if not stream_warned:
                stream_warned = True
                print(f"[paper] website state push failed: {type(exc).__name__}")
        entries = sum(1 for row in evaluations if row["state"] == "entry")
        open_count = len(positions_by_event)
        print(
            f"[paper] {state['ts']} watched={len(evaluations)} "
            f"entries={entries} open={open_count} state={state_path}"
        )
        if args.report:
            _print_performance(state["performance"])

    try:
        while True:
            cycle()
            if not args.watch or args.settle_only:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        save_state(state, state_path, web_path)
        print("[paper] stopped; state and journal saved")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="us-open", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download immutable ATP/WTA history snapshots")
    fetch.add_argument("--start-year", type=int, default=config.DEFAULT_START_YEAR)
    fetch.add_argument("--end-year", type=int, default=2026)
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=command_fetch)

    train = sub.add_parser("train", help="train both tour models")
    train.add_argument("--cutoff", default=config.DEFAULT_TRAIN_CUTOFF)
    train.add_argument("--artifact", default=str(config.ARTIFACTS / "uso_hybrid_v1.joblib"))
    train.set_defaults(func=command_train)

    def add_context(command: argparse.ArgumentParser, include_players: bool = True) -> None:
        if include_players:
            command.add_argument("--tour", choices=["ATP", "WTA"], required=True)
            command.add_argument("--player1", required=True)
            command.add_argument("--player2", required=True)
            command.add_argument("--best-of", type=int, choices=[3, 5])
        command.add_argument("--surface", default="Hard")
        command.add_argument("--tournament", default="US Open")
        command.add_argument("--round", default="R128")
        command.add_argument("--date", default="2026-09-01")
        command.add_argument("--court")
        command.add_argument("--scheduled-start")
        command.add_argument(
            "--match-status",
            choices=[
                "scheduled",
                "pre_match",
                "upcoming",
                "not_started",
                "live",
                "suspended",
                "completed",
            ],
        )
        command.add_argument("--indoor", action="store_true")
        command.add_argument("--temperature-c", type=float)
        command.add_argument("--humidity-pct", type=float)
        command.add_argument("--wind-kph", type=float)

    predict = sub.add_parser("predict", help="predict one match and detailed stats")
    add_context(predict)
    predict.add_argument("--artifact", default=str(config.ARTIFACTS / "uso_hybrid_v1.joblib"))
    predict.add_argument("--sims", type=int, default=20_000)
    predict.add_argument("--seed", type=int, default=20260901)
    predict.add_argument("--kalshi", action="store_true")
    predict.add_argument("--output")
    predict.set_defaults(func=command_predict)

    slate = sub.add_parser("slate", help="predict every row of a fixture CSV")
    add_context(slate, include_players=False)
    slate.add_argument("--fixtures", required=True)
    slate.add_argument("--artifact", default=str(config.ARTIFACTS / "uso_hybrid_v1.joblib"))
    slate.add_argument("--sims", type=int, default=2_000)
    slate.add_argument("--seed", type=int, default=20260901)
    slate.add_argument("--kalshi", action="store_true")
    slate.add_argument("--output")
    slate.add_argument(
        "--web-output",
        help="also write the full slate JSON to a website data path",
    )
    slate.set_defaults(func=command_slate)

    backtest = sub.add_parser("backtest", help="expanding annual walk-forward judge")
    backtest.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026])
    backtest.add_argument("--cutoff", default="2026-12-31")
    backtest.add_argument("--resamples", type=int, default=10_000)
    backtest.add_argument("--output")
    backtest.set_defaults(func=command_backtest)

    agent = sub.add_parser("agent", help="run the pre-match Kalshi paper trading agent")
    agent.add_argument("--slate", default=str(config.OUTPUTS / "us_open_2026_latest_slate.json"))
    agent.add_argument("--state", default=str(config.OUTPUTS / "tennis_agent_state.json"))
    agent.add_argument("--web-output", default=str(config.ROOT / "web/public/data/agent.json"))
    agent.add_argument("--journal")
    agent.add_argument("--watch", action="store_true", help="poll until stopped")
    agent.add_argument("--poll-seconds", type=float, default=30.0)
    agent.add_argument(
        "--cached",
        action="store_true",
        help="use quotes already stored in the slate",
    )
    agent.add_argument(
        "--observe-only",
        action="store_true",
        help="evaluate without opening paper positions",
    )
    agent.add_argument(
        "--settle-only",
        action="store_true",
        help="settle existing positions without evaluating or opening entries",
    )
    agent.add_argument("--report", action="store_true", help="print a paper performance report")
    agent.add_argument(
        "--report-output",
        default=str(config.OUTPUTS / "tennis_agent_report.json"),
    )
    agent.set_defaults(func=command_agent)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
