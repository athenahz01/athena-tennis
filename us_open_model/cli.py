from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .data import fetch_history, load_history
from .evaluation import save_backtest, walk_forward_backtest
from .kalshi import compare_match, snapshot_winner_markets
from .ledger import log_forecast, log_market
from .model import TennisPredictor
from .pipeline import predict_full
from .schema import MatchContext


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _context(args: argparse.Namespace, *, tour: str | None = None,
             best_of: int | None = None, round_name: str | None = None,
             date: str | None = None, court: str | None = None) -> MatchContext:
    resolved_tour = (tour or args.tour).upper()
    return MatchContext(
        tour=resolved_tour,
        surface=getattr(args, "surface", "Hard"),
        best_of=best_of or getattr(args, "best_of", None) or (5 if resolved_tour == "ATP" else 3),
        tournament=getattr(args, "tournament", "US Open"),
        round=round_name or getattr(args, "round", "R128"),
        match_date=date or getattr(args, "date", "2026-09-01"),
        court=court or getattr(args, "court", None),
        indoor=getattr(args, "indoor", False),
        temperature_c=getattr(args, "temperature_c", None),
        humidity_pct=getattr(args, "humidity_pct", None),
        wind_kph=getattr(args, "wind_kph", None),
    )


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
