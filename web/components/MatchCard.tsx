"use client";

import { useState } from "react";
import type { Forecast, PlayerStats, Range } from "@/lib/types";

const pct = (value?: number | null, digits = 0) =>
  value == null ? "—" : `${(value * 100).toFixed(digits)}%`;

const number = (value?: number | null, digits = 1) =>
  value == null ? "—" : value.toFixed(digits);

function ProbabilityCourt({ model, market }: { model: number; market?: number | null }) {
  return (
    <div className="probability-wrap">
      <div className="probability-labels"><span>Player 1</span><span>50 · NET</span><span>Player 2</span></div>
      <div className="probability-court" aria-label={`Athena ${pct(model)}, Kalshi ${pct(market)}`}>
        <span className="prob-net" />
        <span className="prob-service left" />
        <span className="prob-service right" />
        <span className="model-ball" style={{ left: `${Math.min(98, Math.max(2, model * 100))}%` }}>
          <span className="sr-only">Athena {pct(model)}</span>
        </span>
        {market != null && <span className="market-marker" style={{ left: `${market * 100}%` }} />}
      </div>
      <div className="court-legend"><span><i className="legend-ball" />Athena {pct(model, 1)}</span>
        <span><i className="legend-market" />Kalshi {pct(market, 1)}</span></div>
    </div>
  );
}

function Distribution({ rows }: { rows: [string, number][] }) {
  const max = Math.max(...rows.map(([, value]) => value), 0.01);
  return <div className="distribution">{rows.map(([label, value]) => (
    <div className="distribution-row" key={label}>
      <strong>{label}</strong><span><i style={{ width: `${(value / max) * 100}%` }} /></span><em>{pct(value, 1)}</em>
    </div>
  ))}</div>;
}

function StatRow({ label, a, b }: { label: string; a: Range; b: Range }) {
  return <tr><td>{number(a.mean)}</td><th>{label}</th><td>{number(b.mean)}</td></tr>;
}

function PlayerStatTable({ forecast }: { forecast: Forecast }) {
  const a: PlayerStats = forecast.simulation.player1_stats;
  const b: PlayerStats = forecast.simulation.player2_stats;
  return (
    <table className="player-stat-table">
      <thead><tr><th>{forecast.player1.name.split(" ").at(-1)}</th><th>Projected mean</th><th>{forecast.player2.name.split(" ").at(-1)}</th></tr></thead>
      <tbody>
        <StatRow label="Aces" a={a.aces} b={b.aces} />
        <StatRow label="Double faults" a={a.double_faults} b={b.double_faults} />
        <StatRow label="Breaks" a={a.breaks} b={b.breaks} />
        <StatRow label="Break points faced" a={a.break_points_faced} b={b.break_points_faced} />
        <StatRow label="Service points won" a={a.service_points_won} b={b.service_points_won} />
      </tbody>
    </table>
  );
}

function readableFlag(flag: string) {
  const labels: Record<string, string> = {
    ok: "Full history",
    large_model_disagreement: "Models disagree",
    low_match_history: "Limited match history",
    low_surface_history: "Limited hard-court history",
    missing_rank: "Rank unavailable",
    "2026_wta_stack_failed_gate_using_elo_fallback": "WTA · Elo safety fallback",
  };
  return labels[flag] ?? flag.replaceAll("_", " ");
}

export default function MatchCard({ forecast, index }: { forecast: Forecast; index: number }) {
  const [open, setOpen] = useState(false);
  const p1 = forecast.probabilities.final;
  const p2 = 1 - p1;
  const market = forecast.kalshi?.player1?.de_vig_probability;
  const gap = forecast.kalshi?.model_minus_market_p1;
  const likelySet = Object.entries(forecast.simulation.set_score_distribution)
    .sort((a, b) => b[1] - a[1])[0];
  const setRows = Object.entries(forecast.simulation.set_score_distribution)
    .sort((a, b) => b[1] - a[1]);
  const componentRows: [string, number][] = [
    ["Final", forecast.probabilities.final],
    ["Elo", forecast.probabilities.elo],
    ["Point", forecast.probabilities.point_model],
    ["ML", forecast.probabilities.machine_learning],
    ["Stack", forecast.probabilities.stack],
  ];
  const totalsKeys = forecast.context.best_of === 5
    ? ["over_36.5", "over_39.5", "over_42.5"]
    : ["over_20.5", "over_22.5", "over_24.5"];

  return (
    <article className={`match-card ${open ? "is-open" : ""}`} style={{ "--order": index } as React.CSSProperties}>
      <div className="match-card-main">
        <header className="match-meta">
          <div><span className={`tour-pill ${forecast.context.tour.toLowerCase()}`}>{forecast.context.tour}</span>
            <span>{forecast.context.round}</span><span>{forecast.context.best_of === 5 ? "best of 5" : "best of 3"}</span></div>
          <span>{forecast.context.court ?? "Court TBD"}</span>
        </header>

        <div className="players">
          <div className={p1 >= .5 ? "favored" : ""}>
            <span className="rank">{forecast.player1.rank ? `#${Math.round(forecast.player1.rank)}` : "NR"}</span>
            <h3>{forecast.player1.name}</h3><strong>{pct(p1, 1)}</strong>
          </div>
          <div className={p2 > .5 ? "favored" : ""}>
            <span className="rank">{forecast.player2.rank ? `#${Math.round(forecast.player2.rank)}` : "NR"}</span>
            <h3>{forecast.player2.name}</h3><strong>{pct(p2, 1)}</strong>
          </div>
        </div>

        <ProbabilityCourt model={p1} market={market} />

        <div className="match-call">
          <div><span>Athena calls</span><strong>{forecast.winner.name}</strong></div>
          <div className={gap != null && Math.abs(gap) >= .05 ? "gap notable" : "gap"}>
            <span>vs market</span><strong>{gap == null ? "No pair" : `${gap > 0 ? "+" : ""}${Math.round(gap * 100)}¢`}</strong>
          </div>
        </div>

        <div className="quick-stats">
          <div><span>Likely sets</span><strong>{likelySet?.[0] ?? "—"}</strong><small>{pct(likelySet?.[1])}</small></div>
          <div><span>Total games</span><strong>{number(forecast.simulation.total_games.mean)}</strong><small>{forecast.simulation.total_games.p10}–{forecast.simulation.total_games.p90}</small></div>
          <div><span>Tiebreak</span><strong>{pct(forecast.simulation.p_tiebreak)}</strong><small>{number(forecast.simulation.expected_tiebreaks, 2)} expected</small></div>
          <div><span>Duration</span><strong>{Math.round(forecast.simulation.duration_minutes.p50)}m</strong><small>{Math.round(forecast.simulation.duration_minutes.p10)}–{Math.round(forecast.simulation.duration_minutes.p90)}m</small></div>
        </div>

        <div className="quality-row">
          {forecast.data_quality.map((flag) => <span key={flag}>{readableFlag(flag)}</span>)}
        </div>

        <button className="detail-toggle" type="button" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? "Close match book" : "Open match book"}<span aria-hidden="true">{open ? "−" : "+"}</span>
        </button>
      </div>

      {open && (
        <div className="match-detail">
          <section><p className="eyebrow">probability bench</p><Distribution rows={componentRows} />
            <p className="detail-note">Deployed: {forecast.probabilities.champion_component.replaceAll("_", " ")}</p></section>
          <section><p className="eyebrow">set score</p><Distribution rows={setRows} /></section>
          <section className="stat-section"><p className="eyebrow">player output</p><PlayerStatTable forecast={forecast} /></section>
          <section className="scorebook"><div><p className="eyebrow">exact scorebook</p>
            {forecast.simulation.top_exact_scores.slice(0, 5).map((row) => <p key={row.score}><strong>{row.score}</strong><span>{pct(row.probability, 1)}</span></p>)}</div>
            <div><p className="eyebrow">total games</p>{totalsKeys.map((key) => <p key={key}><strong>{key.replace("over_", "Over ")}</strong><span>{pct(forecast.simulation.total_games_probabilities[key])}</span></p>)}</div>
          </section>
          <p className="snapshot-note">{forecast.kalshi?.observed_at ? `Market captured ${new Date(forecast.kalshi.observed_at).toLocaleString()}. ` : "No matching Kalshi pair. "}Simulation: {forecast.simulation.n_sims.toLocaleString()} matches · data through {forecast.training_cutoff}.</p>
        </div>
      )}
    </article>
  );
}
