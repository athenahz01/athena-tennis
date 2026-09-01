"use client";

import { useEffect, useMemo, useState } from "react";

type AgentSide = {
  side: string;
  player: string;
  ticker?: string;
  fair_cents: number;
  bid_cents: number | null;
  ask_cents: number | null;
  spread_cents: number | null;
  edge_cents: number | null;
  contracts: number;
  eligible: boolean;
  reason: string;
};

type AgentMatch = {
  match_id: string;
  event_ticker?: string | null;
  tour: string;
  round: string;
  court?: string | null;
  player1: string;
  player2: string;
  state: string;
  reason: string;
  sides: AgentSide[];
  position?: {
    player: string;
    contracts: number;
    entry_cents: number;
    cost_cents: number;
  };
};

export type AgentState = {
  ts: string;
  mode: string;
  session: {
    buys: number;
    settles: number;
    fees_cents: number;
    realized_cents: number;
    open_cost_cents: number;
    bankroll_cents: number;
  };
  matches: AgentMatch[];
};

const money = (cents: number) =>
  `${cents < 0 ? "-" : ""}$${(Math.abs(cents) / 100).toFixed(2)}`;
const quote = (cents: number | null) => (cents == null ? "No quote" : `${cents.toFixed(1)}c`);

export default function AgentDesk({ fallback }: { fallback: AgentState }) {
  const [state, setState] = useState(fallback);
  const [live, setLive] = useState(false);

  useEffect(() => {
    let mounted = true;
    const pull = async () => {
      try {
        const response = await fetch("/api/agent-state", { cache: "no-store" });
        if (!response.ok) {
          if (mounted) setLive(false);
          return;
        }
        const next = (await response.json()) as AgentState;
        if (mounted) {
          setState(next);
          setLive(Date.now() - Date.parse(next.ts) < 45_000);
        }
      } catch {
        if (mounted) setLive(false);
      }
    };
    pull();
    const timer = window.setInterval(pull, 10_000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  const matches = useMemo(
    () => [...(state.matches || [])].sort((a, b) => {
      const order: Record<string, number> = { position: 0, entry: 1, watch: 2, market_missing: 3 };
      return (order[a.state] ?? 4) - (order[b.state] ?? 4);
    }),
    [state.matches],
  );
  const session = state.session;
  const open = matches.filter((match) => match.state === "position").length;

  return (
    <main id="top" className="agent-page">
      <section className="agent-hero">
        <div>
          <p className="eyebrow">paper trades only</p>
          <h1>The <span>agent</span></h1>
          <p>
            A paper trader watching Kalshi before matches start. It uses the frozen model
            probability, pays the displayed ask, includes taker fees, and caps each match
            at $10. It never places a real order.
          </p>
        </div>
        <div className={`agent-status ${live ? "is-live" : ""}`}>
          <i />
          <span>{live ? "Live runner" : "Last saved scan"}</span>
          <small>{new Date(state.ts).toLocaleString()}</small>
        </div>
      </section>

      <section className="agent-scoreboard" aria-label="Paper agent summary">
        <div><span>Paper P&amp;L</span><strong>{money(session.realized_cents)}</strong><small>settled</small></div>
        <div><span>Entries</span><strong>{session.buys}</strong><small>paper fills</small></div>
        <div><span>Open</span><strong>{open}</strong><small>positions</small></div>
        <div><span>Risk</span><strong>{money(session.open_cost_cents)}</strong><small>currently open</small></div>
        <div><span>Fees</span><strong>{money(session.fees_cents)}</strong><small>charged on entry</small></div>
      </section>

      <section className="agent-board">
        <div className="board-heading">
          <div><p className="eyebrow">current scan</p><h2>{matches.length} matches watched</h2></div>
          <p>{state.mode}</p>
        </div>
        <div className="agent-grid">
          {matches.map((match) => (
            <AgentMatchCard
              key={match.match_id}
              match={match}
              observing={state.mode === "paper observation"}
            />
          ))}
        </div>
      </section>

      <section className="agent-rules">
        <p className="eyebrow">entry rules</p>
        <h2>How entries work</h2>
        <p>
          ATP needs six cents of model edge after the estimated entry fee. WTA needs eight
          because it is running the Elo fallback. The agent refuses wide books, thin model
          histories, prices outside 10c to 90c, and a second position on the same match.
        </p>
      </section>
    </main>
  );
}

function AgentMatchCard({ match, observing }: { match: AgentMatch; observing: boolean }) {
  const badge = match.state === "market_missing"
    ? "no market"
    : observing && match.state === "entry"
      ? "signal"
      : match.state;
  return (
    <article className={`agent-card state-${match.state}`}>
      <header>
        <div><span className={`tour-pill ${match.tour.toLowerCase()}`}>{match.tour}</span><span>{match.round}</span></div>
        <b>{badge}</b>
      </header>
      <h3>{match.player1} <small>vs</small> {match.player2}</h3>
      <p className="agent-reason">{match.reason}</p>
      <div className="agent-sides">
        {match.sides.map((side) => (
          <div key={side.side} className={side.eligible ? "eligible" : ""}>
            <strong>{side.player}</strong>
            <span>fair {quote(side.fair_cents)}</span>
            <span>bid {quote(side.bid_cents)}</span>
            <span>ask {quote(side.ask_cents)}</span>
            <em>{side.edge_cents == null ? "No edge" : `${side.edge_cents > 0 ? "+" : ""}${side.edge_cents.toFixed(1)}c net`}</em>
          </div>
        ))}
      </div>
      {match.position && (
        <div className="position-strip">
          <span>{match.position.contracts} contracts on {match.position.player}</span>
          <strong>{quote(match.position.entry_cents)} entry</strong>
        </div>
      )}
    </article>
  );
}
