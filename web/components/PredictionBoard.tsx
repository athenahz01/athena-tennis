"use client";

import { useMemo, useState } from "react";
import MatchCard from "./MatchCard";
import type { Forecast } from "@/lib/types";

type Tour = "ALL" | "ATP" | "WTA";

export default function PredictionBoard({ forecasts }: { forecasts: Forecast[] }) {
  const [tour, setTour] = useState<Tour>("ALL");
  const [query, setQuery] = useState("");
  const [onlyEdges, setOnlyEdges] = useState(false);

  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return forecasts.filter((match) => {
      const tourMatch = tour === "ALL" || match.context.tour === tour;
      const nameMatch = !needle || `${match.player1.name} ${match.player2.name} ${match.context.court ?? ""}`
        .toLowerCase().includes(needle);
      const gap = match.kalshi?.model_minus_market_p1;
      const edgeMatch = !onlyEdges || (gap != null && Math.abs(gap) >= 0.05);
      return tourMatch && nameMatch && edgeMatch;
    });
  }, [forecasts, onlyEdges, query, tour]);

  return (
    <section id="matches" className="board-section">
      <div className="board-heading">
        <div>
          <p className="eyebrow">september 1 · opening round</p>
          <h2>Today&apos;s court</h2>
        </div>
        <p>{shown.length} of {forecasts.length} matches</p>
      </div>

      <div className="controls" aria-label="Match filters">
        <div className="tour-tabs" role="group" aria-label="Filter by tour">
          {(["ALL", "ATP", "WTA"] as Tour[]).map((value) => (
            <button key={value} type="button" aria-pressed={tour === value}
              className={tour === value ? "active" : ""} onClick={() => setTour(value)}>
              {value === "ALL" ? "All matches" : value}
            </button>
          ))}
        </div>
        <label className="search-field">
          <span className="sr-only">Search players or courts</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)}
            placeholder="Search player or court" />
        </label>
        <label className="edge-toggle">
          <input type="checkbox" checked={onlyEdges}
            onChange={(event) => setOnlyEdges(event.target.checked)} />
          <span>5¢+ gaps</span>
        </label>
      </div>

      {shown.length ? (
        <div className="match-grid">
          {shown.map((forecast, index) => (
            <MatchCard key={`${forecast.context.tour}-${forecast.player1.name}-${forecast.player2.name}`}
              forecast={forecast} index={index} />
          ))}
        </div>
      ) : (
        <div className="empty-state">
          <strong>No matches on this court.</strong>
          <span>Clear the search or turn off the market gap filter.</span>
        </div>
      )}
    </section>
  );
}
