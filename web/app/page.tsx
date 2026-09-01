import PredictionBoard from "@/components/PredictionBoard";
import slate from "@/public/data/slate.json";
import type { Forecast } from "@/lib/types";

function TennisCourt() {
  return (
    <div className="hero-court" aria-hidden="true">
      <span className="court-net" />
      <span className="court-service court-service-left" />
      <span className="court-service court-service-right" />
      <span className="court-center" />
      <span className="hero-ball" />
      <span className="hero-shadow" />
    </div>
  );
}

export default function Home() {
  const forecasts = slate as unknown as Forecast[];
  const first = forecasts[0];
  const marketCount = forecasts.filter((item) => item.kalshi?.player1).length;
  const simulations = forecasts.reduce((sum, item) => sum + item.simulation.n_sims, 0);

  return (
    <main id="top">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">model vs market · every point simulated</p>
          <h1>
            Every Match.
            <br />
            <span>Before First Ball.</span>
          </h1>
          <p className="hero-lede">
            Winner probabilities and the shape of the match—sets, games, tiebreaks,
            serve stats, and time on court—for every 2026 US Open singles matchup.
          </p>
          <a className="primary-button" href="#matches">Read today&apos;s court</a>
        </div>
        <TennisCourt />
      </section>

      <section className="scoreboard" aria-label="Slate summary">
        <div><span>Draw</span><strong>{forecasts.length}</strong><small>matches</small></div>
        <div><span>Tours</span><strong>ATP · WTA</strong><small>singles</small></div>
        <div><span>Simulated</span><strong>{simulations.toLocaleString()}</strong><small>matches</small></div>
        <div><span>Market</span><strong>{marketCount}/{forecasts.length}</strong><small>Kalshi pairs</small></div>
        <div><span>Data through</span><strong>{first?.training_cutoff ?? "—"}</strong><small>{first?.model_version}</small></div>
      </section>

      <PredictionBoard forecasts={forecasts} />

      <section className="method" id="method">
        <p className="eyebrow">how to read the board</p>
        <div className="method-grid">
          <h2>The court is the probability.</h2>
          <div>
            <p>
              The chartreuse ball is Athena&apos;s win probability. The white marker is the
              de-vigged Kalshi midpoint captured after the model forecast. Distance between
              them is disagreement—not an automatic bet.
            </p>
            <p>
              ATP uses the calibrated stack. The 2026 WTA challenger failed its frozen
              log-loss promotion gate, so WTA safely falls back to Elo and says so on every card.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
