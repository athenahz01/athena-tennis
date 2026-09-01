# 2026 US Open predictor

This repository is a full local ATP/WTA prediction pipeline for the 2026 US Open.
It predicts the match winner and a coherent distribution of set, game, tiebreak,
serve, break-point and duration statistics. It can snapshot live Kalshi prices and
compare them with model probabilities without using those prices as training labels.

The design carries forward the best practices from the existing World Cup, club-soccer
and MLB projects in this workspace:

- chronological, point-in-time features only;
- proper scoring rules and calibration before accuracy;
- a simple Elo baseline that every challenger must beat;
- explicit joint simulation for correlated detailed outputs;
- immutable data and market snapshots;
- model-first logging, followed by market comparison;
- no claim of market outperformance without a held-out, paired test.

## Model

The production candidates combine three independent views:

1. overall plus hard-court Elo;
2. opponent-adjusted, shrinkage-stabilized serve/return probabilities mapped through
   tennis scoring rules;
3. a regularized logistic model using only pre-match differences: the two ratings,
   serve/return quality, ranking, recent results, workload/rest, age, height,
   handedness, head-to-head, round and Grand Slam context.

The detailed projection engine then simulates every point. Men's US Open singles use
best-of-five; women's singles use best-of-three. A 10-point, win-by-two tiebreak is used
at 6-6 in the final set and a 7-point tiebreak in other sets.

The pre-tournament deployment gate selected the stack for ATP. The 2026 WTA stack failed
the primary log-loss gate on matches through August 30, so WTA forecasts use Elo as a
safety fallback while emitting all component probabilities and a visible warning.

## Quick start

```powershell
cd C:\AA_Projects\us-open-predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]

# Download frozen 2012-2026 ATP/WTA source files.
us-open fetch --start-year 2012 --end-year 2026

# Train both tours using only matches completed by the cutoff.
us-open train --cutoff 2026-08-30

# Predict one match and attach the current public Kalshi quote.
us-open predict --tour ATP --player1 "Taylor Fritz" --player2 "Darwin Blanch" \
  --best-of 5 --round R128 --date 2026-09-01 --kalshi --sims 20000

# Predict every row in a slate CSV.
us-open slate --fixtures data/fixtures/us_open_2026_2026-09-01.csv --kalshi

# One-command full slate + Kalshi + website data refresh (20,000 simulations/game).
.\scripts\predict_all.ps1

# Honest chronological judge (trains before each test season).
us-open backtest --years 2024 2025 2026
```

If `python` is not on PATH in this workspace, the existing sports-model runtime can run it:

```powershell
& C:\AA_Projects\wc-predictor\.venv\Scripts\python.exe -m us_open_model.cli --help
```

## Fixture contract

CSV columns: `tour,player1,player2,best_of,round,date,court`. Only the first five are
required. Player names are normalized for accents, punctuation and spacing and resolved
against the latest pre-cutoff player state. Unknown players still receive a rank-informed,
tour-level prior and are explicitly flagged as low-data.

## Outputs

Each JSON/CSV result includes:

- winner probabilities from Elo, point model, ML layer and final stack;
- uncertainty and data-quality flags;
- likely match and set score;
- expected total games and game-margin distribution;
- total-games over probabilities on common half-game lines;
- straight-sets, deciding-set and tiebreak probabilities;
- expected aces, double faults, first serves, service points won, breaks and break points;
- expected duration and percentile interval;
- optional Kalshi bid/ask, de-vigged fair probability and model-market gap.

These are forecasts, not guarantees or financial advice. Retirements and late injuries are
not inferable from historical box scores; refresh availability and market data immediately
before first ball.

## Data and licensing

The default downloader uses TennisMyLife's public ATP/WTA CSV mirror, which is based partly
on Jeff Sackmann/Tennis Abstract data. Those sources restrict commercial use and require
attribution. Review the provider's current terms before publishing or monetizing the model.
Kalshi market reads use public, unauthenticated endpoints; trading is intentionally absent.

See [RESEARCH.md](RESEARCH.md) for the evidence review and [MODEL_CARD.md](MODEL_CARD.md)
for evaluation, leakage, uncertainty and deployment requirements.

## Prediction website

The Next.js website reads the same full JSON emitted by the prediction command. After
running `scripts\predict_all.ps1`, start it locally with:

```powershell
cd C:\AA_Projects\us-open-predictor\web
npm.cmd run dev
```

Then open `http://localhost:3000`. The site includes ATP/WTA filters, player search,
model-versus-Kalshi probability courts, score and totals distributions, detailed player
statistics, and model-quality warnings.

### Vercel

The repository is an npm workspace, so Vercel detects Next.js and builds the nested `web`
application when the project points at the repository root. Alternatively, set **Settings
→ Build and Deployment → Root Directory** to `web`; Vercel will then use the web package
directly.
