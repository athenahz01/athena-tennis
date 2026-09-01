# Research synthesis: predicting 2026 US Open singles

## Bottom line

The best technical architecture is a calibrated ensemble around a tennis-specific scoring
model, not a generic deep network. Surface-aware Elo is hard to beat on limited tabular data.
Opponent-adjusted serve and return estimates add interpretable, match-specific information.
A low-capacity supervised model can learn residual effects, while a point simulator is the
right way to turn two serve probabilities into winner, set, total-game and player-stat
distributions.

The market should be evaluated as a separate forecast and, only after enough live evidence,
as a possible ensemble member. Published work repeatedly finds betting prices exceptionally
strong; an impressive favorite-pick accuracy can still lose money.

## Evidence that shaped the implementation

- Bunker et al., *A comparative evaluation of Elo ratings- and machine learning-based
  methods for tennis match result prediction* (2023), used expanding chronological years,
  not shuffled folds, and directly compared weighted Elo with ML.
  DOI: https://doi.org/10.1177/17543371231212235
- Kovalchik, *Searching for the GOAT of tennis win prediction* (2016), compared rating and
  point-based approaches and supports strong Elo/serve-return baselines.
  https://doi.org/10.1515/jqas-2015-0059
- Knottenbelt et al., *A common-opponent stochastic model* (2012), showed why serve and
  return must be opponent-adjusted before a hierarchical scoring model is applied.
  https://doi.org/10.1016/j.camwa.2012.03.005
- Gollub, *Forecasting serve performance in professional tennis matches* (2021), addresses
  small samples and schedule strength with shrinkage and supplies reproducible code.
  https://doi.org/10.3233/JSA-200345
- Wang and Drekic, *Boosting markovian tennis prediction* (2026), finds that point-model
  ensembles close part of the gap to regression and that blindly restricting data to one
  surface can sacrifice useful sample size.
  https://doi.org/10.1177/22150218251412670
- Easton and Uylangco (2010) found very high agreement between point models and in-play
  markets, evidence that the market is a formidable benchmark.
  https://doi.org/10.1016/j.ijforecast.2009.10.004
- Wilkens (2021) reports bookmaker variables among the most important features across
  several ML families. That is useful for a market-aware product, but using the same market
  as both feature and benchmark obscures independent model value.
  https://doi.org/10.3233/JSA-200463

## Data hierarchy

1. Stable player IDs, official result, surface, round, best-of format and date.
2. Point denominators: service points, first serves in, first/second-serve points won,
   aces, double faults, service games and break points.
3. Overall and surface Elo, both time-aware and updated only after a match.
4. Recent form and workload: rest days, matches/minutes in 7/14 days and prior-match length.
5. Biographical context: age, height, handedness and home-country indicator.
6. Current context: court, roof state, temperature/humidity/wind, scheduled local time,
   injury/retirement news and tournament-round workload.
7. Market: Kalshi bid/ask, depth and timestamp, stored after the model forecast.

Jeff Sackmann's ATP/WTA files provide the canonical historical schema and integer match-stat
denominators. The README warns of some missing statistics and applies a CC BY-NC-SA license.
The current local downloader uses TennisMyLife's compatible public files because they include
2026 ATP/WTA results. Data must be snapshotted because current-season files change.

## Factors worth using now

- Hard-court and overall strength together. Pure hard-court filtering is too noisy for
  low-sample players; partial pooling is safer.
- Serve and return separately. Hold percentage alone hides the denominator and opponent.
- Best-of-five format. Small point-skill edges compound more strongly across five sets.
- Rank as a low-data prior, not the central signal.
- Rest and accumulated minutes. Fatigue evidence is real but heterogeneous, so its effect
  belongs in a regularized challenger rather than a hand-written large penalty.
- Age and recent inactivity as mild priors for degradation and injury uncertainty.
- Roof and weather as revision features. Heat affects match characteristics and recovery;
  Arthur Ashe and Louis Armstrong have roofs, so an outdoor forecast cannot be applied after
  a roof change without revision.

## Folk/practitioner claims: useful hypotheses, not facts

Reddit, betting communities and social media repeatedly mention head-to-head dominance,
"momentum," travel, crowd/home advantage, lefty matchups, night-session conditions, court
speed, long prior rounds, visible taping or medical timeouts, and a favorite's tendency to
start slowly. The disciplined treatment is:

- H2H is shrunk heavily; a few meetings are not a stable matchup law.
- Recent form is included, but no ad-hoc hot-hand multiplier is applied.
- Travel is more useful between tournaments than within a two-week Slam; within the US Open,
  accumulated minutes and rest are the cleaner measures.
- Injuries and tactical/visual observations enter as explicit, timestamped overrides with a
  separate model revision, never as silent manual edits.
- Social-media sentiment is not a core feature. It is noisy, manipulable, and often already
  embedded in price. It can be tested later as a market-movement feature.
- High win rate is not profitability. A recent practitioner report described 76% picks but
  negative ROI; price, calibration and execution matter. Another long-running practitioner
  comparison found that entering earlier and shopping prices materially changed returns on
  the same selections. These are self-reported community results, not peer-reviewed evidence,
  but they motivate logging the quote timestamp, spread and closing-line value rather than
  celebrating pick accuracy.

Community examples:

- Full-book 2026 model report (76.2% hit rate, -3.7% ROI):
  https://www.reddit.com/r/algobetting/comments/1uvrmkj/graded_every_pick_my_tennis_model_made_for_2/
- ATP bettor's early-price versus closing-price comparison:
  https://www.reddit.com/r/sportsbook/comments/l2dujt/analysis_of_my_tennis_results_over_the_last_two/
- Spreadsheet discussion that independently surfaces second serve, opponent return,
  surface and indoor/outdoor context:
  https://www.reddit.com/r/sportsbook/comments/ip11qo/tennis_model_excel/

## 2026 US Open specifics

The official ATP schedule lists the event in New York from 30 August through 13 September
2026. The USTA event is played on hard courts. Men play best-of-five and women best-of-three;
the final set uses a first-to-10, win-by-two tiebreak at 6-6. Kalshi's public API exposes
ATP and WTA match-winner series (`KXATPMATCH`, `KXWTAMATCH`) and related totals/props. On
31 August 2026, Kalshi and the USTA announced a US Open partnership; the prices are therefore
especially relevant as a live comparison, not evidence of independent model edge.

Sources:

- Official tournament dates: https://www.usopen.org/en_US/news/articles/2025-12-11/2026_us_open_tournament_dates.html
- ATP daily schedule: https://www.atptour.com/en/scores/current/us-open/560/daily-schedule
- Kalshi API docs: https://docs.kalshi.com/api-reference/market/get-markets
- Kalshi/USTA announcement: https://news.kalshi.com/p/kalshi-us-open-official-prediction-market-partner
- TennisMyLife data catalog: https://stats.tennismylife.org/tennis-match-database
- Jeff Sackmann schema/license mirror: https://github.com/rokoo/tennis_atp
- Heat and productivity: https://doi.org/10.3386/w31650
- Fatigue review: https://doi.org/10.1136/bjsports-2013-093196

## Evaluation protocol

- Build every row immediately before updating either player's state.
- Use expanding walk-forward seasons and an untouched final 2026 test slice.
- Primary winner metric: log loss. Secondary: Brier, calibration error and accuracy.
- Detailed counts: distributional CRPS where available, plus MAE and coverage.
- Compare paired per-match losses; resample by match date, 10,000 times.
- Refit calibration on a later chronological window than the base learner.
- Report ATP and WTA separately and by favorite band, round and player-data coverage.
- Never optimize against the current US Open outcomes and call the same matches a test.
