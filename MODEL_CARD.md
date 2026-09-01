# Model card: `uso-hybrid-v1`

## Intended use

Pre-match ATP/WTA US Open singles forecasts, detailed-stat exploration and a transparent
comparison with timestamped Kalshi prices. It is not designed for doubles, juniors, live
point-by-point trading, medical diagnosis, or automatic order execution.

## Components

- overall/surface Elo baseline;
- empirical-Bayes serve/return and serve-component rates with exponential recency decay;
- opponent adjustment on the log-odds scale;
- regularized logistic residual model and chronological calibration stack;
- point-by-point Monte Carlo under the actual Grand Slam scoring format.

## Known limitations

- Public box scores do not directly identify current injury severity, motivation or tactical
  changes. Retirements are excluded from training and not simulated.
- Point outcomes are treated as conditionally independent given server, aside from explicit
  score state. Momentum and pressure effects are therefore not fully modeled.
- Weather and roof fields are supported but require a current external feed or manual input.
- Low-ranked qualifiers can have sparse main-tour history; they receive strong population and
  ranking priors and a visible low-data warning.
- The detailed-stat layer inherits any bias in serve probability and rate estimates.
- Detailed-stat outputs are coherent simulation projections, but their MAE/interval coverage
  have not yet been validated against a separate 2026 point-stat test set. Treat those
  percentiles as model uncertainty, not proven empirical coverage.
- Data licenses may prohibit commercial use.

## Frozen walk-forward results

Expanding-window tests train only on prior seasons. Log loss and Brier are lower-is-better;
ECE is 10-bin expected calibration error. The comparison gate uses 10,000 paired bootstrap
resamples clustered by match date.

| Test slice | Matches | Stack log loss | Elo log loss | Stack Brier | ECE | 95% CI, stack − Elo | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 2024 ATP | 3,008 | 0.6036 | 0.6158 | 0.2093 | 0.0159 | [-0.0179, -0.0066] | pass |
| 2024 WTA | 2,580 | 0.6050 | 0.6193 | 0.2099 | 0.0124 | [-0.0210, -0.0077] | pass |
| 2025 ATP | 2,307 | 0.6132 | 0.6250 | 0.2131 | 0.0180 | [-0.0178, -0.0057] | pass |
| 2025 WTA | 2,486 | 0.6105 | 0.6201 | 0.2117 | 0.0186 | [-0.0167, -0.0029] | pass |
| 2026 ATP | 2,053 | 0.6099 | 0.6355 | 0.2115 | 0.0114 | [-0.0338, -0.0171] | pass |
| 2026 WTA | 1,894 | 0.6606 | 0.6410 | 0.2134 | 0.0370 | [-0.0209, 0.0714] | **fail** |

Because log loss was declared the primary probability metric, the deployed 2026 WTA forecast
falls back to Elo. Every such forecast is flagged
`2026_wta_stack_failed_gate_using_elo_fallback`; stack, point-model and ML probabilities remain
visible for audit. This freeze used matches completed through 30 August, before the modeled
US Open slate, and must not be revised using subsequent US Open outcomes.

## Promotion gate

No component is called champion until it improves held-out log loss with a positive practical
effect and a date-clustered 95% confidence interval excluding zero, without materially harming
calibration or detailed-stat coverage. Kalshi outperformance must be demonstrated on prices
snapshotted after model logging and before first ball.
