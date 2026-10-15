# BTC/ETH Portfolio Risk and Rebalancing

**Question:** do rebalancing and volatility-based weighting reduce the drawdowns of a BTC/ETH portfolio compared with buy-and-hold, and what do they give up in return and turnover?

**Short answer:** rebalancing between BTC and ETH barely changes drawdowns, because the two assets crash together. Weighting by volatility mostly means holding more BTC. Only moving part of the portfolio into cash (volatility targeting) cut drawdowns materially, and that result depended on how the parameters were chosen.

The full report is [`reports/report.html`](reports/report.html), a single self-contained file. Tables are saved as CSV in [`reports/tables/`](reports/tables/).

## Results (out-of-sample, Jan 2022 → Oct 2026, after 25 bps costs)

| Rule (parameters picked on 2016–2021 data) | CAGR | Max drawdown | Turnover / yr |
|---|---:|---:|---:|
| Buy & hold 50/50 | 4.1% | −68.9% | 0.00 |
| BTC only | 12.6% | −67.0% | 0.00 |
| Calendar rebalance 50/50, quarterly | 3.7% | −68.5% | 0.31 |
| Threshold rebalance 50/50, ±20% band | 3.4% | −68.9% | 0.08 |
| Inverse volatility, 60-day, monthly | 3.5% | −68.2% | 0.75 |
| Minimum variance, 60-day, monthly | 8.6% | −67.1% | 1.99 |
| Volatility target 40% (cash allowed), drawdown-picked | 9.2% | −54.5% | 1.12 |
| Volatility target 80% (cash allowed), Calmar-picked | 2.1% | −69.9% | 0.55 |

- **Correlation is the binding constraint.** The 90-day BTC–ETH correlation had a median of 0.87 in the test period. On days BTC fell 5% or more, ETH also fell 98% of the time, and fell further on average.
- **Minimum variance held about 89% BTC.** It beat buy-and-hold because BTC beat ETH in this period, not because it diversified better.
- **The cash rule's drawdown cut is robust, but its return gain is not.** In 500 block-bootstrap paths, its drawdown was shallower than buy-and-hold's in 97% of paths, but its CAGR was higher in only 48%. Over a walk-forward test from 2019 it fell 54% against 77%, while returning 34% a year against 47%.
- **Parameter tuning did not carry over to the test period.** For volatility targeting, the train-vs-test rank correlation of the Calmar ratio was −0.87.
- **Costs are second-order.** Turnover was low, so at 25 bps the 50/50 rules lost at most 0.19% a year to costs. They trailed buy-and-hold even when trading was free.

## How it works

```
data/raw/*.csv          Coinbase daily USD candles (pinned to END_DATE in config.py)
data/reference/*.csv    Bitstamp candles, used only to cross-check prices
sql/schema.sql          raw price table
sql/features.sql        gap-filled calendar, returns, rolling 30/60/90/180-day vol and
                        correlation (window aggregates), drawdowns
sql/queries/            yearly summary, co-movement by market move, source cross-check
src/portfolio_risk/
  fetch.py              downloads prices
  db.py                 loads CSVs into SQLite and runs the SQL
  strategies.py         allocation rules and the parameter grids
  backtest.py           daily engine: drift, calendar/band triggers, lag, costs
  metrics.py            CAGR, vol, Sharpe, Sortino, max drawdown, Calmar, CVaR, turnover
  evaluate.py           train selection, test, cost/lag sensitivity, walk-forward, bootstrap
  report.py             figures, CSV tables, HTML report
tests/                  engine and data checks (look-ahead, costs, SQL vs pandas)
```

**Design choices**

- **Chronological split.** 180-day warm-up, train from 2016-11-15 to 2021-12-31, test from 2022-01-01 to 2026-10-14. Parameters are chosen on train only, using two criteria reported side by side: best Calmar ratio, and shallowest drawdown.
- **No look-ahead.** Signals at close *t* use data up to *t*, and trades fill at close *t+1*. A test changes future returns and checks that no earlier result moves; the same test catches a rule that deliberately peeks ahead.
- **Costs.** Each trade costs bps × Σ|Δw| across BTC and ETH, tested at 0, 10, 25, 50 and 100 bps, along with each rule's break-even cost.
- **Robustness.** A stationary block bootstrap of the test period (500 paths, 30-day mean blocks, with every signal rebuilt on each path), an annual walk-forward re-selection from 2019, an execution-lag check, and train-vs-test rank stability of the parameters.

## Reproduce

Requires Python 3.9+.

```bash
make setup   # virtualenv + pinned dependencies
make all     # build SQLite db, run tests, regenerate reports/
```

`make fetch` re-downloads prices. Move `END_DATE` in `src/portfolio_risk/config.py` first. The narrative in the findings section is written for the pinned data window; every number in it is filled in from the results.

*Historical research on two assets over about ten years, not investment advice.*
