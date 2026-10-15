"""The study: in-sample selection, out-of-sample test, cost, lag and
walk-forward checks, and a block bootstrap of the out-of-sample period.

Order of operations matters and is the point of the design:
  1. every parameter choice is made on TRAIN (2016-11 → 2021-12),
  2. the chosen rules are run once on TEST (2022-01 → end),
  3. the remaining sections stress the TEST result; none of them feed back
     into the choice of parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest as bt
from . import config
from . import metrics
from . import strategies as st

# Two ways to pick each family's parameters on TRAIN. Calmar (CAGR / |max
# drawdown|) is the headline; "max_drawdown" picks the shallowest training
# drawdown, i.e. what an investor who only cares about drawdowns would choose.
SELECTION_METRIC = "calmar"
SELECTION_METRICS = {"calmar": "Best training Calmar", "max_drawdown": "Shallowest training drawdown"}
WALK_FORWARD_FIRST_YEAR = 2019
BOOTSTRAP_RUNS = 500
BOOTSTRAP_BLOCK_DAYS = 30


@dataclass
class Study:
    panel: pd.DataFrame
    returns: pd.DataFrame
    train_start: str
    end: str

    @classmethod
    def load(cls) -> "Study":
        from . import db
        panel = db.load_panel()
        train_start = panel.index[0] + pd.Timedelta(days=config.WARMUP_DAYS)
        return cls(panel, bt.asset_returns(panel), str(train_start.date()), config.END_DATE)

    def run(self, spec, start, end, cost_bps=config.COST_BPS, lag=config.EXECUTION_LAG):
        return bt.run(spec, self.returns, start, end, cost_bps, lag)

    def train(self, spec, **kw):
        return self.run(spec, self.train_start, config.TRAIN_END, **kw)

    def test(self, spec, **kw):
        return self.run(spec, config.TEST_START, self.end, **kw)


# ── 1–2. selection on train, evaluation on test ─────────────────────────────

def grid_tables(study: Study) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every candidate on both periods. Test numbers are diagnostic only."""
    specs = st.benchmarks(study.panel) + [s for g in st.grids(study.panel).values() for s in g]
    train = metrics.table([study.train(s) for s in specs])
    test = metrics.table([study.test(s) for s in specs])
    return train, test


def select(study: Study, train_table: pd.DataFrame, metric: str = SELECTION_METRIC) -> list[bt.Spec]:
    """Highest training score per family (max_drawdown is negative, so the
    highest value is the shallowest drawdown)."""
    chosen = []
    for family, specs in st.grids(study.panel).items():
        scores = train_table.loc[[s.label() for s in specs], metric]
        chosen.append(next(s for s in specs if s.label() == scores.idxmax()))
    return chosen


def headline(study: Study, chosen: list[bt.Spec]) -> tuple[pd.DataFrame, list[bt.Result]]:
    specs = st.benchmarks(study.panel) + chosen
    results = [study.test(s) for s in specs]
    return metrics.table(results), results


def rank_stability(train: pd.DataFrame, test: pd.DataFrame, metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Within each family, does the train ranking of parameters survive to test?"""
    rows = []
    for family, g in train.groupby("family"):
        if len(g) < 3:
            continue
        a = g[metric]
        b = test.loc[g.index, metric]
        best = a.idxmax()
        rows.append({
            "family": family,
            "candidates": len(g),
            "spearman_train_vs_test": a.rank().corr(b.rank()),
            "chosen": best,
            "chosen_test_rank": int(b.rank(ascending=False)[best]),
            "best_on_test": b.idxmax(),
        })
    return pd.DataFrame(rows).set_index("family")


# ── 3. stress tests on the out-of-sample result ──────────────────────────────

def cost_sensitivity(study: Study, chosen: list[bt.Spec]) -> pd.DataFrame:
    rows = []
    hold = st.buy_and_hold(study.panel)
    for bps in config.COST_GRID_BPS:
        base = metrics.summarize(study.test(hold, cost_bps=bps))
        for spec in chosen:
            s = metrics.summarize(study.test(spec, cost_bps=bps))
            rows.append({
                "family": spec.family, "strategy": spec.label(), "cost_bps": bps,
                "cagr": s["cagr"], "max_drawdown": s["max_drawdown"], "calmar": s["calmar"],
                "cost_drag_per_year": s["cost_drag_per_year"],
                "cagr_vs_hold": s["cagr"] - base["cagr"],
                "drawdown_vs_hold": s["max_drawdown"] - base["max_drawdown"],
            })
    return pd.DataFrame(rows)


def breakeven_cost(study: Study, spec: bt.Spec, hi: float = 2_000.0) -> float:
    """Cost (bps) at which the rule's test CAGR falls to buy-and-hold's.
    NaN if it trails buy-and-hold even when trading is free."""
    hold = metrics.summarize(study.test(st.buy_and_hold(study.panel), cost_bps=0))["cagr"]
    gap = lambda bps: metrics.summarize(study.test(spec, cost_bps=bps))["cagr"] - hold
    if gap(0) <= 0:
        return np.nan
    if gap(hi) > 0:
        return np.inf
    lo = 0.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def lag_sensitivity(study: Study, chosen: list[bt.Spec]) -> pd.DataFrame:
    rows = []
    for spec in chosen:
        for lag in (0, 1):
            s = metrics.summarize(study.test(spec, lag=lag))
            rows.append({"strategy": spec.label(), "lag_days": lag, "cagr": s["cagr"],
                         "max_drawdown": s["max_drawdown"], "calmar": s["calmar"]})
    return pd.DataFrame(rows)


def walk_forward(study: Study, metric: str = SELECTION_METRIC) -> tuple[pd.DataFrame, pd.DataFrame, list[bt.Result]]:
    """Re-select each family's parameters every January on all data before
    that year (expanding window), then trade the year out of sample."""
    idx = study.panel.index
    years = range(WALK_FORWARD_FIRST_YEAR, pd.Timestamp(study.end).year + 1)
    start = f"{WALK_FORWARD_FIRST_YEAR}-01-01"
    picks, results = [], []
    for family, specs in st.grids(study.panel).items():
        stitched = None
        for year in years:
            fit_end = f"{year - 1}-12-31"
            scores = {s.label(): metrics.summarize(study.run(s, study.train_start, fit_end))[metric]
                      for s in specs}
            best = next(s for s in specs if s.label() == max(scores, key=scores.get))
            picks.append({"family": family, "year": year, "chosen": best.label()})
            if stitched is None:
                stitched = bt.Spec(f"wf_{best.name}", family, {},
                                   best.target.copy(), best.rebalance.copy(), best.band.copy())
            else:
                mask = idx.year == year
                stitched.target.loc[mask] = best.target.loc[mask]
                stitched.rebalance.loc[mask] = best.rebalance.loc[mask]
                stitched.band.loc[mask] = best.band.loc[mask]
        results.append(study.run(stitched, start, study.end))
    hold = study.run(st.buy_and_hold(study.panel), start, study.end)
    table = metrics.table([hold] + results)
    return table, pd.DataFrame(picks), [hold] + results


def _stationary_bootstrap_index(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano stationary bootstrap: geometric block lengths, wraps."""
    out = np.empty(n, int)
    out[0] = rng.integers(n)
    restart = rng.random(n) < 1.0 / block
    fresh = rng.integers(n, size=n)
    for t in range(1, n):
        out[t] = fresh[t] if restart[t] else (out[t - 1] + 1) % n
    return out


def bootstrap(study: Study, chosen: list[bt.Spec], runs: int = BOOTSTRAP_RUNS,
              seed: int = 20261015) -> pd.DataFrame:
    """Resample the TEST period's joint BTC/ETH daily returns in blocks (so
    crashes and their co-movement stay intact), rebuild every signal from the
    resampled path, and rerun each chosen rule against buy-and-hold."""
    rng = np.random.default_rng(seed)
    p = study.panel
    test_mask = p.index >= pd.Timestamp(config.TEST_START)
    history = p.loc[~test_mask].iloc[-config.WARMUP_DAYS:]       # real warm-up
    test = p.loc[test_mask]
    specs = [st.buy_and_hold(p)] + chosen
    rows = []
    for k in range(runs):
        pick = _stationary_bootstrap_index(len(test), BOOTSTRAP_BLOCK_DAYS, rng)
        sim = pd.concat([
            history[["btc_ret", "eth_ret", "btc_log", "eth_log"]],
            test[["btc_ret", "eth_ret", "btc_log", "eth_log"]].iloc[pick].set_axis(test.index),
        ])
        sim = sim.join(st.rolling_features(sim["btc_log"], sim["eth_log"]))
        rets = bt.asset_returns(sim)
        base = None
        for spec in specs:
            res = bt.run(st.rebuild(spec, sim), rets, config.TEST_START, study.end,
                         config.COST_BPS, config.EXECUTION_LAG)
            s = metrics.summarize(res)
            row = {"run": k, "strategy": spec.label(), "family": spec.family,
                   "cagr": s["cagr"], "max_drawdown": s["max_drawdown"], "sharpe": s["sharpe"]}
            if base is None:
                base = row
            row["dd_improvement"] = row["max_drawdown"] - base["max_drawdown"]
            row["cagr_diff"] = row["cagr"] - base["cagr"]
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_summary(boot: pd.DataFrame) -> pd.DataFrame:
    g = boot[boot.family != "Buy & hold 50/50"].groupby("strategy")
    return pd.DataFrame({
        "p_shallower_drawdown": g["dd_improvement"].apply(lambda x: (x > 0).mean()),
        "dd_improvement_median": g["dd_improvement"].median(),
        "dd_improvement_p05": g["dd_improvement"].quantile(0.05),
        "dd_improvement_p95": g["dd_improvement"].quantile(0.95),
        "p_higher_cagr": g["cagr_diff"].apply(lambda x: (x > 0).mean()),
        "cagr_diff_median": g["cagr_diff"].median(),
        "cagr_diff_p05": g["cagr_diff"].quantile(0.05),
        "cagr_diff_p95": g["cagr_diff"].quantile(0.95),
    })
