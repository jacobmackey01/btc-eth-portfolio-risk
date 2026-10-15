"""Daily long-only backtest engine for BTC, ETH and (optionally) cash.

Conventions
-----------
* `returns[i]` is the simple return from close i-1 to close i.
* A strategy supplies, for every day, a target weight vector computed from
  data up to close i, a calendar rebalance flag, and a drift band.
* At close i the engine decides whether to trade: yes if the calendar flag is
  set or any weight has drifted more than `band` from target. With
  `lag = 1` the order fills at close i+1 at the weights decided at close i,
  so no decision ever uses the price it trades at.
* Costs: `cost_bps` per unit of notional traded in BTC or ETH, so a full
  switch BTC→ETH (|Δw| summed = 2) costs 2 × cost_bps. Cash legs are free.
* The opening allocation is assumed already in place (no cost), so results
  measure the cost of *maintaining* an allocation, not building one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COLUMNS = ("BTC", "ETH", "CASH")


@dataclass
class Spec:
    """A compiled strategy: what to hold and when to trade."""

    name: str
    family: str
    params: dict
    target: pd.DataFrame        # columns BTC, ETH, CASH; NaN = no view yet
    rebalance: pd.Series        # bool, calendar trade dates
    band: pd.Series             # float, max drift before a forced trade

    def label(self) -> str:
        if not self.params:
            return self.family
        inner = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.family} ({inner})"


@dataclass
class Result:
    spec: Spec
    equity: pd.Series           # portfolio value, starts at 1.0
    returns: pd.Series          # daily portfolio return, net of costs
    weights: pd.DataFrame       # end-of-day weights after any trade
    turnover: pd.Series         # sum |Δw| over BTC and ETH on trade days
    costs: pd.Series            # cost paid, as a fraction of portfolio value
    cost_bps: float
    meta: dict = field(default_factory=dict)


def asset_returns(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    out["BTC"] = panel["btc_ret"]
    out["ETH"] = panel["eth_ret"]
    out["CASH"] = 0.0  # cash earns nothing: a conservative, rate-free baseline
    return out


def run(
    spec: Spec,
    returns: pd.DataFrame,
    start: str,
    end: str,
    cost_bps: float = 25.0,
    lag: int = 1,
) -> Result:
    if lag not in (0, 1):
        raise ValueError("lag must be 0 or 1")

    idx = returns.index
    i0 = idx.searchsorted(pd.Timestamp(start))
    i1 = idx.searchsorted(pd.Timestamp(end), side="right")
    if i0 == 0:
        raise ValueError("start must leave at least one day of history")

    R = returns.loc[:, COLUMNS].to_numpy(float)
    T = spec.target.reindex(idx).loc[:, COLUMNS].to_numpy(float)
    F = spec.rebalance.reindex(idx).fillna(False).to_numpy(bool)
    B = spec.band.reindex(idx).fillna(np.inf).to_numpy(float)
    c = cost_bps / 10_000.0

    if np.isnan(T[i0 - 1]).any():
        raise ValueError(f"{spec.label()}: no target available at {idx[i0 - 1].date()}")

    n = i1 - i0
    values = np.empty(n)
    weights = np.empty((n, 3))
    turnover = np.zeros(n)
    costs = np.zeros(n)

    # Hold the target known at the close before the first evaluated day.
    w = T[i0 - 1].copy()
    v = 1.0
    pending = None  # (fill_index, target_weights)

    def trade(new_w: np.ndarray, k: int) -> None:
        nonlocal w, v
        traded = np.abs(new_w[:2] - w[:2]).sum()
        turnover[k] += traded
        costs[k] += c * traded
        v *= 1.0 - c * traded
        w = new_w.copy()

    for k, i in enumerate(range(i0, i1)):
        growth = w * (1.0 + R[i])
        gross = growth.sum()
        v *= gross
        w = growth / gross

        if pending is not None and pending[0] == i:
            trade(pending[1], k)
            pending = None

        tgt = T[i]
        if not np.isnan(tgt).any() and pending is None:
            drift = np.abs(w - tgt).max()
            if F[i] or drift > B[i]:
                if drift > 1e-12:
                    if lag == 0:
                        trade(tgt, k)
                    elif i + 1 < i1:
                        pending = (i + 1, tgt.copy())

        values[k] = v
        weights[k] = w

    dates = idx[i0:i1]
    equity = pd.Series(values, index=dates, name=spec.name)
    prev = np.concatenate([[1.0], values[:-1]])
    return Result(
        spec=spec,
        equity=equity,
        returns=pd.Series(values / prev - 1.0, index=dates),
        weights=pd.DataFrame(weights, index=dates, columns=COLUMNS),
        turnover=pd.Series(turnover, index=dates),
        costs=pd.Series(costs, index=dates),
        cost_bps=cost_bps,
    )
