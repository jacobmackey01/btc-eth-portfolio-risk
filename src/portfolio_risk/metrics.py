"""Risk, return and trading metrics for a backtest result."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Result
from .config import DAYS_PER_YEAR


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def longest_underwater_days(equity: pd.Series) -> int:
    """Longest stretch (days) spent below a previous peak."""
    under = (drawdown(equity) < 0).to_numpy()
    best = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def cagr(equity: pd.Series, start_value: float = 1.0) -> float:
    years = len(equity) / DAYS_PER_YEAR
    return float((equity.iloc[-1] / start_value) ** (1.0 / years) - 1.0)


def summarize(res: Result) -> dict:
    r = res.returns
    eq = res.equity
    years = len(r) / DAYS_PER_YEAR
    vol = r.std() * np.sqrt(DAYS_PER_YEAR)
    downside = r.clip(upper=0).pow(2).mean() ** 0.5 * np.sqrt(DAYS_PER_YEAR)
    mdd = max_drawdown(eq)
    growth = cagr(eq)
    tail = r.quantile(0.05)
    return {
        "strategy": res.spec.label(),
        "family": res.spec.family,
        "cagr": growth,
        "ann_vol": vol,
        "sharpe": r.mean() / r.std() * np.sqrt(DAYS_PER_YEAR),   # rf = 0
        "sortino": r.mean() * DAYS_PER_YEAR / downside if downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": growth / abs(mdd) if mdd < 0 else np.nan,
        "cvar_95": float(r[r <= tail].mean()),                     # avg of worst 5% days
        "worst_month": float((1 + r).resample("MS").prod().sub(1).min()),
        "longest_underwater_days": longest_underwater_days(eq),
        "turnover_per_year": res.turnover.sum() / years,
        "trades_per_year": (res.turnover > 0).sum() / years,
        "cost_drag_per_year": res.costs.sum() / years,
        "avg_btc_weight": res.weights["BTC"].mean(),
        "avg_cash_weight": res.weights["CASH"].mean(),
        "final_value": float(eq.iloc[-1]),
    }


def table(results: list[Result]) -> pd.DataFrame:
    return pd.DataFrame([summarize(r) for r in results]).set_index("strategy")
