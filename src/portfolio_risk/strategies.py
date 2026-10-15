"""Allocation rules compared in the study, and the parameter grids searched.

Every rule is long-only and fully invested in BTC+ETH except volatility
targeting, which may hold cash. Signals on date t use only rolling statistics
up to close t (see sql/features.sql); the engine adds the execution lag.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .backtest import COLUMNS, Spec

LOOKBACKS = (30, 60, 90, 180)


# ── schedules ────────────────────────────────────────────────────────────────

def calendar_flags(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    """True on the first day of each period: D, W (Monday), M, Q."""
    if freq == "D":
        flags = np.ones(len(index), bool)
    elif freq == "W":
        flags = index.dayofweek == 0
    elif freq == "M":
        flags = index.day == 1
    elif freq == "Q":
        flags = (index.day == 1) & index.month.isin([1, 4, 7, 10])
    else:
        raise ValueError(freq)
    return pd.Series(flags, index=index)


def _never(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(False, index=index)


def _no_band(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.inf, index=index)


def _weights(index: pd.DatetimeIndex, btc, eth, cash=0.0) -> pd.DataFrame:
    df = pd.DataFrame({"BTC": btc, "ETH": eth, "CASH": cash}, index=index)
    return df.loc[:, COLUMNS].astype(float)


def _fixed(index: pd.DatetimeIndex, w_btc: float) -> pd.DataFrame:
    return _weights(index, w_btc, 1.0 - w_btc)


# ── rolling features (pandas twin of sql/features.sql, used for bootstraps) ──

def rolling_features(btc_log: pd.Series, eth_log: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=btc_log.index)
    ann = np.sqrt(365)
    for n in LOOKBACKS:
        out[f"btc_vol_{n}"] = btc_log.rolling(n).std() * ann
        out[f"eth_vol_{n}"] = eth_log.rolling(n).std() * ann
        out[f"corr_{n}"] = btc_log.rolling(n).corr(eth_log)
    return out


# ── rules ────────────────────────────────────────────────────────────────────

def buy_and_hold(panel: pd.DataFrame, w_btc: float = 0.5, name: str | None = None) -> Spec:
    idx = panel.index
    family = {1.0: "BTC only", 0.0: "ETH only"}.get(w_btc, "Buy & hold 50/50")
    return Spec(name or family, family, {}, _fixed(idx, w_btc), _never(idx), _no_band(idx))


def calendar(panel: pd.DataFrame, freq: str) -> Spec:
    idx = panel.index
    return Spec(f"calendar_{freq}", "Calendar rebalance 50/50", {"freq": freq},
                _fixed(idx, 0.5), calendar_flags(idx, freq), _no_band(idx))


def threshold(panel: pd.DataFrame, band: float) -> Spec:
    idx = panel.index
    return Spec(f"band_{band:g}", "Threshold rebalance 50/50", {"band": band},
                _fixed(idx, 0.5), _never(idx), pd.Series(band, index=idx))


def inverse_vol(panel: pd.DataFrame, lookback: int, freq: str) -> Spec:
    """Weight each asset by 1/σ so both contribute similar standalone risk."""
    idx = panel.index
    inv_b = 1.0 / panel[f"btc_vol_{lookback}"]
    inv_e = 1.0 / panel[f"eth_vol_{lookback}"]
    w_btc = inv_b / (inv_b + inv_e)
    return Spec(f"invvol_{lookback}_{freq}", "Inverse volatility",
                {"lookback": lookback, "freq": freq},
                _weights(idx, w_btc, 1.0 - w_btc), calendar_flags(idx, freq), _no_band(idx))


def min_variance(panel: pd.DataFrame, lookback: int, freq: str) -> Spec:
    """Two-asset minimum-variance weight from rolling vols and correlation."""
    idx = panel.index
    sb, se = panel[f"btc_vol_{lookback}"], panel[f"eth_vol_{lookback}"]
    rho = panel[f"corr_{lookback}"]
    cov = rho * sb * se
    w_btc = ((se**2 - cov) / (sb**2 + se**2 - 2 * cov)).clip(0.0, 1.0)
    return Spec(f"minvar_{lookback}_{freq}", "Minimum variance",
                {"lookback": lookback, "freq": freq},
                _weights(idx, w_btc, 1.0 - w_btc), calendar_flags(idx, freq), _no_band(idx))


def vol_target(panel: pd.DataFrame, target_vol: float, lookback: int, freq: str) -> Spec:
    """50/50 BTC/ETH scaled down (into cash) when forecast portfolio vol is
    above `target_vol`. Never levers up: exposure is capped at 100%."""
    idx = panel.index
    sb, se = panel[f"btc_vol_{lookback}"], panel[f"eth_vol_{lookback}"]
    rho = panel[f"corr_{lookback}"]
    port_vol = np.sqrt(0.25 * sb**2 + 0.25 * se**2 + 0.5 * rho * sb * se)
    exposure = (target_vol / port_vol).clip(upper=1.0)
    return Spec(f"voltgt_{target_vol:g}_{lookback}_{freq}", "Volatility target (cash allowed)",
                {"target_vol": target_vol, "lookback": lookback, "freq": freq},
                _weights(idx, 0.5 * exposure, 0.5 * exposure, 1.0 - exposure),
                calendar_flags(idx, freq), pd.Series(0.10, index=idx))


# ── grids searched on the training period ────────────────────────────────────

def benchmarks(panel: pd.DataFrame) -> list[Spec]:
    return [buy_and_hold(panel, 0.5), buy_and_hold(panel, 1.0), buy_and_hold(panel, 0.0)]


def grids(panel: pd.DataFrame) -> dict[str, list[Spec]]:
    return {
        "Calendar rebalance 50/50": [calendar(panel, f) for f in ("D", "W", "M", "Q")],
        "Threshold rebalance 50/50": [threshold(panel, b) for b in (0.025, 0.05, 0.10, 0.15, 0.20)],
        "Inverse volatility": [inverse_vol(panel, n, f) for n, f in product(LOOKBACKS, ("W", "M", "Q"))],
        "Minimum variance": [min_variance(panel, n, f) for n, f in product(LOOKBACKS, ("W", "M", "Q"))],
        "Volatility target (cash allowed)": [
            vol_target(panel, t, n, f)
            for t, n, f in product((0.4, 0.6, 0.8), (30, 60, 90), ("W", "M"))
        ],
    }


BUILDERS = {
    "Calendar rebalance 50/50": calendar,
    "Threshold rebalance 50/50": threshold,
    "Inverse volatility": inverse_vol,
    "Minimum variance": min_variance,
    "Volatility target (cash allowed)": vol_target,
}
HOLD_WEIGHTS = {"Buy & hold 50/50": 0.5, "BTC only": 1.0, "ETH only": 0.0}


def rebuild(spec: Spec, panel: pd.DataFrame) -> Spec:
    """Recreate a strategy with the same rule and parameters on new data."""
    if spec.family in HOLD_WEIGHTS:
        return buy_and_hold(panel, HOLD_WEIGHTS[spec.family])
    return BUILDERS[spec.family](panel, **spec.params)
