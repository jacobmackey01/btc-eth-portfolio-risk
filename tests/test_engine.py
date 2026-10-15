"""Engine and signal checks on small synthetic data, plus the real database."""
import numpy as np
import pandas as pd
import pytest

from portfolio_risk import backtest as bt
from portfolio_risk import config, metrics
from portfolio_risk import strategies as st


def make_panel(btc, eth, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(btc), freq="D")
    p = pd.DataFrame({"btc_ret": btc, "eth_ret": eth}, index=idx)
    p["btc_log"], p["eth_log"] = np.log1p(p.btc_ret), np.log1p(p.eth_ret)
    return p.join(st.rolling_features(p.btc_log, p.eth_log))


def random_panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2)) @ np.linalg.cholesky([[1, 0.8], [0.8, 1]]).T
    return make_panel(0.03 * z[:, 0], 0.04 * z[:, 1])


def window(p, warm=200):
    return str(p.index[warm].date()), str(p.index[-1].date())


def test_buy_and_hold_matches_closed_form():
    p = random_panel()
    start, end = window(p)
    res = bt.run(st.buy_and_hold(p), bt.asset_returns(p), start, end, cost_bps=50)
    grow = (1 + p.loc[start:end, ["btc_ret", "eth_ret"]]).prod()
    assert res.equity.iloc[-1] == pytest.approx(0.5 * grow.btc_ret + 0.5 * grow.eth_ret)
    assert res.turnover.sum() == 0 and res.costs.sum() == 0


def test_daily_rebalance_without_costs_or_lag_matches_fixed_mix():
    p = random_panel()
    start, end = window(p)
    res = bt.run(st.calendar(p, "D"), bt.asset_returns(p), start, end, cost_bps=0, lag=0)
    mix = 0.5 * p.btc_ret + 0.5 * p.eth_ret
    assert res.equity.iloc[-1] == pytest.approx((1 + mix.loc[start:end]).prod())


def test_costs_are_charged_on_traded_notional():
    # BTC doubles on day 1 then nothing moves: 50/50 drifts to 2/3–1/3,
    # one trade of |Δw| = 1/6 + 1/6 = 1/3 brings it back.
    btc = np.zeros(10); btc[5] = 1.0
    p = make_panel(btc, np.zeros(10))
    spec = st.threshold(p, 0.05)
    res = bt.run(spec, bt.asset_returns(p), str(p.index[5].date()), str(p.index[-1].date()),
                 cost_bps=100, lag=0)
    assert res.turnover.sum() == pytest.approx(1 / 3)
    assert res.equity.iloc[-1] == pytest.approx(1.5 * (1 - 0.01 / 3))


def test_lag_delays_the_trade_by_one_day():
    btc = np.zeros(10); btc[5] = 1.0
    p = make_panel(btc, np.zeros(10))
    spec = st.threshold(p, 0.05)
    rets = bt.asset_returns(p)
    start, end = str(p.index[5].date()), str(p.index[-1].date())
    now = bt.run(spec, rets, start, end, cost_bps=0, lag=0)
    later = bt.run(spec, rets, start, end, cost_bps=0, lag=1)
    assert now.turnover.iloc[0] > 0 and later.turnover.iloc[0] == 0
    assert later.turnover.iloc[1] == pytest.approx(now.turnover.iloc[0])


def test_no_lookahead_future_shock_does_not_change_the_past():
    """Changing returns after day k must leave every result up to day k
    untouched, for every strategy family."""
    p = random_panel(seed=1)
    shocked = p[["btc_ret", "eth_ret"]].copy()
    k = 320
    shocked.iloc[k:] *= -3.0
    shocked = shocked.clip(lower=-0.9)
    q = make_panel(shocked.btc_ret.to_numpy(), shocked.eth_ret.to_numpy())
    start, end = window(p)
    cut = p.index[k - 1]
    specs = st.benchmarks(p) + [g[0] for g in st.grids(p).values()] + [g[-1] for g in st.grids(p).values()]
    for spec in specs:
        a = bt.run(spec, bt.asset_returns(p), start, end)
        b = bt.run(st.rebuild(spec, q), bt.asset_returns(q), start, end)
        pd.testing.assert_series_equal(a.equity[:cut], b.equity[:cut], check_names=False)


def test_weights_are_long_only_and_sum_to_one():
    p = random_panel()
    start, end = window(p)
    for specs in st.grids(p).values():
        for spec in specs:
            w = bt.run(spec, bt.asset_returns(p), start, end).weights
            assert (w >= -1e-12).all().all()
            assert np.allclose(w.sum(axis=1), 1.0)


def test_inverse_vol_tilts_to_the_calmer_asset():
    p = random_panel()  # ETH is built 4/3 as volatile as BTC
    w = st.inverse_vol(p, 90, "M").target["BTC"].dropna()
    assert (w > 0.5).mean() > 0.95


def test_vol_target_never_levers_and_holds_cash_when_vol_is_high():
    p = make_panel(np.r_[np.full(200, 0.001), np.tile([0.08, -0.08], 100)],
                   np.r_[np.full(200, 0.001), np.tile([0.08, -0.08], 100)])
    t = st.vol_target(p, 0.4, 30, "M").target.dropna()
    assert (t[["BTC", "ETH"]].sum(axis=1) <= 1 + 1e-12).all()
    assert t["CASH"].iloc[-1] > 0.5


def test_max_drawdown_and_underwater():
    eq = pd.Series([1, 2, 1, 1.5, 2.5, 2], index=pd.date_range("2020", periods=6))
    assert metrics.max_drawdown(eq) == pytest.approx(-0.5)
    assert metrics.longest_underwater_days(eq) == 2


@pytest.mark.skipif(not config.DB_PATH.exists(), reason="run `make db` first")
def test_sql_rolling_stats_match_pandas():
    from portfolio_risk import db
    p = db.load_panel()
    ref = st.rolling_features(p.btc_log, p.eth_log)
    for col in ref.columns:
        both = pd.concat([p[col], ref[col]], axis=1).dropna()
        assert len(both) > 3000
        assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-10), col


@pytest.mark.skipif(not config.DB_PATH.exists(), reason="run `make db` first")
def test_calendar_is_complete_and_gaps_are_flagged():
    from portfolio_risk import db
    p = db.load_panel()
    assert (p.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert p.index[-1] == pd.Timestamp(config.END_DATE)
    assert p.any_filled.sum() <= 5
