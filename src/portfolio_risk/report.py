"""Run the whole study and write tables, figures and the HTML report.

Run:  python -m portfolio_risk.report        (after `python -m portfolio_risk.db`)
"""
from __future__ import annotations

import base64
import html
from datetime import date

import matplotlib
import matplotlib.patheffects
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config, db, metrics  # noqa: E402
from . import evaluate as ev  # noqa: E402
from . import strategies as st  # noqa: E402

TABLES = config.REPORT_DIR / "tables"
FIGURES = config.REPORT_DIR / "figures"

# Reference palette (dataviz skill): fixed categorical order, recessive chrome.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
FAINT = "#d6d5ce"
STROKE = matplotlib.patheffects.withStroke(linewidth=3, foreground=SURFACE)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.3, "legend.frameon": False, "axes.titlelocation": "left",
    "axes.titlesize": 11, "axes.titleweight": "bold",
})


# ── run everything ───────────────────────────────────────────────────────────

def run_all() -> dict:
    study = ev.Study.load()
    train, test = ev.grid_tables(study)
    picks = {m: ev.select(study, train, m) for m in ev.SELECTION_METRICS}

    chosen, picked_by = [], {}
    for m, specs in picks.items():
        for s in specs:
            picked_by.setdefault(s.label(), []).append(ev.SELECTION_METRICS[m])
            if s.label() not in {c.label() for c in chosen}:
                chosen.append(s)

    head, head_results = ev.headline(study, chosen)
    head.insert(0, "picked_by", [", ".join(picked_by.get(i, ["Benchmark"])) for i in head.index])
    train_head = metrics.table([study.train(s) for s in st.benchmarks(study.panel) + chosen])

    wf = {m: ev.walk_forward(study, m) for m in ev.SELECTION_METRICS}
    boot = ev.bootstrap(study, chosen + [st.buy_and_hold(study.panel, 1.0)])
    stress = {
        name: db.query((config.SQL_DIR / "queries" / "stress_correlation.sql").read_text()
                       .replace(":start", f"'{a}'").replace(":end", f"'{b}'"))
        for name, (a, b) in {"train": (study.train_start, config.TRAIN_END),
                             "test": (config.TEST_START, config.END_DATE)}.items()
    }
    return {
        "study": study, "train_grid": train, "test_grid": test, "picks": picks,
        "chosen": chosen, "headline": head, "headline_results": head_results,
        "train_headline": train_head,
        "rank_stability": {m: ev.rank_stability(train, test, m) for m in ev.SELECTION_METRICS},
        "costs": ev.cost_sensitivity(study, chosen),
        "breakeven": pd.Series({s.label(): ev.breakeven_cost(study, s) for s in chosen}, name="breakeven_bps"),
        "lag": ev.lag_sensitivity(study, chosen),
        "walk_forward": wf,
        "bootstrap": boot, "bootstrap_summary": ev.bootstrap_summary(boot),
        "yearly": db.query_file("yearly_summary.sql"),
        "stress": stress,
        "sources": db.query_file("source_check.sql"),
    }


def save_tables(r: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    r["train_grid"].to_csv(TABLES / "train_all_candidates.csv")
    r["test_grid"].to_csv(TABLES / "test_all_candidates.csv")
    r["headline"].to_csv(TABLES / "test_headline.csv")
    r["train_headline"].to_csv(TABLES / "train_headline.csv")
    for m, t in r["rank_stability"].items():
        t.to_csv(TABLES / f"rank_stability_{m}.csv")
    r["costs"].to_csv(TABLES / "cost_sensitivity.csv", index=False)
    r["breakeven"].to_csv(TABLES / "breakeven_cost.csv")
    r["lag"].to_csv(TABLES / "lag_sensitivity.csv", index=False)
    for m, (t, picks, _) in r["walk_forward"].items():
        t.to_csv(TABLES / f"walk_forward_{m}.csv")
        picks.to_csv(TABLES / f"walk_forward_picks_{m}.csv", index=False)
    r["bootstrap_summary"].to_csv(TABLES / "bootstrap_summary.csv")
    r["yearly"].to_csv(TABLES / "yearly_market_summary.csv", index=False)
    for k, t in r["stress"].items():
        t.to_csv(TABLES / f"stress_correlation_{k}.csv", index=False)
    r["sources"].to_csv(TABLES / "source_check.csv", index=False)


# ── figures ──────────────────────────────────────────────────────────────────

def _featured(r: dict) -> list[tuple[str, str]]:
    """(label, colour) for the lines drawn on the headline charts."""
    picks = r["picks"]
    by_family = lambda m, fam: next(s.label() for s in picks[m] if s.family == fam)
    return [
        ("Buy & hold 50/50", INK_2),
        (by_family("calmar", "Calendar rebalance 50/50"), SLOT[0]),
        (by_family("max_drawdown", "Minimum variance"), SLOT[1]),
        (by_family("max_drawdown", "Volatility target (cash allowed)"), SLOT[2]),
    ]


def _short(label: str) -> str:
    return (label.replace("Volatility target (cash allowed)", "Vol target")
                 .replace("Calendar rebalance 50/50", "Calendar 50/50")
                 .replace("Threshold rebalance 50/50", "Threshold 50/50")
                 .replace("target_vol=", "σ*=").replace("lookback=", "L=").replace("freq=", ""))


def _pct_axis(ax, axis="y", decimals=0):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _end_label(ax, series: pd.Series, text: str, color: str, dy: float = 0):
    ax.annotate(text, (series.index[-1], series.iloc[-1]), xytext=(6, dy), textcoords="offset points",
                color=INK_2, fontsize=8.5, va="center", path_effects=[STROKE])


def fig_market(r: dict) -> str:
    p = r["study"].panel
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1, 1]})
    for ax, (col, title) in zip(axes, [("close", "Daily close, USD (log scale)"),
                                       ("vol_90", "Rolling 90-day volatility, annualised"),
                                       ("corr_90", "Rolling 90-day BTC–ETH correlation")]):
        ax.set_title(title)
        if col == "corr_90":
            ax.plot(p.index, p["corr_90"], color=INK_2)
            ax.set_ylim(-0.2, 1)
            continue
        for asset, color in (("btc", SLOT[0]), ("eth", SLOT[1])):
            s = p[f"{asset}_{col}"].dropna()
            ax.plot(s.index, s, color=color, label=asset.upper())
        if col == "close":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
            ax.legend(loc="upper left", ncols=2)
        else:
            _pct_axis(ax)
    for ax in axes:
        ax.axvline(pd.Timestamp(config.TEST_START), color=AXIS, lw=1)
    axes[0].annotate("  out-of-sample period", (pd.Timestamp(config.TEST_START), 1), xycoords=("data", "axes fraction"),
                     va="top", fontsize=8.5, color=INK_2)
    fig.tight_layout()
    return _save(fig, "market.png")


def fig_test_paths(r: dict) -> str:
    res = {x.spec.label(): x for x in r["headline_results"]}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    a1.set_title("Growth of $1 out-of-sample, from Jan 2022, after 25 bps costs")
    a2.set_title("Drawdown from previous peak")
    for i, (label, color) in enumerate(_featured(r)):
        eq = res[label].equity
        a1.plot(eq.index, eq, color=color, label=_short(label))
        a2.plot(eq.index, metrics.drawdown(eq), color=color, lw=1.1)
    a1.axhline(1, color=AXIS, lw=0.8)
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    a1.legend(loc="upper left")
    _pct_axis(a2)
    fig.tight_layout()
    return _save(fig, "test_paths.png")


def fig_risk_return(r: dict) -> str:
    test = r["test_grid"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, m in zip(axes, ev.SELECTION_METRICS):
        ax.set_title(f"Highlighted: picked by {ev.SELECTION_METRICS[m][0].lower() + ev.SELECTION_METRICS[m][1:]}", fontsize=10)
        ax.scatter(test.max_drawdown, test.cagr, s=26, color=FAINT, label="all candidates", zorder=2)
        chosen = [s.label() for s in r["picks"][m]]
        ax.scatter(test.loc[chosen, "max_drawdown"], test.loc[chosen, "cagr"], s=60, color=SLOT[0],
                   edgecolor=SURFACE, linewidth=2, zorder=4, label="picked on training data")
        crowd = []
        for lab in chosen:
            x, y = test.loc[lab, "max_drawdown"], test.loc[lab, "cagr"]
            if abs(x - test.loc["Buy & hold 50/50", "max_drawdown"]) < 0.02 and abs(y - test.loc["Buy & hold 50/50", "cagr"]) < 0.02:
                crowd.append(_short(lab).split(" (")[0])
                continue
            ax.annotate(_short(lab), (x, y), xytext=(0, 9), textcoords="offset points",
                        ha="right" if x > -0.6 else "center",
                        fontsize=8, color=INK, path_effects=[STROKE])
        for lab, mk in (("Buy & hold 50/50", "s"), ("BTC only", "^"), ("ETH only", "v")):
            ax.scatter(test.loc[lab, "max_drawdown"], test.loc[lab, "cagr"], s=55, marker=mk,
                       color=INK_2, edgecolor=SURFACE, linewidth=1.5, zorder=5)
            ax.annotate(lab, (test.loc[lab, "max_drawdown"], test.loc[lab, "cagr"]),
                        xytext=(-8, 0), textcoords="offset points", fontsize=8, color=INK_2,
                        ha="right", va="center", path_effects=[STROKE])
        if crowd:
            x, y = test.loc["Buy & hold 50/50", ["max_drawdown", "cagr"]]
            ax.annotate("\n".join(crowd), (x, y), xytext=(28, -34), textcoords="offset points", fontsize=8,
                        color=INK, path_effects=[STROKE],
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=6))
        _pct_axis(ax, "x"); _pct_axis(ax)
        ax.set_xlabel("Max drawdown (right = shallower)")
    axes[0].set_ylabel("CAGR, out-of-sample")
    axes[1].legend(loc="lower right", fontsize=8)
    for ax in axes:
        ax.set_xlim(test.max_drawdown.min() - 0.02, test.max_drawdown.max() + 0.02)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    fig.suptitle("Out-of-sample return vs drawdown for every candidate rule", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, "risk_return.png")


def fig_costs(r: dict) -> str:
    c = r["costs"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_title("CAGR minus buy-and-hold CAGR, out-of-sample, by trading cost")
    ax.axhline(0, color=AXIS, lw=1)
    colors = dict(_featured(r))
    for lab, g in c.groupby("strategy", sort=False):
        g = g.set_index("cost_bps")["cagr_vs_hold"]
        color = colors.get(lab, FAINT)
        ax.plot(g.index, g, marker="o", ms=4, color=color, zorder=3 if lab in colors else 2,
                label=_short(lab) if lab in colors else None)
    ax.plot([], [], color=FAINT, marker="o", ms=4, label="other picked rules")
    _pct_axis(ax, decimals=1)
    ax.set_xlabel("Cost per unit traded (bps)")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    return _save(fig, "costs.png")


def fig_weights(r: dict) -> str:
    res = {x.spec.label(): x for x in r["headline_results"]}
    feat = _featured(r)
    fig, axes = plt.subplots(len(feat), 1, figsize=(9, 6.4), sharex=True, sharey=True)
    for ax, (lab, color) in zip(axes, feat):
        w = res[lab].weights
        ax.stackplot(w.index, w.BTC, w.ETH, w.CASH, colors=[SLOT[0], SLOT[1], FAINT], lw=0)
        ax.set_title(_short(lab), fontsize=9.5)
        ax.set_ylim(0, 1); _pct_axis(ax); ax.grid(False)
    axes[0].legend(["BTC", "ETH", "Cash"], ncols=3, loc="lower left", fontsize=8,
                   bbox_to_anchor=(0, 1.25))
    fig.suptitle("Portfolio weights, out-of-sample", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, "weights.png")


def fig_bootstrap(r: dict) -> str:
    b = r["bootstrap"]
    b = b[b.family != "Buy & hold 50/50"]
    labs = list(dict.fromkeys(b.strategy))
    fig, axes = plt.subplots(len(labs), 1, figsize=(8, 1.15 * len(labs) + 0.8), sharex=True)
    lo, hi = b.dd_improvement.quantile([0.005, 0.995])
    bins = np.linspace(lo, hi, 50)
    for ax, lab in zip(axes, labs):
        x = b.loc[b.strategy == lab, "dd_improvement"]
        ax.hist(x, bins=bins, color=INK_2, rwidth=0.85)
        ax.axvline(0, color=INK_2, lw=1)
        ax.set_yticks([]); ax.grid(False); ax.spines["left"].set_visible(False)
        ax.text(0, 0.95, f"{_short(lab)}   P(shallower) = {(x > 0).mean():.0%}", transform=ax.transAxes,
                fontsize=8.5, va="top", color=INK, path_effects=[STROKE])
    _pct_axis(axes[-1], "x")
    axes[-1].set_xlabel("Max drawdown improvement vs buy-and-hold (percentage points; right = better)")
    fig.suptitle(f"Block bootstrap of the out-of-sample period ({ev.BOOTSTRAP_RUNS} paths)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, "bootstrap.png")


def _save(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=160)
    plt.close(fig)
    # Inline the PNG so report.html is a single self-contained file to share.
    return "data:image/png;base64," + base64.b64encode((FIGURES / name).read_bytes()).decode()


# ── HTML ─────────────────────────────────────────────────────────────────────

PCT = {"cagr", "ann_vol", "max_drawdown", "cvar_95", "worst_month", "cost_drag_per_year",
       "avg_btc_weight", "avg_cash_weight", "total_return", "worst_drawdown", "dd_improvement_median",
       "dd_improvement_p05", "dd_improvement_p95", "p_shallower_drawdown", "p_higher_cagr",
       "cagr_diff_median", "cagr_diff_p05", "cagr_diff_p95", "cagr_vs_hold", "drawdown_vs_hold",
       "mean_abs_gap", "max_abs_gap", "share_days_gap_over_1pct", "avg_btc", "avg_eth", "same_direction"}
NAMES = {
    "cagr": "CAGR", "ann_vol": "Volatility", "sharpe": "Sharpe", "sortino": "Sortino",
    "max_drawdown": "Max drawdown", "calmar": "Calmar", "cvar_95": "CVaR 95% (daily)",
    "worst_month": "Worst month", "longest_underwater_days": "Longest underwater (days)",
    "turnover_per_year": "Turnover / yr", "trades_per_year": "Trades / yr",
    "cost_drag_per_year": "Cost drag / yr", "avg_btc_weight": "Avg BTC weight",
    "avg_cash_weight": "Avg cash", "picked_by": "Picked by",
    "p_shallower_drawdown": "P(shallower drawdown)", "dd_improvement_median": "Drawdown gain, median (pp)",
    "dd_improvement_p05": "Drawdown gain, 5th pct", "dd_improvement_p95": "Drawdown gain, 95th pct",
    "p_higher_cagr": "P(higher CAGR)", "cagr_diff_median": "CAGR gain, median", "cagr_diff_p05": "CAGR gain, 5th pct",
    "cagr_diff_p95": "CAGR gain, 95th pct", "spearman_train_vs_test": "Spearman rank corr, train vs test",
    "chosen_test_rank": "Chosen rule's test rank", "avg_btc": "Avg BTC return", "avg_eth": "Avg ETH return",
    "same_direction": "ETH moved same way", "avg_corr_90": "Avg 90-day corr",
    "cagr_lag0": "CAGR, same close", "cagr_lag1": "CAGR, next close",
    "max_drawdown_lag0": "Max drawdown, same close", "max_drawdown_lag1": "Max drawdown, next close",
}


def _fmt(col: str, v) -> str:
    if isinstance(v, str):
        return html.escape(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, float) and np.isinf(v):
        return "∞"
    if col in PCT:
        return f"{v:.1%}"
    if col in {"longest_underwater_days", "days", "overlapping_days", "candidates", "chosen_test_rank"}:
        return f"{int(v):,d}"
    if col.startswith("Break-even"):
        return f"{v:,.0f}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.2f}"
    return html.escape(str(v))


def _table(df: pd.DataFrame, cols: list[str] | None = None, index=True, short=True) -> str:
    df = df if cols is None else df[cols]
    head = "".join(f"<th>{html.escape(NAMES.get(c, c.replace('_', ' ')))}</th>" for c in df.columns)
    rows = []
    for i, row in df.iterrows():
        name = _short(str(i)) if short else str(i)
        cells = "".join(f"<td>{_fmt(c, row[c])}</td>" for c in df.columns)
        rows.append(f"<tr><th scope='row'>{html.escape(name)}</th>{cells}</tr>" if index else f"<tr>{cells}</tr>")
    first = "<th></th>" if index else ""
    return f"<div class='scroll'><table><thead><tr>{first}{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def write_html(r: dict, figs: dict) -> None:
    h = r["headline"]
    hold = h.loc["Buy & hold 50/50"]
    btc = h.loc["BTC only"]
    vt = next(s.label() for s in r["picks"]["max_drawdown"] if s.family.startswith("Volatility"))
    vt_cal = next(s.label() for s in r["picks"]["calmar"] if s.family.startswith("Volatility"))
    mv = next(s.label() for s in r["picks"]["max_drawdown"] if s.family == "Minimum variance")
    cal = next(s.label() for s in r["picks"]["calmar"] if s.family.startswith("Calendar"))
    rebal = h[h.family.isin(["Calendar rebalance 50/50", "Threshold rebalance 50/50", "Inverse volatility"])]
    stress_test = r["stress"]["test"].set_index("regime")
    bs = r["bootstrap_summary"]
    wf_dd = r["walk_forward"]["max_drawdown"][0]
    wf_vt = next(i for i in wf_dd.index if i.startswith("Volatility"))
    rs = r["rank_stability"]["calmar"]

    main_cols = ["picked_by", "cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "cvar_95",
                 "longest_underwater_days", "turnover_per_year", "cost_drag_per_year",
                 "avg_btc_weight", "avg_cash_weight"]
    short_cols = ["cagr", "sharpe", "max_drawdown", "calmar", "turnover_per_year"]

    cost_tbl = r["costs"].pivot_table(index="strategy", columns="cost_bps", values="cagr_vs_hold", sort=False)
    cost_tbl.columns = [f"{c} bps" for c in cost_tbl.columns]
    for c in cost_tbl.columns:
        PCT.add(c)
    be = r["breakeven"].rename("Break-even cost (bps)").round(0).to_frame()

    lag = r["lag"].pivot_table(index="strategy", columns="lag_days", values=["cagr", "max_drawdown"], sort=False)
    lag.columns = [f"{a}_lag{b}" for a, b in lag.columns]
    PCT.update(lag.columns)

    yearly = r["yearly"].pivot(index="year", columns="asset", values=["total_return", "ann_vol", "worst_drawdown"])
    nice = {"total_return": "total return", "ann_vol": "volatility", "worst_drawdown": "worst drawdown from ATH"}
    yearly.columns = [f"{b} {nice[a]}" for a, b in yearly.columns]
    yearly["avg_corr_90"] = r["yearly"].groupby("year")["avg_corr_90"].first()
    PCT.update(c for c in yearly.columns if c != "avg_corr_90")
    stress_view = r["stress"]["test"].set_index("regime")
    stress_view.index = [i.split(". ", 1)[1] for i in stress_view.index]

    picks_wf = r["walk_forward"]["max_drawdown"][1].pivot(index="year", columns="family", values="chosen")
    picks_wf = picks_wf.apply(lambda col: col.map(lambda s: s.split("(", 1)[-1].rstrip(")") if "(" in s else s))

    dd = lambda x: f"{abs(x):.0%}"
    lag_w = r["lag"].pivot_table(index="strategy", columns="lag_days", values=["cagr", "max_drawdown"])
    lag_cagr = (lag_w["cagr"][0] - lag_w["cagr"][1]).abs().max()
    lag_dd = (lag_w["max_drawdown"][0] - lag_w["max_drawdown"][1]).abs().max()
    vt_cfg = r["picks"]["max_drawdown"][[s.family for s in r["picks"]["max_drawdown"]].index("Volatility target (cash allowed)")].params
    vt_cal_cfg = r["picks"]["calmar"][[s.family for s in r["picks"]["calmar"]].index("Volatility target (cash allowed)")].params
    freq_word = {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly"}
    findings = f"""
<ol class="findings">
<li><b>Rebalancing between BTC and ETH did not reduce drawdowns.</b> Out of sample, buy-and-hold 50/50
fell {dd(hold.max_drawdown)} from peak to trough. The calendar, threshold and inverse-volatility rules all fell between
{dd(rebal.max_drawdown.max())} and {dd(rebal.max_drawdown.min())}, within about one percentage point of it. The two assets fell together:
on days BTC dropped 5% or more, ETH also fell {stress_test.loc['1. BTC down 5%+', 'same_direction']:.0%} of the time, and fell
further on average ({stress_test.loc['1. BTC down 5%+', 'avg_eth']:.1%} vs {stress_test.loc['1. BTC down 5%+', 'avg_btc']:.1%}).</li>
<li><b>Weighting by volatility mostly meant holding more BTC.</b> ETH was the more volatile asset almost throughout,
so minimum variance held about {h.loc[mv, 'avg_btc_weight']:.0%} BTC on average. Its {dd(h.loc[mv, 'max_drawdown'])} drawdown
is about the same as holding only BTC ({dd(btc.max_drawdown)}). It beat buy-and-hold ({h.loc[mv, 'cagr']:.1%} vs {hold.cagr:.1%} CAGR)
because BTC beat ETH in this period, not because it diversified better.</li>
<li><b>Only holding cash cut drawdowns materially, and how the parameters were picked decided whether it did.</b>
Picked for the shallowest training drawdown, volatility targeting (a {vt_cfg['target_vol']:.0%} target, {vt_cfg['lookback']}-day lookback,
{freq_word[vt_cfg['freq']]} rebalancing) fell {dd(h.loc[vt, 'max_drawdown'])}, against {dd(hold.max_drawdown)} for buy-and-hold, holding
{h.loc[vt, 'avg_cash_weight']:.0%} cash on average. Picked for the best training Calmar ratio, the same rule chose {'an' if str(round(vt_cal_cfg['target_vol']*100)).startswith(('8', '11', '18')) else 'a'}
{vt_cal_cfg['target_vol']:.0%} target. Volatility after 2022 rarely exceeded that target, so the rule held almost no cash
({h.loc[vt_cal, 'avg_cash_weight']:.1%}) and fell {dd(h.loc[vt_cal, 'max_drawdown'])}, like buy-and-hold. The training period
rewarded staying fully invested, because 2017 and 2020–21 were huge rallies.</li>
<li><b>The drawdown reduction from holding cash is robust; the return gain is not.</b> In {ev.BOOTSTRAP_RUNS} block-bootstrapped
versions of the test period, the drawdown-picked volatility target had a shallower drawdown than buy-and-hold in
{bs.loc[vt, 'p_shallower_drawdown']:.0%} of paths (median {bs.loc[vt, 'dd_improvement_median']*100:.0f} points shallower). It had a higher
CAGR in only {bs.loc[vt, 'p_higher_cagr']:.0%} of them, so its {h.loc[vt, 'cagr']:.1%} vs {hold.cagr:.1%} CAGR in the actual test period was
partly luck of timing. Over the longer walk-forward test from 2019, where parameters were re-picked each January, it fell
{dd(wf_dd.loc[wf_vt, 'max_drawdown'])} vs {dd(wf_dd.loc['Buy & hold 50/50', 'max_drawdown'])}, but returned
{wf_dd.loc[wf_vt, 'cagr']:.0%} a year against {wf_dd.loc['Buy & hold 50/50', 'cagr']:.0%}. That is the trade-off: shallower falls, smaller rallies.
Quarterly 50/50 rebalancing had a shallower drawdown in only {bs.loc[cal, 'p_shallower_drawdown']:.0%} of bootstrap paths.</li>
<li><b>Parameter rankings did not carry over from training to testing.</b> For volatility targeting, the Spearman rank
correlation of Calmar between training and test was {rs.loc['Volatility target (cash allowed)', 'spearman_train_vs_test']:.2f},
so the order almost reversed. For inverse volatility it was {rs.loc['Inverse volatility', 'spearman_train_vs_test']:.2f}. Fine-tuning
lookbacks and rebalancing frequencies mostly fits noise. The choice of rule matters far more than its settings.</li>
<li><b>Trading costs were small next to these effects.</b> The 50/50 rebalancing rules traded
{rebal.turnover_per_year.min():.1f}–{rebal.turnover_per_year.max():.1f}× the portfolio a year, costing at most
{rebal.cost_drag_per_year.max():.2%} a year at 25 bps. They trailed buy-and-hold even when trading was free. Minimum variance
and the drawdown-picked volatility target stayed ahead of buy-and-hold up to about {r['breakeven'][mv]:.0f} and
{r['breakeven'][vt]:.0f} bps per unit traded. Trading at the signal's own close instead of the next day's changed CAGR
by at most {lag_cagr * 100:.1f} points and max drawdown by at most {lag_dd * 100:.1f} points.</li>
</ol>"""

    body = f"""
<header>
<p class="eyebrow">Portfolio risk study · Python, SQL · data to {config.END_DATE}</p>
<h1>Does rebalancing make a BTC/ETH portfolio safer?</h1>
<p class="lede">This study tests whether rebalancing, or weighting by volatility, reduces the drawdowns of a two-asset
BTC/ETH portfolio compared with buying and holding, and what it costs in return and trading. Rules were tuned
on {r['study'].train_start[:7]} to {config.TRAIN_END[:7]} and then judged once on {config.TEST_START[:7]} to {config.END_DATE[:7]}.</p>
</header>

<section class="tiles">
<div class="tile"><span>Buy & hold 50/50, max drawdown</span><b>{hold.max_drawdown:.0%}</b></div>
<div class="tile"><span>Best pure rebalancing rule</span><b>{rebal.max_drawdown.max():.0%}</b></div>
<div class="tile"><span>Volatility target, drawdown-picked</span><b>{h.loc[vt, 'max_drawdown']:.0%}</b></div>
<div class="tile"><span>90-day correlation, test median</span><b>{r['study'].panel.loc[config.TEST_START:, 'corr_90'].median():.2f}</b></div>
</section>

<h2>Findings</h2>
{findings}

<h2>Out-of-sample results ({config.TEST_START} → {config.END_DATE})</h2>
<p>Benchmarks, plus each rule family's parameters as picked on training data by each of the two selection criteria.
All results are after 25 bps per unit traded, with a one-day execution lag.</p>
{_table(h, main_cols)}
<figure><img src="{figs['paths']}" alt="Out-of-sample equity and drawdown curves"></figure>
<figure><img src="{figs['weights']}" alt="Portfolio weights over time"></figure>
<figure><img src="{figs['risk']}" alt="Return versus drawdown scatter for all candidates"></figure>

<h2>Market context</h2>
<figure><img src="{figs['market']}" alt="Prices, rolling volatility and rolling correlation"></figure>
<h3>Calendar years (from <code>sql/queries/yearly_summary.sql</code>)</h3>
<p>The worst drawdown is measured from the all-time high, so it can carry over from a peak set in an earlier year.</p>
{_table(yearly, short=False)}
<h3>Co-movement by market move (from <code>sql/queries/stress_correlation.sql</code>), test period</h3>
<p>Compare the "BTC down 5%+" and "BTC up 5%+" rows with each other, not with "All days": splitting returns into
buckets shrinks the correlation within every bucket.</p>
{_table(stress_view, short=False)}

<h2>Stress tests</h2>
<h3>Trading-cost sensitivity: CAGR minus buy-and-hold CAGR</h3>
{_table(cost_tbl)}
<figure><img src="{figs['costs']}" alt="Cost sensitivity"></figure>
<h3>Break-even cost (the cost at which a rule's CAGR falls to buy-and-hold's)</h3>
<p>"–" means the rule trails buy-and-hold even when trading is free.</p>
{_table(be)}
<h3>Execution lag: trading at the signal's own close (0) vs the next close (1)</h3>
{_table(lag)}
<h3>Block bootstrap of the test period</h3>
<p>Each of the {ev.BOOTSTRAP_RUNS} paths resamples the test period's joint BTC/ETH daily returns in blocks averaging
{ev.BOOTSTRAP_BLOCK_DAYS} days. This keeps crashes and their co-movement intact. Every signal is then rebuilt from
the resampled path. The bootstrap tests whether a result depends on the order of events. It cannot test a change of
regime, because every path reuses the test period's own returns.</p>
{_table(bs, ["p_shallower_drawdown", "dd_improvement_median", "dd_improvement_p05", "dd_improvement_p95",
             "p_higher_cagr", "cagr_diff_median", "cagr_diff_p05", "cagr_diff_p95"])}
<figure><img src="{figs['boot']}" alt="Bootstrap distributions"></figure>
<h3>Walk-forward: parameters re-picked every January, 2019 → {config.END_DATE[:4]}</h3>
<p>Each January, every family's parameters were re-picked on all data up to the previous 31 December, then held
for the year. The period runs longer than the fixed split and includes the 2019–21 cycle.</p>
<p><b>Picked for shallowest drawdown</b></p>
{_table(wf_dd, short_cols)}
<p><b>Picked for best Calmar</b></p>
{_table(r['walk_forward']['calmar'][0], short_cols)}
<p>Parameters picked each year (shallowest-drawdown criterion):</p>
{_table(picks_wf, short=False)}
<h3>Did training rankings survive to the test period? (Calmar)</h3>
{_table(rs, ["candidates", "spearman_train_vs_test", "chosen_test_rank"], short=False)}

<h2>Method</h2>
<ul>
<li><b>Question.</b> Do rebalancing and volatility-based weighting reduce a BTC/ETH portfolio's drawdowns compared with
buy-and-hold, and what do they give up in return and turnover?</li>
<li><b>Data.</b> Daily USD candles (00:00 UTC close) from Coinbase Exchange, {config.START_DATE} → {config.END_DATE}.
Two missing ETH days in May 2016, inside the warm-up period, carry the previous close forward. The Coinbase closes were
cross-checked against Bitstamp: the daily-return correlation was {r['sources'].return_correlation.min():.3f} or higher, and
the average absolute price gap was {r['sources'].mean_abs_gap.max():.2%} or less. The largest gaps were in the December 2017 peak.</li>
<li><b>SQL layer.</b> SQLite builds the gap-filled calendar, the returns, the rolling 30/60/90/180-day volatility and correlation
(window aggregates), the drawdowns and the summary queries. A test checks the SQL statistics against pandas to 1e-10.</li>
<li><b>Rules.</b> Buy-and-hold 50/50 (weights drift); calendar rebalancing to 50/50 (daily, weekly, monthly, quarterly);
threshold rebalancing when BTC's weight drifts beyond a band (2.5–20%); inverse volatility; two-asset minimum variance
(clipped to 0–100%); and a 50/50 volatility target that moves into cash when forecast volatility is above target (no leverage, cash earns 0%).
In total {len(r['train_grid']) - 3} candidates across 5 families.</li>
<li><b>No look-ahead.</b> Signals at close t use data up to close t, and trades fill at close t+1. A test changes the
future returns and checks that no earlier result moves; the check also catches a rule that deliberately peeks ahead.</li>
<li><b>Out-of-sample.</b> {config.WARMUP_DAYS}-day warm-up, then training {r['study'].train_start} → {config.TRAIN_END}
and testing {config.TEST_START} → {config.END_DATE}. Parameters were chosen on training data only, then run once on the test period.</li>
<li><b>Costs and turnover.</b> Turnover is the sum of |Δweight| across BTC and ETH, so swapping 10% of the portfolio
from ETH to BTC counts as 0.2. The cost is that turnover times the bps rate. The opening allocation is free for every rule.</li>
<li><b>Annualisation.</b> 365 days a year. Sharpe uses a 0% risk-free rate.</li>
</ul>
<p class="caveat">This is historical research on two assets over one decade, not investment advice. Ten years of
crypto includes only about three full boom-and-bust cycles, so every conclusion here rests on a small number of crashes.</p>
<footer>Generated {date.today().isoformat()} by <code>make report</code>. Tables as CSV in <code>reports/tables/</code>.</footer>
"""
    css = """
:root { color-scheme: light; --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --grid:#e1e0d9; --accent:#2a78d6; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
       font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 40px 16px 64px; background: var(--surface); }
h1 { font-size: 30px; line-height: 1.2; margin: 4px 0 12px; text-wrap: balance; }
h2 { font-size: 20px; margin: 44px 0 10px; padding-top: 16px; border-top: 1px solid var(--grid); }
h3 { font-size: 15px; margin: 28px 0 8px; }
.eyebrow { color: var(--muted); font-size: 13px; margin: 0; }
.lede { color: var(--ink2); font-size: 16px; max-width: 70ch; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr)); gap: 12px; margin: 24px 0; }
.tile { border: 1px solid var(--grid); border-radius: 8px; padding: 14px 16px; }
.tile span { display:block; color: var(--ink2); font-size: 13px; }
.tile b { font-size: 28px; font-weight: 600; }
.findings li { margin-bottom: 10px; max-width: 78ch; }
.scroll { overflow-x: auto; margin: 8px 0 16px; }
table { border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; width: 100%; }
th, td { padding: 6px 8px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
thead th { color: var(--ink2); font-weight: 600; vertical-align: bottom; white-space: normal; min-width: 64px; }
tbody th { text-align: left; font-weight: 500; }
figure { margin: 16px 0 24px; }
img { max-width: 100%; height: auto; display: block; }
code { font-size: 12.5px; background: #f0efec; padding: 1px 4px; border-radius: 3px; }
.caveat { color: var(--ink2); font-size: 13.5px; border-left: 3px solid var(--grid); padding-left: 12px; }
footer { margin-top: 40px; color: var(--muted); font-size: 12.5px; }
"""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC/ETH Rebalancing Study</title><style>{css}</style></head>
<body><main>{body}</main></body></html>"""
    (config.REPORT_DIR / "report.html").write_text(page)


def main() -> None:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    r = run_all()
    save_tables(r)
    figs = {"market": fig_market(r), "paths": fig_test_paths(r), "risk": fig_risk_return(r),
            "costs": fig_costs(r), "weights": fig_weights(r), "boot": fig_bootstrap(r)}
    write_html(r, figs)
    pd.set_option("display.width", 220)
    print(r["headline"][["picked_by", "cagr", "max_drawdown", "calmar", "turnover_per_year"]].round(3).to_string())
    print(f"\nwrote {config.REPORT_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
